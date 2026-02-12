import time
import threading
import numpy as np
from mpi4py import MPI
# import matplotlib.pyplot as plt

from src.agent.planning.state_inference import timed_infer_state
from src.game.game import Game
from src.utils import get_chunk, custom_round, logsumexp
import os

epsilon = 0.01
size = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

def get_avatar_pos(obs_state):
    for col in obs_state:
        for cell in col:
            for obj in cell:
                if obj['name'] == 'avatar':
                    return obj['pos']

    return None


class LikelihoodComputer:
    """
    Class for computing the likelihood given a particular game description
    """

    def __init__(self, comm_engine, params, time_tracker):
        self.params = params
        self.comm_engine = comm_engine                  # LLM wrapper
        self.time_tracker = time_tracker
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))

    # Computes likelihood by simulating
    def compute_likelihood(self, theories, generative_model, data, save_img=False, debug=False):

        episodes, stepss, traj_episodes, obj_episodes, _, event_episodes, data_indexes, linguistic_data = data

        if size > 1: theories = comm.bcast(theories, root=0)
        if not isinstance(theories, list):
            theories = [theories]

        if self.params['exp_params']['use_interaction_likelihood']:

            n_simulations = self.params['agent']['thinker']['n_simulations_likelihood']

            obj_names = sorted(theories[0].names)
            # concatenated_obs_event_episodes = []
            # concatenated_sim_event_episodes = [[] for _ in range(n_simulations)]
            # concatenated_obs_imgs = []
            # concatenated_triggered_event_episodes = []
            # all_imgs = []
            # scatter jobs across workers
            # n_particles x n_simulations simulations in total

            # Dispatch simulation to different CPUs
            indexes = [(i, j) for i in range(len(theories)) for j in range(n_simulations)]
            indexes_here = get_chunk(indexes, size, rank)

            game_dict = dict()
            results = dict()

            # Create games based on each theory
            theory_ids = sorted(set([idx[0] for idx in indexes_here]))
            for theory_id in theory_ids:
                game = Game(self.params, rules=theories[theory_id])
                game_dict[theory_id] = game
                results[theory_id] = dict(obs_events=[], sim_events=dict(), obs_imgs=[], sim_imgs=dict(), actions=[])

                simulation_ids = [idx[1] for idx in indexes_here if idx[0] == theory_id]
                n_sims_here = len(simulation_ids)
                eps_to_care_about = [idx[0] for idx in data_indexes]
                t_init_all = time.time()
                t_reset, t_infer, t_simulate, t_convert = [], [], [], []
                skip_this_theory = False
                for i_ep, obs_traj_episode, obs_obj_episode, obs_event_episode in zip(range(len(traj_episodes)), traj_episodes, obj_episodes, event_episodes):

                    if i_ep in eps_to_care_about:
                        t_name = time.time()
                        last_step_to_care_about = 0
                        for datapoint in data_indexes:
                            if datapoint[0] == i_ep and datapoint[1] > last_step_to_care_about:
                                last_step_to_care_about = datapoint[1]
                        last_step_to_care_about += 1  # we want to care about the next state to compute win/lose
                        obs_state_episode = obs_traj_episode['state'][:last_step_to_care_about + 1]
                        actions = obs_traj_episode['action'][:last_step_to_care_about + 1]
                        steps = obs_traj_episode['step'][: last_step_to_care_about + 1]

                        lvl = obs_traj_episode['lvl'][0]
                        game_format = self.params['true_game_info']['formats'][lvl]

                        # Option to save the visual image of each transition / observation
                        if save_img:
                            all_observed_imgs = obs_traj_episode['img']
                            observed_imgs = [(obs_img_prev, obs_img_post) for obs_img_prev, obs_img_post in zip(all_observed_imgs[:-1], all_observed_imgs[1:])]
                            simulated_imgs = [[] for _ in range(n_sims_here)]

                        # Reset game to current level    
                        t_init = time.time()
                        # Infer hidden states for the whole trajectory
                        self.time_tracker.tic('likelihood_infer_hidden')
                        current_episode = dict(traj=obs_traj_episode, objs=obs_obj_episode)
                        if not skip_this_theory:
                            try:
                                game, hidden_states = timed_infer_state(lvl, game, current_episode, n_simulations=n_sims_here, last_step=last_step_to_care_about)
                                hidden_states = hidden_states['hidden_states']
                            except TimeoutError:
                                hidden_states = [None for _ in range(len(steps))]
                                skip_this_theory = True
                                print(f'INFERENCE TOO SLOW on rank {rank}:\n{theories[theory_id]}, {time.time() - t_init} secs')
                        else:
                            hidden_states = [None for _ in range(len(steps))]
                        self.time_tracker.toc('likelihood_infer_hidden')
                        t_infer.append(time.time() - t_init)
                        t_init = time.time()
                        # Simulate each inferred (hidden) state
                        prev_simulated_state_episode = []
                        next_simulated_state_episode = []
                        all_actions = []
                        sim_wins = []
                        sim_loses = []
                        sim_rewards = []
                        # simulated_triggered_events_episode = []
                        selected_steps = []
                        if time.time() - t_name > 15:
                            print('tinfer', t_infer[-1])
                            print('total', time.time() - t_name)
                            print('skip', skip_this_theory)
                            print(theories[theory_id])

                        for local_i_step, i_step, hidden_state, obs_state, action in zip(range(len(steps)), steps, hidden_states, obs_state_episode, actions):

                            if (i_ep, i_step) in data_indexes:
                                selected_steps.append(i_step)
                                prev_simulated_state_episode.append(obs_state)
                                all_actions.append(action)

                                if hidden_state is None:
                                    next_simulated_state = [None for _ in range(n_sims_here)]
                                    next_simulated_state_episode.append(next_simulated_state)
                                    sim_wins.append([None for _ in range(n_sims_here)])
                                    sim_loses.append([None for _ in range(n_sims_here)])
                                    sim_rewards.append([None for _ in range(n_sims_here)])
                                else:
                                    full_hidden_states = [dict(i_step=i_step, obj_hidden_state=hidden_state[i_sim], shape=game_format , names=obj_names) for i_sim in range(n_sims_here)]

                                    # check whether setting the hidden state creates the observed state (optional)
                                    if debug:
                                         game.set_state(full_hidden_states[0], obs_state)

                                    # simulate next step
                                    name = "simulates"
                                    t_name = time.time()
                                    self.time_tracker.tic('likelihood_simulate')
                                    next_simulated_outputs = game.env.simulate(full_hidden_states, action, n_sims_here, with_img=save_img)
                                    next_full_hidden_state = dict(i_step=i_step+1, obj_hidden_state=hidden_states[i_step + 1][0], shape=game_format , names=obj_names)
                                    win, lose = game.env.get_win_and_lose(next_full_hidden_state)
                                    self.time_tracker.toc('likelihood_simulate')
                                    # extract info
                                    next_simulated_state = [output['state'] if output is not None else None for output in next_simulated_outputs[-1]]
                                    next_simulated_state_episode.append(next_simulated_state)
                                    sim_wins.append([win for _ in range(n_sims_here)])
                                    sim_loses.append([lose for _ in range(n_sims_here)])
                                    sim_rewards.append([output['reward'] if output is not None else None for output in next_simulated_outputs[-1]])

                                    if save_img:
                                        prev_simulated_img = observed_imgs[local_i_step][0]
                                        next_simulated_imgs = next_simulated_outputs[0]
                                        for i_sim in range(n_sims_here):
                                            simulated_imgs[i_sim].append((prev_simulated_img, next_simulated_imgs[i_sim]))
                                        # if i_step in [0, 1]:
                                        #     print_obs_and_sim(prev_simulated_img, prev_simulated_img, next_simulated_imgs, observed_imgs[local_i_step+1][0])
                                        #     stop = 1
                        t_simulate.append(time.time() - t_init)
                        t_init = time.time()
                        # compute the likelihood information by comparing the observed obs with simulated obs
                        if len(next_simulated_state_episode) > 0:
                            self.time_tracker.tic('likelihood_convert_event')
                            simulated_events_episode = convert_simulated_traj_into_events(states=next_simulated_state_episode,
                                                                                          prev_states=prev_simulated_state_episode,
                                                                                          wins=sim_wins,
                                                                                          loses=sim_loses,
                                                                                          rewards=sim_rewards,
                                                                                          true_steps=selected_steps,
                                                                                          several=True, verbose=False)

                            self.time_tracker.toc('likelihood_convert_event')
                            obs_event_episode = [obs_ev for obs_ev, i_step in zip(obs_event_episode[1:], range(len(obs_event_episode[1:]))) if (i_ep, i_step) in data_indexes]
                            # obs_event_episode = obs_event_episode[1:]

                            for i_sim, simulation_id in enumerate(simulation_ids):
                                if simulation_id == 0:
                                    results[theory_id]['obs_events'] += obs_event_episode
                                    if save_img: results[theory_id]['obs_imgs'] += observed_imgs
                                    results[theory_id]['actions'] += all_actions
                                if simulation_id not in results[theory_id]['sim_events'].keys():
                                    results[theory_id]['sim_events'][simulation_id] = []
                                    results[theory_id]['sim_imgs'][simulation_id] = []
                                results[theory_id]['sim_events'][simulation_id] += simulated_events_episode[i_sim]
                                if save_img:
                                    results[theory_id]['sim_imgs'][simulation_id] += simulated_imgs[i_sim]
                        t_convert.append(time.time() - t_init)
                # Sanity check    
                if debug:
                    obs_events = results[theory_id]['obs_events']
                    sim_events = list(results[theory_id]['sim_events'].values())
                    obs_imgs = results[theory_id]['obs_imgs']
                    sim_imgs = list(results[theory_id]['sim_imgs'].values())
                    i_sim = 0
                    for i in range(len(obs_events)):
                        if len(set(obs_events[i]) - set(sim_events[i_sim][i])) > 0 or len(set(sim_events[i_sim][i]) - set(obs_events[i])) > 0:
                            print(f'step {i}')
                            print(f'act: {results[theory_id]["actions"][i]}')
                            print(set(obs_events[i]) - set(sim_events[i_sim][i]))
                            print(set(sim_events[i_sim][i]) - set(obs_events[i]))
                    stop = 1

            # bcast simulation results
            # Collect simulation results on base CPU
            self.time_tracker.tic('likelihood_res_bcast')
            if size > 1:
                all_results = comm.gather(results, root=0)
                results = dict()
                if rank == 0:
                    for i_theory in range(len(theories)):
                        results[i_theory] = dict(obs_events=[], sim_events=dict())
                        for i_res, res in enumerate(all_results):
                            if i_theory in res.keys():
                                results[i_theory]['obs_events'] += res[i_theory]['obs_events']
                                results[i_theory]['sim_events'].update(res[i_theory]['sim_events'])
                results = comm.bcast(results, root=0)
            self.time_tracker.toc('likelihood_res_bcast')

            # One particle -> one theory - multiple simulations of each theory
            # scatter particles across workers
            self.time_tracker.tic('likelihood_aggregate')
            interaction_loglikelihoods = []
            feedbacks = []
            particles_ids = get_chunk(list(range(len(theories))), size, rank)
            for particles_id in particles_ids:
                results_here = results[particles_id]
                results_here['sim_events'] = list(results_here['sim_events'].values())

                interaction_loglikelihood, feedback = compute_likelihood_per_object(results_here['sim_events'], results_here['obs_events'])
                # feedback['sim_triggered_events_all'] = concatenated_triggered_event_episodes
                # feedback['all_imgs'] = all_imgs
                interaction_loglikelihoods.append(interaction_loglikelihood)
                feedbacks.append(feedback)
            if size > 1:
                all_interaction_loglikelihoods = comm.gather(interaction_loglikelihoods, root=0)
                if rank == 0:
                    interaction_loglikelihoods = sum(all_interaction_loglikelihoods, [])
                else:
                    interaction_loglikelihoods = None
                all_feeds = comm.gather(feedbacks, root=0)
                if rank == 0:
                    feedbacks = sum(all_feeds, [])
                else:
                    feedbacks = None
            self.time_tracker.toc('likelihood_aggregate')

        else:
            feedbacks = [None] * len(theories)
            interaction_loglikelihoods = np.zeros(len(theories))

        # compute likelihood of linguistic data
        self.time_tracker.tic('likelihood_language')
        language_loglikes = self.compute_language_likelihood(generative_model, theories, linguistic_data)
        self.time_tracker.toc('likelihood_language')

        if rank == 0:
            for theory, interaction_loglike, feedback, language_loglike in zip(theories, interaction_loglikelihoods, feedbacks, language_loglikes):
                theory.feedback = feedback
                theory.interaction_loglikelihood = interaction_loglike
                theory.language_loglikelihood = language_loglike
        return theories

    def compute_language_likelihood(self, generative_model, theories, linguistic_data):
        if rank == 0 and len(linguistic_data) > 0 and self.params['exp_params']['use_language_likelihood']:
            if True: # self.params['exp_params']['language_likelihood_type'] == 'direct_likelihood':
                language_loglikes = self.comm_engine.compute_loglike_from_theories(theories, linguistic_data, generative_model.names)
            elif self.params['exp_params']['language_likelihood_type'] == 'inverted_proposal':
                proposal = generative_model.print_prior(to_print=False)[1]  # update generative model
                all_keys = set()
                all_win = set()
                all_lose = set()
                for theory in theories:
                    all_keys = all_keys.union(set(list(theory.vgdl_lines.dict.keys())))
                    all_win = all_win.union(set(theory.terminations['win']))
                    all_lose = all_lose.union((theory.terminations['lose']))
                language_loglikes = []
                for i_theory, theory in enumerate(theories):
                    probs = []
                    theory_dict = theory.vgdl_lines.dict
                    for w in all_win:
                        confidence = proposal['win'][w][-1]
                        if w not in theory.terminations['win']:
                            probs.append((1 - proposal['win'][w][1]) ** confidence)
                        else:
                            probs.append((proposal['win'][w][1]) ** confidence)
                    for l in all_lose:
                        confidence = proposal['lose'][l][-1]
                        if l not in theory.terminations['lose']:
                            probs.append((1 - proposal['lose'][l][1]) ** confidence)
                        else:
                            probs.append((proposal['lose'][l][1]) ** confidence)

                    for key in all_keys:
                        key_type = theory_dict[key].type if key in theory_dict.keys() else 'noInteraction'
                        if key_type in ['stepBack', 'bounceForward', 'reverseDirection', 'teleportToExit', 'addResource', 'turnAround']:
                            key_type = 'noInteraction'
                        # add prob of the type
                        confidence = proposal[key][key_type][-1]
                        probs.append(proposal[key][key_type][1] * confidence)
                        key_params = theory_dict[key].params if key in theory_dict.keys() else {}
                        # add prob of the params
                        param_keys = get_param_key(key, key_type, key_params, theory_dict)
                        confidence = proposal[key][key_type][4]
                        for param_key in param_keys:
                            idx = proposal[key][key_type][5].index(param_key)
                            probs.append(proposal[key][key_type][3][idx] * confidence)
                    language_loglikes.append(np.sum(np.log(probs)))
            else: raise NotImplementedError
        else:
            language_loglikes = np.zeros(len(theories))

        return language_loglikes

def convert_simulated_traj_into_events(states, prev_states, wins, loses, rewards, true_steps, several=False, verbose=False):
    map_of_prev_state_elements = []
    for i_step, step in enumerate(prev_states):
        map_of_prev_state_elements.append(dict())
        # if i_step > 0:
        for i_col, column in enumerate(step):
            for i_line, cell in enumerate(column):
                for element in cell:
                    map_of_prev_state_elements[-1][element['obj_id']] = element

    if not several:
        states = [[o] for o in states]
    n_ids = len(states[0])
    all_events = []
    for id in range(n_ids):
        all_events.append([])  # list of events that occurred
        obj_names = dict()
        for i_step, step in enumerate(states):
            if verbose: print(f'Events for transition {i_step}')
            new_obj_positions = dict()
            all_events[id].append([])  # list events at this timestep
            all_objects_encountered_this_step = set()
            all_objects_previous_step = set()
            if step[id] is not None:
                # go through the previous map
                for i_col, column in enumerate(prev_states[i_step]):
                    for i_line, cell in enumerate(column):
                        for element in cell:
                            if element['name'] == 'wall':
                                continue
                            # for current element and current step
                            if element['obj_id'] not in obj_names.keys(): obj_names[element['obj_id']] = element['name']
                            all_objects_previous_step.add(element['obj_id'])  # track all objects ever encountered
                if wins[i_step][id]:
                    assert not loses[i_step][id]
                    all_events[id][i_step].append(('win', None, None, None))
                elif loses[i_step][id]:
                    all_events[id][i_step].append(('lose', None, None, None))
                all_events[id][i_step].append((f'rew', rewards[i_step][id], None, None))

                # go through the map
                for i_col, column in enumerate(step[id]):
                    for i_line, cell in enumerate(column):
                        for element in cell:
                            # for current element and current step

                            if element['name'] == 'wall':
                                continue
                            pos = element['pos']
                            resources = element['resources']
                            new_obj_positions[element['obj_id']] = pos
                            all_objects_encountered_this_step.add(element['obj_id'])
                            prev_element = map_of_prev_state_elements[i_step].get(element['obj_id'], None)
                            # prev_element = find_in_map(prev_states[i_step], element['obj_id'])  # find that object in the previous map
                            if prev_element is None:  # it wasn't here, the obj was just born
                                prev_pos = None
                                prev_resources = None
                            else:  # we found it
                                prev_pos = prev_element['pos']
                                prev_resources = prev_element['resources']

                            if prev_pos is None:  # the object was just born
                                if verbose: print(f'  {element["name"]} appeared in {pos}')
                                all_events[id][i_step].append(('birth', element['name'], None, pos))
                            else:  # the objects moved or didn't move
                                new_pos = np.array(pos)
                                old_pos = np.array(prev_pos)
                                mvt = (custom_round(new_pos[0] - old_pos[0]), custom_round(new_pos[1] - old_pos[1]))
                                if mvt != (0, 0):
                                    if verbose: print(f'  {element["name"]} #{element["obj_id"]} moved from {prev_pos} to {pos}')
                                    all_events[id][i_step].append(('mvt', element['name'], element['obj_id'], prev_pos, pos, mvt))
                                else:
                                    all_events[id][i_step].append(('no_mvt', element['name'], element['obj_id'], prev_pos, pos, mvt))
                                resources_names = list(set(list(prev_resources.keys()) + list(resources.keys())))
                                for r_name in resources_names:
                                    prev_val = prev_resources.get(r_name, 0)
                                    new_val = resources.get(r_name, 0)
                                    if prev_val != new_val:
                                        all_events[id][i_step].append(('resource_change', element['name'], element['obj_id'], r_name, prev_val, new_val, new_val - prev_val))
                # track object deaths
                for obj_id in all_objects_previous_step - all_objects_encountered_this_step:
                    if true_steps[i_step] > 0:
                        prev_element = map_of_prev_state_elements[i_step].get(obj_id, None)
                        # prev_element = find_in_map(prev_states[i_step], obj_id)  # find that object in the previous map
                        if prev_element is None:  # it was already dead
                            pass
                        else:
                            assert prev_element['pos'] is not None
                            prev_pos = prev_element['pos']
                            if verbose: print(f'  {obj_names[obj_id]} #{obj_id} just died (prev_pos={prev_pos})')
                            all_events[id][i_step].append(('death', obj_names[obj_id], obj_id, prev_pos))
    if not several:
        return all_events[0]
    else:
        return all_events
    

def print_obs_and_sim(sim_prev_obs, obs_prev_obs, sim_obs, obs_obs):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 3)
    axs[0, 0].imshow(sim_prev_obs)
    axs[0, 2].imshow(obs_prev_obs)
    for i in range(min(len(sim_obs), 3)):
        axs[1, i].imshow(sim_obs[i])
    axs[2, 1].imshow(obs_obs)
    plt.show(block=False)


# Loop over each event and determine whether it fit theory (boolean) 
def compute_likelihood_per_object(simulated_events_all, observed_events_all):
    # compute likelihood of observed events
    likelihood_info = dict(per_name=dict(),
                           like_per_name=dict(),
                           win=[],
                           lose=[],
                           rew=dict(),
                           loglikelihood=0,
                           # simulated_events_all=simulated_events_all,
                           # observed_events_all=observed_events_all
                           )
    for i_step in range(len(observed_events_all)):
        all_events = set(observed_events_all[i_step])
        for i_sim in range(len(simulated_events_all)):
            all_events = all_events.union(set(simulated_events_all[i_sim][i_step]))
        
        # events
        for event in sorted(all_events):
            observed = event in observed_events_all[i_step]

            # Percentage of times that observed event matched simulated event
            simulated_frac = np.mean([event in simulated_events_all[i_sim][i_step] for i_sim in range(len(simulated_events_all))])
            likelihood = simulated_frac if observed else (1 - simulated_frac)
            if likelihood == 0:
                likelihood = epsilon     # Min likelihood value

            # Filtering by event type
            event_type = event[0]
            if event_type in ['win', 'lose']:
                likelihood_info[event_type].append(likelihood)
            elif event_type == 'rew':                           # Reward
                rew = event[1]
                if str(rew) not in likelihood_info[event_type].keys():
                    likelihood_info[event_type][str(rew)] = []
                likelihood_info[event_type][str(rew)].append(likelihood)
            else:
                event_type, obj_name, obj_id = event[:3]
                if obj_name not in likelihood_info['per_name'].keys():
                    likelihood_info['per_name'][obj_name] = dict(all_events=[], birth=[], death=[], mvt=[], no_mvt=[], resource_change=[])
                    likelihood_info['like_per_name'][obj_name] = dict(birth=None, death=None, mvt=None, no_mvt=None, resource_change=None)
                # Break down by object and different factors of object (for later reweighting)
                likelihood_info['per_name'][obj_name][event_type].append(likelihood)
                likelihood_info['per_name'][obj_name]['all_events'].append(likelihood)

    # we want to balance the likelihood of each category of agent and number of objects as a function of event counts
    likelihood_factor = 5
    for obj_name, obj_like in likelihood_info['per_name'].items():
        for event_cat, event_like in obj_like.items():
            if event_cat != 'all_events' and len(event_like) > 0:
                lambda_ = likelihood_factor / (len(event_like)) ** (1 / 2)
                # print(lambda_)
                for like in event_like:
                    if likelihood_info['like_per_name'][obj_name][event_cat] is None:
                        likelihood_info['like_per_name'][obj_name][event_cat] = 0
                    likelihood_info['like_per_name'][obj_name][event_cat] += np.log(like) * lambda_
                    likelihood_info['loglikelihood'] += np.log(like) * lambda_
    for key in ['win', 'lose']:
        for value in likelihood_info[key]:
            likelihood_info['loglikelihood'] += np.log(value) * likelihood_factor
    for rew, rew_like in likelihood_info['rew'].items():
        lambda_ = likelihood_factor / (len(rew_like)) ** (1 / 2)
        for like in rew_like:
            likelihood_info['loglikelihood'] += np.log(like) * lambda_

    return likelihood_info['loglikelihood'], likelihood_info


def accept_or_reject(particle, mutated_particle, generative_model, linguistic_data):
    # compute ratio of loglikelihood for RELEVANT events
    removed, added = [], []
    for key, value in particle.vgdl_lines.dict.items():
        if value not in mutated_particle.vgdl_lines:
            removed.append(value)
    for key, value in mutated_particle.vgdl_lines.dict.items():
        if value not in particle.vgdl_lines:
            added.append(value)
    different_values = removed + added
    event_types = []
    if particle.params['exp_params']['use_interaction_likelihood']:

        # list event types that may be impacted by this change
        for value in different_values:
            if value.line_type == 'obj':
                # if it's a change in object, let's look at how it moves and when it dies
                event_types.append((value.name, 'mvt'))
                event_types.append((value.name, 'no_mvt'))
                event_types.append((value.name, 'death'))
                if value.spawns():
                    # if there is a change in spawning object, look at the birth of the spawn object
                    event_types.append((value.params['stype'], 'birth'))
            elif value.line_type == 'int':
                if value.type in 'stepBack':
                    # look at no mvts of the objects
                    event_types.append((value.name[0], 'no_mvt'))
                    # event_types.append((value.name[1], 'no_mvt'))
                if value.type == 'reverseDirection':
                    event_types.append((value.name[0], 'mvt'))
                    event_types.append((value.name[0], 'no_mvt'))
                if value.type == 'turnAround':
                    event_types.append((value.name[0], 'mvt'))
                    event_types.append((value.name[0], 'no_mvt'))
                if value.type == 'bounceForward':
                    event_types.append((value.name[0], 'mvt'))
                    event_types.append((value.name[0], 'no_mvt'))
                    event_types.append((value.name[1], 'mvt'))
                    event_types.append((value.name[1], 'no_mvt'))
                if value.kills_a():
                    # look at death
                    event_types.append((value.name[0], 'death'))
                if value.teleports_a():
                    event_types.append((value.name[0], 'mvt'))
                if value.spawns():
                    # if spawning interaction, look at birth of spawned object
                    event_types.append((value.params['stype'], 'birth'))
                if value.changes_resource():
                    # if interaction changes resource, look at the resource change events
                    event_types.append((value.name[1], 'resource_change'))
                if 'scoreChange' in value.params.keys():
                    event_types.append((None, 'rew'))
        # are terminations different?
        if particle.terminations['win'] != mutated_particle.terminations['win']:
            event_types.append((None, 'win'))
        if particle.terminations['lose'] != mutated_particle.terminations['lose']:
            event_types.append((None, 'lose'))

        # now integrate the likelihood of all of these
        event_types = [ev_type for ev_type in event_types if ev_type[0] != 'wall']  # filter events about walls
        # if len(event_types) == 0:
        #     print(different_values, particle.terminations, mutated_particle.terminations)
        #     assert False
        event_types = list(set(event_types))
        feedback = particle.feedback
        mutated_feedback = mutated_particle.feedback
        int_loglikelihood, int_mutated_loglikelihood = 0, 0
        likelihood_factor = 5
        for event_type in event_types:
            name, ev_type = event_type
            if ev_type in ['win', 'lose']:
                event_likes = feedback[ev_type]
                mutated_events_likes = mutated_feedback[ev_type]
                for like in event_likes:
                    int_loglikelihood += np.log(like)
                for like in mutated_events_likes:
                    int_mutated_loglikelihood += np.log(like)
            elif ev_type == 'rew':
                for event_likes in feedback['rew'].values():
                    lambda_ = likelihood_factor / (len(event_likes)) ** (1 / 2)
                    for like in event_likes:
                        int_loglikelihood += np.log(like) * lambda_
                for mutated_events_likes in mutated_feedback['rew'].values():
                    lambda_ = likelihood_factor / (len(mutated_events_likes)) ** (1 / 2)
                    for like in mutated_events_likes:
                        int_mutated_loglikelihood += np.log(like) * lambda_
            else:
                event_likes = feedback['per_name'][name][ev_type]
                mutated_events_likes = mutated_feedback['per_name'][name][ev_type]
                if len(event_likes) > 0:
                    # print(name, ev_type)
                    lambda_ = likelihood_factor / (len(event_likes)) ** (1 / 2)
                    for like in event_likes:
                        int_loglikelihood += np.log(like) * lambda_
                if len(mutated_events_likes) > 0:
                    # print(name, ev_type)
                    lambda_ = likelihood_factor / (len(mutated_events_likes)) ** (1 / 2)
                    for like in mutated_events_likes:
                        int_mutated_loglikelihood += np.log(like) * lambda_
    else:
        int_loglikelihood, int_mutated_loglikelihood = 0, 0
    proposal_info = dict()
    added_probs = []
    removed_probs = []
    if particle.params['exp_params']['use_language_likelihood'] and len(linguistic_data) > 0:
        if particle.params['exp_params']['language_likelihood_type'] == 'inverted_proposal' and particle.params['exp_params']['use_language_proposal']:
            proposal = generative_model.print_prior(to_print=False)[1]
            mutated_dict = mutated_particle.vgdl_lines.dict
            parent_dict = particle.vgdl_lines.dict
            removed_probs = []
            removed_lines = []
            added_probs = []
            added_lines = []
            changed_keys = []
            confidences = []
            keys = list(set(list(parent_dict.keys()) + list(mutated_dict.keys())))
            for win_lose in ['win', 'lose']:
                for w in particle.terminations[win_lose]:
                    if w not in mutated_particle.terminations[win_lose]:
                        confidence = proposal[win_lose][w][-1]
                        if confidence > 0.5:
                            removed_lines.append(w)
                            added_lines.append(None)
                            changed_keys.append(win_lose)
                            removed_probs.append(proposal[win_lose][w][1])
                            added_probs.append(1 - proposal[win_lose][w][1])
                            confidences.append(proposal[win_lose][w][3])
                for w in mutated_particle.terminations[win_lose]:
                    if w not in particle.terminations[win_lose]:
                        confidence = proposal[win_lose][w][-1]
                        if confidence > 0.5:
                            removed_lines.append(None)
                            added_lines.append(w)
                            changed_keys.append(win_lose)
                            removed_probs.append(1 - proposal[win_lose][w][1])
                            added_probs.append(proposal[win_lose][w][1])
                            confidences.append(proposal[win_lose][w][3])

            for key in keys:
                if key in changed_keys:
                    continue
                parent_type = parent_dict[key].type if key in parent_dict.keys() else 'noInteraction'
                if parent_type in ['stepBack', 'bounceForward', 'reverseDirection', 'teleportToExit', 'addResource', 'turnAround']:
                    parent_type = 'noInteraction'
                mutated_type = mutated_dict[key].type if key in mutated_dict.keys() else 'noInteraction'
                if mutated_type in ['stepBack', 'bounceForward', 'reverseDirection', 'teleportToExit', 'addResource', 'turnAround']:
                    mutated_type = 'noInteraction'

                if parent_type == mutated_type == 'noInteraction':
                    # both are noInteraction
                    continue
                elif parent_type == mutated_type:
                    parent_params = parent_dict[key].params
                    mutated_params = mutated_dict[key].params
                    parent_param_key = get_param_key(key, parent_type, parent_params, parent_dict)
                    mutated_param_key = get_param_key(key, mutated_type, mutated_params, mutated_dict)
                    if parent_param_key == mutated_param_key:
                        # nothing changed
                        continue
                    else:
                        # params changed
                        confidence = proposal[key][mutated_type][4]
                        if confidence > 0.5:
                            changed_keys.append(key)
                            removed_probs_values = []
                            added_probs_values = []
                            for param_key in parent_param_key:
                                parent_idx = proposal[key][parent_type][5].index(param_key)
                                removed_probs_values.append(proposal[key][parent_type][3][parent_idx])
                            for param_key in mutated_param_key:
                                mutated_idx = proposal[key][mutated_type][5].index(param_key)
                                added_probs_values.append(proposal[key][mutated_type][3][mutated_idx])
                            removed_probs.append(np.min(removed_probs_values))
                            added_probs.append(np.min(added_probs_values))
                            removed_lines.append(parent_dict[key])
                            added_lines.append(mutated_dict[key])
                            confidences.append(confidence)
                else:
                    # type changed
                    confidence = proposal[key][mutated_type][-1]
                    if confidence > 0.5:
                        changed_keys.append(key)
                        removed_probs.append(proposal[key][parent_type][1] * max(0.75, confidence))
                        added_probs.append(proposal[key][mutated_type][1] * max(0.75, confidence))
                        removed_lines.append(parent_dict.get(key))
                        added_lines.append(mutated_dict.get(key))
                        confidences.append(confidence)
            proposal_info = dict(removed_lines=removed_lines, added_lines=added_lines, confidences=confidences, added_probs=added_probs, removed_probs=removed_probs,
                                 changed_keys=changed_keys)
            assert len(added_probs) == len(removed_probs)
            added_probs = np.clip(added_probs, 1e-6, 1)
            removed_probs = np.clip(removed_probs, 1e-6, 1)
            ratio_lang_loglikelihood = (np.sum(np.log(np.array(added_probs))) - np.sum(np.log(np.array(removed_probs)))) * 5
        elif particle.params['exp_params']['language_likelihood_type'] == 'description_likelihood' or not particle.params['exp_params']['use_language_proposal']:
            ratio_lang_loglikelihood = (mutated_particle.language_loglikelihood - particle.language_loglikelihood)
        else:
            raise NotImplementedError
    else:
        ratio_lang_loglikelihood = 0

    # compute ratio of priors
    ratio_logprior = mutated_particle.logprior - particle.logprior
    ratio_int_loglikelihood = int_mutated_loglikelihood - int_loglikelihood
    logratio = ratio_logprior + ratio_int_loglikelihood + ratio_lang_loglikelihood
    if logratio < -200:
        ratio = 0
    elif logratio > 200:
        ratio = 1
    else:
        ratio = np.exp(logratio)
    accept = np.random.rand() < ratio
    # print('\nMUTATION')
    # print('mutation:', mutated_particle.mutation_strs)
    # print('removed', removed.__str__())
    # print('added', added.__str__())
    # print('ratio_int_loglikelihood', ratio_int_loglikelihood)
    # print('ratio_lang_loglikelihood', ratio_lang_loglikelihood)
    # print('ratio_logprior', ratio_logprior)
    # print('logratio', logratio)
    # print('accept', accept)
    accept_info = dict(accept=accept, ratio_logprior=ratio_logprior, ratio_int_loglikelihood=ratio_int_loglikelihood,
                       ratio_lang_loglikelihood=ratio_lang_loglikelihood,
                       ratio_lang_full_loglikelihood=mutated_particle.language_loglikelihood - particle.language_loglikelihood,
                       ratio=ratio,
                       mutated=mutated_particle.copy(no_feedback=True),
                       parent=particle.copy(no_feedback=True),
                       event_types=event_types ,
                       different_values=different_values,
                       added_probs=added_probs,
                       removed_probs=removed_probs,
                       removed=removed,
                       added=added,
                       proposal_info=proposal_info,
                       mutated_logprior=mutated_particle.logprior,
                       mutated_int_local_loglike=int_mutated_loglikelihood,
                       mutated_lang_loglike=mutated_particle.language_loglikelihood,
                       mutated_int_loglike=mutated_particle.interaction_loglikelihood,

                       parent_logprior=particle.logprior,
                       parent_int_local_loglike=int_loglikelihood,
                       parent_lang_loglike=particle.language_loglikelihood,
                       parent_int_loglike=particle.interaction_loglikelihood,
                       )
    return accept_info

def get_param_key(key, type, params, vgdl_dict):
    if type == 'transformTo':
        param_keys = [params['stype']]
    elif type == 'Portal':
        param_keys = []
        for k, v in vgdl_dict.items():
            if isinstance(k, tuple) and k[1] == key and v.type == 'teleportToExit':
                param_keys.append((k[0], params['stype']))
                # break
    elif type == 'Passive':
        param_keys = []
        for k, v in vgdl_dict.items():
            if isinstance(k, tuple) and k[0] == key and v.type == 'bounceForward':
                param_keys.append(k[1])
                # break
    else:
        param_keys = []
    return set(param_keys)
