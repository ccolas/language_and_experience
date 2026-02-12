import os
import numpy as np

from src.utils import pickle_save, convert_traj_into_object_mvt, convert_traj_into_events, get_count_objects_per_step, AVATAR_NAME
from src.agent.thinking.theory_pieces.object_types import MovingAvatar

class DataStore:
    """
    
    """
    def __init__(self, params):
        
        # self.new_sensorimotor_data = None
        self.linguistic_data = []
        self.traj_episodes = []     # save episodes as complete map states
        self.obj_episodes = []      # save episodes as dictionaries of objects
        self.event_episodes = []    # save episodes as the events at each time step 
        self.nn_episodes = []       # nearest neighbors to each object within radius
        self.steps_processed = []   # 
        self.format_episodes = []   # dimensions of map
        self.layouts_episodes = []

        self.time_tracker = params['time_tracker']
        self.verbose = params['verbose']
        self.block_size = params['game_params']['block_size']
        self.colors = params['true_game_info']['colors']
        self.layouts = params['true_game_info']['layouts']
        self.character_mapping = params['true_game_info']['character_mapping']
        self.step_by_step = params['true_game_info']['step_by_step']
        self.n_transitions_likelihood = params['agent']['thinker']['n_transitions_likelihood']
        self.data_path = params['exp_path'] + 'dumps/interaction_data/'
        os.makedirs(self.data_path, exist_ok=True)
        self.params = params
        
        self.latest_sensorimotor_data_start_at = (0, 0)  # indexes of the start of the latest batch of sensorimotor data (episode_id, step_id)
        self.latest_linguistic_data_starts_at = 0
        self.new_data_in = False

        self.memory_events = dict()

        self.names = set()
        self.avatar_name = AVATAR_NAME
        self.last_episode_saved = 0

    # Load data
    def load(self, sensorimotor_data=None, linguistic_data=None):

        # if reset=True, the first element of sensorimotor_data is the first observation
        if isinstance(sensorimotor_data, dict):
            sensorimotor_data = [sensorimotor_data]
        if isinstance(linguistic_data, str):
            linguistic_data = [linguistic_data]

        # 
        if sensorimotor_data:
            self.new_data_in = True
            assert len(sensorimotor_data) > 0,  "sensorimotor data is empty"
            first_is_reset = sensorimotor_data[0]['action'] is None    # Decide whether to discard first episode
            current_episode = len(self.traj_episodes) - 1 + int(first_is_reset)

            for i_step, transition in enumerate(sensorimotor_data):
                reset = transition['action'] is None
                if reset:
                    new_traj = dict(zip(transition.keys(), [[] for _ in range(len(transition.keys()))]))
                    self.traj_episodes.append(new_traj)
                    self.format_episodes.append((len(transition['state']), len(transition['state'][0])))
                    self.layouts_episodes.append(self.layouts[transition['lvl']])

                for k in transition.keys():
                    if transition[k] is not None:
                        self.traj_episodes[-1][k].append(transition[k])

            # compute event-based and object-based representations
            for i_ep in range(current_episode, len(self.traj_episodes)):

                prev_obj_episode = self.obj_episodes[i_ep] if i_ep < len(self.obj_episodes) else None
                prev_nn_episode = self.nn_episodes[i_ep] if i_ep < len(self.nn_episodes) else None
                prev_events = self.event_episodes[i_ep] if i_ep < len(self.event_episodes) else None
                new_events_start = len(prev_events) if prev_events else 0
                steps_processed = self.steps_processed[i_ep] if i_ep < len(self.steps_processed) else 0

                obj_episode, nn_episode = convert_traj_into_object_mvt(self.traj_episodes[i_ep]['state'], self.params['game_params']['block_size'],
                                                           objects=prev_obj_episode, nn=prev_nn_episode, steps_processed=steps_processed)
                
                event_episode = convert_traj_into_events(self.traj_episodes[i_ep]['state'], [None] + self.traj_episodes[i_ep]['state'][:-1],
                                                         wins=self.traj_episodes[i_ep]['won'], loses=self.traj_episodes[i_ep]['lose'],
                                                         rewards=self.traj_episodes[i_ep]['reward'], events=prev_events, steps_processed=steps_processed)
                
                self.count_events(event_episode[new_events_start:], ([None] + self.traj_episodes[i_ep]['action'])[new_events_start:], obj_episode, new_events_start)
                if i_ep >= len(self.obj_episodes):
                    self.obj_episodes.append(obj_episode)
                    self.nn_episodes.append(nn_episode)
                    self.event_episodes.append(event_episode)
                    self.steps_processed.append(len(self.traj_episodes[i_ep]['state']))
                else:
                    self.obj_episodes[i_ep] = obj_episode
                    self.nn_episodes[i_ep] = nn_episode
                    self.event_episodes[i_ep] = event_episode
                    self.steps_processed[i_ep] = len(self.traj_episodes[i_ep]['state'])
            assert len(self.traj_episodes) == len(self.event_episodes) == len(self.obj_episodes) == len(self.nn_episodes)
            new_names = self.compute_extra_info(current_episode)
        else:
            new_names = False

        if linguistic_data:
            self.linguistic_data += linguistic_data

        # if linguistic_data or sensorimotor_data:
        #     self.dump_data()
        # if self.verbose and (n_steps_loaded > 0 or n_linguistic_loaded > 0):
        #     print(f'    > new datapoints loaded: {f"{n_steps_loaded} sensorimotor steps" if n_steps_loaded > 0 else ""}'
        #           f'{" and " if n_steps_loaded>0 and n_linguistic_loaded>0 else ""}'
        #           f'{f"{n_linguistic_loaded} linguistic feedback" if n_linguistic_loaded>0 else ""}')

        return new_names

    def get_linguistic_data(self):
        self.latest_linguistic_data_starts_at = len(self.linguistic_data)
        return self.linguistic_data

    # keep track of event counts to compute sampling priorities
    def count_events(self, event_episode, actions, obj_episode, first_i_step):
        for events, action, i_step in zip(event_episode, actions, range(first_i_step, first_i_step + len(actions))):
            for event in events:
                ev_key = self.get_event_key(event, action, obj_episode, i_step)
                if ev_key not in self.memory_events.keys():
                    self.memory_events[ev_key] = 0
                self.memory_events[ev_key] += 1

    # returns the new batch of data in all its representations, could start in the middle of an episode
    def get_latest_sensorimotor_data(self):
        traj_episodes = []
        obj_episodes = []
        nn_episodes = []
        event_episodes = []
        steps = []
        episodes = []
        for i_ep in range(self.latest_sensorimotor_data_start_at[0], len(self.traj_episodes)):
            start_step = self.latest_sensorimotor_data_start_at[1] if i_ep == self.latest_sensorimotor_data_start_at[0] else 0
            if len(self.event_episodes[i_ep]) > start_step:
                if start_step > 0:
                    start_step -= 1  # so we don't miss the transition between last batch and current batch
                steps.append(list(range(start_step, len(self.event_episodes[i_ep]))))
                episodes.append(i_ep)
                traj_episode = self.start_episode_at(self.traj_episodes[i_ep], start_step, episode_rep='traj')
                if traj_episode:
                    traj_episodes.append(traj_episode)
                obj_episode = self.start_episode_at(self.obj_episodes[i_ep], start_step, episode_rep='obj')
                if obj_episode:
                    obj_episodes.append(obj_episode)
                nn_episode = self.start_episode_at(self.nn_episodes[i_ep], start_step, episode_rep='nn')
                if nn_episode:
                    nn_episodes.append(nn_episode)
                event_episode = self.start_episode_at(self.event_episodes[i_ep], start_step, episode_rep='event')
                if event_episode:
                    event_episodes.append(event_episode)

        # track the start of the data that was not ingested (i_ep, i_step)
        self.latest_sensorimotor_data_start_at = (len(self.traj_episodes) - 1, self.n_steps_in_last_episode)

        return episodes, steps, traj_episodes, obj_episodes, nn_episodes, event_episodes


    def get_event_key(self, event, action, obj_episode, i_step):
        ev_key = f"{event[1]}_{event[0]}"
        if event[0] == 'mvt':
            ev_key += f'_{event[-1]}'
        # if obj event is pushed or avatar, use the action in the key
        if AVATAR_NAME in str(event[1]):
            ev_key += f"_act{action}"
        elif event[0] in ['mvt', 'no_mvt'] and self.agent_tries_push(event, action, obj_episode, i_step):
            ev_key += f"_act{action}"
        return ev_key

    #
    def sample_data_for_likelihood(self, all=False):

        if isinstance(self.n_transitions_likelihood, int) and not all:
            # compute priorities
            data_indexes = []
            event_keys = []
            data_idx_info = dict()
            event_key_to_data_idx = dict()
            data_idx_to_event_key = dict()
            for i_ep, event_episode, traj_episode, obj_episode in zip(range(len(self.event_episodes)), self.event_episodes, self.traj_episodes, self.obj_episodes):
                if len(event_episode) > 1:
                    for i_step, events, action, won, lose in zip(range(len(event_episode)), event_episode, [None] + traj_episode['action'],
                                                                 [None] + traj_episode['won'], [None] + traj_episode['lose']):
                        if i_step > 0:
                            step_prios = []
                            step_keys = []
                            for event in events:
                                ev_key = self.get_event_key(event, action, obj_episode, i_step)
                                step_keys.append(ev_key)
                                if 'win' in ev_key:
                                    assert won
                                    step_prios.append(2)
                                elif 'lose' in ev_key:
                                    assert lose
                                    step_prios.append(2)
                                else:
                                    count = self.memory_events[ev_key]
                                    step_prios.append(1 / count)
                            event_keys.append(step_keys)
                            data_idx = (i_ep, i_step - 1)
                            data_indexes.append(data_idx)
                            data_idx_info[data_idx] = sum(step_prios)  # the info of a transition is the sum of the rarity of the events it contains
                            data_idx_to_event_key[data_idx] = step_keys
                            for event_key in step_keys:
                                if event_key not in event_key_to_data_idx.keys():
                                    event_key_to_data_idx[event_key] = []
                                event_key_to_data_idx[event_key].append(data_idx)

                            #     # if won or lose:
                            # #     self.priorities.append(2)
                            # # else:
                            # #     prios = []
                            #     # for event in events:
                            #         ev_key = self.get_event_key(event, action, obj_episode, i_step)
                            #         count = self.memory_events[ev_key]
                            #         prios.append(1 / count)
                            #     # self.priorities.append(np.max(prios))

            assert len(event_key_to_data_idx.keys()) < self.n_transitions_likelihood
            event_counts_dict = dict(zip(event_key_to_data_idx.keys(), [len(v) for v in event_key_to_data_idx.values()]))
            print('events counts in data', event_counts_dict)
            # for key, val in event_key_to_data_idx.items():
            #     print(key, len(val))
            events_keys = list(event_key_to_data_idx.keys())

            count_sampled = dict(zip(events_keys, [0] * len(event_key_to_data_idx)))
            all_sampled_data_idx = []
            while len(all_sampled_data_idx) < self.n_transitions_likelihood:
                # find the least represented event
                events_keys = list(event_key_to_data_idx.keys())
                if len(events_keys) == 0:
                    break
                np.random.shuffle(events_keys)
                counts_sampled_events = [count_sampled[ev_key] for ev_key in events_keys]
                ev_key = events_keys[np.argmin(counts_sampled_events)]
                # now sample a transition containing it
                candidate_data_idx = list(set(event_key_to_data_idx[ev_key]) - set(all_sampled_data_idx))
                np.random.shuffle(candidate_data_idx)
                if len(candidate_data_idx) == 0:
                    # we've sampled all transitions containing that event
                    del event_key_to_data_idx[ev_key]
                else:
                    values = [data_idx_info[data_idx] for data_idx in candidate_data_idx]
                    idx_idx = np.argmax(values) #np.random.randint(len(values))
                    sampled_data_idx = candidate_data_idx[idx_idx]
                    all_sampled_data_idx.append(sampled_data_idx)
                    for ev_key in data_idx_to_event_key[sampled_data_idx]:
                        count_sampled[ev_key] += 1
            print('events counts sampled', count_sampled)
            all_sampled_data_idx = sorted(all_sampled_data_idx)
            # print(len(all_sampled_data_idx))
            assert len(all_sampled_data_idx) == min(self.n_transitions_likelihood, len(data_indexes))
        else:
            all_sampled_data_idx = []
            for i_ep, event_episode in enumerate(self.event_episodes):
                if len(event_episode) > 1:
                    for i_step, events in enumerate(event_episode):
                        all_sampled_data_idx.append((i_ep, i_step))

        return self.get_sensory_data() + (all_sampled_data_idx, self.linguistic_data)

    def agent_tries_push(self, event, action, obj_episode, i_step):
        # whether we expect the agent to push the object or not
        mvt_dict = MovingAvatar(None, self.params).action_id_mapping
        mvt = mvt_dict[action]
        if np.sum(np.abs(mvt)) > 0:
            prev_pos = obj_episode[self.avatar_name + '.1']['pos'][i_step-1]
            new_pos = obj_episode[self.avatar_name + '.1']['pos'][i_step]
            if prev_pos and new_pos and new_pos == prev_pos:
                expected_pos = np.array(prev_pos) + mvt
                if np.linalg.norm(expected_pos - np.array(event[4]), ord=2) < np.sqrt(2):
                    return True
        return False

    def compute_extra_info(self, current_episode):
        names = set()
        for i_ep in range(current_episode, len(self.traj_episodes)):
            count_objs, names = get_count_objects_per_step(self.traj_episodes[i_ep], names)
            self.traj_episodes[i_ep]['count_objs'] = count_objs
            diff_count = dict(zip(names, [[0] + list(np.diff([count_objs[i_step][obj_name] for i_step in range(len(count_objs))])) for obj_name in names]))
            self.traj_episodes[i_ep]['diff_count'] = diff_count
        if len(names - self.names) > 0:
            new_names = True
            # print('new names')
            # print(names - self.names)
            self.names = self.names.union(names)
        else:
            new_names = False
        avatar_name = [name for name in self.names if AVATAR_NAME in name]
        assert len(avatar_name) == 1
        self.avatar_name = avatar_name[0]
        return new_names

    def get_latest_linguistic_data(self):
        linguistic_data = self.linguistic_data[self.latest_linguistic_data_starts_at:]
        self.latest_linguistic_data_starts_at = len(self.linguistic_data)
        return linguistic_data

    def start_episode_at(self, episode, start_step, episode_rep):
        assert episode_rep in ['traj', 'obj', 'event', 'nn']
        if episode_rep == 'traj':
            len_episode = len(episode['state'])
        elif episode_rep == 'obj':
            len_episode = len(episode[list(episode.keys())[0]]['pos'])
        elif episode_rep == 'nn':
            len_episode = len(episode)
        elif episode_rep == 'event':
            len_episode = len(episode)
        else: raise ValueError

        if start_step == 0:
            return episode
        elif start_step == len_episode:
            return None
        else:
            if episode_rep == 'traj':
                ep = dict()
                for k in episode.keys():
                    if isinstance(episode[k], list):
                        ep[k] = episode[k][start_step:]
                    else:
                        ep[k] = episode[k]
            elif episode_rep == 'nn':
                ep = episode[start_step:]
            elif episode_rep == 'obj':
                ep = dict()
                for k in episode.keys():
                    ep[k] = dict()
                    for k2 in episode[k].keys():
                        if isinstance(episode[k][k2], list):
                            ep[k][k2] = episode[k][k2][start_step:]
                        else:
                            ep[k][k2] = episode[k][k2]
            elif episode_rep == 'event':
                ep = episode[start_step:]
            else:
                raise ValueError
        return ep

    def append_chunk(self, episode_chunk):
        for k in self.traj_episodes[-1].keys():
            if isinstance(self.traj_episodes[-1][k], list):
                self.traj_episodes[-1][k] += episode_chunk[k]
            else:
                assert self.traj_episodes[-1][k] == episode_chunk[k]

    def get_current_episode(self):
        return dict(traj=self.traj_episodes[-1], objs=self.obj_episodes[-1], events=self.event_episodes[-1])

    def dump_data(self, life_step_tracker):
        # dump latest episode
        to_dump = life_step_tracker.copy()
        first_ep_to_save = self.last_episode_saved
        if self.params['agent']['thinker']['alg'] != 'dqn':
            to_dump.update(dict(traj_episode=self.traj_episodes[first_ep_to_save:],
                                obj_episode=self.obj_episodes[first_ep_to_save:],
                                events_episode=self.event_episodes[first_ep_to_save:],
                                linguistic_data=self.linguistic_data))
        self.last_episode_saved = len(self.traj_episodes)
        pickle_save(to_dump, self.data_path + f'generation_{life_step_tracker["gen"]}_life_{life_step_tracker["life"]}_lvl_solved_{life_step_tracker["n_levels_solved"]}.pkl')

    def get_sensory_data(self):
        episodes = list(range(len(self.traj_episodes)))
        steps = [list(range(len(traj_episode['state']))) for traj_episode in self.traj_episodes]
        return episodes, steps, self.traj_episodes, self.obj_episodes, self.nn_episodes, self.event_episodes
    @property
    def n_episodes(self):
        return len(self.traj_episodes)

    @property
    def n_steps(self):
        return sum([len(ep['state']) for ep in self.traj_episodes])

    @property
    def n_steps_in_last_episode(self):
        return len(self.traj_episodes[-1]['state']) if len(self.traj_episodes) > 0 else 0

    def log(self):
        return dict(datastore=dict(n_steps_stored=self.n_steps, n_episodes_stored=self.n_episodes))