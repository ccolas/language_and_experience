import time
import numpy as np
from mpi4py import MPI
from scipy.stats import rankdata
# import concurrent.futures


class Evolution:
    """
    Maintain a population of plans
    """
    def __init__(self, hidden_state, prev_state, state, mental_model, reward_function, planning_horizon, step_info, time_tracker, params):
        self.params = params
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.time_tracker = time_tracker
        self.step_info = step_info
        self.reward_function = reward_function
        self.mental_model = mental_model #[mental_model for _ in range(params['n_cpus'])]
        self.planning_horizon = planning_horizon
        self.hidden_state = hidden_state
        self.is_stochastic = mental_model.is_stochastic  # whether the mental model is stochastic
        self.n_expansions = 1 if self.is_stochastic else 1
        self.mcts_exploration_param = self.params['agent']['planner']['mcts_exploration_param']
        self.action_space = self.params['true_game_info']['valid_actions']
        # self.budget = params['agent']['planner']['budget']
        self.step_by_step = self.params['true_game_info']['step_by_step']
        self.null_action = self.params['true_game_info']['null_action']
        self.stickiness = params['agent']['planner']['stickiness']
        self.prev_state = prev_state
        self.state = state
        self.win_reward = 100
        self.lose_cost = -1000
        self.gen_size = 150
        self.nb_gens = 3

    def sample_actions(self, n, actions=[]):
        avatar_type = self.mental_model.rules.obj.type(self.mental_model.rules.avatar_name)
        if avatar_type == 'ShootAvatar':
            dir_actions = [0, 1, 2, 3, 4, 5]
        elif avatar_type == 'MovingAvatar':
            dir_actions = [0, 1, 2, 3, 4]
        elif avatar_type == 'FlakAvatar':
            dir_actions = [2, 3, 4, 5]
        else:
            raise NotImplementedError

        if len(actions) >= n:
            return actions[:n].copy()
        for i in range(len(actions), n):
            if i == 0 or actions[-1] not in dir_actions or np.random.rand() < (1 - self.stickiness):
                actions.append(np.random.choice(dir_actions))
            else:
                # implement stickiness
                actions.append(actions[-1])
        return actions.copy()

    def sample_gen(self):
        n_here = self.gen_size // self.size
        if self.rank != 0 and self.rank - 1 < (self.gen_size % self.size):
            n_here += 1
        return np.array([self.sample_actions(n=self.planning_horizon, actions=[]) for _ in range(n_here)])

    # Cut randomly or max reward 
    def mutate(self, selected_sim_results):
        mutated_actions = []
        for sim_res in selected_sim_results:
            actions = sim_res['actions']
            goal_rewards = sim_res['goal_rewards']
            if np.random.rand() < 0.5:
                cut = np.random.randint(len(actions) - 1)
            else:
                cut = np.argmax(goal_rewards)
                if cut == len(actions) - 1:
                    cut = np.random.randint(len(actions) - 1)
            mutated_action = self.sample_actions(self.planning_horizon, actions.copy()[:cut])
            mutated_actions.append(list(mutated_action))

        return mutated_actions

    # Select plans from population
    def select(self, simulation_results):
        total_goal_rewards = np.array([res['selection_reward'] for res in simulation_results])
        indexes = np.argsort(total_goal_rewards)[-len(total_goal_rewards) // 10:]
        ranks = rankdata(total_goal_rewards[indexes])
        norm_ranks = (ranks / np.max(ranks)) ** 3
        probs = norm_ranks / sum(norm_ranks)

        selected_idx = np.random.choice(indexes, size=self.gen_size, p=probs)
        # print(len(np.unique(selected_idx)) / len(total_goal_rewards))
        best_idx = np.argmax(total_goal_rewards)
        selected_idx[:len(total_goal_rewards) // 10] = best_idx
        selected_sim_results = [simulation_results[i] for i in selected_idx]
        best_sim_results = simulation_results[best_idx]

        return selected_sim_results, best_sim_results

    def run(self):
        new_gen = self.sample_gen()
        # look for optimal actions.
        success, init_goal_reward, actions = self.reward_function.eval(self.state)
        if self.rank == 0:
            optimal_actions = []
            # if np.random.rand() < 0.5:
            #     for act in actions:
            #         if act is not None:
            #             optimal_actions += act
            if len(optimal_actions) > 0:
                actions = self.sample_actions(self.planning_horizon, optimal_actions)
                new_gen = np.concatenate([np.array(actions).reshape(1, self.planning_horizon), new_gen], axis=0)

        best_score = - np.inf
        best_sim_results = None
        simulated_steps = 0
        for i_gen in range(self.nb_gens):
            simulation_results = []
            for actions in new_gen:
                simulation_results.append(self.bundle_sim_results([self.simulate(self.hidden_state, actions,
                                                                                 self.prev_state, self.state, init_goal_reward=init_goal_reward)
                                                                   for _ in range(self.n_expansions)]))
            simulation_results = self.comm.gather(simulation_results, root=0)
            if self.rank == 0:
                simulation_results = sum(simulation_results, [])
                selected_sim_results, gen_best_sim_results = self.select(simulation_results)
                if gen_best_sim_results['selection_reward'] > best_score:
                    best_score = gen_best_sim_results['selection_reward']
                    best_sim_results = gen_best_sim_results
                if self.params['verbose']: print(f"      > gen {i_gen+1}/{self.nb_gens}, best score: {best_score:.2f}", end='\r')
                new_gen = self.mutate(selected_sim_results)
                new_gen += [best_sim_results['actions']]
                for sim_res in simulation_results:
                    simulated_steps += sim_res['simulated_steps']
                # scatter new_gen
                n_per_ranks = len(new_gen) // self.size
                gen_ids_per_rank = [list(range(i * n_per_ranks, (i + 1) * n_per_ranks)) for i in range(self.size)]
                for i in range(len(gen_ids_per_rank)):
                    if i > 0 and (i - 1) < (len(new_gen) % self.size):
                        gen_ids_per_rank[i].append(n_per_ranks * self.size + (i - 1))
                new_gen_per_ranks = [[new_gen[i] for i in gen_ids] for gen_ids in gen_ids_per_rank]
            else:
                new_gen_per_ranks = []
            new_gen = self.comm.scatter(new_gen_per_ranks, root=0)

        return best_sim_results, simulated_steps

    def bundle_sim_results(self, sim_res):
        max_len1 = np.max([len(res['game_rewards']) for res in sim_res])
        game_rew = list(np.mean(np.array([res['game_rewards'] + [0] * (max_len1 - len(res['game_rewards'])) for res in sim_res]), axis=0))
        goal_rew_diff = list(np.mean(np.array([res['goal_rewards_diff'] + [0] * (max_len1 - len(res['goal_rewards_diff'])) for res in sim_res]), axis=0))
        goal_rew = list(np.mean(np.array([res['goal_rewards'] + [0] * (max_len1 - len(res['goal_rewards'])) for res in sim_res]), axis=0))
        bundle = dict(game_rewards=game_rew,
                      goal_rewards=goal_rew,
                      goal_rew_diff=goal_rew_diff,
                      selection_reward=np.mean([res['selection_reward'] for res in sim_res]),
                      total_goal_reward=np.mean([res['total_goal_reward'] for res in sim_res]),
                      actions=sim_res[0]['actions'],
                      simulated_steps=np.sum([res['simulated_steps'] for res in sim_res]),
                      total_game_reward = np.mean([res['total_game_reward'] for res in sim_res]),
                      success = np.mean([res['success'] for res in sim_res]),
                      failure = np.mean([res['failure'] for res in sim_res]),
                      )
        return bundle

    def get_fresh_reward_func(self):
        return self.reward_function.copy()

    def safety_trigger(self, mental_model, state):
        enemies_poss = []
        enemy_speeds = []
        avatar_pos = None
        if len(mental_model.rules.enemies) > 0:
            for col in state:
                for cell in col:
                    for obj in cell:
                        if obj['name'] in mental_model.rules.enemies.keys():
                            enemies_poss.append(obj['pos'])
                            enemy_speeds.append(mental_model.rules.enemies[obj['name']])
                        if obj['name'] == mental_model.rules.avatar_name:
                            avatar_pos = np.array(obj['pos'])
            if avatar_pos is not None and len(enemies_poss) > 0:
                for enemie_pos, enemy_speed in zip(enemies_poss, enemy_speeds):
                    # how many enemy steps away from collision? (dist - 1 < 0)
                    # we want to flag if the enemy is less than 2 steps away
                    n_steps_x = np.ceil(max(0, (np.abs(enemie_pos[0] - avatar_pos[0]) - 0.999)) / enemy_speed)
                    n_steps_y = np.ceil(max(0, (np.abs(enemie_pos[1] - avatar_pos[1]) - 0.999)) / enemy_speed)
                    if (n_steps_x + n_steps_y) < 2:
                        # print(f'DANGER: {avatar_pos} == {enemie_pos} (speed={enemy_speed})')
                        return 1

        return 0

    def compute_colocation_cost(self, state):
        for col in state:
            for cell in col:
                for obj in cell:
                    if obj['name'] == 'avatar':
                        filtered_cell = [c for c in cell if c['name'] != self.mental_model.rules.obj.dict['avatar'].params.get('stype')]
                        if len(filtered_cell) > 1:
                            return - 0.5
                        else:
                            return 0
        return 0

    def noisy_action(self, action, past_actions, step_info):
        if len(past_actions) > 0 and (past_actions[-1] == action or past_actions[-1] in [-1, 4]):
            # players don't make mistake often when they didn't act at the previous step, or are repeating the same action
            prob_noisy = 0.01
        else:
            # players get better as they play more
            prob_noisy = np.linspace(0.1, 0.01, 15)[step_info['life']]
        noisy = np.random.rand() < prob_noisy
        if noisy:
            action = np.random.randint(5)
        return action

    def simulate(self, hidden_state, actions, prev_state, state, init_goal_reward, return_first_state=False, render=False, debug=False):
        self.time_tracker.tic('simulate')
        self.mental_model.set_state(hidden_state=hidden_state, state=state)  # set the hidden state once

        # print(f'Starting new simulation with score: {self.mental_model.env.unwrapped.game.score}')
        if render: self.mental_model.render()
        game_rewards = []
        goal_rewards = []
        first_state = None
        success, failure = False, False
        remaining_steps = 0
        simulated_steps = 0
        reward_func = self.get_fresh_reward_func()
        safety_trigger = 0
        past_actions = []
        for i_action, action in enumerate(actions):
            if self.params['exp_params']['noisy_action']:
                action = self.noisy_action(action, past_actions, self.step_info)
            past_actions.append(action)
            # plan for a given planning horizon, stop if you reach the end (either you die or timelimit)
            n_steps_to_run = self.mental_model.get_steps_to_run(action)
            # init = time.time()
            game_rewards.append(0)
            goal_rewards_this_action = []
            safety_trigger_this_step = 0
            for i_step_to_run in range(n_steps_to_run):
                self.time_tracker.tic('simulate_env_step')
                obs, reward, isOver, truncated = self.mental_model.step(action, return_state_pre_effect=True)
                self.time_tracker.toc('simulate_env_step')
                if render: self.mental_model.render()
                safety_trigger_this_step += self.safety_trigger(self.mental_model, obs['state'])
                colocation_cost = self.compute_colocation_cost(obs['state'])
                # if safety_trigger_this_step > 0:
                #     self.safety_trigger(self.mental_model, obs['state'])
                game_rewards[-1] += reward + colocation_cost

                # print(f'new reward: {reward}')
                action = self.null_action
                simulated_steps += 1
                self.time_tracker.tic('simulate_rew')
                goal_reached, goal_reward = reward_func.eval(obs['state'], events_triggered=obs['events_triggered'])[:2]
                goal_rewards_this_action.append(goal_reward)
                self.time_tracker.toc('simulate_rew')
                if isOver or goal_reached:
                    break
            goal_rewards.append(np.mean(goal_rewards_this_action))

            if i_action == 0 and return_first_state:
                first_state = obs['state']
            if obs['lose']:
                goal_rewards[-1] = self.lose_cost
            elif obs['won']:
                goal_rewards[-1] = self.win_reward
            elif safety_trigger_this_step > 0:
                goal_rewards[-1] += self.lose_cost / 2
            if debug: time.sleep(1)
            if isOver or goal_reached:
                remaining_steps = len(actions) - (i_action + 1)
                break

        if obs['won']:
            goal_rewards += [self.win_reward] * remaining_steps
            game_rewards += [0] * remaining_steps
            success = True
        elif isOver:
            goal_rewards += [self.lose_cost] * remaining_steps
            game_rewards += [0] * remaining_steps
            if obs['lose']:
                failure = True
        elif goal_reached:
            goal_rewards += [goal_rewards[-1]] * remaining_steps
            game_rewards += [0] * remaining_steps

        gamma = 0.95
        total_game_reward = np.sum([r * gamma ** (exponent) for exponent, r in zip(range(len(game_rewards)), game_rewards)])
        total_goal_reward = np.sum([(r - init_goal_reward) * gamma ** (exponent) for exponent, r in zip(range(len(goal_rewards)), goal_rewards)])
        total_selection_reward = total_goal_reward
        if goal_rewards[-1] > 0 or init_goal_reward == 0:  # if we make progress towards the goal
            total_selection_reward += total_game_reward


        simulation_results = dict(game_rewards=game_rewards,
                                  goal_rewards=goal_rewards,
                                  goal_rewards_diff=list(np.array(goal_rewards) - init_goal_reward),
                                  actions=list(actions),
                                  simulated_steps=simulated_steps,
                                  failure=failure,
                                  success=success,
                                  first_state=first_state,
                                  selection_reward=total_selection_reward,
                                  total_goal_reward=total_goal_reward,
                                  total_game_reward=total_game_reward)
        self.time_tracker.toc('simulate')

        return simulation_results
