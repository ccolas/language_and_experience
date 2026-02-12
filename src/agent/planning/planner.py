import itertools
import os

import pygame
import numpy as np
from mpi4py import MPI

from src.utils import get_obj_ids_from_map, pickle_save
from src.agent.planning.reward_functions import RewardFunction
from src.agent.planning.evolution import Evolution
from src.agent.planning.state_inference import infer_state
from src.agent.planning.goal_extraction import extract_goals
from src.game.game import Game

class Planner:
    """
    
    """
    def __init__(self, params, thinker):
        self.params = params
        self.thinker = thinker
        self.gen_model = thinker.generative_model
        self.time_tracker = params['time_tracker']
        self.data_path = params['exp_path'] + 'dumps/'
        os.makedirs(self.data_path, exist_ok=True)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size

        self.verbose = params['verbose']
        self.action_space = params['true_game_info']['action_space']
        self.valid_actions = params['true_game_info']['valid_actions']
        self.planner_type = self.params['agent']['planner']['planner_type']
        self.n_particles = self.params['agent']['thinker']['n_particles']
        self.mental_models = None    # Games you can play (as opposed to rules)

        self.safety_distance = params['agent']['planner']['safety_distance']
        self.safety_trials = params['agent']['planner']['safety_trials']
        self.invalid_dist_rew = params['agent']['planner']['invalid_dist_rew']  # Reward assigned when you can't find a path to the reward
        self.simulated_steps = 0
        self.planning_horizon_info = params['agent']['planner']['planning_horizon_info']  # How far to plan (adaptive)
        self.planning_horizon = self.planning_horizon_info[-1]
        self.beta_explore_exploit = params['agent']['planner']['beta_explore_exploit']
        self.history = []

        self.goal_tracker = dict()
        self.current_goal = None
        self.reset(lvl=0)
        self.first_states = None
        self.evo = None

    # Update the set of mental models based on rules in the posterior
    def update_mental_model(self, posterior_rules, id_rules_to_plan, lvl):

        # if self.mental_models is None or hash(self.mental_models) != hash(mental_model):
        self.mental_models = [Game(self.params, rules=rules) for rules in posterior_rules]
        for mental_model in self.mental_models:
            mental_model.reset(lvl=lvl)

        # Set mental model to highest posterior probability theory
        self.id_rules_to_plan = id_rules_to_plan
        self.mental_model = self.mental_models[id_rules_to_plan]
        if self.verbose:
            print(f'      > new mental model: {self.mental_model.rules}')
            if self.params['exp_params']['use_language_likelihood'] and self.thinker.comm_engine is not None:
                print(self.mental_model.rules.str_llm(self.thinker.comm_engine.colors))

        # Update current underlying state
        self.state_memory = dict(hidden_states=[], last_hidden_state=None, mem_n_spawned_per_step=dict())

        for goal in self.goal_tracker.keys():
            # recreate reward function
            self.goal_tracker[goal]['reward_func'] = RewardFunction(goal, self.mental_model, self.params)
        if self.current_goal is not None:
            self.current_rew_func = self.goal_tracker[self.current_goal]['reward_func']
        elif self.current_rew_func is not None:
            self.current_rew_func.mental_model = self.mental_model
            self.current_rew_func.teleportations = dict()

        if self.evo: self.evo.mental_model = self.mental_model

    # Reset everything for new level
    def reset(self, lvl=None):
        self.state_memory = dict(hidden_states=[], last_hidden_state=None, mem_n_spawned_per_step=dict()) # current underlying state
        self.plan = None
        self.current_goal = None
        self.goal_tracker = dict()
        self.current_rew_func = None
        self.planning_horizon = self.planning_horizon_info[-1]
        self.actions = []
        # if we keep the same mental models, we need to change their level
        if self.mental_models and lvl is not None and lvl != self.mental_model.lvl:
            for mental_model in self.mental_models:
                mental_model.reset(lvl=lvl)
            self.mental_model = self.mental_models[self.id_rules_to_plan]

    def is_plan_valid(self):
        # a plan is valid if:
        # the goal is still valid
        # there are still actions to take
        # you don't lose
        # the reward you expected is not out of distribution of your short term safety simulations

        if self.size > 1: self.plan = self.comm.bcast(self.plan, root=0)  # bcast plan to evaluate
        if self.plan['goal'] is None:
            if self.rank == 0:
                goal_available = len([goal for goal, info in self.goal_tracker.items() if info['is_active']]) > 0
            else:
                goal_available = None
            goal_available = self.comm.bcast(goal_available, root=0)
            if goal_available:
                return False, 'goals are now available'
        if self.plan['goal'] is not None and self.plan['goal'] not in self.goals:
            return False, 'previous goal is invalid'  # the goal is not achievable anymore
        elif len(self.plan['actions']) < self.safety_distance:
            self.adapt_planning_horizon('increment')
            return False, 'few actions left in the plan, replan for security'  # we've reached the end of the plan
        else:
            # is this plan safe in the short term?
            # does it increase reward?
            sim_results = []
            safety_trials = self.safety_trials if self.mental_model.is_stochastic else 1
            self.evo.reward_function = self.current_rew_func
            # we consider game rewards only if we make progress towards the goal, such that we're not distracted by goal-irrelevant rewards
            # init_goal_reward measures the initial goal-oriented reward so we can compute progress
            if self.rank == 0:
                init_goal_reward = self.goal_tracker[self.current_goal]['rewards'][-1] if self.current_goal else 0
            else:
                init_goal_reward = None
            init_goal_reward = self.comm.bcast(init_goal_reward, root=0)
            # scatter the work across cpus
            safety_trials_per_rank = [safety_trials // self.size for _ in range(self.size)]
            for i in range(safety_trials % self.size):
                safety_trials_per_rank[i] += 1
            safety_trials = safety_trials_per_rank[self.rank]

            for i_trial in range(safety_trials):
                sim_result = self.evo.simulate(self.state_memory['last_hidden_state'], self.plan['actions'][:self.safety_distance], self.prev_state,
                                               self.state, init_goal_reward, return_first_state=True)
                sim_results.append(sim_result)
                self.simulated_steps += sim_result['simulated_steps']

            # is the action sequence safe?
            fails = any([res['failure'] for res in sim_results])
            self.first_states = [res['first_state'] for res in sim_results]
            if self.size > 1:
                all_fails = self.comm.gather(fails, root=0)
                self.first_states = self.comm.gather(self.first_states, root=0)
                if self.rank == 0:
                    fails = any(all_fails)
                    self.first_states = sum(self.first_states, [])
                fails = self.comm.bcast(fails, root=0)

            if fails:
                self.adapt_planning_horizon('decrement')
                return False, 'i might die'

            # make sure the expected future rewards are in the distribution of short term forward looking (ie that we're on track)
            # if not, it means planning was too optimistic and something happened that makes that optimistic trajectory unlikely right now
            expected_sum_goal_reward = np.sum(self.plan['goal_rewards'][:self.safety_distance])  # reward expected from the plan
            sum_goal_rewards = [np.sum(res['goal_rewards']) for res in sim_results]  # samples of rewards in the safety simulations
            if self.size > 1:
                sum_goal_rewards = self.comm.gather(sum_goal_rewards, root=0)
                if self.rank == 0:
                    sum_goal_rewards = sum(sum_goal_rewards, [])
                    max_sum_rewards = np.max(sum_goal_rewards)
                else:
                    max_sum_rewards = None
                max_sum_rewards = self.comm.bcast(max_sum_rewards, root=0)
            else:
                max_sum_rewards = np.max(sum_goal_rewards)
            if expected_sum_goal_reward <= max_sum_rewards * 1.1:
                return True, 'plan is safe and in distribution'
            else:
                self.adapt_planning_horizon('decrement')
                return False, 'old plan has derailed'

    def convert_actions(self, actions):
        act_strs = []
        for act in actions:
            if len(self.params['true_game_info']['action_space'][act].keys) > 0:
                act_key = self.params['true_game_info']['action_space'][act].keys[0]
                act_str = pygame.key.name(act_key)
            else:
                act_str = 'noop'
            act_strs.append(act_str)
        return act_strs

    # Use the composite reward to determine which goal to pursue
    def select_goal(self, print_shift):
        # should we keep the current goal?
        if self.current_goal and not self.resample_goal:
            # print('N ATTEMPTS', self.n_attempts)
            if self.goal_tracker[self.current_goal]['is_active'] and self.n_attempts < 2:  # the goal needs to be still active
                rew = self.goal_tracker[self.current_goal]['rewards_when_targeted']  # rewards achieved when this goal was targeted by the agent

                if len(rew) < 20:  # keep on pursuing that goal if we've pursued it for fewer than 20 actions
                    if self.verbose: print(f'{" " * print_shift}  > keeping the old goal (not explored enough yet)')
                    return
                elif np.mean(rew[-10:]) > np.mean(rew[-20:-10]) > 0:  # we're making progress
                    rew_prog = np.mean(rew[-10:]) > np.mean(rew[-20:-10])
                    if self.verbose: print(f"{' ' * print_shift}  > keeping the old goal (we're making progress: {rew_prog:.2f})")
                    return
                else:
                    if self.verbose: print(f'{" " * print_shift}  > no progress made on the old goal, {rew}')

            else:
                if not self.goal_tracker[self.current_goal]['is_active']:
                    if self.verbose: print(f'{" " * print_shift}  > old goal now inactive')
                else:
                    if self.verbose: print(f'{" " * print_shift}  > old goal too risky right now')

        # we either don't have a goal yet, or are not making progress towards it, let's sample another
        # compute sampling probabilities
        goals = [goal for goal, info in self.goal_tracker.items() if info['is_active']]  # list active goals
        if self.current_goal in goals:
            goals.remove(self.current_goal)
        if len(goals) == 0:
            # no active goal, we don't have a goal
            self.current_goal = None
            if self.verbose: print(f'{" " * print_shift}  > no goal available')
            return
        elif len(goals) == 1:
            self.current_goal = goals[0]
        else:
            probs = self.compute_goal_probabilities(goals, print_probs=True)
            goal_id = np.random.choice(range(len(probs)), p=probs)
            self.current_goal = goals[goal_id]
            if self.verbose: print(f'{" " * print_shift}  > sampling new goal')

        if self.verbose: print(f'{" " * print_shift}  > current goal: {self.current_goal}')

    def compute_goal_probabilities(self, goals, discard_current_goal=True, print_probs=False):

        if discard_current_goal:
            current_goal = self.current_goal
        else:
            current_goal = None
        # compute priorities for each goal
        prios = np.array([self.goal_tracker[goal]['prio'] for goal in goals]) ** 2

        # current rewards towards each of them
        latest_rewards = np.array([self.goal_tracker[goal]['rewards'][-1] for goal in goals]) ** 3
        if latest_rewards.max() == latest_rewards.min():
            latest_rewards_tweaked = np.ones(len(self.goals))
        else:
            latest_rewards_tweaked = ((latest_rewards - latest_rewards.min()) / (latest_rewards.max() - latest_rewards.min())) + 0.5
        modulated_reward_info = []

        # modulate rewards by the progress (no progress = 0.25*reward)
        for goal, latest_rew, prio in zip(goals, latest_rewards_tweaked, prios):
            rew = self.goal_tracker[goal]['rewards_when_targeted']
            if len(rew) > 10:
                reward_progress = 1 if np.mean(rew[-5:]) > np.mean(rew[-10:-5]) else 0.25
            else:
                reward_progress = 1
            modulated_reward_info.append(reward_progress * latest_rew * prio)
            # select the closest goal unless you make no progress towards it
        modulated_reward_info = np.array(modulated_reward_info)


        # hierarchical sampling of goals
        goal_categories = dict()
        current_goal_cat = tuple(tuple(el.split('.')[0] for el in sub_g) for sub_g in current_goal) if current_goal is not None else None
        assert len(goals) == len(modulated_reward_info)
        for g, mod_rew in zip(goals, modulated_reward_info):
            g_cat = tuple(tuple(el.split('.')[0] for el in sub_g) for sub_g in g)
            if g_cat not in goal_categories.keys():
                goal_categories[g_cat] = dict(goals=[],
                                              scores=[])
            goal_categories[g_cat]['goals'].append(g)
            if g == current_goal:
                goal_categories[g_cat]['scores'].append(0)
            else:
                goal_categories[g_cat]['scores'].append(mod_rew)
        g_cats = list(goal_categories.keys())
        # sample goal category
        max_scores = np.array([max(goal_categories[g_cat]['scores']) for g_cat in g_cats])
        if current_goal in goals and current_goal_cat in g_cats:
            max_scores[g_cats.index(current_goal_cat)] = 0
        if max_scores.sum() == 0:
            probs_cat = np.ones(len(g_cats)) / len(g_cats)
        else:
            probs_cat = max_scores / max_scores.sum()
        if print_probs:
            print('proba goal categories')
            for g, p in zip(g_cats, probs_cat):
                print(g, "proba", p)
        for goal_cat in g_cats:
            low_scores = np.array(goal_categories[goal_cat]['scores'])
            if low_scores.sum() == 0:
                low_probs = np.ones(len(goals)) / len(goals)
            else:
                low_probs = low_scores / low_scores.sum()
            goal_categories[goal_cat]['probs'] = low_probs
        proba_goals = dict()
        if print_probs: print('goal probs info')
        for goal, rew, prio in zip(goals, latest_rewards_tweaked, prios):
            g_cat = tuple(tuple(el.split('.')[0] for el in sub_g) for sub_g in goal)
            goals_in_cat = goal_categories[g_cat]['goals']
            probs_in_cat = goal_categories[g_cat]['probs']
            p_goal = probs_cat[g_cats.index(g_cat)] * probs_in_cat[goals_in_cat.index(goal)]
            proba_goals[goal] = p_goal
            if print_probs:
                print(goal, "prob", p_goal, "prio", prio, "rew", rew)

        return np.array([proba_goals[g] for g in goals])

    def set_rew_func(self):
        if self.current_goal == None:
            self.current_rew_func = RewardFunction(None, self.mental_model, self.params)
        else:
            if self.current_goal not in self.goal_tracker.keys():
                self.goal_tracker[self.current_goal] = dict(reward_func=RewardFunction(self.current_goal, self.mental_model, self.params))
            self.current_rew_func = self.goal_tracker[self.current_goal]['reward_func']


    def infer_state_memory(self, current_episode):
        # compute hidden states so far
        if self.rank == 0:
            # set the previous and current state
            self.time_tracker.tic('planner_infer_state')
            if len(current_episode['traj']['state']) > 1:
                self.prev_state = current_episode['traj']['state'][-2]
            else:
                self.prev_state = 0
            self.state = current_episode['traj']['state'][-1]

            # infer hidden states (either just latest, or since beginning) + keep memory
            self.state_memory = infer_state(self.mental_model, current_episode, state_memory=self.state_memory, return_orientation=True)
            # set the hidden state (and check the states match)
            self.mental_model.set_state(self.state_memory['last_hidden_state'], self.state)
            self.time_tracker.toc('planner_infer_state')
        else:
            self.prev_state = None
            self.state = None
            self.state_memory = dict(last_hidden_state=None)

        if self.size > 1:
            self.prev_state = self.comm.bcast(self.prev_state, root=0)  # broadcast to others
            self.state = self.comm.bcast(self.state, root=0)  # broadcast to others
            self.state_memory['last_hidden_state'] = self.comm.bcast(self.state_memory['last_hidden_state'], root=0)  # broadcast to others

    def noisy_action(self, action, step_info):
        if len(self.actions) > 0 and (self.actions[-1] == action or self.actions[-1] in [-1, 4]):
            # players don't make mistake often when they didn't act at the previous step, or are repeating the same action
            prob_noisy = 0.01
        else:
            # players get better as they play more
            prob_noisy = np.linspace(0.1, 0.01, 15)[step_info['life']]
        noisy = np.random.rand() < prob_noisy
        if noisy:
            action = np.random.randint(5)
        return action

    # select a goal and plan towards it
    def act(self, current_episode, step_info, i_step, print_shift):
        # if self.verbose: print(f'{" " * print_shift}> act')

        if self.rank == 0:
            self.time_tracker.tic('planner_goals')
            # get available goals
            self.goals = self.get_goals(current_episode)
            # check whether they are active, and compute current rewards towards each
            self.reason_about_goals(i_step, self.prev_state, self.state)
            self.time_tracker.toc('planner_goals')

        else:
            self.goals = None

        if self.size > 1:
            self.goals = self.comm.bcast(self.goals, root=0)

        self.n_attempts = 0  # planning attempts, make sure you still act if all your plans are too risky
        need_new_plan, reason = self.need_new_plan()
        if not need_new_plan:
            if self.params['verbose']: print(f'{" " * print_shift}  > keeping old plan')
        while need_new_plan:
            if self.rank == 0:
                if self.params['verbose']: print(f'{" " * print_shift}  > ({self.n_attempts}) make a new plan because {reason}')
                # if i have no plan or the plan is deemed unsafe, i should replan
                self.time_tracker.tic('planner_goals')
                self.select_goal(print_shift)  # sample a goal
                self.time_tracker.toc('planner_goals')
            self.current_goal = self.comm.bcast(self.current_goal, root=0)  # broadcast to others
            self.set_rew_func()

            self.time_tracker.tic('planner_plan')
            self.make_plan(self.state_memory['last_hidden_state'], self.prev_state, self.state, self.current_goal, self.current_rew_func, step_info, print_shift + 2)  # find a plan
            # towards that goal
            self.time_tracker.toc('planner_plan')

            self.just_planned = True
            need_new_plan, reason = self.need_new_plan()
            self.n_attempts += 1
        if self.params['verbose']: print(f'{" " * print_shift}  > ({self.n_attempts}) going with this plan because: {reason}')

        # then I should execute the plan
        if self.rank == 0:
            action = self.plan['actions'].pop(0)
            self.plan['goal_rewards'].pop(0)
            self.plan['game_rewards'].pop(0)
            print(f'{" " * print_shift}  > taking action {self.convert_actions([action])}')
        else:
            action = None
        self.just_planned = False
        self.append_history(step_info)
        if self.params['exp_params']['noisy_action']:
            action = self.noisy_action(action, step_info)
        self.actions.append(action)
        return action

    def need_new_plan(self):
        self.time_tracker.tic('planner_validate_plan')
        check_plan, need_plan, reason = None, None, None
        self.resample_goal = False
        if self.rank == 0:
            # did new goals appear? if so, do they have much high priority?
            if len(self.newly_added_goals) > 0 and self.current_goal is not None and self.goal_tracker[self.current_goal]['is_active']:
                goals = [goal for goal, info in self.goal_tracker.items() if info['is_active']]  # list active goals
                if len(goals) > 1:
                    probs = self.compute_goal_probabilities(goals, discard_current_goal=False)
                    current_prob = probs[goals.index(self.current_goal)]
                    newly_added_probs = [p for p, g in zip(probs, goals) if g in self.newly_added_goals]
                    if len(newly_added_probs) > 0:
                        max_prob = np.max(newly_added_probs)
                        if max_prob > 10 * current_prob:
                            prio_goal = self.newly_added_goals[np.argmax(newly_added_probs)]
                            check_plan = False
                            need_plan, reason = True, "new high priority goal appeared"
                            print(f'new high priority goal: {prio_goal} (p={max_prob:.2f} vs old_p={current_prob:.2f})')
                            self.resample_goal = True
            if check_plan is None:
                # do we need to check the current plan?
                if self.n_attempts > 7:
                    check_plan = False
                    need_plan, reason = False, 'too many attempts'
                elif self.plan is None:
                    check_plan = False
                    need_plan, reason = True, 'there was no plan'
                else:
                    check_plan = True
        check_plan = self.comm.bcast(check_plan, root=0)
        assert check_plan is not None
        if check_plan:
            plan_is_valid, reason = self.is_plan_valid()
            need_plan = not plan_is_valid
        else:
            need_plan = self.comm.bcast(need_plan, root=0)
            reason = self.comm.bcast(reason, root=0)
        self.time_tracker.toc('planner_validate_plan')
        return need_plan, reason

    def append_history(self, step_info):
        assert step_info['lvl'] == self.mental_model.lvl
        to_save = step_info.copy()
        to_save.update(dict(n_simulated_step=self.simulated_steps, goal=self.current_goal))
        self.history.append(to_save)

    def dump_data(self, life_step_tracker):
        if self.rank == 0:
            name = f'planning_output_generation_{life_step_tracker["gen"]}_life_{life_step_tracker["life"]}_lvl_solved_{life_step_tracker["n_levels_solved"]}.pkl'
            pickle_save(self.history, self.data_path + name)
            self.history = []

    def make_plan(self, hidden_state, prev_state, state, goal, goal_reward_func, step_info, print_shift):
        if self.planner_type == 'evo':
            self.evo = Evolution(hidden_state, prev_state, state, self.mental_model, goal_reward_func, self.planning_horizon, step_info, self.time_tracker, self.params)
        else:
            raise ValueError(f"Unknown planner_type: {self.planner_type}. Only 'evo' is supported.")
        self.plan, simulated_steps = self.evo.run()
        if self.rank == 0:
            self.simulated_steps += simulated_steps
            self.plan['goal'] = goal
            act_str = ', '.join(self.convert_actions(self.plan['actions']))
            failure_str = '' if not self.plan['failure'] else ', expected death'
            if self.verbose: print(f'{" " * print_shift}  > found new plan (score={self.plan["selection_reward"]:.2f}): [{act_str}]{failure_str}')

    # List active goals and measure current reward towards them
    def reason_about_goals(self, i_step, prev_state, new_state):
        
        goal_list = [goal for goal in self.goal_tracker.keys() if self.goal_tracker[goal]['is_active']]
        self.newly_added_goals = []
        for goal, prio in self.goals.items():
            if goal not in self.goal_tracker.keys():
                self.goal_tracker[goal] = dict(i_step=i_step, goal=goal, rewards=[], prio=prio,
                                               rewards_when_targeted=[],
                                               reward_func=RewardFunction(goal, self.mental_model, self.params))
            elif goal in goal_list:
                goal_list.remove(goal)
            goal_reached, reward = self.goal_tracker[goal]['reward_func'].eval(new_state)[:2]
            if reward == self.invalid_dist_rew or goal_reached:
                # print(goal, goal_reached, reward)
                self.goal_tracker[goal]['is_active'] = False  # deactivate these goals
            else:
                if not self.goal_tracker[goal].get('is_active', False):
                    print(f'newly added, {goal}')
                    self.newly_added_goals.append(goal)
                self.goal_tracker[goal]['is_active'] = True  # activate these goals
            if goal == self.current_goal:
                self.goal_tracker[goal]['rewards_when_targeted'].append(reward)
            self.goal_tracker[goal]['rewards'].append(reward)
            self.goal_tracker[goal]['prio'] = prio
        # deactivate goals that are not listed anymore
        for goal in goal_list:
            # print('inactive goals', goal)
            self.goal_tracker[goal]['is_active'] = False

    def print_goals(self, all_task_goals, all_exploration_goals):
        print('Extracted goals')
        for task_goals, explo_goals in zip(all_task_goals, all_exploration_goals):
            print('---')
            if len(task_goals) > 0:
                print('Task goals:')
                for goal in task_goals:
                    print(f"{goal['actor']}, {goal['controlled']} ({goal['type']})), prio: {goal['prio']}")
            if len(explo_goals):
                print('Explo goals:')
                for goal in explo_goals:
                    print(f"{goal['actor']}, {goal['controlled']} ({goal['type']})), prio: {goal['prio']}")

    def get_goals(self, current_episode):

        # get list of goals from the environment and compute concrete goals in the current state
        # first get subgoals from each of the particles
        goals = extract_goals(self.mental_models, self.beta_explore_exploit)
        # if self.verbose: print(f'        > {len(goals)} abstract goals found')
        # filter goals that can't be achieved (one object is missing)
        # expand goals if there are several objects of one kind
        current_state = current_episode['traj']['state'][-1]
        extended_goals = dict()
        name_to_ids = dict()
        for goal in goals:
            # get indices of all object names in that goal
            names_in_subgoals = set()
            for subgoal in goal['goals']:
                name1, name2 = subgoal[1:3]
                if name1 is not None and subgoal[0] != 'shoot_at':
                    names_in_subgoals.add(name1)
                names_in_subgoals.add(name2)
                if name1 is not None and name1 not in name_to_ids.keys():
                    name_to_ids[name1] = get_obj_ids_from_map(current_state, name1)
                if name2 not in name_to_ids.keys():
                    name_to_ids[name2] = get_obj_ids_from_map(current_state, name2)

            # now expand all possible specific instances of that goal (replacing obj names by obj ids)
            names_in_subgoals = sorted(names_in_subgoals)
            obj_ids = [name_to_ids[name.split('+')[0]] for name in names_in_subgoals]
            # Use itertools.product to generate all combinations
            all_combinations = itertools.product(*obj_ids)
            # Iterate over each combination and use them as needed
            for combination in all_combinations:
                extended_goal = []
                for subgoal in goal['goals']:
                    if subgoal[1] is None:
                        name1 = None
                    elif subgoal[0] == 'shoot_at':
                        name1 = subgoal[1]
                    else:
                        name1 = combination[names_in_subgoals.index(subgoal[1])]
                    name2 = combination[names_in_subgoals.index(subgoal[2])]
                    if (name1 == name2) and (subgoal[1] != subgoal[2]):
                        break
                    extended_goal.append((subgoal[0], name1, name2))
                if len(extended_goal) < len(goal['goals']):
                    continue
                extended_goals[tuple(extended_goal)] = goal['prio']
        # if self.verbose: print(f'        > {len(goals)} concrete goals found')
        return extended_goals

    def adapt_planning_horizon(self, dir):
        # reduce planning horizon when your plans derail
        # increase planning horizon when your plans are implemented up to the end
        old_planning_horizon = self.planning_horizon
        if dir == 'increment':
            self.planning_horizon = min(self.planning_horizon + 1, self.planning_horizon_info[1])
        elif dir == 'decrement':
            self.planning_horizon = max(self.planning_horizon - 1, self.planning_horizon_info[0])
        else:
            raise NotImplementedError
        if self.verbose and old_planning_horizon != self.planning_horizon: print(f'        > new planning horizon: {self.planning_horizon}')

    def log(self):
        logs = dict(simulated_steps=self.simulated_steps)
        return dict(planner=logs)
