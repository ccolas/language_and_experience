import json
import os
import shutil
import time
import random
import sys
import argparse
import pickle

repo_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/'
sys.path.append(repo_path)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gym")

from mpi4py import MPI
import numpy as np
import src.vgdl as vgdl
import pygame
from datetime import datetime

from src.game.game import Game
from src.agent.agent import Agent
from src.baselines.llm_agent import LLMAgent
from src.baselines.rl_agent import DQNAgent
from src.utils import get_repo_path, find_experiment_path, mpi_fork, load_vgdl, BG_NAME, is_step_by_step, TimeDict, Logger, is_run_locally, prepare_run_continuation, clear_neighbor_cache

"""
Main experiment loop for "Language and Experience" paper.

Definitions:
- Episode: Ends when you win, die, or after max steps
- Life: One attempt at the game
- Generation: In chain mode, one agent's complete learning period

The main loop handles:
1. Setting up the game environment
2. Running the agent through episodes/lives
3. Coordinating inference across MPI workers
4. Logging results
"""

num_cpus = os.cpu_count()
if MPI.COMM_WORLD.rank == 0:
    print("Number of available CPUs:", num_cpus)

games = [
    "avoidGeorge",
    "beesAndBirds",
    "preconditions",
    "portals",
    "pushBoulders",
    "relational",
    "plaqueAttack",
    "aliens",
    "missile_command",
    "jaws",
    "test"
]
game = "relational"

# =============================================================================
# Default Parameters
# =============================================================================
# These parameters control the experiment, agent behavior, and inference settings.
# For most experiments, use run_experiment.py which provides a cleaner interface.

default_params = dict(
    # --- Global Settings ---
    verbose=True,           # Print detailed progress information
    debug=False,            # Enable debug mode with additional logging
    seed=None,              # Random seed (None = random)

    # --- Experiment Parameters ---
    exp_params=dict(
        exp_id="test",                                      # Experiment identifier for logging
        exp_path=get_repo_path() + 'data/inference_data/',  # Output directory for results
        trial_id=0,                                         # Trial number (for multiple runs)

        # Language settings
        language_likelihood_type='inverted_proposal',  # Method for computing P(language|theory)
        msg_to_load='machine_no_feedback',             # Source of language descriptions:
                                                       #   'human_individual', 'machine_no_feedback', etc.

        # Learning conditions (main experimental manipulations)
        use_interaction_likelihood=True,   # Use experience: P(observations|theory)
        use_language_likelihood=False,     # Use language in likelihood: P(description|theory)
        use_language_proposal=False,       # Use language in prior: bias hypothesis generation
        use_data_proposal=True,            # Use experience to guide hypothesis generation

        # Oracle/debugging settings
        use_oracle_data=False,  # Skip interaction, use pre-recorded trajectories
        data_to_load=None,      # Path to pre-recorded trajectory data

        # Episode/life settings
        stop_when_solved=True,       # End experiment when all levels completed
        comparison_immortal=False,   # Don't reset level on death (for baselines)
        n_lives_per_gen=15,          # Number of lives per generation

        # Chain/generational learning settings
        chain=False,        # Enable generational learning (multiple agents)
        n_gens=1,           # Number of generations (>1 for chain learning)
        agent_reset=True,   # Reset agent between generations (vs. cumulative learning)

        # Execution settings
        n_cpus=8,           # Number of CPU cores for parallel inference
        render=True,        # Display game visually
        noisy_action=False, # Add noise to action selection
    ),

    # --- Agent Parameters ---
    agent=dict(
        # Inference (theory learning) settings
        thinker=dict(
            alg='smc',              # Inference algorithm: 'smc' (main), 'llm', 'dqn' (baselines)
            schedule='every_20',    # When to run inference: 'every_N' = every N steps

            # SMC (Sequential Monte Carlo) settings
            n_smc_steps=1,                   # SMC update iterations per inference call
            n_particles=10,                  # Number of theory hypotheses to maintain
            n_mcmc_steps=5,                  # MCMC rejuvenation steps per particle
            n_simulations_likelihood=10,     # Simulation rollouts per particle for likelihood
            n_transitions_likelihood=250,    # Max timesteps to consider for likelihood

            # Prior settings
            prior_prob_no_int=0.75,          # Prior probability of no interaction between object types
            prior_prob_low=0.1,              # Prior probability for rare interaction types

            # Language model settings
            beta_softmax_lang_proposal=0.25, # Temperature for language-based proposal distribution
            aggregation_strategy='standard', # (Deprecated) Particle aggregation method
            llm_params=dict(
                llm_model=get_repo_path() + 'data/models/deepseek-coder-1.3b-instruct'
            )
        ),

        # Planning (action selection) settings
        planner=dict(
            planner_type='evo',              # Planning algorithm: 'evo' (evolutionary search)
            time_budget=2,                   # Seconds allowed for planning per decision
            max_subgoals_to_plan_for=5,      # Max number of subgoals to consider

            # Action selection
            stickiness=0.15,                 # Probability of repeating previous action
            discount=0.99,                   # Reward discount factor
            beta_explore_exploit=2,          # Exploration-exploitation trade-off

            # Planning horizon (adaptive)
            planning_horizon_info=(4, 10, 5), # (min, max, initial) planning depth

            # Safety checks (avoid dangerous states)
            safety_trials=20,     # Lookahead simulations for safety checking
            safety_distance=3,    # Steps to look ahead for safety
            invalid_dist_rew=-1000,  # Penalty for unsafe states

            # MCTS settings (not used with 'evo' planner)
            mcts_exploration_param=1.5,
        ),

        # Timing settings
        reaction_time=4,  # Frames between agent actions (simulates human reaction time)
        warmup=6,         # Initial observation period before first action
    ),

    # --- Game Display Settings ---
    game_params=dict(
        block_size=100  # Pixel size of game tiles
    )
)


class MainLoop:
    """
    Class supporting complete control of an experiment
    Can be used to run simulations with specified agent, game, etc. 
    """

    def __init__(self, params, game, linguistic_data=None):
        self.start_time = datetime.now()
        self.params = params
        self.params['linguistic_data'] = linguistic_data
        self.verbose = params['verbose']
        self.fork()                                                     # mpi fork
        self.n_gens = params['exp_params']['n_gens']
        self.n_lives_per_gen = params['exp_params']['n_lives_per_gen']
        self.render = params['exp_params']['render']
        self.game_id = game
        self.load_true_vgdl()
        self.setup_exp_folder()
        self.first_level = 0                                           # level to start with (default=0)
        self.logs = dict()

        if self.rank > 0:
            self.verbose = False
            params['verbose'] = False
            self.render = False
            self.logs = None

        self.time_tracker = TimeDict(rank=self.rank)               # track times
        self.params['time_tracker'] = self.time_tracker

        self.game = self.create_game()                             # true game
        self.agent = self.create_agent()

        if params['linguistic_data'] is not None and self.rank == 0:
            self.agent.store(linguistic_data=params['linguistic_data'])
        self.save_linguistic_data = None

    # Create the different MPI processes
    def fork(self):
        if not is_run_locally():
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.size
            self.n_cpus = self.size
            print(f'Nb cpus used: {self.n_cpus}')
        else:
            self.n_cpus = self.params['exp_params']['n_cpus']
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.size
            if self.n_cpus > 1 and self.comm.size == 1:
                if self.verbose: print(f"Forking into {self.n_cpus} processes...")
                whoami = mpi_fork(self.n_cpus, ['--bind-to', 'core']) #, oversubscribe=True)
                if whoami == 'parent':
                    sys.exit(0)
                self.comm = MPI.COMM_SELF.Spawn(command='python', args=[__file__], maxprocs=self.n_cpus)

    # create folder to store experiment results
    def setup_exp_folder(self):
        
        if self.params['seed'] is None:
            self.params['seed'] = np.random.randint(1e6)
        np.random.seed(self.params['seed'] + self.rank)
        random.seed(self.params['seed'] + self.rank)

        # create exp folder and save params
        self.exp_id = self.params['exp_params']['exp_id']
        self.trial_id = self.params['exp_params']['trial_id']
        continue_if_possible = self.params['exp_params']['chain'] and self.params['exp_params']['agent_reset']
        if self.rank == 0:
            self.exp_path, self.trial_id = find_experiment_path(self.params['exp_params']['exp_path'], self.game_id, self.exp_id, self.trial_id,
                                                                overwrite=continue_if_possible)
            self.continuing = os.path.exists(self.exp_path + 'theories_and_descriptions')  # continuing earlier run
            os.makedirs(self.exp_path + 'theories_and_descriptions', exist_ok=True)
        else:
            self.exp_path = None
        self.params['exp_params']['trial_id'] = self.trial_id
        self.exp_path = self.comm.bcast(self.exp_path, root=0)
        self.params['path'] = self.path
        self.params['exp_path'] = self.exp_path
        self.params['game'] = self.game_id

        self.start_gen = 0
        if self.rank == 0:
            if self.continuing:
                # need to find which generation to start from and to clean data from ongoing generation
                self.start_gen, description = prepare_run_continuation(self.exp_path, self.params)
                if description:
                    self.params['linguistic_data'] = description
            with open(self.exp_path + 'params.json', 'w') as f:
                json.dump(self.params, f)
            if self.verbose:
                if self.continuing:
                    print(f'continuing from {self.exp_path}')
                    print(f'generation: {self.start_gen}\nwith description:\n', description)
                else: print(f'logging to {self.exp_path}')
                print(self.params)
        self.params['linguistic_data'] = self.comm.bcast(self.params['linguistic_data'], root=0)
        self.start_gen = self.comm.bcast(self.start_gen, root=0)

    # Load VGDL script of the true game
    def load_true_vgdl(self):

        self.path = get_repo_path()
        true_vgdl_script = load_vgdl(self.path + f"games/{self.game_id}_v0/{self.game_id}.txt")

        # Local and store the level layouts in an array
        true_layouts = []
        formats = []
        lvl = 0
        while True:
            level_path = self.path + f"games/{self.game_id}_v0/{self.game_id}_lvl{lvl}.txt"
            if os.path.exists(level_path):
                with open(level_path, 'r') as f:
                    layout = f.read()
                layout_split = layout.split('\n')
                if len(layout_split[-1]) == 0:
                    layout_split = layout_split[:-1]
                formats.append((len(layout_split[0]), len(layout_split)))
                true_layouts.append(layout)
            else:
                break
            lvl += 1

        sprites, interactions, mapping, terminations, args = vgdl.VGDLParser().parse_game_for_theory(true_vgdl_script)

        character_mapping = dict()
        for map in mapping:
            names, character = map
            names = [n for n in names if n != BG_NAME]
            if len(names) > 0:
                character_mapping[character] = names
        assert "A" in character_mapping.keys()

        colors = dict()
        for sprite in sprites:
            if sprite[0] != BG_NAME:
                colors[sprite[0]] = sprite[2]['img'].split('/')[1]

        self.params['true_game_info'] = dict(vgdl_script=true_vgdl_script,
                                             layouts=true_layouts,
                                             formats=formats,
                                             character_mapping=character_mapping,
                                             step_by_step=is_step_by_step(self.game_id),
                                             colors=colors,
                                             fps=20)
        self.step_by_step = self.params['true_game_info']['step_by_step']
        self.fps = self.params['true_game_info']['fps']

    # Create the true game environment the agent interacts with
    def create_game(self):

        true_game = Game(self.params, vgdl_script=self.params['true_game_info']['vgdl_script'], is_true=True)

        # Create dictionaries mapping between action id and string
        action_space = true_game.get_action_space()
        valid_actions = self.valid_actions = [i for i in range(len(action_space))]      # Assign each action an integer id
        self.act_str_to_act_id = dict()
        self.act_id_to_act_str = dict()
        for act_id in valid_actions:
            if len(action_space[act_id].keys) > 0:
                act_key = action_space[act_id].keys[0]
                act_str = pygame.key.name(act_key)
            else:
                act_str = 'noop'
            self.act_id_to_act_str[act_id] = act_str
            self.act_str_to_act_id[act_str] = act_id
        self.null_action = np.argwhere(np.array([len(act.keys) == 0 for act in action_space])).flatten()[0]

        self.params['true_game_info']['action_space'] = action_space
        self.params['true_game_info']['valid_actions'] = valid_actions
        self.params['true_game_info']['act_str_to_act_id'] = self.act_str_to_act_id
        self.params['true_game_info']['act_id_to_act_str'] = self.act_id_to_act_str
        self.params['true_game_info']['null_action'] = self.null_action

        if self.render: true_game.render()
        return true_game
    
    # Create the agent
    def create_agent(self):
        if self.params['agent']['thinker']['alg'] == 'llm':
            return LLMAgent(self.game, self.params)
        elif self.params['agent']['thinker']['alg'] == 'dqn':
            return DQNAgent(self.game, self.params)
        else:
            return Agent(self.game, self.params)

    def test(self, sensory_data):
        if self.rank == 0:
            output_file = open(self.exp_path + "output.txt", "w")
            sys.stdout = Logger(sys.stdout, output_file)
            self.agent.store(sensorimotor_data=sensory_data)
        self.test_true_theory()
        self.agent.thinker.test()
        if self.rank == 0:
            output_file.close()

    # Compute the likelihood of the ground truth VGDL game description
    def test_true_theory(self):

        if self.rank == 0:
            all_data = self.agent.datastore.sample_data_for_likelihood()
        else:
            all_data = None

        if self.size > 1:
            all_data = self.comm.bcast(all_data, root=0)

        self.game.rules = self.agent.thinker.likelihood_computer.compute_likelihood(self.game.rules, self.agent.thinker.generative_model, all_data)[0]
        if self.verbose:
            print(f'Loglikelihood true theory: int: {self.game.rules.interaction_loglikelihood:.3f}, lang: {self.game.rules.language_loglikelihood:.3f}, '
                  f'total: {self.game.rules.loglikelihood:.3f}')

    # Run inference without planning
    def infer_only(self, sensory_data):
        if self.rank == 0:
            output_file = open(self.exp_path + "output.txt", "w")
            sys.stdout = Logger(sys.stdout, output_file)
            self.agent.store(sensorimotor_data=sensory_data)
        self.params['current_gen'] = 0
        self.test_true_theory()
        self.agent.thinker.think(step_info={}, n_steps=self.params['agent']['thinker']['n_smc_steps'])

    # Run one interaction episode
    def run_one_episode(self, step_info):
        
        # Initialize observation and transition data
        if self.rank == 0:
            obs = self.game.reset(with_img=False, lvl=self.lvl)
            if self.render: self.game.render()
            transition = dict(action=None, reward=None, done=None, step=0, img=obs['img'], state=obs['state'], won=None, lose=None,
                              events_triggered=None, lvl=self.lvl, reset=True)
            self.agent.store(sensorimotor_data=transition)  # store initial (null) 'transition'
        else:
            transition = None
        intermediate_reward = 0

        # Reset agent (including mental game and other internal variables)
        self.agent.reset(self.lvl)  

        # interact with the game for an episode
        done, env_step, actions, total_reward = False, 0, 0, 0
        lvl_solved = None
        while not done:

            self.time_tracker.tic('main_step')
            actions += 1
            step_info['env_step'] = env_step
            step_info['actions'] = actions
            step_info = self.comm.bcast(step_info, root=0)

            # get action from agent
            self.time_tracker.tic('main_act')
            if self.rank == 0: transition['reward'] = intermediate_reward
            act = self.agent.act(step_info, transition)
            intermediate_reward = 0
            self.time_tracker.toc('main_act')

            if self.rank == 0:
                # Collect and display step info
                time_since_start = time.time() - step_info['t_init']
                t_hours = int(time_since_start // 3600)
                t_min = int((time_since_start % 3600) // 60)
                t_sec = int(time_since_start - t_hours * 3600 - t_min * 60)
                lvl = step_info['lvl']
                i_life = step_info['life']
                i_episode = step_info['episode']
                avatar_pos = (self.game.env.unwrapped.game.sprite_registry.get_avatar().rect.x / 100, self.game.env.unwrapped.game.sprite_registry.get_avatar().rect.y / 100)
                if self.verbose: print(f'    > gen #{step_info["gen"]}, lvl #{lvl}, episode #{i_episode+1}, life #{i_life+1}, action #{actions}, env step #{env_step}, '
                                       f'time:{t_hours}:{t_min}: {t_sec}, n_cpus: {self.size}, pos: {avatar_pos}')

                # Execute action in the environment
                n_steps_to_run = self.game.get_steps_to_run(act)  # set 'reaction-time delay' between actions
                for _ in range(n_steps_to_run):

                    self.time_tracker.tic('main_true_env_step')
                    obs, reward, done, truncated = self.game.step(act, with_img=False)
                    self.time_tracker.toc('main_true_env_step')

                    self.time_tracker.tic('main_true_env_render')
                    if self.render: self.game.render(self.agent)
                    self.time_tracker.toc('main_true_env_render')

                    env_step += 1
                    transition = dict(step=env_step, action=act, reward=reward, done=done, img=obs['img'], state=obs['state'], won=obs['won'], lose=obs['lose'],
                                      events_triggered=obs['events_triggered'], lvl=lvl, reset=False)
                    lvl_solved = obs['won']
                    total_reward += reward
                    intermediate_reward += reward
                    self.agent.store(sensorimotor_data=transition.copy())
                    act = self.null_action                          # set action to null and continue simulating environment

                    if done:
                        if self.params['agent']['thinker']['alg'] in ['llm', 'dqn']:
                            transition['reward'] = intermediate_reward
                            self.agent.act(None, transition, end_episode=True)
                        break

            self.time_tracker.toc('main_step')

            self.time_tracker.step()  # advance the time tracker

            # bcast the step info and done signal
            self.agent.i_steps = self.comm.bcast(self.agent.i_steps, root=0)
            done = self.comm.bcast(done, root=0)

        if self.verbose: print(f"{' ' * 6}> agent reached the end of the episode")
        step_info = self.comm.bcast(step_info,  root=0)
        self.agent.think(step_info)
        self.agent.think(step_info)

        episode_step_tracker = dict(env_steps=env_step, actions=actions)       # Total time steps of simulation and actions taken by agent
        [lvl_solved, episode_step_tracker] = self.comm.bcast([lvl_solved, episode_step_tracker], root=0)

        return lvl_solved, episode_step_tracker

    # Run one agent life
    def run_one_life(self, step_info):

        # cancel language if stuck
        if self.rank == 0:
            if self.lvl < 2 and np.sum(np.array(self.levels) == self.lvl) > 3 and self.save_linguistic_data is None and not self.params['exp_params']['chain']:
                print('Ignoring language for now')
                # save language and ignore it if you're stuck
                self.save_linguistic_data = self.agent.datastore.linguistic_data
                self.agent.datastore.linguistic_data = []
                prior_file_path = self.exp_path + "prior.txt"
                if os.path.exists(prior_file_path):
                    shutil.copy(prior_file_path, prior_file_path.replace('prior', f'prior_start_life_{step_info["life"]}'))
            elif self.save_linguistic_data is not None and np.sum(np.array(self.levels) == self.lvl) == 1:
                print('Restoring language')
                # restore language when you move to the next level
                self.agent.datastore.linguistic_data = self.save_linguistic_data
                self.save_linguistic_data = None
        life_step_tracker = dict(episodes=0, env_steps=0, actions=0)
        
        dead, game_solved = False, False
        n_levels_solved = self.lvl
        while not (dead or game_solved):

            step_info['episode'] = life_step_tracker['episodes']
            step_info['lvl'] = self.lvl

            lvl_solved, episode_step_tracker = self.run_one_episode(step_info)
            for k, v in episode_step_tracker.items():
                life_step_tracker[k] += v

            life_step_tracker['episodes'] += 1

            if lvl_solved:
                if self.verbose: print(f'\n\n###########\n  > LVL {self.lvl} SOLVED!!\n#########\n\n')
                if self.verbose: print(f'  > lvl {self.lvl} solved in life {step_info["life"]+1}!')
                n_levels_solved += 1
                if n_levels_solved == len(self.params['true_game_info']['layouts']):
                    if self.verbose: print(f'\n\n###########\n  > GAME SOLVED!!\n#########\n\n')
                    game_solved = True
                else:
                    self.lvl += 1
            else:
                dead = True
        if self.agent.comm_engine is not None:
            print(f'Number of tokens used so far: {self.agent.comm_engine.total_tokens}')
        return n_levels_solved, game_solved, life_step_tracker

    # Play until the agent runs out of lives or solves the game
    def run_one_generation(self, step_info):

        gen_step_tracker = dict(episodes=0, env_steps=0, actions=0, lives=0)
        if not self.params['exp_params']['comparison_immortal']:
            self.lvl = self.first_level
        n_levels_solved = 0
        game_solved = False

        self.levels = []
        for i_life in range(self.n_lives_per_gen):
            self.levels.append(self.lvl)
            if self.verbose: print(f'> life #{i_life+1}/{self.n_lives_per_gen}, level {self.lvl}')
            step_info['life'] = i_life

            # Run one life
            n_levels_solved, game_solved, life_step_tracker = self.run_one_life(step_info)
            for k, v in life_step_tracker.items():
                gen_step_tracker[k] += v
            gen_step_tracker['lives'] += 1

            if self.rank == 0:
                life_step_tracker.update(dict(life=i_life,
                                              gen=step_info['gen'],
                                              n_levels_solved=n_levels_solved))
                print('\n\n', n_levels_solved)
                self.agent.dump_data(life_step_tracker)  # save interaction data

            if game_solved:
                break

        return n_levels_solved, game_solved, gen_step_tracker

    # Play a game for the number of generations stored in self.n_gens
    def run(self):
        print(self.game.rules.str_obj_llm(self.params['true_game_info']['colors']))
        t_init = time.time()

        if self.rank == 0:
            output_file = open(self.exp_path + "output.txt", "w")
            sys.stdout = Logger(sys.stdout, output_file)

        if self.params['exp_params']['chain']:
            assert self.n_gens > 1
        else:
            assert self.n_gens == 1

        self.lvl = self.first_level
        if self.first_level != 0:
            print('FIRST LEVEL IS NOT 0!!!')

        total_step_tracker = dict(episodes=0, env_steps=0, actions=0, lives=0)

        for i_gen in range(self.start_gen, self.n_gens):
            self.params['current_gen'] = i_gen
            
            total_step_tracker['generation'] = i_gen

            if self.verbose: print(f'-----\n\nGENERATION {i_gen+1}/{self.n_gens} -- game {self.game_id}')
            step_info = dict(gen=i_gen, t_init=t_init)

            n_levels_solved, game_solved, gen_step_tracker = self.run_one_generation(step_info)

            for k, v in gen_step_tracker.items():
                total_step_tracker[k] += v

            # log stuff
            if self.verbose: print(f'END OF GENERATION {i_gen+1}/{self.n_gens} -- game {self.game_id}\n'
                                   f'{n_levels_solved} levels solved\n')
            if self.params['exp_params']['stop_when_solved'] and game_solved:
                break


            if self.params['exp_params']['chain'] and self.params['exp_params']['agent_reset']:
                assert self.params['exp_params']['use_language_proposal'] or self.params['exp_params']['use_language_likelihood']
                self.agent.think_more = True
                self.agent.think(step_info)
                # describe the best theory and save it
                if self.rank == 0:
                    description = self.agent.comm_engine.generate_description(self.agent.thinker.best_ever)
                    with open(self.params['exp_path'] + f'theories_and_descriptions/description_gen_{i_gen}.txt', 'w') as f:
                        f.write(description)
                    with open(self.params['exp_path'] + f'theories_and_descriptions/theory_gen_{i_gen}.txt', 'w') as f:
                        f.write(self.agent.thinker.best_ever.vgdl_script)
                    # reset the agent with this description as input
                else:
                    description = None
                description = self.comm.bcast(description, root=0)
                self.params['linguistic_data'] = description
                clear_neighbor_cache()  # clear neighbor cache
                del self.agent
                self.agent = self.create_agent()
                if description is not None and self.rank == 0:
                    if self.verbose: print("\n\n----------------------\nResetting the agent")
                    if self.verbose: print(f'\n\ntransmitted description:\n{description}\n\n')
                    self.agent.store(linguistic_data=description)


        if self.verbose:
            print('Experiment is complete')
            end_time = datetime.now()
            elapsed = end_time - self.start_time
            # Format as hours:minutes:seconds
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"Total elapsed time: {hours:02d}:{minutes:02d}:{seconds:02d}")



# =============================================================================
# Direct Execution (for backwards compatibility)
# =============================================================================
# For most use cases, prefer run_experiment.py which provides a cleaner interface.
# This __main__ block is kept for direct execution and testing.

if __name__ == '__main__':

    repo_path = get_repo_path()
    models_path = os.environ.get("MODELS_PATH", os.path.join(repo_path, "data/models/"))
    exp_path = os.path.join(repo_path, "data/inference_data/")

    # Max steps per game (episode terminates after this many steps)
    game_max_steps = dict(
        avoidGeorge=510, beesAndBirds=1000, preconditions=500, relational=350,
        portals=1000, pushBoulders=500, plaqueAttack=1200, aliens=1300,
        jaws=510, missile_command=400, test=350
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run main experiment loop directly")
    parser.add_argument('--game', type=str, required=True, choices=games,
                        help="Game to play")
    parser.add_argument('--exp', type=str, default='test',
                        help="Experiment identifier")
    parser.add_argument('--trial', type=int, default=0,
                        help="Trial number")
    parser.add_argument('--msg-to-load', type=str, default=None,
                        help="Description source folder (e.g., 'machine_no_feedback')")
    parser.add_argument('--language-likelihood', action='store_true',
                        help="Use language in likelihood computation")
    parser.add_argument('--language-proposal', action='store_true',
                        help="Use language to guide hypothesis generation")
    parser.add_argument('--data-proposal', action='store_true', default=True,
                        help="Use experience to guide hypothesis generation")
    parser.add_argument('--chain', action='store_true',
                        help="Enable generational/chain learning")
    parser.add_argument('--thinker-alg', type=str, default='smc',
                        choices=['smc', 'llm', 'dqn'],
                        help="Learning algorithm")
    parser.add_argument('--llm-model', type=str, default='deepseek-coder-1.3b-instruct',
                        help="LLM model name (relative to MODELS_PATH)")
    parser.add_argument('--n-particles', type=int, default=20,
                        help="Number of SMC particles")
    parser.add_argument('--n-lives', type=int, default=15,
                        help="Number of lives per generation")
    parser.add_argument('--render', action='store_true',
                        help="Display game visually")
    parser.add_argument('--verbose', action='store_true', default=True,
                        help="Print detailed progress")
    args = parser.parse_args()

    # Configure parameters from arguments
    params = default_params.copy()
    params['verbose'] = args.verbose
    params['exp_params']['exp_id'] = args.exp
    params['exp_params']['trial_id'] = args.trial
    params['exp_params']['exp_path'] = exp_path
    params['exp_params']['render'] = args.render
    params['exp_params']['use_language_likelihood'] = args.language_likelihood
    params['exp_params']['use_language_proposal'] = args.language_proposal
    params['exp_params']['use_data_proposal'] = args.data_proposal
    params['exp_params']['msg_to_load'] = args.msg_to_load
    params['exp_params']['n_lives_per_gen'] = args.n_lives
    params['agent']['thinker']['alg'] = args.thinker_alg
    params['agent']['thinker']['n_particles'] = args.n_particles
    params['agent']['thinker']['llm_params']['llm_model'] = os.path.join(models_path, args.llm_model)

    # Configure chain learning
    if args.chain:
        params['exp_params']['chain'] = True
        params['exp_params']['n_gens'] = 10
        params['exp_params']['n_lives_per_gen'] = 2
        params['exp_params']['stop_when_solved'] = False
        params['exp_params']['agent_reset'] = True
    else:
        params['exp_params']['chain'] = False
        params['exp_params']['n_gens'] = 1
        params['exp_params']['stop_when_solved'] = True
        params['exp_params']['agent_reset'] = False

    # DQN baseline uses more lives
    if args.thinker_alg == 'dqn':
        params['exp_params']['n_lives_per_gen'] = 3000

    # Set max steps for this game
    params['max_steps'] = game_max_steps[args.game]

    # Load linguistic data if using language
    linguistic_data = None
    if args.language_likelihood or args.language_proposal:
        if args.msg_to_load:
            description_path = os.path.join(
                repo_path, 'data_input/descriptions', args.msg_to_load,
                f"{args.game}_{args.trial % 100}.txt"
            )
            if os.path.exists(description_path):
                with open(description_path, 'r') as f:
                    linguistic_data = f.read().strip()
            else:
                print(f"Warning: No description found at {description_path}")

    # Run experiment
    main = MainLoop(params, args.game, linguistic_data)
    main.run()

