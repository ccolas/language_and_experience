import numpy as np
from mpi4py import MPI

from src.agent.thinking.thinker import SMCThinker, OracleThinker
from src.agent.planning.planner import Planner
from src.agent.datastore import DataStore
from src.game.game import Game
from src.utils import find_in_map
from src.agent.communicating.llm import CommunicationEngine


class Agent:
    """
    An agent for modeling and playing a VGDL game
    """
    def __init__(self, game, params):

        self.comm = MPI.COMM_WORLD                                      # MPI
        self.rank = self.comm.Get_rank()
        self.verbose = params['verbose'] if self.rank == 0 else False

        self.game = game
        self.params = params
        self.datastore = self.setup_data_store()                         # create the data manager
        self.time_tracker = params['time_tracker']

        self.thinking_schedule = params['agent']['thinker']['schedule']  # how often should the agent update its mental model? (periodic schedule)
        self.n_smc_steps = params['agent']['thinker']['n_smc_steps']
        self.think_more = False                                         # If true sets SMC steps to 10
        self.comm_engine = self.setup_comm_engine()                     # LLM interface
        self.thinker = self.setup_inference()                           # create the inference engine

        self.valid_actions = params['true_game_info']['valid_actions']
        self.random_actions = [0, 4, 4, 2, 5]
        np.random.shuffle(self.random_actions)
        self.planner = self.setup_planner()                             # create the planner

        self.lvl = 0
        self.known_lvls = []
        self.new_names = False
        self.reset(lvl=self.lvl)
     
    def setup_data_store(self):
        if self.rank == 0:
            return DataStore(self.params)
        else:
            return None
        
    def setup_comm_engine(self):
        if self.rank == 0:
            if self.params['exp_params']['use_language_likelihood'] or self.params['exp_params']['use_language_proposal']:
                return CommunicationEngine(self.time_tracker, self.params)
            else:
                return None
        else:
            return None

    def setup_inference(self):
        if self.params['agent']['thinker']['alg'] == 'oracle':
            return OracleThinker(self.params, self.game, self.datastore, self.comm_engine)
        elif self.params['agent']['thinker']['alg'] == 'smc':
            return SMCThinker(self.params, self.game, self.datastore, self.comm_engine)
        else:
            raise NotImplementedError
        
    def setup_planner(self):
        return Planner(self.params, self.thinker)

    # Reset planner to given level
    def reset(self, lvl):
        self.lvl = lvl
        if self.comm_engine:
            self.comm_engine.lvl = lvl
        if self.lvl not in self.known_lvls:
            # will need to warm up and generate a new mental model
            self.last_thought = None
            self.planner.reset()
        else:
            self.last_thought = 0
            self.planner.reset(lvl)
        self.i_steps = 0
    
    # Decide whether to think, then take the next action
    def act(self, step_info, transition):

        self.i_steps += 1

        # Check whether we are in warm-up phare
        if self.lvl not in self.known_lvls:
            if self.i_steps < self.params['agent']['warmup']:
                # warm up (take random actions) at each new level to collect minimum data to build a mental model
                return self.random_actions[self.i_steps - 1]
            else:
                self.known_lvls.append(self.lvl)  # add lvl to known levels once the warmup phase has ended

        # Think
        self.time_tracker.tic('agent_think')
        if self.think_now(transition):
            self.think(step_info)
        self.time_tracker.toc('agent_think')

        if self.rank == 0:
            current_episode = self.datastore.get_current_episode()
        else:
            current_episode = None
        
        for _ in range(10):
            # make sure we're not in a state where we think the game is lost already
            self.planner.infer_state_memory(current_episode)
            if self.rank == 0:
                _, lose = self.planner.mental_model.get_win_and_lose()
            else:
                lose = None
            lose = self.comm.bcast(lose, root=0)
            if lose:
                if self.verbose: print('Mental model thinks we should have lost -- need to think more')
                # if you think the game is lost but it isn't, you should think more
                self.time_tracker.tic('agent_think')
                self.think_more = False
                self.think(step_info)
                self.time_tracker.toc('agent_think')
            else:
                break

        # Plan
        self.time_tracker.tic('agent_plan')
        self.planner.infer_state_memory(current_episode)
        action = self.planner.act(current_episode, step_info, self.i_steps, print_shift=6)
        self.time_tracker.toc('agent_plan')

        return action

    # Determine whether to think
    def think_now(self, transition, print_shift=6):

        # let first rank decide whether to think
        if self.rank == 0:

            # think if new object appeared
            if self.new_names:                                                              
                self.think_more = True
                if self.verbose: print(f"{' ' * print_shift}> agent must think because a new object appeared")
                think_now = True
                self.new_names = False

            # think if the level is new
            elif self.last_thought is None:                                                 
                if self.verbose: print(f"{' ' * print_shift}> agent must think because it's a new level")
                think_now = True

            # if there is unexpected movement, think 30% of time
            elif not self.is_current_state_expected(transition) and np.random.rand() < 0.3:
                if self.verbose: print(f'{" " * print_shift}> agent must think because of unpredicted avatar mvt')
                think_now = True

            # think if you haven't thought in a while
            elif 'every' in self.thinking_schedule:                                         
                every = int(self.thinking_schedule.split('_')[1])
                if self.i_steps - self.last_thought >= every:
                    if self.verbose: print(f"{' ' * print_shift}> agent must think because it hasn't in a while")
                    think_now = True
                else:
                    think_now = False
            else:
                raise NotImplementedError
        else:
            think_now = False

        think_now = self.comm.bcast(think_now, root=0)  # broadcast to others
        self.think_more = self.comm.bcast(self.think_more, root=0)

        return think_now
    
    # Checks if the agent's position matches what we expected 
    def is_current_state_expected(self, obs):

        # Find agent's true position
        state = obs['state']
        agent_obj = find_in_map(state, self.datastore.avatar_name + '.1')
        if agent_obj:
            true_agent_pos = agent_obj['pos']
        else:
            true_agent_pos = None

        # Find agent's position in the planned states
        if self.planner.first_states is not None:
            agent_poss = []
            for state in self.planner.first_states:
                agent_obj = find_in_map(state, self.datastore.avatar_name + '.1')
                if agent_obj:
                    agent_poss.append(agent_obj['pos'])
                else:
                    agent_poss.append(None)
            if true_agent_pos in agent_poss:
                return True
            else:
                return False
        else:
            return True

    # Actually think (run inference of the game rules)
    def think(self, step_info):

        # Set number of SMC steps
        if self.think_more: # and self.params['exp_params']['use_language_likelihood']:
            n_smc_steps = 20
            self.think_more = False
        else:
            n_smc_steps = self.n_smc_steps

        self.thinker.think(step_info, n_steps=n_smc_steps, print_shift=6)

        # sample from the posterior
        posterior_rules = self.thinker.particles
        # if self.verbose: print(f'      > new rules: {posterior_rules}')
        posterior_rules = self.comm.bcast(posterior_rules, root=0)  # broadcast to others

        # update planner's mental model of the game
        self.planner.update_mental_model(posterior_rules, self.thinker.idx_rules_to_plan, self.lvl)
        self.last_thought = self.i_steps

    # Push transition data to datastore, store new names locally
    def store(self, sensorimotor_data=None, linguistic_data=None):
        self.time_tracker.tic('main_store_data')
        new_names = self.datastore.load(linguistic_data=linguistic_data, sensorimotor_data=sensorimotor_data)
        self.new_names = new_names or self.new_names
        self.time_tracker.toc('main_store_data')

    def dump_data(self, life_step_tracker):
        self.datastore.dump_data(life_step_tracker)
        self.thinker.dump_data(life_step_tracker)
        self.planner.dump_data(life_step_tracker)

    def log(self):
        planner_log = self.planner.log()
        thinker_log = self.thinker.log()
        datastore_log = self.datastore.log()
        agent_log = planner_log.copy()
        agent_log.update(thinker_log)
        agent_log.update(datastore_log)
        return agent_log
    