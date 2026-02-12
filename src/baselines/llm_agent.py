"""
Pure LLM Baseline Agent.

This module implements a pure large language model agent that uses text-based
reasoning to play games directly, without explicit Bayesian inference over
game theories. The agent receives state descriptions and uses prompting to
decide on actions.

This baseline is described in the paper's experimental comparison section.
"""

import time
import numpy as np
from src.agent.datastore import DataStore
from src.utils import get_repo_path, AVATAR_NAME, pickle_save
import os
import pygame
from mpi4py import MPI

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    return prompt


class LLMAgent:
    def __init__(self, game, params):
        self.game = game
        self.comm = MPI.COMM_WORLD  # MPI
        self.rank = self.comm.Get_rank()
        self.verbose = params['verbose'] if self.rank == 0 else False
        prompt_path = get_repo_path() + 'data_input/vgdl_doc/'
        self.sys_prompt = load(prompt_path + 'prompt_system.txt')
        self.agent_prompt = load(prompt_path + 'prompt_llm_agent')
        self.params = params
        self.colors = params['true_game_info']['colors']
        if self.rank == 0:
            self.setup_llm(self.params)
        self.datastore = self.setup_data_store()                         # create the data manager
        self.time_tracker = params['time_tracker']
        self.lvl = 0
        self.get_last_n = 3
        self.reset(lvl=self.lvl)
        self.random_actions = [0, 4, 4, 2, 5]
        self.known_lvls = []
        self.wall_description = dict()
        self.new_names = False
        self.planner = None
        self.scratchpad = ""
        self.comm_engine = None
        self.history = []
        self.data_path = params['exp_path'] + 'dumps/'

    def setup_data_store(self):
        if self.rank == 0:
            return DataStore(self.params)
        else:
            return None

    def reset_cache_proposal(self):
        self.cache_proposal = dict()

    def setup_llm(self, params):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        model_path = params['agent']['thinker']['llm_params']['llm_model']
        # Use full path instead of model_id
        print(f'using model path: {model_path}')

        # Initialize tokenizer with full path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        n_gpus = 1 # if '70b' in model_path.lower() else 1
        if 'deepseek' in model_path.lower():
            self.max_len = 3800
            gpu_use = 0.5
        elif '70b' in model_path.lower():
            self.max_len = 4500
            gpu_use = 0.6
        else:
            self.max_len = 4000
            gpu_use = 0.6

        import os
        import torch
        import gc

        # Print initial state
        print("\n=== Initial State ===")
        print(f"CUDA available: {torch.cuda.is_available()}")
        memory = 0
        allocated = 0
        max_allocated = 0
        device_count = torch.cuda.device_count()

        for i in range(device_count):
            memory += torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
            max_allocated += torch.cuda.max_memory_allocated(i) / 1024 ** 3
        print(f"Total GPU memory: {memory:.2f} GB")
        print(f"Currently allocated: {allocated:.2f} GB")
        print(f"Max allocated: {max_allocated:.2f} GB")

        # Set allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        # Clear memory
        print("\n=== Clearing Memory ===")
        torch.cuda.empty_cache()
        gc.collect()
        allocated = 0
        for i in range(device_count):
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
        print(f"After clear - Currently allocated: {allocated:.2f} GB")

        # Try loading model
        print("\n=== Loading Model ===")
        self.model = LLM(
            model=model_path,
            tensor_parallel_size=n_gpus,
            max_model_len=self.max_len,
            gpu_memory_utilization=gpu_use,
            max_num_batched_tokens=self.max_len + 100,
            max_num_seqs=1,
            enforce_eager=True,
            max_seq_len_to_capture=self.max_len,
            enable_prefix_caching=False,  # Disabled to avoid logprob computation issues
        )

        # Set sampling parameters to be memory efficient
        self.max_gen_tokens = 800
        self.sampling_params = SamplingParams(
            max_tokens=self.max_gen_tokens ,
            temperature=0.5,
            top_k=50,
            top_p=0.95,
            # stop_token_ids=stop_token_ids
        )

        # Print final state
        print("\n=== Final State ===")
        allocated = 0
        max_allocated = 0
        for i in range(device_count):
            allocated += torch.cuda.memory_allocated(i) / 1024 ** 3
            max_allocated += torch.cuda.max_memory_allocated(i) / 1024 ** 3
        print(f"Final allocated: {allocated:.2f} GB")
        print(f"Final max allocated: {max_allocated:.2f} GB")

    def store(self, sensorimotor_data=None, linguistic_data=None):
        self.time_tracker.tic('main_store_data')
        new_names = self.datastore.load(linguistic_data=linguistic_data, sensorimotor_data=sensorimotor_data)
        self.new_names = new_names or self.new_names
        self.time_tracker.toc('main_store_data')

    def reset(self, lvl):
        self.lvl = lvl
        self.i_steps = 0

    def count_tokens(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def construct_prompt(self, step_info, end_episode):
        if step_info and step_info['env_step'] == 0:
            self.history = []
        if end_episode:
            end_instruction = "The episode has terminated. Reflect on what caused you to win or lose and update your scratchpad. Now answer using the following format (analysis, append/update). No need to generate an action since the episode has ended."
        else:
            end_instruction = "Now answer following the format above (analysis, append / update, act)"
        scratchpad = self.scratchpad if len(self.scratchpad) > 0 else "The scratchpad is currently empty."
        if len(scratchpad.split(' ')) > 400:
            scratchpad = ' '.join(scratchpad.split(' ')[:400]) + '\nThe scratchpad is full, please keep it under 400 words.'
        if len(self.datastore.linguistic_data) > 0:
            linguistic_data = "\n".join(self.datastore.linguistic_data)
            linguistic_data = f"#### Help\n\nHere is a description of the game sent by another player:\n{linguistic_data}\n\n"
        else:
            linguistic_data = ""

        # get data
        # crop prompt if needed
        for get_last_n in range(self.get_last_n + 1 , 0, -1):
            data_str, steps_strs, state_str = self.format_data(end_episode, get_last_n=get_last_n)
            steps_str = ''.join(steps_strs)
            user_prompt = (f"{self.agent_prompt}\n\n"
                           f"### TASK\n\n"
                           f"{linguistic_data}"
                           f"#### Recent history:\n\n"
                           f"{data_str}\n"
                           f"{steps_str}\n"
                           f"#### Old scratchpad\n"
                           f"{scratchpad}\n\n"
                           f"{end_instruction}")
            messages = [{'role': 'system', "content": self.sys_prompt},
                        {'role': 'user', 'content': user_prompt}]

            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = self.count_tokens(prompt)
            if tokens + self.max_gen_tokens < self.max_len:
                print('n_tokens', tokens)
                break
            else:
                print(self.count_tokens(self.agent_prompt), self.count_tokens(data_str), self.count_tokens(steps_str), self.count_tokens(scratchpad), self.count_tokens(end_instruction))
                print('n_tokens', tokens, f'too long! removing one step ({len(steps_strs)} left)')
        return prompt, user_prompt, scratchpad, state_str

    def act(self, step_info, transition, end_episode=False):
        if self.rank == 0:
            self.i_steps += 1

            if self.lvl not in self.known_lvls:
                if self.i_steps < self.params['agent']['warmup']:
                    # warm up (take random actions) at each new level to collect minimum data to build a mental model
                    return self.random_actions[self.i_steps - 1]
                else:
                    self.known_lvls.append(self.lvl)  # add lvl to known levels once the warmup phase has ended
            prompt, user_prompt, scratchpad, state_str = self.construct_prompt(step_info, end_episode)
            # Generate text
            outputs = self.model.generate(prompt, self.sampling_params, use_tqdm=False)
            text = outputs[0].outputs[0].text
            action_id, parsed_output = self.parse_outputs(text, end_episode)
            parsed_output['scratchpad'] = self.scratchpad
            if self.verbose:
                if end_episode:
                    print('end episode')
                    print(user_prompt)
                else:
                    print(state_str)
                print('\nanalysis:', parsed_output['analysis'])
                print('\nscratchpad:', self.scratchpad)
                print('\naction:', parsed_output['act'])
            parsed_output['user_prompt'] = user_prompt
            self.history.append(parsed_output)

            # action_id = np.random.randint(4)
        else:
            action_id = None
        return action_id

    def parse_outputs(self, text, end_episode):
        parsing_errors = []
        info = dict()
        action_space = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NOOP', 'SPACE_BAR']
        expected_keys = ['analysis', 'append', 'replace', 'act']
        if end_episode:
            expected_keys.remove('act')
            info['act'] = None
        for key in expected_keys:
            parsed_text = None
            if f"<{key}>" in text:
                parsed_text = text.split(f'<{key}>')[1].strip()
                if f"</{key}>" in parsed_text:
                    parsed_text = parsed_text.split(f'</{key}>')[0].strip()
                else:
                    parsing_errors.append(key)
            else:
                parsing_errors.append(key)
            if key == 'act' and parsed_text not in action_space:
                if 'act' not in parsing_errors: parsing_errors.append(key)
                info[key] = None
            else:
                info[key] = parsed_text

        if len(parsing_errors) > 1 or len(parsing_errors) > 0 and parsing_errors[0] not in ['append', 'replace']:
            if self.verbose:
                print(f'Parsing errors in llm: {parsing_errors}:\n{text}')
        if info['act'] not in action_space:
            info['action_id'] = np.random.randint(5)
        else:
            info['action_id'] = action_space.index(info['act'])
        if info['append']:
            self.scratchpad += f' {info["append"]}'
        if info['replace']:
            self.scratchpad = info['replace']
        return info['action_id'], info


    def format_pos(self, obj_name, positions):
        if len(positions) == 0:
            return ""
        pos_str = f"{obj_name.capitalize()} objects are located in: "
        for pos in positions:
            x, y = pos
            x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
            y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
            pos_str += f'({x}, {y}), '
        return pos_str[:-2]

    def format_data(self, end_episode, get_last_n=None):
        get_last_n = get_last_n or self.get_last_n
        episode = self.datastore.get_current_episode()
        if self.lvl not in self.wall_description.keys():
            init_state = episode['traj']['state'][0]
            wall_positions = []
            for col in init_state:
                for cell in col:
                    for obj in cell:
                        if obj['name'] == 'wall':
                            wall_positions.append(obj['pos'])
            self.wall_description[self.lvl] = self.format_pos("walls", wall_positions)
        wall_str = self.wall_description[self.lvl] + '\nWalls are permanent and remain in these positions during the whole game.'

        last_n_states = get_last_n * 4
        total_len = len(episode['traj']['state'])
        start_step = total_len - min(last_n_states, total_len)
        last_states = episode['traj']['state'][start_step:]
        last_actions = episode['traj']['action'][start_step:] + [4]
        last_dones = [False] + episode['traj']['done'][start_step:]
        if end_episode:
            assert last_dones[-1]
        last_wins = [False] + episode['traj']['won'][start_step:]
        last_loses = [False] + episode['traj']['lose'][start_step:]
        last_rews = [0] + episode['traj']['reward'][start_step:]
        if start_step == 0:
            data_str = f"The game was just reset to a new episode, you're playing level {self.lvl}.\n\n{wall_str}\n"
        else:
            data_str = f"Here is the recent history of game states and actions.\n\n{wall_str}\n"

        steps_strs = []

        for i in range(len(last_states)):
            state = last_states[i]
            action, done, win, lose, rew = last_actions[i], last_dones[i], last_wins[i], last_loses[i], last_rews[i]
            action = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NOOP', 'SPACE_BAR'][action]
            step = start_step + i
            step_str = ""
            if step > 0:
                # print mvt
                step_str += self.format_events(step, episode['events'][step], done, win, lose, rew)
            if action != 'NOOP' or i in [0, len(last_states) - 1]:
                # print state
                state_str = self.format_state(step, state, done, win, lose)
                step_str += state_str
            if action != 'NOOP':
                # print action
                step_str += f"\nYou took action {action}.\n"
            steps_strs.append(step_str)

        return data_str, steps_strs, state_str

    def format_events(self, step, events, done, win, lose, rew):
        events_str = f"\n---\nHere is what happened between step {step-1} and step {step}:\n"
        event_dict = dict()
        # group events
        for event in events:
            if (event[0] in ['mvt', 'birth', 'death', 'resource_change']):
                event_key = event[0] + '_' + event[1]
                if event_key not in event_dict.keys():
                    event_dict[event_key] = []
                event_dict[event_key].append(event)
        if len(event_dict) == 0:
            events_str = f"\n---\nNothing happened between step {step-1} and step {step}\n"
            return events_str
        keys = sorted(event_dict.keys())
        avatar_keys = []
        for key in keys:
            if 'avatar' in key:
                avatar_keys.append(key)
                keys.remove(key)
        keys = avatar_keys + keys
        for key in keys:
            evs = event_dict[key]
            event_type, obj_name = evs[0][:2]
            if obj_name == 'avatar':
                obj_color = 'You'
            else:
                obj_color = f"{self.colors[obj_name].capitalize()} objects"
            if event_type == 'mvt':
                mvt_str = f'- {obj_color} have moved: '
                for ev in evs:
                    x, y = ev[3]
                    x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
                    y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
                    prev_pos = f"({x}, {y})"
                    x, y = ev[4]
                    x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
                    y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
                    new_pos = f"({x}, {y})"
                    mvt_str += f'from {prev_pos} to {new_pos}, '
                events_str += mvt_str[:-2] + '\n'
            elif event_type == 'birth':
                birth_str = f'- {obj_color} have appeared at: '
                for ev in evs:
                    x, y = ev[3]
                    x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
                    y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
                    birth_str += f"({x}, {y}), "
                events_str += birth_str[:-2] + '\n'
            elif event_type == 'death':
                death_str = f'- {obj_color} have died at: '
                for ev in evs:
                    x, y = ev[3]
                    x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
                    y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
                    death_str += f"({x}, {y}), "
                events_str += death_str[:-2] + '\n'
            elif event_type == 'resource':
                resource_str = f'- {obj_color} now have: '
                for ev in evs:
                    r_name = ev[5]
                    resource_str += f'{r_name}, '
                events_str += resource_str[:-1] + 'resources\n'
        if rew > 0:
            events_str += f"You gained {rew} points!\n"
        elif rew < 0:
            events_str += f"You lost {rew} points!\n"
        if done and win:
            events_str += "The episode has terminated, you won!\n"
        elif done and lose:
            events_str += "The episode has terminated, you lost!\n"
        elif done:
            events_str += "The episode has terminated, you lost because of a timeout!\n"
        else:
            events_str += ""

        return events_str

    def format_state(self, step, state, done, win, lose):


        state_str = f"\n---------GAME STATE AT STEP {step}"
        if done and win:
            state_str += " (wining state)"
        if done and lose:
            state_str += " (losing state)"
        state_str += '\n'


        position_dict = dict()
        your_position = None
        your_resource = None
        for col in state:
            for cell in col:
                for obj in cell:
                    if obj['name'] != 'wall':
                        if obj['name'] == 'avatar':
                            your_position = obj['pos']
                            your_resource = obj['resources']
                        else:
                            obj_color = self.colors[obj['name']].lower()
                            if obj_color not in position_dict.keys():
                                position_dict[obj_color] = []
                            position_dict[obj_color].append(obj['pos'])
        if your_position:
            x, y = your_position
            x = f"{int(x)}" if int(x) == x else f"{x:.1f}"
            y = f"{int(y)}" if int(y) == y else f"{y:.1f}"
            your_position_str = f'({x}, {y})'
            state_str += f"- You are located at {your_position_str}\n"
            resource_str = ""
            for key, val in your_resource.items():
                if val > 0:
                    resource_str += f"{val} {self.colors[key].lower()} objects, "
            if len(resource_str) > 0:
                state_str += "- You have collected " + resource_str[:-2] + '\n'
        else:
            state_str += "- You are dead.\n"
        keys = sorted(position_dict.keys())
        for key in keys:
            pos = position_dict[key]
            state_str += f'- {self.format_pos(key, pos)}\n'
        return state_str


    def think(self, step_info):
        pass

    def dump_data(self, life_step_tracker):
        self.datastore.dump_data(life_step_tracker)
        name = f'thinking_output_generation_{life_step_tracker["gen"]}_life_{life_step_tracker["life"]}_lvl_solved_{life_step_tracker["n_levels_solved"]}.pkl'
        pickle_save(self.history, self.data_path + name)
