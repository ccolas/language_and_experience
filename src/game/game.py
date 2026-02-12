import os
import shutil
import gym

# import matplotlib
# matplotlib.use('TkAgg')  # As an example, switch to Qt5Agg before importing pyplot
import matplotlib.pyplot as plt

from src.utils import load_vgdl, register_game
from src.game.rules import Rules
from src.game.play_game import play_game
from src.agent.planning.state_inference import *

class Game:
    """
    Object supporting a game created from a VGDL script.
    It can be simulated, it can infer underlying hidden states and more.
    """

    def __init__(self, params, vgdl_script=None, vgdl_path=None, rules=None, is_true=False):
        assert vgdl_script is not None or vgdl_path is not None or rules is not None

        self.rules = None                                   # ? This is convoluted
        if vgdl_script is None:
            if vgdl_path is None:
                assert rules is not None                    # Not necessary
                self.rules = rules
                self.vgdl_script = self.rules.vgdl_script
            else:
                self.vgdl_script = load_vgdl(vgdl_path)
        else:
            self.vgdl_script = vgdl_script

        self.is_true = is_true                              # Indicates whether this is the true game or a mental game

        self.params = params
        if self.rules is None:
            self.rules = self.create_constraints_obj()      # Create rules from VGDL script
        self.character_mapping, self.colors, self.layouts = [self.params['true_game_info'][k] for k in ['character_mapping', 'colors', 'layouts']]
        self.goals = None                                   # Set of goals afforded by the game
        self.compute_info_for_binary_rep()                  # Compute info for state feature representations

        self.time_tracker = params['time_tracker']
        self.fig = None                                     # ?

        self.build_env()                                    # Create gym environment

    # Generates an object corresponding to VGDL game - object types and interaction rules
    def create_constraints_obj(self):
        return Rules(self.params, vgdl_script=self.vgdl_script)

    # Builds gym environment
    def build_env(self):

        # Make file path and write VGDL script to text file
        self.game_name = str(hash(self.rules)) + str(os.getpid())
        self.game_path = self.params['path'] + f'games/generated/{self.game_name}_v0/'
        os.makedirs(self.game_path, exist_ok=True)
        self.vgdl_script_path = self.game_path + f'{self.game_name}.txt'
        with open(self.vgdl_script_path, 'w') as f:
            f.write(self.vgdl_script)

        # Iterate through layouts and write to .txt files 
        # Create a map from level number to level filepath (?)
        self.i_ep_to_level_path = dict()
        layouts_to_lvl = dict()
        for i_ep, layout in enumerate(self.layouts):
            if layout not in layouts_to_lvl.keys():
                lvl = len(layouts_to_lvl)
            else:
                lvl = layouts_to_lvl[layout]
            level_path = self.game_path + f'{self.game_name}_lvl{lvl}.txt'
            with open(level_path, 'w') as f:
                f.write(layout)
            layouts_to_lvl[layout] = lvl
            self.i_ep_to_level_path[i_ep] = level_path

        # Register game using gym (?)
        self.game_id = f'{self.game_name}-v0'
        self.lvl = 0
        register_game(self.game_name, generated=True)
        self.env = gym.make(self.game_id, level_file=self.i_ep_to_level_path[self.lvl])
        shutil.rmtree(self.game_path)
        self.is_stochastic = self.env.unwrapped.game.is_stochastic
        self.max_steps = self.params['max_steps']
        self.reset()

    # 
    def make_game(self, lvl=0):
        os.makedirs(self.game_path, exist_ok=True)
        for i_lvl, layout in enumerate(self.layouts):
            level_path = self.game_path + f'{self.game_name}_lvl{i_lvl}.txt'
            with open(level_path, 'w') as f:
                f.write(layout)
        vgdl_script_path = self.game_path + f'{self.game_name}.txt'
        with open(vgdl_script_path, 'w') as f:
            f.write(self.vgdl_script)
        self.env = gym.make(self.game_id, level_file=self.i_ep_to_level_path[lvl])
        shutil.rmtree(self.game_path)

    # Resets gym environment to the given level
    def reset(self, with_img=False, lvl=0):
        
        if lvl == self.lvl:
            obs, info = self.env.reset(with_img=with_img)
        else:
            self.lvl = lvl
            if self.fig:
                plt.close(self.fig)
            self.fig = None
            self.goals = None
            self.make_game(self.lvl)
            obs, info = self.env.reset(with_img=with_img)
        info['img'] = obs

        return info
    
    def get_steps_to_run(self, action):
        step_by_step = self.params['true_game_info']['step_by_step']
        null_action = self.params['true_game_info']['null_action']
        if action == null_action or step_by_step:
            n_steps_to_run = 2
        else:
            n_steps_to_run = self.params['agent']['reaction_time']
        return n_steps_to_run

    def get_action_space(self):
        if 'action_space' in self.params['true_game_info'].keys():
            return self.params['true_game_info']['action_space']
        else:
            return self.env.unwrapped._action_keys

    def step(self, action, no_obs=False, return_state_pre_effect=False, with_img=False):
        state, reward, isOver, truncated, info = self.env.unwrapped.step(action, no_obs=no_obs, with_img=with_img, return_state_pre_effect=return_state_pre_effect)
        info['img'] = state
        if self.env.unwrapped.game.time >= self.max_steps:
            if self.is_true:
                print('TIMEOUT')
            isOver = True
        return info, reward, isOver, truncated

    def render(self, agent=None):
        # self.time_tracker.tic('render_setup')

        block_size = self.params['game_params']['block_size']
        render_block_size = block_size / 2
        subsampling_factor = 4

        self.env.render()
        # if self.fig is None:
        #     self.fig, self.ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        #     title = 'True game' if self.is_true else 'Mental Game'
        #     img = self.env.unwrapped.render(mode='rgb_array')
        #     subimg = img[::subsampling_factor, ::subsampling_factor]
        #     self.img = self.ax.imshow(subimg)
        #     self.text_labels = []
        #     self.ax.set_title(title)
        #     y_ticks = np.arange(render_block_size / 2, img.shape[0], render_block_size) / subsampling_factor
        #     y_ticks_labels = np.arange(0, img.shape[0] // render_block_size).astype(int)
        #     self.ax.set_yticks(y_ticks)
        #     self.ax.set_yticklabels(y_ticks_labels)
        #     x_ticks = np.arange(render_block_size / 2, img.shape[1], render_block_size) / subsampling_factor
        #     x_ticks_labels = np.arange(0, img.shape[1] // render_block_size).astype(int)
        #     self.ax.set_xticks(x_ticks)
        #     self.ax.set_xticklabels(x_ticks_labels)
        # else:
        #     img = None
        #
        # # self.time_tracker.toc('render_setup')
        # # self.time_tracker.tic('render_img_update')
        #
        # if img is None:
        #     img = self.env.unwrapped.render(mode='rgb_array')
        #     subimg = img[::subsampling_factor, ::subsampling_factor]
        #     self.img.set_data(subimg)
        #
        # # self.time_tracker.toc('render_img_update')
        # # self.time_tracker.tic('render_goal')
        # # plot goal info
        #
        # if agent is not None and agent.planner is not None:
        #     current_goal = agent.planner.current_goal
        #     while len(self.text_labels) > 0:
        #         self.text_labels.pop().remove()
        #     if current_goal is not None:
        #         self.ax.set_xlabel(current_goal, fontweight='bold')
        #         for g in current_goal:
        #             for c in g[1:]:
        #                 try:
        #                     short_key = c.split('.')[0][:3] + '.' + c.split('.')[1]
        #                     rect = self.env.unwrapped.game.sprite_registry._sprite_by_id[c].rect
        #                     pos_x, pos_y = np.array([rect.x, rect.y]) / self.env.unwrapped.game.block_size * block_size
        #                     pos = np.array(((pos_x + block_size / 2) / 2, (pos_y + block_size / 2) / 2)) / subsampling_factor
        #                     new_text = self.ax.text(pos[0], pos[1], short_key, ha='center', va='center',
        #                                  fontsize=12, color='k', fontweight='bold')
        #                     self.text_labels.append(new_text)
        #                 except:
        #                     pass
        #
        # # self.time_tracker.toc('render_goal')
        # # self.time_tracker.tic('render_canvas_update')
        # self.fig.canvas.draw_idle()
        # # self.time_tracker.toc('render_canvas_update')
        # # self.time_tracker.tic('render_pause')
        # plt.pause(0.00000001)  # Short pause to allow the figure to update
        # self.time_tracker.toc('render_pause')
        
    #
    def compute_info_for_binary_rep(self):
        self.names2id = dict()
        self.avatar2id = dict()
        self.resources = []
        self.orientations = dict()
        i_avatar = 0
        i_name = 0
        for name in self.rules.names:
            if self.rules.obj.type(name) == 'ResourcePack':
                self.resources.append(name)
            if name in self.rules.avatar_names:
                if self.rules.obj.type(name) == 'ShootAvatar':
                    self.orientations[name] = True
                else:
                    self.orientations[name] = False
                self.avatar2id[name] = i_avatar
                i_avatar += 1
            else:
                self.names2id[name] = i_name
                i_name += 1

    # Compute binary feature representation of game state
    def compute_binary_rep(self, hidden_state, state):
        width, height = len(state), len(state[0])
        binary_dict = dict()
        for name in self.rules.names:
            if name != 'wall':
                if name in self.rules.avatar_names:
                    array = np.zeros
                    if self.orientations[name]:
                        n_orientations = 4
                    else:
                        n_orientations = 1
                    array = np.zeros([width, height, n_orientations] + [self.env.unwrapped.resources_max_info[r] for r in self.resources])
                else:
                    array = np.zeros([width, height])
                binary_dict[name] = array
        for i_row, row in enumerate(state):
            for i_col, cell in enumerate(row):
                for obj in cell:
                    if obj['name'] != 'wall':
                        if obj['name'] in self.rules.avatar_names:

                            if self.orientations[obj['name']]:
                                try:
                                    orientation_vec = hidden_state['agent_orientation'][obj['obj_id']]
                                except:
                                    print( hidden_state['agent_orientation'])
                                    stop = 1
                                orientation_vec = (np.sign(orientation_vec.x), np.sign(orientation_vec.y))
                                orientation = ['UP', 'DOWN', 'LEFT', 'RIGHT'].index(inv_dir_dict[orientation_vec])
                            else:
                                orientation = 0
                            resources = [obj['resources'][r] for r in range(len(self.resources))]
                            coordinates = [i_row, i_col, orientation] + resources
                            binary_dict[obj['name']][tuple(coordinates)] = 1
                        else:
                            binary_dict[obj['name']][i_row, i_col] = 1
        binary_rep = np.concatenate([array.flatten() for array in binary_dict.values()])
        return binary_rep

    #
    def set_state(self, hidden_state, state=None):
        if state:
            img, obs = self.env.unwrapped.set_state(hidden_state, return_obs=True, with_img=False)
            if not check_hidden_inference(obs['state'], state):
                img, obs = self.env.unwrapped.set_state(hidden_state, return_obs=True, with_img=False)
                assert False
        else:
            self.env.unwrapped.set_state(hidden_state)

    def get_win_and_lose(self):
        self.env.unwrapped.game._check_terminations()
        return self.env.unwrapped.game.won, self.env.unwrapped.game.lose
    #
    def get_state(self, return_orientation=False):
        return self.env.unwrapped.get_state(return_orientation=return_orientation)


    def play(self, lvl=0):
        print(f'Playing game {self.game_id}')
        register_game(self.game_id, level=lvl, fast=False)
        play_game(self.game_id, lvl=lvl, step_by_step=self.params['true_game_info'].get('step_by_step', False))


    def copy(self):
        return Game(self.params, vgdl_script=self.vgdl_script)
    
    #
    def save_to(self, path):
        sub_dirs = os.listdir(path)
        for sub_dir in sub_dirs:
            shutil.rmtree(path + sub_dir)
        game_path = path + f'{self.game_name}_v0/'
        os.makedirs(game_path, exist_ok=True)
        for i_lvl, layout in enumerate(self.layouts):
            level_path = game_path + f'{self.game_name}_lvl{i_lvl}.txt'
            with open(level_path, 'w') as f:
                f.write(layout)
        vgdl_script_path = game_path + f'{self.game_name}.txt'
        with open(vgdl_script_path, 'w') as f:
            f.write(self.vgdl_script)


# Sanity check of observations
def check_hidden_inference(reconstructed_state, state):

    for i_column in range(len(reconstructed_state)):
        for i_row in range(len(reconstructed_state[i_column])):
            observed_cell = state[i_column][i_row]
            reconstructed_cell = reconstructed_state[i_column][i_row]
            if len(observed_cell) != len(reconstructed_cell):
                stop = 1
                return False
            for obs_el in observed_cell:
                valid = False
                for sim_el in reconstructed_cell:
                    if sim_el['name'] == obs_el['name'] and tuple(obs_el['pos']) == sim_el['pos']:
                        valid = True
                        break
                if not valid:
                    return False
            for sim_el in reconstructed_cell:
                valid = False
                for obs_el in observed_cell:
                    if sim_el['name'] == obs_el['name'] and tuple(obs_el['pos']) == sim_el['pos']:
                        valid = True
                        break
                if not valid:
                    return False
                
    return True
