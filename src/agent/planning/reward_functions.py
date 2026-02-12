from collections import deque
import time

import numpy as np
# import matplotlib.pyplot as plt

from src.utils import find_in_map, get_obj_ids_from_map, find_all_names_in_map, AVATAR_NAME

class RewardFunction:
    """
    Reward functions are used to guide the planner towards its goals.
    There are different types of goals:
    > avatar kills an object
    > avatar pushes an object onto another one to be killed
    > avatar pushed the obj to kill onto its killer
    > avatar spawns the killer (eg missile or flicker)
    for the three first, the value function is the distance between the killer and obj to kill (a star dist)
    for the last one, it's just 1 is the killed object disappeared, 0 if not
    """
    def __init__(self, goals, mental_model, params):
        self.goals = goals
        self.mental_model = mental_model
        self.params = params
        self.teleportations = dict()
        self.invalid_dist_rew = params['agent']['planner']['invalid_dist_rew']

    def copy(self):
        reward_func = RewardFunction(self.goals, self.mental_model, self.params)
        reward_func.teleportations = self.teleportations
        return reward_func

    #
    def eval(self, state, events_triggered=[]):
        if self.goals is None:
            return False, 0, []
        rewards = []
        optimal_action_seqs = []

        # print(self.goals)
        for goal in self.goals:
            if goal[0] in ['go_one_step_away', 'go_to', 'push_to', 'shoot_at']:
                # if goal[0] == 'shoot_at':
                #     stop = 1
                actor_obj = find_in_map(state, goal[1])
                target_obj = find_in_map(state, goal[2])
                distance = None
                if target_obj and actor_obj and target_obj['pos'] == actor_obj['pos']:
                    distance = 0
                    optimal_actions = []
                if distance is None:
                    for ev in events_triggered:
                        if {ev[1][1], ev[2][1]} == {goal[1], goal[2]}:
                            distance = 0
                            optimal_actions = []
                            # print(f'reward found in triggered events,{ev}')
                            break
                if goal[0] == 'shoot_at':
                    # is any killer on target?
                    reward = None
                    for ev in events_triggered:
                        if ({ev[1][0], ev[2][1]} == {goal[1], goal[2]}) or ({ev[1][1], ev[2][0]} == {goal[1], goal[2]}):
                            # print(f'reward found in triggered events, {ev}')
                            reward = 1
                            break
                    if reward is None:
                        actor_objs = find_all_names_in_map(state, goal[1])
                        if target_obj in actor_objs:
                            actor_objs.remove(target_obj)
                        if target_obj and len(actor_objs) > 0:
                            reward = int(any([obj['pos'] == target_obj['pos'] for obj in actor_objs]))
                        else:
                            reward = 0
                    optimal_action_seqs.append(None)
                    rewards.append(reward)
                else:
                    if distance is None:
                        if actor_obj:
                            resources = actor_obj['resources']
                        else:
                            resources = {}
                        tele_key = (self.mental_model.lvl, goal)
                        if tele_key not in self.teleportations.keys():
                            map = self.convert_map(state, actor=goal[1], target=goal[2])
                            self.teleportations[tele_key] = self.compute_teleportations(state, map)
                        map = self.convert_map(state, actor=goal[1], target=goal[2])
                        distance, optimal_actions = self.get_distance(map, self.teleportations[tele_key],
                                                                      resources=resources, pushing=goal[0]=='push_to')
                    if goal[0] in ['push_to', 'spawn_onto']:
                        optimal_action_seqs.append(None)
                    else:
                        optimal_action_seqs.append(optimal_actions)

                    if distance is None:
                        rewards.append(self.invalid_dist_rew)
                    elif distance < (3 if goal[0] == 'go_one_step_away' else 1):
                        rewards.append(1)
                    else:
                        reward = (len(state) + len(state[0]) - distance) / (len(state) + len(state[0]))
                        rewards.append(reward)
            else:
                raise NotImplementedError
            
        if rewards[-1] == 1:
            success = True
        else:
            success = False
        if np.any(np.array(rewards) == self.invalid_dist_rew):
            total_rewards = self.invalid_dist_rew
        elif len(self.goals) == 1:
            total_rewards = 2 * np.sum(rewards) * 3
        else:
            total_rewards = np.sum(rewards) * 3

        return success, total_rewards, optimal_action_seqs

    # create a new map with objects that don't move on their own and block the moving object (killer or killed)
    # this will be used to compute A* distance
    def convert_map(self, state, actor=None, target=None):
        
        # we code 1 for walls and blockers, 0 for empty spaces, 'g' for goal and 'k' for the killer
        clean_map = []
        assert target is not None
        if actor:
            moving_object_name = actor.split('.')[0]
        else:
            moving_object_name = None
        for row in state:
            clean_map.append([])
            for cell in row:
                obj_names = []
                obj_ids = []
                for obj in cell:
                    obj_names.append(obj['name'])
                    obj_ids.append(obj['obj_id'])
                if actor in obj_ids or actor in obj_names:  # the killer is here
                    character = 'k'
                elif target in obj_ids:  # the obj to kill is here
                    character = 'g'
                elif 'wall' in obj_names:
                    character = '1'
                else:
                    if moving_object_name is None:
                        character = '0'
                    else:
                        blocks = []
                        for obj in obj_names:
                            if not AVATAR_NAME in obj:
                                this_blocks, resource = self.does_block(obj, moving_object_name)
                                if this_blocks:
                                    if resource is None:
                                        blocks.append('1')
                                    else:
                                        blocks.append(resource)
                        blocks = sorted(set(blocks))
                        if len(blocks) == 0:
                            character = '0'
                        elif '1' in blocks:
                            character = '1'
                        elif len(blocks) == 1:
                            character = blocks[0]
                        else:
                            raise ValueError
                clean_map[-1].append(character)
        clean_map = np.array(clean_map)
        assert np.sum(clean_map == 'g') <= 1
        assert np.sum(clean_map == 'k') <= 1

        return clean_map

    def compute_teleportations(self, state, map):
        # for each entrance, we want to assign the exit that is the furthest away from the goal
        # to avoid getting stuck trying to optimistically find portals and exits
        entrances = []
        exits = []
        for obj, obj_line in self.mental_model.rules.obj.dict.items():
            if obj_line.type == 'Portal':
                entrances.append(obj)
                exits.append(obj_line.params['stype'])

        # now find teleportations
        done = False
        teleportations = dict()
        while not done:
            done = True
            all_failed = True
            for entrance, exit in zip(entrances, exits):
                entrances_ids, entrances_pos = get_obj_ids_from_map(state, obj_name=entrance, return_pos=True)
                exits_ids, exits_pos = get_obj_ids_from_map(state, obj_name=exit, return_pos=True)
                dists = [self.get_distance(from_pos=exit_pos, map=map, teleportations=teleportations)[0] for exit_pos in exits_pos]
                if all([d is None for d in dists]):
                    done = False
                    continue
                else:
                    for i_d in range(len(dists)):
                        if dists[i_d] is None:
                            dists[i_d] = 0
                    for entrance_pos in entrances_pos:
                        exit_pos = exits_pos[np.argmax(dists)]  # pick the worst exit
                        if entrance_pos not in teleportations.keys():
                            teleportations[entrance_pos] = exit_pos
                            all_failed = False
            if all_failed:
                done = True

        return teleportations

    def does_block(self, candidate_blocker, name_to_block):
        # whether candidate_block blocks name_to_block
        if self.mental_model.rules.obj.dict[candidate_blocker].moves() and \
            (self.mental_model.rules.obj.dict[candidate_blocker].type != 'Passive' or self.mental_model.rules.obj.dict.get((candidate_blocker, name_to_block)) == 'bounceForward'):
            return False, None
        else:
            if (candidate_blocker, name_to_block) in self.mental_model.rules.int.dict.keys():
                blocks = self.mental_model.rules.int.dict[(candidate_blocker, name_to_block)].blocks(who='second')
                if blocks:
                    return True, None
            if (name_to_block, candidate_blocker) in self.mental_model.rules.int.dict.keys():

                blocks = self.mental_model.rules.int.dict[(name_to_block, candidate_blocker)].blocks(who='first')
                if blocks:
                    if self.mental_model.rules.int.dict[(name_to_block, candidate_blocker)].type == 'killIfHasLess':
                        # can go through with the right resource
                        return True, self.mental_model.rules.int.dict[(name_to_block, candidate_blocker)].params['resource']
                    else:
                        return True, None
            return False, None

    def plot_map(self, map):

        code_to_rgb = {'0': (255, 255, 255),
                       '1': (0, 0, 0),
                       'g': (0, 255, 0),
                       'k': (255, 0, 0)}
        rgb_array = np.array([code_to_rgb[code] for code in map.T.flatten()]).reshape(*map.T.shape, 3)
        plt.figure()
        plt.imshow(rgb_array)
        plt.show()

    # A*
    def get_distance(self, map, teleportations, resources={}, from_pos=None, pushing=False):
        if np.sum(map == 'g') == 0:
            # goal has been killed
            return 0, []

        if from_pos is None:
            if np.sum(map == 'k') == 0:
                # killer is dead
                return None, []
            else:
                from_pos = tuple(np.argwhere(map == 'k').flatten())
                map[from_pos] = '0'
        else:
            from_pos = tuple(from_pos)

        goal = tuple(np.argwhere(map == 'g').flatten())
        rows, cols = len(map), len(map[0])
        if self.mental_model.rules.obj.type(self.mental_model.rules.avatar_name) == 'FlakAvatar':
            directions = [(0, 1), (1, 0)]  # right, down, left, up
        else:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        actions_str = ['down', 'right', 'up', 'left']
        actions_ids = [self.params['true_game_info']['act_str_to_act_id'][act_str] for act_str in actions_str]
        visited = set()
        queue = deque()
        queue.append((from_pos, 0, resources.copy(), []))  # (position, distance)
        visited.add(from_pos)
        if from_pos == goal:
            return 0, []
        
        while queue:
            (x, y), dist, res, actions = queue.popleft()
            for dir, action_id in zip(directions, actions_ids):
                dx, dy = dir
                # is pushing, make sure this direction can be taken
                if pushing:
                    pusher_pos = x - dx, y - dy
                    if pusher_pos[0] >= len(map) or pusher_pos[0] < 0 or pusher_pos[1] >= len(map[0]) or pusher_pos[1] < 0:
                        continue
                    if map[pusher_pos[0]][pusher_pos[1]] != '0':
                        continue
                res_dir = res.copy()
                actions_here = actions.copy()
                actions_here.append(action_id)
                nx, ny = x + dx, y + dy
                (nx, ny) = teleportations.get((nx, ny), (nx, ny))  # map portals to exit
                if 0 <= nx < rows and 0 <= ny < cols:  # within bounds and not visited
                    if map[nx][ny] not in ['0', 'g', 'k', '1']:  # here we have a barrier we can go through with resources
                        if map[nx][ny] in res_dir.keys():
                            res_dir[map[nx][ny]] -= 1 # remove one resource
                            if res_dir[map[nx][ny]] < 0:
                                continue  # the actor doesn't have enough resources to go through

                    if map[nx][ny] != '1' and (nx, ny, str(res_dir)) not in visited:  # can go through?
                        visited.add((nx, ny, str(res_dir)))
                        queue.append(((nx, ny), dist + 1, res_dir.copy(), actions_here.copy()))
                        if (nx, ny) == goal:  # test whether goal is reached
                            return dist + 1, actions_here.copy()
                        
        return None, []  # Return -1 if 'g' is not reachable from from_pos
