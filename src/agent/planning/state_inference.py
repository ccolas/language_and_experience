import numpy as np
from pygame.math import Vector2
from src.utils import detect_collisions_in_map, find_id_closest_in_map, inv_dir_dict, dir_dict, timeout
import time
from mpi4py import MPI
import time
shoot_avatar_mapping = {0: np.array([0, -1]),
                        1: np.array([0, 1]),
                        2: np.array([-1, 0]),
                        3: np.array([1, 0])}

size = MPI.COMM_WORLD.size
rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

# Infer the number of new objects spawned at each step during this episode
def infer_nb_spawned(params, obj_id, obj_episode, state_episode, obj_state):
    n_spawned_per_step = [0]
    obj = obj_episode[obj_id]
    pos = obj['pos']
    width, height = len(state_episode[0]), len(state_episode[0][0])
    block_size =  params['game_params']['block_size']
    format_episode = dict(w=width, h=height, bs=block_size)

    # search for candidate spawned objects
    for i_step, p in enumerate(pos):
        spawned = False
        if i_step > 0:
            prev_pos = pos[i_step - 1]
            if prev_pos is not None and (p is None or p == prev_pos):
                if obj_state['type'] == 'ShootAvatar':
                    speed = obj_state['params']['speed']
                    spawner_ids = []
                    # the spawnee must be one step away from the previous position
                    possible_poss = [tuple(np.array(prev_pos) + mov) for mov in np.array([[0, speed], [0, -speed], [-speed, 0], [speed, 0]])]
                    for possible_pos in possible_poss:
                        spawner_ids += detect_collisions_in_map(possible_pos, state_episode[i_step], i_step, format_episode,
                                                                obj_episode, obj_id=obj['obj_id'], external_collisions=False)
                else:
                    spawner_ids = detect_collisions_in_map(prev_pos, state_episode[i_step], i_step, format_episode,
                                                           obj_episode, obj_id=obj['obj_id'], external_collisions=False)
            else:
                spawner_ids = []

            for spawner_id in spawner_ids:
                if obj_episode[spawner_id]['name'] == obj_state['params']['stype']:
                    # was it just born?
                    spanwer_pos = obj_episode[spawner_id]['pos']
                    if spanwer_pos[i_step - 1] is None and spanwer_pos[i_step] is not None:  # it just appeared
                        spawned = True
                        break

        n_spawned_per_step.append(n_spawned_per_step[-1] + int(spawned))

    return n_spawned_per_step

# Search for the given object in the hidden state
def search_hidden_state(hidden_states, i_step, obj_id):
    for hidden_state in hidden_states[i_step][0]:
        if hidden_state['obj_id'] == obj_id:
            return hidden_state
    assert False

# Infer the orienation of the given object
def infer_orientation(rules, obj, i_step, state_episode, obj_episode, hidden_states, pos, obj_id, depth=0):
    orientation_vector = None
    if obj['mov'][i_step] is None:
        # obj just appeared, use initial orientation
        if 'orientation' in rules.obj.params(obj['name']).keys():
            orientation = rules.obj.params(obj['name'])['orientation']
        else:
            # find oriented spawner
            spawner = None
            for o_name, o in rules.obj.dict.items():
                if 'stype' in o.params.keys():
                    if o.params['stype'] == obj['name']:
                        spawner = o_name
                        break
            if spawner:
                spawner_id = find_id_closest_in_map(state_episode[i_step], spawner, pos)
                if spawner_id == None or depth > 0:
                    orientation = 'UP'
                else:
                    spawner_obj = obj_episode[spawner_id]
                    spawner_pos = spawner_obj['pos'][i_step]
                    orientation_vector = infer_orientation(rules, spawner_obj, i_step, state_episode, obj_episode, hidden_states, spawner_pos, spawner_id, depth=depth+1)
            else:
                orientation = 'UP'  # assume default orientation
    elif np.sum(np.abs(obj['mov'][i_step])) > 0:
        if rules.obj.type(obj['name']) in ['Bomber', 'Missile'] and rules.obj.params(obj['name']).get('orientation') in ['RIGHT', 'LEFT'] and obj['mov'][i_step] == (0, 1):
            # might have turned around
            past_orientation = search_hidden_state(hidden_states, i_step - 1, obj_id)['params']['orientation']
            past_orientation = np.array([past_orientation.x, past_orientation.y])
            expected_pos = tuple(np.array(obj['pos'][i_step]) + past_orientation)
            width, height = len(state_episode[0]), len(state_episode[0][0])
            block_size = rules.params['game_params']['block_size']
            format_episode = dict(w=width, h=height, bs=block_size)
            objs_facing = detect_collisions_in_map(expected_pos, state_episode[i_step], i_step, format_episode, obj_episode)
            for obj_facing in objs_facing:
                interaction = rules.int.dict.get((obj['name'], obj_facing.split('.')[0]))
                if interaction and interaction.type == 'turnAround':
                    orientation_vector = Vector2(-past_orientation[0], -past_orientation[1])
                    break
        if orientation_vector is None:
            orientation_vector = Vector2(np.sign(obj['mov'][i_step][0]), np.sign(obj['mov'][i_step][1]))
    else:
        # find last movement and assume same orientation
        orientation_vector = None
        cooldown = rules.obj.params(obj['name']).get('cooldown', 1)
        first_mov = np.argwhere(np.array([mov is not None for mov in obj['mov']])).flatten()[0]
        for i_s in range(i_step):
            mov = obj['mov'][i_step - i_s]
            step_here = i_step - i_s
            if mov is not None:
                if rules.obj.type(obj['name']) in ['Bomber', 'Missile'] and rules.obj.params(obj['name']).get('orientation') in ['RIGHT', 'LEFT'] and mov == (0, 1):
                    # it turned around
                    orientation_vector = search_hidden_state(hidden_states, i_step - 1, obj_id)['params']['orientation']
                    break
                elif 0 < np.sum(np.abs(mov)) <= 1:
                    # orientation following last mov
                    orientation_vector = Vector2(np.sign(mov[0]), np.sign(mov[1]))
                    break
                elif np.sum(np.abs(mov)) == 0:
                    should_have_moved = (step_here - first_mov) % cooldown == 0
                    if should_have_moved:
                        # could it have reverse dir?
                        try:
                            past_orientation = search_hidden_state(hidden_states, i_step - 1, obj_id)['params']['orientation']
                            past_orientation = np.array([past_orientation.x, past_orientation.y])
                            expected_pos = tuple(np.array(obj['pos'][i_step]) + past_orientation)
                            width, height = len(state_episode[0]), len(state_episode[0][0])
                            block_size = rules.params['game_params']['block_size']
                            format_episode = dict(w=width, h=height, bs=block_size)
                            objs_facing = detect_collisions_in_map(expected_pos, state_episode[i_step], i_step, format_episode, obj_episode)
                            for obj_facing in objs_facing:
                                interaction = rules.int.dict.get((obj['name'], obj_facing.split('.')[0]))
                                if interaction and interaction.type == 'reverseDirection':
                                    orientation_vector = Vector2(-past_orientation[0], -past_orientation[1])
                                    break
                        except:
                            pass
                    if orientation_vector is not None:
                        break

        # same orientation as before
        if orientation_vector is None:
            try:
                orientation_vector = search_hidden_state(hidden_states, i_step - 1, obj_id)['params']['orientation']
            except:
                orientation_vector = Vector2(*tuple(dir_dict['RIGHT']))
                stop=1

    if orientation_vector is None:
        direction = dir_dict[orientation]
        orientation_vector = Vector2(*tuple(direction))

    return orientation_vector

# Infer missing elements of hidden game state from past trajectory of observation and actions
# Operates on existing theory

# Usage:
@timeout(seconds=10)
def timed_infer_state(lvl, game, current_episode, state_memory=None, n_simulations=1, last_step=None, return_orientation=False):
    game.reset(lvl=lvl)
    hidden_state = infer_state(game, current_episode, state_memory, n_simulations, last_step, return_orientation)
    return game, hidden_state

def infer_state(game, current_episode, state_memory=None, n_simulations=1, last_step=None, return_orientation=False):
    t_init_1 = time.time()
    traj_episode, obj_episode = current_episode['traj'], current_episode['objs']
    if last_step is None:
        last_i_step = traj_episode['step'][-1]
    else:
        last_i_step = last_step
    actions = traj_episode['action']
    state_episode = traj_episode['state']
    if state_memory is None:
        state_memory = dict(hidden_states=[], mem_n_spawned_per_step=dict())  # Memory associated with this theory
    hidden_states = state_memory['hidden_states'].copy()             # Copy previous and continue filling
    mem_n_spawned_per_step = state_memory['mem_n_spawned_per_step'].copy()
    hidden_states += [[[] for _ in range(n_simulations)] for _ in range(last_i_step + 1 - len(hidden_states))]
    if return_orientation:
        orientations = dict()
    # print('t prep', time.time() - t_init_1)
    times = dict(orientation=[], spawn=[], search=[], total=[], objs=dict())
    t_init_3 = time.time()
    for obj_id, obj in obj_episode.items():
        t_init_2 = time.time()
        positions = obj['pos']
        name = obj['name']
        if name == 'wall':
            continue
        for i_step in range(len(state_memory['hidden_states']), last_i_step + 1):
            if len(positions) > i_step:
                pos = positions[i_step]
                if pos is not None:  # only track hidden state of object that exist
                    otype = game.rules.obj.type(name)
                    # fill layout
                    obj_state = dict(name=name,
                                     type=otype,
                                     pos=pos,
                                     img=f"colors/{game.colors[obj['name']]}",
                                     obj_id=obj_id,
                                     params=game.rules.obj.params(name).copy())
                    obj_state['resources'] = obj['resources'][i_step].copy()

                    # Type of object tells us what information we need to fill in
                    if otype in ['Passive', 'Immovable', 'VerticalAvatar', 'HorizontalAvatar', 'MovingAvatar', 'FlakAvatar', 'Portal']:
                        pass
                    elif otype == 'ResourcePack':
                        obj_state['params']['value'] = 1
                        obj_state['params']['res_type'] = None
                    elif otype == 'Missile':
                        t_init_4 = time.time()
                        obj_state['params']['orientation'] = infer_orientation(game.rules, obj, i_step, state_episode, obj_episode, hidden_states, pos, obj_id)
                        times['orientation'].append(time.time() - t_init_4)
                    elif otype == 'Flicker':
                        birth = np.argwhere(np.array([pos is not None for pos in obj_episode[obj_id]['pos']])).flatten()[0]
                        obj_state['params']['_age'] = i_step - birth
                    elif otype in ['Chaser', 'AStarChaser', 'RandomNPC']:
                        obj_state['params']['is_stochastic'] = True
                    elif otype == 'Bomber':
                        # infer how many it shot already? counter
                        if obj_id in mem_n_spawned_per_step.keys() and i_step < len(mem_n_spawned_per_step[obj_id]):
                            obj_state['params']['counter'] = mem_n_spawned_per_step[obj_id][i_step]
                        else:
                            t_init_4 = time.time()
                            n_spawned_per_step = infer_nb_spawned(game.params, obj_id, obj_episode, state_episode, obj_state)
                            times['spawn'].append(time.time() - t_init_4)
                            mem_n_spawned_per_step[obj_id] = n_spawned_per_step
                            obj_state['params']['counter'] = n_spawned_per_step[i_step]
                        t_init_4 = time.time()
                        obj_state['params']['orientation'] = infer_orientation(game.rules, obj, i_step, state_episode, obj_episode, hidden_states, pos, obj_id)
                        times['orientation'].append(time.time() - t_init_4)
                    elif otype == 'SpawnPoint':
                        # infer how many it shot already? counter
                        if obj_id in mem_n_spawned_per_step.keys() and i_step < len(mem_n_spawned_per_step[obj_id]):
                            obj_state['params']['counter'] = mem_n_spawned_per_step[obj_id][i_step]
                        else:
                            t_init_4 = time.time()
                            n_spawned_per_step = infer_nb_spawned(game.params, obj_id, obj_episode, state_episode, obj_state)
                            times['spawn'].append(time.time() - t_init_4)
                            mem_n_spawned_per_step[obj_id] = n_spawned_per_step
                            obj_state['params']['counter'] = n_spawned_per_step[i_step]
                    elif otype == 'ShootAvatar':
                        # infer current orientation
                        if i_step > 0:
                            prev_actions = actions[:i_step - 1]
                            ind_dir_actions = np.argwhere(np.logical_and(0 <= np.array(prev_actions), np.array(prev_actions) < 4)).flatten()
                            if ind_dir_actions.size > 0:
                                last_dir_action = prev_actions[ind_dir_actions[-1]]
                                direction = shoot_avatar_mapping[last_dir_action]
                                orientation = Vector2(*tuple(direction))
                                obj_state['params']['orientation'] = orientation
                    else:
                        raise NotImplementedError

                    # If this object has a cooldown, determine how far from next possible move
                    if 'cooldown' in obj_state['params'].keys():
                        if i_step == 0 or positions[i_step - 1] is None:  # the object just got born
                            obj_state['params']['lastmove'] = 0
                        else:
                            t_init_4 = time.time()
                            obj_prev_state = search_hidden_state(hidden_states, i_step - 1, obj_id)
                            times['search'].append(time.time() - t_init_4)
                            obj_state['params']['lastmove'] = (obj_prev_state['params']['lastmove'] + 1) % obj_prev_state['params']['cooldown']
                    hidden_states[i_step][0].append(obj_state)

                    # Update / save
                    for i_sim in range(1, n_simulations):
                        obj_state_copy = obj_state.copy()
                        params_copy = {}
                        for k, v in obj_state['params'].items():
                            try:
                                params_copy[k] = obj_state['params'][k].copy()
                            except:
                                params_copy[k] = obj_state['params'][k]
                        obj_state_copy['params'] = params_copy
                        hidden_states[i_step][i_sim].append(obj_state_copy)
                        
                    if return_orientation and i_step == last_i_step:
                        if 'orientation' in obj_state['params'].keys() and 'Avatar' in otype:
                            orientations[obj_state['obj_id']] = obj_state['params']['orientation']
        t = time.time() - t_init_2
        if obj['name'] not in times['objs'].keys():
            times['objs'][obj['name']] = []
        times['objs'][obj['name']].append(t)
        # if t > 1:
        #     print(t)
        #     print(game.rules.obj.type(name))
        #     print(positions, obj_id)
    total_time = time.time() - t_init_1
    if total_time > 15:
        print('total inference', time.time() - t_init_1)
        for k, v in times['objs'].items():
            o_type = game.rules.obj.type(k)
            print(f"{k} ({o_type}), total: {np.sum(v):.2f}, mean: {np.sum(v):.2f}")
        for k, v in times.items():
            if k != 'objs':
                print(k, f"total: {np.sum(v):.2f}")
    state_memory['hidden_states'] = hidden_states
    names = set(sorted(game.rules.names))
    names.add('floor')
    state_memory['last_hidden_state'] = dict(i_step=last_i_step, obj_hidden_state=hidden_states[-1][0], names=names,
                                             shape=(len(state_episode[0]), len(state_episode[0][0])))
    if return_orientation:
        state_memory['last_hidden_state']['agent_orientation'] = orientations
    state_memory['mem_n_spawned_per_step'] = mem_n_spawned_per_step

    return state_memory
