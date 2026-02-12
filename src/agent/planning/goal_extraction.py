import numpy as np
from src.utils import AVATAR_NAME

# Compute a reward for each interaction
def get_task_score(rules, name1, name2):

    avatar_name = rules.avatar_name
    to_kill_for_win = set(rules.terminations['win'])
    stuff_to_protect = list(rules.avatar_names) + list(rules.terminations['lose'])
    tools_to_kill = set()
    enemies_to_kill = set()
    allies_to_spawn = set()
    for int_names, int_line in rules.int.dict.items():
        if int_names[0] in stuff_to_protect and int_line.kills_a():
            enemies_to_kill.add(int_names[1])
        if int_line.spawns() and int_line.params['stype'] in stuff_to_protect:
            allies_to_spawn.add(int_names[1])
        if int_line.kills_a() and int_names[0] in to_kill_for_win:
            tools_to_kill.add(int_names[1])
    for _ in range(2):
        for int_names, int_line in rules.int.dict.items():
            if int_line.spawns() and int_line.params['stype'] in tools_to_kill:
                tools_to_kill.add(int_names[1])
                tools_to_kill.add(int_names[0])

    enemies_to_kill = list(enemies_to_kill)
    allies_to_spawn = list(allies_to_spawn)
    tools_to_kill = list(tools_to_kill)
    int_line = rules.int.dict.get((name1, name2))
    reverse_int_line = rules.int.dict.get((name2, name1))
    int_score = 0 if int_line is None else int_line.params.get('scoreChange', 0)
    reverse_int_score = 0 if reverse_int_line is None else reverse_int_line.params.get('scoreChange', 0)
    useless_interactions = ['noInteraction', 'bounceForward', 'teleportToExit', 'stepBack', 'reverseDirection']
    avatar_stype = rules.obj.dict.get(avatar_name).params.get('stype')
    if avatar_stype is not None and avatar_stype in [name1, name2] and rules.obj.dict[avatar_stype].type in ['Chaser', 'Missile'] and avatar_name not in [name1, name2]:
        shoot_at_something = True
    else:
        shoot_at_something = False

    # is this interaction spawning a useful object?
    spawned_objs = []
    if int_line and int_line.spawns():
        spawned_objs.append(int_line.params['stype'])
    if reverse_int_line and reverse_int_line.spawns():
        spawned_objs.append(reverse_int_line.params['stype'])

    if shoot_at_something:
        task_score = 0
    elif (int_line and int_line.kills_a() and name1 in to_kill_for_win) or (reverse_int_line and reverse_int_line.kills_a() and name2 in to_kill_for_win):
        # getting closer to solve the game is very good
        task_score = 5
    elif any([obj in (allies_to_spawn + tools_to_kill) for obj in spawned_objs]):
        # spawning allies is good
        task_score = 4
    elif any([obj in stuff_to_protect for obj in spawned_objs]):
        # spawning obj to protect is good
        task_score = 3
    elif (int_score + reverse_int_score) > 0:
        # getting positive rewards is good
        task_score = 1
    elif (int_line and name2 == avatar_name and int_line.type == 'addResource') or (reverse_int_line and name1 == avatar_name and reverse_int_line.type == 'addResource'):
        # collecting resources is good
        task_score = 1
    elif (int_line and int_line.kills_a() and name1 in enemies_to_kill) or (reverse_int_line and reverse_int_line.kills_a() and name2 in enemies_to_kill):
        # killing enemies is good
        task_score = 0.5
    elif (int_line and int_line.spawns() and int_line.params['stype'] not in enemies_to_kill) or (reverse_int_line and reverse_int_line.spawns() and reverse_int_line.params['stype'] not in enemies_to_kill):
        # spawning new objects is good
        task_score = 0.5
    elif (int_score + reverse_int_score) < 0 or (int_line and int_line.kills_a() and int_line.name[0] == avatar_name) or (reverse_int_line and reverse_int_line.kills_a() and
                                                                                                                          reverse_int_line.name[0] == avatar_name):
        # paying costs (neg reward) is bad
        task_score = 0
    elif (int_line and int_line.type not in useless_interactions) or (reverse_int_line and reverse_int_line.type not in useless_interactions):
        # everything else has default low value
        task_score = 0.25
    elif (int_line and int_line.type == 'stepBack') or (reverse_int_line and reverse_int_line.type == 'stepBack'):
        # stepback could be wrong (eg confounded with bounceforward + block)
        task_score = 0.1
    # elif shoot_at_something:
    #     task_score = 0
    else:
        # no interaction between object has 0 task value
        task_score = 0
    return task_score

# Check for interactions we don't have data for
def get_explo_score(all_rules, name1, name2):

    n_particles = len(all_rules)
    reverse_values = [rules.int.dict.get((name1, name2)) for rules in all_rules]
    values = [rules.int.dict.get((name2, name1)) for rules in all_rules]

    # score is the min disagreement value
    values_str = []
    for value, rules in zip(values, all_rules):
        if value is None:
            continue
        if value.type == 'teleportToExit':
            values_str.append(value.type + '_' + rules.obj.dict.get(name2).type)
        else:
            values_str.append(value.type)
    if len(values_str) > 0:
        bins, counts = np.unique(values_str, return_counts=True)
        max_count = np.max(counts)
    else:
        max_count = 0
    forward_score = 1 - (max_count / n_particles)

    values_str = []
    for value, rules in zip(reverse_values, all_rules):
        if value is None:
            continue
        if value.type == 'teleportToExit':
            values_str.append(value.type + '_' + rules.obj.dict.get(name2).type)
        else:
            values_str.append(value.type)
    if len(values_str) > 0:
        bins, counts = np.unique(values_str, return_counts=True)
        max_count = np.max(counts)
    else:
        max_count = 0
    reverse_score = 1 - (max_count / n_particles)
    explo_score = (forward_score + reverse_score) / 2
    if name1 == name2:
        explo_score = 0.5 * explo_score
    return explo_score

# Is this something you can push, spawn, or is it agent?
def get_controllability(rules, name1, name2):
    avatar_name = rules.avatar_name
    avatar_obj = rules.vgdl_lines.dict[avatar_name]
    controllability = dict(name1=[], name2=[], is_controllable=0.)

    interaction = rules.vgdl_lines.dict.get((name1, name2))
    reverse_interaction = rules.vgdl_lines.dict.get((name2, name1))
    if name1 == avatar_name:
        obj2 = rules.vgdl_lines.dict[name2]
        if avatar_obj.spawns() and avatar_obj.params['stype'] == name2 and (obj2.moves() or obj2.type == 'Flicker'):
            pass
        else:
            controllability['name1'].append('avatar')
    if name2 == avatar_name:
        obj1 = rules.vgdl_lines.dict[name1]
        if avatar_obj.spawns() and avatar_obj.params['stype'] == name1 and (obj1.moves() or obj1.type == 'Flicker'):
            pass
        else:
            controllability['name2'].append('avatar')
    for int_name, int in rules.int.dict.items():
        if int_name[1] == avatar_name and int.type == 'bounceForward':
            if name1 == int_name[0]:
                controllability['name1'].append('pushed')
            if name2 == int_name[0]:
                controllability['name2'].append('pushed')
    stype = rules.obj.dict[avatar_name].params.get('stype')
    if stype:
        if name1 == stype and name2 not in [name1, avatar_name]: # and rules.obj.type(name1) not in ['Chaser', 'Missile']:
            controllability['name1'].append('spawned')
        if name2 == stype and name1 not in [name2, avatar_name]: # and rules.obj.type(name2) not in ['Chaser', 'Missile']:
            controllability['name2'].append('spawned')
    if len(controllability['name1']) > 0 or len(controllability['name2']) > 0:
        if (interaction and interaction.blocks('second')) or reverse_interaction and reverse_interaction.blocks('second'):
            controllability['is_controllable'] = 0.1
        else:
            controllability['is_controllable'] = 1.0
    return controllability

# Look at all pairs of objects, including agent and something you can push or shoot
def extract_goals(games,  beta_explore_exploit):

    names = sorted(games[0].rules.names)
    avatar_name = games[0].rules.avatar_name

    goals = dict()
    best_game_id = np.argmax([game.rules.logjoint for game in games])
    game = games[best_game_id]
    # loop over every pair of names
    for i_name, name1 in enumerate(names):
        for name2 in names[i_name:]:
            goals[(name1, name2)] = dict()

            rules = game.rules
            # get controllability
            controllability = get_controllability(rules, name1, name2)
            goals[(name1, name2)]['controllability'] = controllability
            # get task score

            task_score = get_task_score(rules, name1, name2)
            goals[(name1, name2)]['task_score'] = task_score
            interaction = game.rules.vgdl_lines.dict.get((name1, name2))
            reverse_interaction = game.rules.vgdl_lines.dict.get((name2, name1))
            goals[(name1, name2)]['dangerous_here'] = False
            if name1 == avatar_name and (interaction and interaction.kills_a()):
                goals[(name1, name2)]['dangerous_here'] = True
            elif name2 == avatar_name and (reverse_interaction and reverse_interaction.kills_a()) or ():
                goals[(name1, name2)]['dangerous_here'] = True
            elif avatar_name in [name1, name2]:
                if avatar_name == name1:
                    obj2 = game.rules.vgdl_lines.dict[name2]
                    if obj2.type == 'SpawnPoint':
                        spawn = obj2.params['stype']
                        avatar_int = game.rules.vgdl_lines.dict.get((avatar_name, spawn), None)
                        if avatar_int and avatar_int.kills_a():
                            goals[(name1, name2)]['dangerous_here'] = True
                else:
                    obj1 = game.rules.vgdl_lines.dict[name1]
                    if obj1.type == 'SpawnPoint':
                        spawn = obj1.params['stype']
                        avatar_int = game.rules.vgdl_lines.dict.get((avatar_name, spawn), None)
                        if avatar_int and avatar_int.kills_a():
                            goals[(name1, name2)]['dangerous_here'] = True

            # get explo score
            goals[(name1, name2)]['explo_score'] = get_explo_score([g.rules for g in games], name1, name2)

    # aggregate scores
    actionnable_goals = []
    for key, goal in goals.items():
        is_controllable = goal['controllability']['is_controllable']   # Only a goal if controllable
        goal['score'] = is_controllable * (goal['explo_score'] + beta_explore_exploit * goal['task_score']) * (1 - int(goal['dangerous_here']))

        if goal['score'] > 0:
            # list subgoals
            name1, name2 = key
            if name2 == 'wall':
                continue
            if name1 == name2:
                name2_alias = name2 + '+other'
            else:
                name2_alias = name2
            controllability = goal['controllability']
            interactions = []
            for key in ((name1, name2), (name2, name1)):
                interaction = games[best_game_id].rules.vgdl_lines.dict.get(key)
                if interaction is not None and interaction.type != 'noInteraction':
                    interactions.append(interaction)
            if 'avatar' in controllability['name1']:
                subgoals = (('go_to', name1, name2_alias),)
                actionnable_goals.append(dict(type='go_to', actor=name1, controlled=name2, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here'] ))
            if 'avatar' in controllability['name2']:
                subgoals = (('go_to', name2_alias, name1),)
                actionnable_goals.append(dict(type='go_to', actor=name2, controlled=name1, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here']))
            if 'spawned' in controllability['name1'] and avatar_name != name2_alias:
                subgoals = (('go_one_step_away', avatar_name, name2_alias), ('shoot_at', name1, name2_alias),)
                actionnable_goals.append(dict(type='spawn_onto', actor=name1, controlled=name2, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here']))
            if 'spawned' in controllability['name2'] and avatar_name != name1:
                subgoals = (('go_one_step_away', avatar_name, name1), ('shoot_at', name2_alias, name1),)
                actionnable_goals.append(dict(type='spawn_onto', actor=name2, controlled=name1, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here']))
            if 'pushed' in controllability['name1']:
                subgoals = (('go_one_step_away', avatar_name, name1), ('push_to', name1, name2_alias),)
                actionnable_goals.append(dict(type='push_to', actor=name1, controlled=name2, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here']))
            if 'pushed' in controllability['name2']:
                subgoals = (('go_one_step_away', avatar_name, name2_alias), ('push_to', name2_alias, name1),)
                actionnable_goals.append(dict(type='push_to', actor=name2, controlled=name1, goals=subgoals, interactions=interactions, prio=goal['score'],
                                              task_score=goal['task_score'], explo_score=goal['explo_score'], dangerous=goal['dangerous_here']))
    
    # # are any of these goals intermediate goals to higher goals?
    # # let's first list elements we need to win
    # elements = set()
    # for g in actionnable_goals:
    #     if g['task_score'] == 3:
    #         elements.add(g['actor'])
    #         elements.add(g['controlled'])
    # # now
    #
    # # transformation goals can be
    # # let's list transformation outcomes
    # stypes = []
    # id_goal = []
    # for i_g, g in enumerate(actionnable_goals):
    #     for interaction in g['interactions']:
    #         if interaction.type == 'transformTo':
    #             stype = interaction.params['stype']
    #             stypes.append(stype)
    #             id_goal.append(i_g)

    # for g in actionnable_goals:
    #     print(g['goals'], g['explo_score'], g['task_score'], g['prio'])
    # now what are the subgoals?
    # for goal in goals

    return actionnable_goals


# def compute_explo_goal_score(all_rules, name1, name2):
#     reverse_values = [rules.int.dict.get((name1, name2)) for rules in all_rules]
#     values = [rules.int.dict.get((name2, name1)) for rules in all_rules]
#     # score is the min disagreement value
#     values_str = [el.type for el in values if el is not None]
#     if len(values_str) > 0:
#         bins, counts = np.unique(values_str, return_counts=True)
#         max_count = np.max(counts)
#     else:
#         max_count = 0
#     forward_score = 1 - (max_count / len(values))
#     values_str = [el.type for el in reverse_values if el is not None]
#     if len(values_str) > 0:
#         bins, counts = np.unique(values_str, return_counts=True)
#         max_count = np.max(counts)
#     else:
#         max_count = 0
#     reverse_score = 1 - (max_count / len(reverse_values))
#     score = (forward_score + reverse_score) / 2
#     return  score
# #
#
# def get_goal(game, actor, target, infos=None):
#     if infos is None:
#         is_controlled, infos = game.is_controlled(actor)
#     else:
#         is_controlled = True
#     goals = []
#     if is_controlled:
#         for info in infos:
#             if info['type'] == 'avatar':
#                 goal = (('go_to', actor, target), )  # go to the target
#             elif info['type'] == 'pushed':
#                 goal1 = ('go_one_step_away', info['pusher'], actor)  # first go near the actor to push
#                 goal2 = ('go_to', actor, target)  # then push it to the target
#                 goal = (goal1, goal2)
#             elif info['type'] == 'spawned':
#                 if info['spawned_type'] == 'Flicker':
#                     goal1 = ('go_one_step_away', info['spawner'], target)  # first go one step away from the target
#                 elif info['spawned_type'] == 'Missile':
#                     goal1 = ('align_with', info['spawner'], target)  # first go one step away from the target
#                 else: raise NotImplementedError
#                 goal2 = ('kill', None, target)  # the kill the target
#                 goal = (goal1, goal2)
#             elif info['type'] == 'transformed':
#                 # we first need to obtain the object by transformation
#                 transformation_goals = game.get_goal(info['transformer'], info['transformed'])
#                 assert len(transformation_goals) == 1
#                 transformation_goals = transformation_goals[0]
#                 filtered_info = [info for info in infos if info['type'] != 'transformed']
#                 filtered_goals = game.get_goal(actor, target, infos=filtered_info)
#                 if len(filtered_goals) > 0:
#                     goal = transformation_goals + filtered_goals[0]
#                 else:
#                     goal = transformation_goals
#             else:
#                 raise NotImplementedError
#             goals.append(goal)
#     return goals
#
# def is_controlled(game, obj_name, depth=0):
#     control = False
#     control_info = []
#     if obj_name in game.rules.avatar_names:
#         control = True
#         control_info.append(dict(type='avatar'))
#     for avatar_name in game.rules.avatar_names:
#         spawned_name = game.rules.obj.params(avatar_name).get('stype', None)
#         if spawned_name == obj_name:
#             control = True
#             control_info.append(dict(type='spawned', spawner=avatar_name, spawned_type=game.rules.obj.type(obj_name)))
#     for int_names, int_line in game.rules.int.dict.items():
#         if int_line.type == 'bounceForward' and int_names[1] in game.rules.avatar_names and int_names[0] == obj_name:
#             control = True
#             control_info.append(dict(type='pushed', pusher=int_names[1]))
#         elif int_line.type == 'transformTo' and int_line.params['stype'] == obj_name:
#             if depth == 0:
#                 if game.is_controlled(int_names[0], depth=depth +1)[0]:
#                     control = True
#                     control_info.append(dict(type='transformed', transformed=int_names[0], transformer=int_names[1], control='transformed'))
#                 elif game.is_controlled(int_names[1], depth=depth +1)[0]:
#                     control = True
#                     control_info.append(dict(type='transformed', transformed=int_names[0], transformer=int_names[1], control='transformer'))
#     return control, control_info
