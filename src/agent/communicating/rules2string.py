
def format_name_list(a_list, colors, avatar_name, obj_name='objects', last='and'):
    if avatar_name in a_list:
        a_list.remove(avatar_name)
        a_list = [avatar_name] + a_list

    if a_list == [avatar_name]:
        return "you"
    else:
        color_list = [colors[obj] for obj in a_list]
        if len(color_list) > 1:
            a_list_str = ", ".join(color_list[:-1]) + f' {last} ' + color_list[-1]
        else:
            a_list_str = color_list[0]
    if obj_name != '':
        return a_list_str + f' {obj_name}'
    else:
        return a_list_str

def format_win(obj_names, avatar_name, colors):
    if len(obj_names) == 0:
        s = f"You win the game by surviving long enough."
    else:
        win_list = format_name_list(obj_names, colors, avatar_name, obj_name='objects')
        s = f"You win the game by either reaching or killing all {win_list}."
    return s

def format_lost(obj_names, avatar_name, colors):
    if len(obj_names) == 0:
        s = f"You lose if you haven't solved the game before the timeout."
    else:
        lost_list = format_name_list(obj_names, colors, avatar_name, obj_name='objects', last='or')
        if lost_list == 'you':
            s = f"You lose if you die."
        elif lost_list[:3] == 'you':
            lost_list = lost_list.replace('you, ', '').replace('you or ', '')
            s = f"You lose if you or all {lost_list} die."
        else:
            s = f"You lose if all {lost_list} die."
    return s

def format_prompt(rules, obj_names, colors, avatar_name, remaining_keys):
    vgdl_lines = rules.vgdl_lines
    type, params = line_(rules, obj_names).type, line_(rules, obj_names).params

    if type == 'Portal':
        portal_takers = []
        exit = stype_(rules, obj_names)
        remaining_keys.remove(obj_names)
        for key, vgdl_line in vgdl_lines.dict.items():
            if vgdl_line.type == 'teleportToExit' and vgdl_line.name[1] == obj_names:
                portal_takers.append(vgdl_line.name[0])
                if key in remaining_keys: remaining_keys.remove(key)
        portal_takers_str = format_name_list(portal_takers, colors, avatar_name)
        exit_str = format_name_list([exit], colors, avatar_name, obj_name='squares')
        portal_str = format_name_list([obj_names], colors, avatar_name, obj_name='squares')
        s = f"{portal_takers_str} can teleport from {portal_str} to {exit_str}."
    elif type == "ResourcePack":
        collectors = []
        remaining_keys.remove(obj_names)
        for key, vgdl_line in vgdl_lines.dict.items():
            if vgdl_line.type == 'addResource' and vgdl_line.name[0] == obj_names:
                collectors.append(vgdl_line.name[1])
                if key in remaining_keys: remaining_keys.remove(key)
        collectors_str = format_name_list(collectors, colors, avatar_name, obj_name='objects')
        resource_str = format_name_list([obj_names], colors, avatar_name, obj_name='')
        s = f"{collectors_str} can collect {resource_str} resources."
    elif type == 'Passive':
        pushers = []
        remaining_keys.remove(obj_names)
        for key, vgdl_line in vgdl_lines.dict.items():
            if vgdl_line.type == 'bounceForward' and vgdl_line.name[0] == obj_names:
                pushers.append(vgdl_line.name[1])
                if key in remaining_keys: remaining_keys.remove(key)
        pushed_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        if len(pushers) > 0:
            pushers_str = format_name_list(pushers, colors, avatar_name, obj_name='objects')
            s = f"{pushed_str} can be pushed by {pushers_str}."
        else:
            s = ""
    elif type == 'Chaser':
        chaser_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        target = vgdl_lines.dict[obj_names].params['stype']
        flee = vgdl_lines.dict[obj_names].params.get('fleeing', False)
        target_str = format_name_list([target], colors, avatar_name, obj_name='object')
        if flee:
            s  = f"{chaser_str} run away from the nearest {target_str}."
        else:
            s  = f"{chaser_str} chase the nearest {target_str}."
    elif type == 'SpawnPoint':
        spawner_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        spawned = vgdl_lines.dict[obj_names].params['stype']
        spawned_str = format_name_list([spawned], colors, avatar_name, obj_name='objects')
        s = f"{spawner_str} generate {spawned_str}."
    elif type == 'Missile':
        missile_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        s = f"{missile_str} move along one direction."
    elif type == 'Bomber':
        spawned = vgdl_lines.dict[obj_names].params['stype']
        spawned_str = format_name_list([spawned], colors, avatar_name, obj_name='objects')
        bomber_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        s = f"{bomber_str} move along one direction and generate {spawned_str}"
    elif type == 'Flicker':
        flicker_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        s = f"{flicker_str} do not move and disappear after a short time."
    elif type == 'Immovable':
        immov_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        s = f"{immov_str} do not move."
    elif type == 'RandomNPC':
        random_str = format_name_list([obj_names], colors, avatar_name, obj_name='objects')
        s = f"{random_str} move around randomly."
    elif type == 'killSprite':
        remaining_keys.remove(obj_names)
        killer_str = format_name_list([obj_names[1]], colors, avatar_name, obj_name='objects')
        killed_str = format_name_list([obj_names[0]], colors, avatar_name, obj_name='objects')
        if killed_str == 'you':
            s = f"{killer_str} can kill {killed_str}."
        else:
            s = f"{killer_str} can kill {killed_str} by touching them."
    elif type == 'transformTo':
        remaining_keys.remove(obj_names)
        spawned = None
        for key, vgdl_line in vgdl_lines.dict.items():
            if vgdl_line.type == 'transformTo' and vgdl_line.name == obj_names:
                spawned = vgdl_line.params['stype']
        killer_str = format_name_list([obj_names[1]], colors, avatar_name, obj_name='objects')
        killed_str = format_name_list([obj_names[0]], colors, avatar_name, obj_name='objects')
        spawned_str = format_name_list([spawned], colors, avatar_name, obj_name='objects')
        s = f"{killer_str} can convert {killed_str} into {spawned_str} by touching {killed_str}."
    elif type == 'removeResource':
        remaining_keys.remove(obj_names)
        resource = vgdl_lines.dict.get(obj_names).params['resource']
        collector_str = format_name_list([obj_names[1]], colors, avatar_name, obj_name='objects')
        taker_str = format_name_list([obj_names[0]], colors, avatar_name, obj_name='objects')
        resource_str = format_name_list([resource], colors, avatar_name, obj_name='')
        if collector_str == 'you':
            s = f"{collector_str} can kill {taker_str} if you touch them but {taker_str} will take a {resource_str} resource from {collector_str}."
        else:
            s = f"{collector_str} can kill {taker_str} if both touch but {taker_str} will take a {resource_str} resources from {collector_str}."
    elif type == "killIfHasLess":
        remaining_keys.remove(obj_names)
        killer_str = format_name_list([obj_names[1]], colors, avatar_name, obj_name='objects')
        killed_str = format_name_list([obj_names[0]], colors, avatar_name, obj_name='object')
        if killed_str == 'you':
            s = f"{killer_str} can kill {killed_str} by touching you if {killed_str} don't have sufficient resources"
        else:
            s = f"{killer_str} can kill {killed_str} by touching them if the {killed_str} doesn't have sufficient resources"
    else:
        print(type)
        assert False
    if s[:3] == 'you':
        s = s.capitalize()
    return s, remaining_keys



def line_(rules, key):
    return rules.vgdl_lines.dict[key]

def stype_(rules, key):
    return line_(rules, key).params.get('stype')

def resource_(rules, key):
    return line_(rules, key).params.get('resource')

def type_(rules, key):
    return line_(rules, key).type

def format_teleports(portals, exits, colors, avatar_name):
    s = f"You can teleport"
    i = 0
    for portal, exit in zip(portals, exits):
        portal_str = format_name_list([portal], colors, avatar_name, obj_name='objects')
        exit_str = format_name_list([exit], colors, avatar_name, obj_name='objects')
        if i == 0:
            s += ' '
        elif i == len(portals) - 1:
            s += ' and '
        else:
            s += ', '
        s += f"from {portal_str} to {exit_str}"
        i += 1
    s += '.'
    return s

def format_transform(transformed, colors, avatar_name):
    killed = [tr[0] for tr in transformed]
    results = [tr[1] for tr in transformed]

    s = f"You can transform"
    i = 0
    for portal, exit in zip(killed, results):
        portal_str = format_name_list([portal], colors, avatar_name, obj_name='objects')
        exit_str = format_name_list([exit], colors, avatar_name, obj_name='objects')
        if i == 0:
            s += ' '
        elif i == len(killed) - 1:
            s += ' and '
        else:
            s += ', '
        s += f"{portal_str} into {exit_str}"
        i += 1
    return s

def from_rules_to_obj_str(rules, base_colors):
    colors = base_colors.copy()
    moving_objects = ['Chaser', 'RandomNPC', 'Bomber', 'Missile', "Passive"]
    other_static_objects = ['Immovable', 'SpawnPoint', 'Flicker', 'Portal', 'ResourcePack']
    avatar_name = rules.avatar_name
    avatar_line = line_(rules, avatar_name)
    avatar_color = colors[avatar_name]
    colors[avatar_name] = 'you'

    # list keys to mention in the description, skip avatar and keys to hide
    keys_to_mention = [k for k in rules.vgdl_lines.dict.keys() if isinstance(k, str) and k!= 'wall' and k != avatar_name]

    keys_moving_objs = list()
    keys_static_objs = list()
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if isinstance(key, str) and vgdl_line.type in moving_objects:
            keys_moving_objs.append(key)
        if isinstance(key, str) and vgdl_line.type in other_static_objects and not key == 'wall':
            keys_static_objs.append(key)

    theory_str = "Known Objects:\n"
    if avatar_line.type == 'FlakAvatar':
        theory_str += f"- You control the horizontal position of the {avatar_color} square with left and right arrow keys.\n"
    elif avatar_line.type in ['MovingAvatar', 'ShootAvatar']:
        theory_str += f"- You control the position of the {avatar_color} square with arrow keys.\n"
    else:
        raise NotImplementedError

    if avatar_line.type in ['ShootAvatar', 'FlakAvatar']:
        shot_name = avatar_line.params["stype"]
        theory_str += f'- You can shoot {colors[shot_name]} objects by pressing space bar.\n'

    # what you can push
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'bounceForward' and rules.avatar_name == key[1]:
            theory_str += f'- You can push {colors[key[0]]} objects.\n'
            # if key[0] in keys_to_mention: keys_to_mention.remove(key[0])

    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'addResource' and rules.avatar_name == key[1]:
            theory_str += f'- You can collect {colors[key[0]]} resources.\n'
            if key[0] in keys_to_mention: keys_to_mention.remove(key[0])

    # what you can teleport through
    portals = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'teleportToExit' and rules.avatar_name == key[0]:
            portals.append(key[1])
            if key[1] in keys_to_mention: keys_to_mention.remove(key[1])

    exits = []
    for portal in portals:
        exit = rules.vgdl_lines.dict[portal].params['stype']
        exits.append(exit)
    if len(portals) > 0:
        s = format_teleports(portals, exits, colors, avatar_name)
        theory_str += f'- {s.capitalize()}\n'


    for key in keys_moving_objs:
        if key in keys_to_mention:
            s, _ = format_prompt(rules, key, colors, avatar_name, keys_to_mention)
            if len(s) > 0:
                if key in keys_to_mention: keys_to_mention.remove(key)
                theory_str += '- ' + s.capitalize() + '\n'

    if len(keys_static_objs) > 0:
        for key in keys_static_objs:
            if key in keys_to_mention: keys_to_mention.remove(key)
        theory_str += f'- {format_name_list(keys_static_objs, colors, avatar_name).capitalize()} have not moved yet\n'

    if len(keys_to_mention) > 0:
        print(keys_to_mention)
        print(rules)
        assert False
    theory_str += '- Darkgray objects are walls.'
    theory_str.replace('nearest you', 'you')
    return theory_str






def from_rules_to_str(rules, base_colors):
    colors = base_colors.copy()
    to_hide = ['stepBack', 'reverseDirection', 'MovingAvatar', 'ShootAvatar', 'FlakAvatar', 'noInteraction', 'bounceForward', 'teleportToExit', 'addResource',
               'SpawnPoint', 'Chaser', 'Missile', 'Bomber', 'RandomNPC', 'Immovable', 'Flicker', 'turnAround']
    moving_objects = ['Chaser', 'RandomNPC', 'Bomber', 'Missile']
    other_static_objects = ['Immovable', 'SpawnPoint', 'Flicker', 'Portal', 'ResourcePack']
    avatar_name = rules.avatar_name
    avatar_line = line_(rules, avatar_name)
    avatar_color = colors[avatar_name]
    colors[avatar_name] = 'you'

    # list keys to mention in the description, skip avatar and keys to hide
    keys_to_mention = set(list(rules.vgdl_lines.dict.keys()))
    keys_moving_objs = list()
    keys_static_objs = list()
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type in to_hide or (isinstance(key, tuple) and key[1] == 'wall') or key[1] == 'EOS':
            keys_to_mention.remove(key)
        if isinstance(key, str) and vgdl_line.type in moving_objects:
            keys_moving_objs.append(key)
        if isinstance(key, str) and vgdl_line.type in other_static_objects:
            keys_static_objs.append(key)

    # in order:
    # who you are, what you shoot
    # what you can push (aggregate)
    # what you can teleport through (aggregate)
    # what you can kill (aggregate)
    # what you can transform (aggregate)
    # what the things you control can
    # win and lose conditions
    # first print info about the avatar
    theory_str = "Here are the objects you've seen so far: darkblue (avatar), "
    for key in rules.names:
        color = rules.colors[key]
        color = color.lower()
        if color not in ['darkblue', 'darkgray']:
            theory_str += f'{color}, '
    theory_str = theory_str[:-2] + ' and darkgray (walls).\n\n'
    theory_str += "Who you are and how you move:\n"
    if avatar_line.type == 'FlakAvatar':
        theory_str += f"- You control the horizontal position of the {avatar_color} square with left and right arrow keys.\n"
    elif avatar_line.type in ['MovingAvatar', 'ShootAvatar']:
        theory_str += f"- You control the position of the {avatar_color} square with arrow keys.\n"
    else:
        raise NotImplementedError
    # what you can teleport through
    portals = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'teleportToExit' and rules.avatar_name == key[0]:
            portals.append(key[1])
            if key in keys_to_mention: keys_to_mention.remove(key)
            if key[1] in keys_to_mention: keys_to_mention.remove(key[1])

    exits = []
    for portal in portals:
        exits.append(rules.vgdl_lines.dict[portal].params['stype'])
    if len(portals) > 0:
        s = format_teleports(portals, exits, colors, avatar_name)
        theory_str += f'- {s.capitalize()}\n'

    # then print win and lose conditions
    theory_str += "How you win and lose:\n"
    keys_used = set()
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if isinstance(key, tuple) and key not in keys_used:
            reward = vgdl_line.params.get('scoreChange', 0)
            if reward > 0:
                actor_str = format_name_list([key[1]], colors, avatar_name, obj_name='objects')
                actee_str = format_name_list([key[0]], colors, avatar_name, obj_name='objects')
                if vgdl_line.type == 'killSprite':
                    # is there a reverse killSprite interaction?
                    inverted_key = (key[1], key[0])
                    inverted_rule = rules.vgdl_lines.dict.get(inverted_key, None)
                    if inverted_rule and inverted_rule.type == 'killSprite':
                        theory_str += f"- You get points when {actor_str} and {actee_str} kill each other.\n"
                        keys_used.add(inverted_key)
                    else:
                        theory_str += f"- You get points when {actor_str} kill {actee_str}.\n"
                elif vgdl_line.type == 'transformTo':
                    transformed_str = format_name_list([vgdl_line.params['stype']], colors, avatar_name, obj_name='objects')
                    theory_str += f"- You get points when {actor_str} transform {actee_str} into {transformed_str}.\n"
                elif vgdl_line.type == 'addResource':
                    theory_str += f"- You get points when {actor_str} collect {actee_str}.\n"
                elif vgdl_line.type == 'removeResource':
                    resource_str = format_name_list([vgdl_line.params['resource']], colors, avatar_name, obj_name='resources')
                    theory_str += f"- You get points when {actee_str} take {resource_str} from {actor_str}.\n"
                else:
                    assert False, f"invalid vgdl type for a reward {vgdl_line.type}"
            elif reward < 0:
                actor_str = format_name_list([key[1]], colors, avatar_name, obj_name='objects')
                actee_str = format_name_list([key[0]], colors, avatar_name, obj_name='objects')
                if vgdl_line.type == 'killSprite':
                    # is there a reverse killSprite interaction?
                    inverted_key = (key[1], key[0])
                    inverted_rule = rules.vgdl_lines.dict.get(inverted_key, None)
                    if inverted_rule and inverted_rule.type == 'killSprite':
                        theory_str += f"- You lose points when {actor_str} and {actee_str} kill each other.\n"
                        keys_used.add(inverted_key)
                    else:
                        theory_str += f"- You lose points when {actor_str} kill {actee_str}.\n"
                elif vgdl_line.type == 'transformTo':
                    transformed_str = format_name_list([vgdl_line.params['stype']], colors, avatar_name, obj_name='objects')
                    theory_str += f"- You lose points when {actor_str} transform {actee_str} into {transformed_str}.\n"
                elif vgdl_line.type == 'addResource':
                    theory_str += f"- You lose points when {actor_str} collect {actee_str}.\n"
                elif vgdl_line.type == 'removeResource':
                    resource_str = format_name_list([vgdl_line.params['resource']], colors, avatar_name, obj_name='resources')
                    theory_str += f"- You lose points when {actee_str} take {resource_str} from {actor_str}.\n"
                else:
                    assert False, f"invalid vgdl type for a reward {vgdl_line.type}"
    theory_str += '- ' + format_win(list(rules.terminations['win']), avatar_name, colors) + '\n'
    theory_str += '- ' + format_lost(list(rules.terminations['lose']), avatar_name, colors) + '\n'

    theory_str += "What you can do:\n"
    something_added = False

    if avatar_line.type in ['ShootAvatar', 'FlakAvatar']:
        shot_name = avatar_line.params["stype"]
        shot = [shot_name]
        theory_str += f'- You can shoot {colors[shot_name]} squares by pressing space bar.\n'
        something_added = True
    else: shot = []

    # what you can push
    pushed = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'bounceForward' and rules.avatar_name == key[1]:
            pushed.append(key[0])
            if key in keys_to_mention: keys_to_mention.remove(key)

    # what you can kill
    killers = dict()
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'killSprite' and key[1] in [avatar_name] + pushed + shot and key[0] != avatar_name:
            if key[1] not in killers:
                killers[key[1]] = []
            killers[key[1]].append(key[0])
            if key in keys_to_mention: keys_to_mention.remove(key)
        elif vgdl_line.type == 'killSprite' and key[0] in pushed and key[1] != 'EOS':
            if key[1] not in killers:
                killers[key[1]] = []
            killers[key[1]].append(key[0])
            if key in keys_to_mention: keys_to_mention.remove(key)
    if len(killers) > 0:
        killer_keys = sorted(list(killers.keys()))
        if avatar_name in killer_keys:
            killer_keys.remove(avatar_name)
            killer_keys = [avatar_name] + killer_keys
        for killer in killer_keys:
            killed = killers[killer]
            something_added = True

            if killer == avatar_name:
                killed_str = format_name_list(killed, colors, avatar_name, obj_name='objects')
                theory_str += f"- You can kill {killed_str} by touching them.\n"
            elif killer in pushed:
                killed_str = format_name_list(killed, colors, avatar_name, obj_name='objects')
                pushed_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                theory_str += f"- You can kill {killed_str} by pushing {pushed_str} onto them.\n"
            elif killer in shot:
                killed_str = format_name_list(killed, colors, avatar_name, obj_name='objects')
                shot_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                theory_str += f"- You can kill {killed_str} by shooting {shot_str} at them.\n"
            elif any([name in pushed for name in killed]):
                killed = [name for name in killed if name in pushed]
                killed_str = format_name_list(killed, colors, avatar_name, obj_name='objects')
                killer_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                theory_str += f"- You can kill {killed_str} by pushing them onto {killer_str}.\n"
            else:
                raise NotImplementedError

    # what you can transform
    transformed = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'transformTo' and rules.avatar_name == key[1]:
            transformed.append((key[0], vgdl_line.params['stype']))

    transformed = dict()
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'transformTo' and key[1] in [avatar_name] + pushed + shot and key[0] != avatar_name:
            if key[1] not in transformed.keys():
                transformed[key[1]] = []
            transformed[key[1]].append((key[0], vgdl_line.params['stype']))
            if key in keys_to_mention: keys_to_mention.remove(key)
        elif vgdl_line.type == 'transformTo' and key[0] in pushed:
            if key[1] not in transformed.keys():
                transformed[key[1]] = []
            transformed[key[1]].append((key[0], vgdl_line.params['stype']))
            if key in keys_to_mention: keys_to_mention.remove(key)
    if len(transformed) > 0:
        transformed_keys = sorted(list(transformed.keys()))
        if avatar_name in transformed_keys:
            transformed_keys.remove(avatar_name)
            transformed_keys = [avatar_name] + transformed_keys
        for killer in transformed_keys:
            something_added = True
            for transformation in transformed[killer]:
                target_str = format_name_list([transformation[0]], colors, avatar_name)
                if killer == avatar_name:
                    theory_str += ('- ' + format_transform([transformation], colors, avatar_name) +
                                   (f" by touching {target_str}.\n").capitalize())
                elif killer in pushed:
                    pushed_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                    theory_str += '- ' + format_transform([transformation], colors, avatar_name) + f" by pushing {pushed_str} onto {target_str}.\n".capitalize()
                elif killer in shot:
                    shot_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                    theory_str += '- ' + format_transform([transformation], colors, avatar_name) + f' by shooting {shot_str} at {target_str}.\n'.capitalize()
                elif transformation[0] in pushed:
                    killer_str = format_name_list([killer], colors, avatar_name, obj_name='objects')
                    theory_str += '- ' + format_transform([transformation], colors, avatar_name) + f' by pushing {target_str} onto {killer_str}.\n'.capitalize()
                else:
                    raise NotImplementedError

    if not something_added:
        theory_str += "- You can't do anything special\n"

    # what can kill you
    killers = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'killSprite' and rules.avatar_name == key[0] and key[1] != 'EOS':
            killers.append(key[1])
            if key in keys_to_mention: keys_to_mention.remove(key)
    theory_str += 'What can kill you:\n'
    something_added = False

    if len(killers) > 0:
        killers_str = format_name_list(killers, colors, avatar_name, obj_name='objects')
        theory_str += f"- {killers_str.capitalize()} will kill you if you touch them.\n"
        something_added = True

    # what can kill you if resource
    killers_resource = []
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if vgdl_line.type == 'killIfHasLess' and rules.avatar_name == key[0]:
            killers_resource.append(key[1])
            if key in keys_to_mention: keys_to_mention.remove(key)

    if len(killers_resource) > 0:
        killers_str = format_name_list(killers_resource, colors, avatar_name, obj_name='objects')
        theory_str += f"- {killers_str.capitalize()} will kill you if you touch them and have insufficient resources to protect yourself.\n"
        something_added = True
    if not something_added:
        theory_str += f"- Nothing can kill you.\n"

    theory_str += "Other possible interactions:\n"
    something_added = False
    # then print interactions
    for key, vgdl_line in rules.vgdl_lines.dict.items():
        if key in keys_to_mention:
            inverted_line = None
            inverted_key = None
            if isinstance(key, tuple):
                inverted_key = (key[1], key[0])
                inverted_line = rules.vgdl_lines.dict.get(inverted_key, None)
            if inverted_line is not None:
                inverted_type = inverted_line.type
            else:
                inverted_type = None
            if vgdl_line.type == 'killSprite' and inverted_type == 'killSprite':
                obj1 = format_name_list([key[0]], colors, avatar_name, obj_name='objects')
                obj2 = format_name_list([key[1]], colors, avatar_name, obj_name='objects')
                s = f"{obj1} and {obj2} kill each other when they touch."
                if 'you' in s:
                    s = s.replace('they', 'you')
                keys_to_mention.remove(key)
                if inverted_key in keys_to_mention:
                    keys_to_mention.remove(inverted_key)
            else:
                s, keys_to_mention = format_prompt(rules, key, colors, avatar_name, keys_to_mention)
            if len(s) > 0:
                theory_str += '- ' + s.capitalize() + '\n'
                something_added = True
    if not something_added:
        theory_str += "- No other interaction.\n"
    theory_str += "Other objects:\n"
    something_added = False
    for key in keys_moving_objs:
        something_added = True
        s, _ = format_prompt(rules, key, colors, avatar_name, keys_to_mention)
        if len(s) > 0:
            theory_str += '- ' + s.capitalize() + '\n'

    if len(keys_static_objs) > 0:
        something_added = True
        theory_str += f'- {format_name_list(keys_static_objs, colors, avatar_name)} do not move.\n'
    if not something_added:
        theory_str += "- No other object.\n"

    theory_str = theory_str.replace('all you', 'you')
    theory_str.replace('nearest you', 'you')

    return theory_str