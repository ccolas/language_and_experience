import numpy as np
import src.vgdl as VGDL

from src.utils import inv_dir_dict, hashdict, AVATAR_NAME, BG_NAME, BG_COLOR, normalize_log_joints, format_prompt
from src.agent.communicating.rules2string import from_rules_to_str, from_rules_to_obj_str


class VGDLLine:
    """
    Stores:
        line_type - object or interaction
        name - id of this VGDL line, either the string name for an object OR a tuple of object names for an interaction
        type - string name of supertype this object inherits from, or the effect this interaction generates
        params - any additional parameters
    """
    def __init__(self, key, v_type, v_params={}):
        if isinstance(key, str):
            self.line_type = 'obj'
        else:
            self.line_type = 'int'
        self.name = key
        self.type = v_type
        self.params = hashdict(v_params)
      
    ## Boolean functions to check whether this line posseses a particular property ##

    def is_avatar(self):
        assert self.line_type == 'obj'
        return 'Avatar' in self.type

    def moves(self):
        assert self.line_type == 'obj'
        return self.type in ['Missile', 'Passive', 'RandomNPC', 'Chaser', 'MovingAvatar', 'ShootAvatar', 'FlakAvatar', 'Bomber']

    def spawns(self):
        return self.type in ['SpawnPoint', 'ShootAvatar', 'transformTo', 'FlakAvatar', 'Bomber']

    def changes_resource(self):
        return self.type in ['addResource', 'removeResource']

    def teleports_a(self):
        return self.type in ['teleportToExit']
    
    def kills_a(self):
        assert self.line_type == 'int'
        return self.type in ['killSprite', 'transformTo', 'addResource', 'removeResource', 'killIfHasLess', 'killIfOtherHasMore', 'killIfFromAbove']

    def blocks(self, who=None):
        assert self.line_type == 'int'
        if who == 'first' or who is None:
            return self.type in ['stepBack', 'killSprite', 'transformTo', 'addResource', 'removeResource', 'killIfHasLess', 'killIfOtherHasMore', 'killIfFromAbove']
        elif who == 'second':
            return self.type in ['stepBack']
        else:
            raise NotImplementedError

    def __hash__(self):
        return hash((self.name, self.type, hashdict(self.params)))

    def __repr__(self):
        return f"{self.name}: {self.type} {self.params}"

    def str_llm(self, color_dict, vgdl_lines):
        if 'EOS' in self.name:
            return ""
        if isinstance(self.name, str):
            name = [self.name]
        else:
            name = self.name
        return format_prompt(name, self.type, self.params, color_dict, vgdl_lines)

    def __str__(self):
        return f"    > {self.name}: {self.type} {self.params}"

    def __eq__(self, other):
        return hash(self) == hash(other)


class VGDLLines:
    """
    Dictionary mapping from line names (object or object pair) to the VGDL line storing additional information
    Not used for level mappings or termination conditions
    """
    def __init__(self, from_dict=None, from_list=None):
        if from_dict:
            self.dict = from_dict
        elif from_list:
            self.dict = dict()
            for vgdl_line in from_list:
                self.add(vgdl_line)
        else:
            self.dict = dict()

    def add(self, vgdl_line):
        self.dict[vgdl_line.name] = vgdl_line

    def type(self, name):
        return self.dict[name].type

    def params(self, name):
        return self.dict[name].params.copy()

    def keys(self):
        keys = list(self.dict.keys())
        key_to_sort = [key if isinstance(key, str) else key[0] + '_' + key[1] for key in keys]
        inds = np.argsort(key_to_sort)
        sorted_keys = [keys[i] for i in inds]
        return sorted_keys

    def __add__(self, other):
        if len(set(other.keys()).intersection(set(self.keys()))) > 0:
            print('key already present:', set(other.keys()).intersection(set(self.keys())))
        new_dict = self.dict.copy()
        new_dict.update(other.dict)
        new_vgdl_lines = VGDLLines()
        new_vgdl_lines.dict = new_dict
        return new_vgdl_lines

    @property
    def obj_names(self):
        return [name for name in self.keys() if self.dict[name].line_type=='obj']

    @property
    def int_names(self):
        return [name for name in self.keys() if self.dict[name].line_type=='int']

    @property
    def int(self):
        return VGDLLines(dict(zip(self.int_names, [self.dict[name] for name in self.int_names])))

    @property
    def obj(self):
        return VGDLLines(dict(zip(self.obj_names, [self.dict[name] for name in self.obj_names])))

    def __contains__(self, vgdl_lines):
        if isinstance(vgdl_lines, VGDLLine):
            v = vgdl_lines
            k = vgdl_lines.name
            if k not in self.dict.keys() and v.type != 'noInteraction':
                return False
            elif k in self.dict.keys() and self.dict[k] != v:
                return False
            else:
                return True
        elif isinstance(vgdl_lines, VGDLLines):
            for k, v in vgdl_lines.dict.items():
                if k not in self.dict.keys() and v.type != 'noInteraction':
                    return False
                elif k in self.dict.keys() and self.dict[k] != v:
                    return False
            return True
        else:
            raise NotImplementedError

    @property
    def no_int_dict(self):
        no_int_dict = dict()
        for k, v in self.dict.items():
            if v.type != 'noInteraction':
                no_int_dict[k] = v
        return no_int_dict

    def __hash__(self):
        return hash(hashdict(self.no_int_dict))

    def __repr__(self):
        s = "VGDLLines("
        n_objs = len(self.obj_names)
        n_ints = len(self.int_names)
        if n_objs > 0:
            s += f"{n_objs} objs"
            if n_ints > 0:
                s += ", "
        if n_ints > 0:
            s += f"{len(self.int_names)} ints"
        s += ")"
        return s

    def __str__(self):
        s = ''
        obj_names = self.obj_names
        int_names = self.int_names
        if len(obj_names) > 0:
            s += "  Objects:\n"
            for o in obj_names:
                s += str(self.dict[o]) + '\n'
        if len(int_names) > 0:
            s += "  Interactions:\n"
            for i in int_names:
                if self.dict[i].type != 'noInteraction':
                    s += str(self.dict[i]) + '\n'
        return s

    def str_llm(self, color_dict):
        s = ''
        obj_names = self.obj_names
        int_names = self.int_names
        if len(obj_names) > 0:
            for o in obj_names:
                prompt = self.dict[o].str_llm(color_dict, self)
                if prompt is not None:
                    s += prompt + '\n'
        if len(int_names) > 0:
            for i in int_names:
                if self.dict[i].type != 'noInteraction':
                    prompt = self.dict[i].str_llm(color_dict, self)
                    if prompt is not None:
                        s += prompt + '\n'
        return s

    def __eq__(self, other):
        keys = set(list(self.keys()))
        other_keys = set(list(other.keys()))
        for k in keys.union(other_keys):
            if k in self.keys() and k in other.keys():
                if hash(self.dict[k]) != hash(other.dict[k]):
                    return False
            elif k in self.keys():
                if self.dict[k].type != 'noInteraction':
                    return False
            elif k in other.keys():
                if other.dict[k].type != 'noInteraction':
                    return False
        return True


class Rules:
    """
    A complete description of a VGDL game
    """
    def __init__(self, params, vgdl_obj=None, vgdl_script=None):
        assert not (vgdl_obj is None and vgdl_script is None)
        self.params = params
        self.character_mapping, self.colors, self.layouts = [self.params['true_game_info'][k] for k in ['character_mapping', 'colors', 'layouts']]
        if vgdl_script is None:
            assert vgdl_obj is not None
            self.vgdl_obj = vgdl_obj
            self.obj, self.names, self.int, self.terminations = [vgdl_obj[k] for k in ['obj', 'names', 'int', 'terminations']]
            self.names = set(self.names)
            self.vgdl_script = self.write_vgdl()
        elif vgdl_obj is None:
            assert vgdl_script is not None
            self.vgdl_script = vgdl_script
            self.read_vgdl(vgdl_script)

        self.enemies, self.enemies_blockers = self.get_enemies()
        self.check_names()

        self.logprior, self.interaction_loglikelihood, self.language_loglikelihood, self.feedback = np.nan, np.nan, np.nan, None
        self.prompt = ''

    # Parse a VGDL script into a VGDL object, i.e. a dictionary storing two VGDLLines objects and termination conditions
    def read_vgdl(self, vgdl_script):
        
        sprites, interactions, mapping, terminations, args = VGDL.VGDLParser().parse_game_for_theory(vgdl_script)

        # Store sprite information in VGDLLines object
        for c in sprites:
            if 'orientation' in c[2]:
                c[2]['orientation'] = inv_dir_dict[(c[2]['orientation'].x, c[2]['orientation'].y)]

        # Create game objects
        self.obj = VGDLLines()
        for sprite in sprites:
            name, obj_type, params = sprite[:-1]
            if name != 'floor':
                obj_type = obj_type.__name__
                for k in ['img', 'autotiling', 'color', 'shrinkfactor', 'frameRate', 'cons', 'hidden', 'invisible']:
                    if k in params.keys():
                        del params[k]
                self.obj.add(VGDLLine(name, obj_type, params))
        self.names = set(self.obj.keys())
        if 'floor' in self.names:
            self.names.remove('floor')

        # Store game interactions, pruning unnecessary parameters
        self.int = VGDLLines()
        for interaction in interactions:
            int_type, (name1, name2), params = interaction
            int_type = int_type.__name__
            for k in ['img', 'autotiling', 'color', 'shrinkfactor', 'frameRate', 'cons', 'hidden', 'invisible']:
                if k in params.keys():
                    del params[k]
            self.int.add(VGDLLine((name1, name2), int_type, params))
        for name1 in self.obj.obj_names:                        # Interaction pairs with no-interaction
            for name2 in self.obj.obj_names:
                if (name1, name2) not in self.int.int_names:
                    self.int.add(VGDLLine((name1, name2), 'noInteraction'))
                if (name2, name1) not in self.int.int_names:
                    self.int.add(VGDLLine((name2, name1), 'noInteraction'))

        # Create a dictionary of win and loss conditions
        self.terminations = dict(lose=(), win=()) # (?)
        for t in terminations:
            if t[1]['win']:
                if t[0].__name__ == 'Timeout':
                    self.terminations['win_timeout'] = t[1]['limit']
                else:
                    if 'stype1' in t[1].keys():
                        stypes = []
                        i = 1
                        while 'stype' + str(i) in t[1].keys():
                            stypes.append(t[1]['stype'+str(i)])
                            i += 1
                        stypes = tuple(stypes)
                    else:
                        stypes = (t[1]['stype'], )
                    self.terminations['win'] = tuple(stypes)
            else:
                if t[0].__name__ == 'Timeout':
                    self.terminations['lose_timeout'] = t[1]['limit']
                else:
                    if 'stype1' in t[1].keys():
                        stypes = []
                        i = 1
                        while 'stype' + str(i) in t[1].keys():
                            stypes.append(t[1]['stype'+str(i)])
                            i += 1
                        stypes = tuple(stypes)
                    else:
                        stypes = (t[1]['stype'], )
                    self.terminations['lose'] = stypes
        self.terminations = hashdict(self.terminations)

        self.vgdl_obj = hashdict(dict(obj=self.obj, names=tuple(self.names), int=self.int, terminations=self.terminations))

    # Convert game rules representation back into VGDL script
    def write_vgdl(self):
        objs, ints, names, terminations = [self.vgdl_obj[k] for k in ['obj', 'int', 'names', 'terminations']]

        avatar_names = self.avatar_names
        tab = '    '
        vgdl_script = f'BasicGame block_size={self.params["game_params"]["block_size"]}\n'
        vgdl_script += f'{tab}SpriteSet\n'
        vgdl_script += f'{tab}{tab}{BG_NAME} > Immovable img=colors/{BG_COLOR} hidden=True\n'  # add background

        for name, obj in objs.dict.items():
            if name not in avatar_names:
                param_str = ''
                if obj.params is not None:
                    for k, v in obj.params.items():
                        param_str += f' {k}={v}'
                vgdl_script += f'{tab}{tab}{name} > {obj.type}{param_str} img=colors/{self.colors[name]}\n'

        for avatar_name in avatar_names:
            param_str = ''
            obj = objs.dict[avatar_name]
            if obj.params is not None:
                for k, v in obj.params.items():
                    param_str += f' {k}={v}'
            vgdl_script += f'{tab}{tab}{avatar_name} > {obj.type}{param_str} img=colors/{self.colors[avatar_name]}\n'  # add

        # level mapping
        vgdl_script += f'{tab}LevelMapping\n'
        for character, names in self.character_mapping.items():
            vgdl_script += f'{tab}{tab}{character} > {" ".join(names)}\n'

        # enforce consistency in resource interactions
        for names, interation in ints.dict.copy().items():
            if interation.type == 'killIfHasLess':
                reversed_names = (names[1], names[0])
                ints.dict[reversed_names] = VGDLLine(reversed_names, 'removeResource', {'resource': interation.params['resource']})
            # if interation.type == 'removeResource':
            #     reversed_names = (names[1], names[0])
            #     ints.dict[reversed_names] = VGDLLine(reversed_names, 'killIfHasLess', {'resource': interation.params['resource'], 'limit': 0})

        vgdl_script += f'{tab}InteractionSet\n'
        for names, interaction in ints.dict.items():
            if interaction.type != 'noInteraction':
                param_str = ''
                if interaction.params is not None:
                    for k, v in interaction.params.items():
                        param_str += f' {k}={v}'
                vgdl_script += f'{tab}{tab}{interaction.name[0]} {interaction.name[1]} > {interaction.type}{param_str}\n'
        vgdl_script += '\n\n'  # add two lines before adding the EOS interactions for readability
        for name in objs.dict.keys():
            vgdl_script += f'{tab}{tab}{name} EOS > killSprite\n'

        vgdl_script += f'{tab}TerminationSet\n'

        # you lose if either of these objects die
        if 'avatar' in terminations['lose']:
            vgdl_script += f'{tab}{tab}SpriteCounter stype=avatar win=False\n'

        # for stype in terminations['lose']:
        #     vgdl_script += f'{tab}{tab}SpriteCounter stype={stype} win=False\n'
        lose_stypes = [stype for stype in terminations['lose'] if stype != 'avatar']
        if len(lose_stypes) == 1:
            vgdl_script += f'{tab}{tab}SpriteCounter stype={lose_stypes[0]} win=False\n'
        elif len(lose_stypes) > 1:
            stypes_str = ''
            for i_stype, stype in enumerate(lose_stypes):
                if stype != 'avatar':
                    stypes_str += f'stype{i_stype + 1}={stype} '
            vgdl_script += f'{tab}{tab}{"MultiSpriteCounter"} {stypes_str}win=False\n'

        # you win when all of these objects die
        if len(terminations['win']) > 0:
            stypes_str = ''
            for i_stype, stype in enumerate(terminations['win']):
                stypes_str += f'stype{i_stype + 1}={stype} '
            vgdl_script += f'{tab}{tab}{"MultiSpriteCounter"} {stypes_str}win=True\n'
        else:
            vgdl_script += f'{tab}{tab}Timeout limit=3000 win=True'

        if 'win_timeout' in terminations.keys():
            vgdl_script += f'{tab}{tab}Timeout limit={terminations["win_timeout"]} win=True'
        if 'lose_timeout' in terminations.keys():
            vgdl_script += f'{tab}{tab}Timeout limit={terminations["lose_timeout"]} win=False'
        return vgdl_script

    # Read interaction rules to determine agents that can kill player
    def get_enemies(self):
        enemies = dict()
        enemies_blockers = dict()
        for int_names, int in self.int.dict.items():
            if int_names[1] not in enemies.keys() and int_names[1] != 'EOS':
                obj = self.obj.dict[int_names[1]]
            if self.avatar_name == int_names[0] and int.kills_a() and obj.type in ['RandomNPC', 'Chaser']:
                if int_names[1] not in enemies.keys():
                    # distance the enemy can travel between two agent steps
                    enemies[int_names[1]] = obj.params.get('speed', 1) * self.params['agent']['reaction_time'] / obj.params.get('cooldown', 1)
        for enemy in enemies.keys():
            enemies_blockers[enemy] = []
            for int_names, int in self.int.dict.items():
                if enemy in int_names and int.type == 'stepBack':
                    blocker = int_names[1] if enemy == int_names[0] else int_names[0]
                    enemies_blockers[enemy].append(blocker)
        return enemies, enemies_blockers
    
    #
    def check_names(self):
        names = set()
        for int_names in self.int.keys():
            names.add(int_names[0])
            names.add(int_names[1])
        if 'EOS' in names:
            names.remove('EOS')

    @property
    def avatar_names(self):
        return [name for name in self.names if AVATAR_NAME in name]

    @property
    def avatar_name(self):
        avatar_names = self.avatar_names
        assert len(avatar_names) == 1
        avatar_name = list(avatar_names)[0]
        return avatar_name

    @property
    def vgdl_lines(self):
        return self.obj + self.int

    def find(self, c_type, c_key):
        pass

    def copy(self, no_feedback=False):
        copied_rules = Rules(self.params, vgdl_obj=self.vgdl_obj)
        copied_rules.logprior = self.logprior
        copied_rules.interaction_loglikelihood = self.interaction_loglikelihood
        copied_rules.language_loglikelihood = self.language_loglikelihood
        if not no_feedback:
            copied_rules.feedback = self.feedback
        copied_rules.prompt = self.prompt
        return copied_rules

    def __repr__(self):
        s = "\n-----Rules-----\n"
        full = self.int + self.obj
        s += full.__str__()
        s += "Termination\n"
        s += str(self.terminations)
        s += f'\nlogprior: {self.logprior:.3f}'
        s += f'\nloglike: {self.loglikelihood:.3f}'
        s += f'\ninteraction_loglikelihood: {self.interaction_loglikelihood:.3f}'
        s += f'\nlanguage_loglike: {self.language_loglikelihood:.3f}'
        s += f'\nlogjoint: {self.logjoint:.3f}'
        s += '\n--------------\n'
        return s

    # Converts to natural language description for LLM prompting
    def str_llm(self, color_dict):
        return from_rules_to_str(self, color_dict)
        # full = self.int + self.obj
        # s = full.str_llm(color_dict)
        # win = self.terminations['win']
        # if len(win) == 0:
        #     s += format_prompt(None, 'win', None, color_dict, None) + '\n'
        # else:
        #     s += format_prompt(win, 'win', None, color_dict, None)  + '\n'
        # lose = self.terminations['lose']
        # if len(lose) == 0:
        #     s += format_prompt(None, 'lose', None, color_dict, None) + '\n'
        # else:
        #     s += format_prompt(lose, 'lose', None, color_dict, None) + '\n'
        # return s
    def str_obj_llm(self, color_dict):
        return from_rules_to_obj_str(self, color_dict)


    @property
    def logjoint(self):
        return self.loglikelihood + self.logprior

    @property
    def loglikelihood(self):
        return self.interaction_loglikelihood + self.language_loglikelihood

    def __str__(self):
        return self.__repr__()
        # s = "Constraint: "
        # s += str(self.int + self.obj)
        # s += "Termination"
        # s += str(self.terminations)
        # return s

    def __hash__(self):
        return abs(hash(self.vgdl_obj))


    def __eq__(self, other):
        return hash(self) == hash(other)


class Particles(list):
    """
    A set of theories (rules objects)
    """

    def __init__(self, *args):
        super().__init__(*args)
    @property
    def loglikelihoods(self):
        return [el.loglikelihood for el in self]

    @property
    def interaction_loglikelihoods(self):
        return [el.interaction_loglikelihood for el in self]

    @property
    def language_loglikelihoods(self):
        return [el.language_loglikelihood for el in self]

    @property
    def logjoints(self):
        return [el.logjoint for el in self]

    @property
    def logpriors(self):
        return [el.logprior for el in self]

    @property
    def feedbacks(self):
        return [el.feedback for el in self]

    @property
    def posteriors(self):
        return np.exp(normalize_log_joints(np.array(self.logjoints) / 10))

    @property
    def n_particles(self):
        return len(self)

    @property
    def best_index(self):
        return np.argmax(self.logjoints)

    def copy_best(self, criterion='logjoint'):
        if criterion == 'loglikelihood':
            best_idx = np.argmax(self.loglikelihoods)
        elif criterion == 'logjoint':
            best_idx = np.argmax(self.logjoints)
        else:
            raise NotImplementedError
        return self[best_idx].copy()

    def __copy__(self):
        return [self[i].copy() for i in range(self.n_particles)]

    def resample(self, indexes):
        self[:] = [self[i].copy() for i in indexes]

    def replace_with(self, mutated_particles, accepts):
        assert len(self) == len(mutated_particles)
        for i, new, accept in zip(range(self.n_particles), mutated_particles, accepts):
            if accept:
                self[i] = new.copy()

    def keep_best(self, best, criterion='logjoint'):
        if criterion == 'logjoint':
            if best.logjoint > np.max(self.logjoints):
                idx_worse = np.argmin(self.logjoints)
                self[idx_worse] = best.copy()
        elif criterion == 'loglikelihood':
            if best.loglikelihood > np.max(self.loglikelihoods):
                idx_worse = np.argmin(self.loglikelihoods)
                self[idx_worse] = best.copy()
        else:
            raise NotImplementedError


    # Sample theories proportionally to their posteriors
    def sample(self, strategy='argmax'):
        if strategy == 'proportional':
            posteriors = self.posteriors
            print(posteriors)
            return np.random.choice(range(self.n_particles), p=posteriors)
        elif strategy == 'argmax':
            return np.argmax(self.posteriors)
        else:
            raise NotImplementedError

    def __repr__(self):
        s = ''
        for i_p, p in enumerate(self):
            s += f'Particle {i_p + 1}/{len(self)}' + p.__repr__() + '\n'
        return s

    def __str__(self):
        return self.__repr__()
