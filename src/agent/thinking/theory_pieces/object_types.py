from abc import abstractmethod

import numpy as np

from src.utils import inv_dir_dict, find_closest_target_in_map, AVATAR_NAME, get_neighbors, format_prompt, just_changed_direction_left_right, compute_probability_and_confidence
from src.game.rules import VGDLLine

low_not_impossible = 0.04

avatar_int_to_dir = {0: 'UP',
                     1: 'DOWN',
                     2: 'LEFT',
                     3: 'RIGHT'}

class ObjectType:
    """
    Theory piece abstract type - each represents a belief that can determine its own probability
    """
    
    def __init__(self, name, params):
        self.name = name
        self.params = params
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))
        self.prior_prob_low = params['agent']['thinker']['prior_prob_low']
        self.count = 0
        self.mvts_mag = []  # list mvt magnitudes to infer the speed
        self.last_steps = dict()  # record last mvt for each obj in each episode
        self.steps_btw_mvts = []  # list time between movements to infer cooldown
        self.game_names = set()
        self.blockers = dict()
        self.i_ep = set()
        self.counters = dict()
        self.linguistic_scores = dict()
        self.beta_softmax_lang_proposal = params['agent']['thinker']['beta_softmax_lang_proposal']

    # Extracting information from transition and determining whether the current type can explain data
    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        self.count += 1
        if i_ep not in self.i_ep:
            self.last_steps = dict()
            self.i_ep.add(i_ep)
        name = obj['name']
        obj_id = obj['obj_id']
        mvt = obj['mov'][i_step]
        prev_pos, pos = obj['pos'][i_step - 1], obj['pos'][i_step]
        still_alive = prev_pos is not None and pos is not None
        just_died = prev_pos is not None and pos is None
        if pos:
            pos = np.array(pos)
            if i_ep not in self.counters.keys():
                self.counters[i_ep] = dict()
            if i_abs_step not in self.counters[i_ep].keys():
                self.counters[i_ep][i_abs_step] = set()
            self.counters[i_ep][i_abs_step].add(obj_id)
        if prev_pos: prev_pos = np.array(prev_pos)
        # track appearance as a step
        if i_ep not in self.last_steps.keys():
            self.last_steps[i_ep] = dict()
        if obj_id not in self.last_steps[i_ep].keys():
            self.last_steps[i_ep][obj_id] = None

        if still_alive and mvt != (0, 0):  # no mvt
            self.mvts_mag.append(np.sum(np.abs(mvt)))
            if self.last_steps[i_ep][obj_id] is not None:
                self.steps_btw_mvts.append(i_abs_step - self.last_steps[i_ep][obj_id])
            self.last_steps[i_ep][obj_id] = i_abs_step
        return name, obj_id, prev_pos, pos, mvt, still_alive, just_died


    def get_linguistic_probs(self, comm_engine, linguistic_data, params):
        # return None
        if len(linguistic_data) == 0 or len(params) == 0:
            return None, None, None
        linguistic_data = tuple(linguistic_data)
        loglikes = []
        uncertain_str = "I don't know"
        candidates = [self.get_str(param=p) for p in params] + [uncertain_str]
        for p in params:
            prompt_to_eval = self.get_str(param=p)
            if linguistic_data not in self.linguistic_scores.keys():
                self.linguistic_scores[linguistic_data] = dict()
            if p not in self.linguistic_scores[linguistic_data].keys():
                self.linguistic_scores[linguistic_data][p] = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates, self.name)
            loglike = self.linguistic_scores[linguistic_data][p]
            loglikes.append(loglike)
        if uncertain_str not in self.linguistic_scores[linguistic_data].keys():
            unsure_loglike = comm_engine.get_linguistic_score(uncertain_str, linguistic_data, self.game_names, candidates, self.name)
            self.linguistic_scores[linguistic_data][uncertain_str] = unsure_loglike
        else:
            unsure_loglike = self.linguistic_scores[linguistic_data][uncertain_str]

        sampling_probabilities, confidence = compute_probability_and_confidence(loglikes, unsure_loglike)
        return sampling_probabilities, confidence, params


    def get_params_and_probs(self, comm_engine, linguistic_data):
        params, interaction_probs = self.get_params_and_probs_from_interaction_data()

        if len(params) > 0 and self.params['exp_params']['use_language_proposal'] and len(linguistic_data) > 0:
            linguistic_probs, confidence, _ = self.get_linguistic_probs(comm_engine, linguistic_data, params)
            if confidence < 0.5:
                probs = interaction_probs
            else:
                probs = (interaction_probs + 2 * confidence * linguistic_probs) / (1 + 2 * confidence)
        else:
            probs = interaction_probs

        return params, probs


    @property
    def get_str(self, param=None):
        pass

    def get_params_and_probs_from_interaction_data(self):
        return [], []

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        pass

    # Get the probability that this object is a blocker
    def get_prob_stepback(self, blocker, vgdl_lines):

        if blocker not in self.blockers.keys() or (vgdl_lines and ((self.name, blocker) in vgdl_lines.int_names or (blocker, self.name) in vgdl_lines.int_names)):
            return 0
        if blocker == 'wall':
            return 0.9
        blockers, probs = list(self.blockers.keys()), np.array(list(self.blockers.values()), dtype=float)
        if self.params['exp_params']['use_data_proposal']:
            probs[np.where(probs > 0)] = 0.5
            probs[np.where(probs == 0)] = low_not_impossible
        else:
            probs[:] = 0.33
        return probs[blockers.index(blocker)]

    def update_with_new_names(self, new_names):
        self.game_names = self.game_names.union(new_names)

    @abstractmethod
    def update_with_no_int(self, theory_pairs):
        pass

    @property
    def cooldown(self):
        if len(self.steps_btw_mvts) == 0:
            return 1
        else:
            return np.min(self.steps_btw_mvts)

    @property
    def speed(self):
        if len(self.mvts_mag) == 0:
            return 1
        else:
            return np.min(self.mvts_mag)

    def singleton(self, ref_theory):
        all_counts = sum([[len(val) for val in counters.values()] for counters in self.counters.values()], [])
        if len(all_counts) == 0:
            return True
        if np.max(all_counts) > 1:
            return False
        elif AVATAR_NAME in self.name:
            return True
        else:
            if ref_theory and self.name in ref_theory.obj_names:
                return ref_theory.dict[self.name].params['singleton']
            else:
                singleton = True
                return singleton

    def add_bonus_for_robustness(self, scores):
        # add small score such that no option has probability 0
        if np.argwhere(scores==0).sum() > 0:
            min_non_zero = np.min(scores[np.where(scores!=0)])
            return scores + low_not_impossible * min_non_zero
        else:
            return scores
    

class Immovable(ObjectType):
    """
    Theory piece indictating belief that this is an immovable object
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Immovable'
        self.move = 0

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        return [VGDLLine(self.name, 'Immovable', {'singleton': self.singleton(ref_theory)})], np.log(1)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
        if still_alive and mvt != (0, 0):
            self.move += 1

    def update_with_new_names(self, new_names):
        pass

    def update_with_no_int(self, theory_pairs):
        pass

    def get_score(self):
        return int(self.move == 0)


class Portal(ObjectType):
    """
    Parameters - whether it teleports and where it teleports
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Portal'
        self.move = 0
        self.teleported_exits = dict()

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        # objects that could have gone through the portal
        if prev_pos is not None:
            neighbors = set(get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=2)) - \
                        set(get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=1))

            # searching for teleportations
            for n_id in neighbors:
                mov = obj_episode[n_id]['mov'][i_step]
                if mov is not None and np.sum(np.abs(mov)) > 1:  # neighbor teleported
                    teleported = obj_episode[n_id]['name']
                    exits = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", n_id, np.array(obj_episode[n_id]['pos'][i_step]), 0)
                    for exit_obj in exits:
                        if (teleported, obj_episode[exit_obj]['name']) in self.teleported_exits.keys():
                            self.teleported_exits[(teleported, obj_episode[exit_obj]['name'])] += 1

        if still_alive and mvt != (0, 0):  # no mvt
            self.move += 1

    def get_score(self):
        if self.move > 0:
            return 0
        else:
            if len(self.teleported_exits) == 0:
                return 0
            elif np.sum(list(self.teleported_exits.values())) == 0:
                return 0
            else:
                return 1

    def get_params_and_probs_from_interaction_data(self):
        params = list(self.teleported_exits.keys())
        scores = np.array(list(self.teleported_exits.values()))
        if np.sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            if not self.params['exp_params']['use_data_proposal']:
                scores[np.where(scores> 0)] = 1
            probs = scores / sum(scores)
        return params, probs

    # Sample the teleportation destination
    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        params, probs = self.get_params_and_probs(comm_engine, linguistic_data)

        sampled_idx = None
        if ref_theory:
            for int_name in ref_theory.int_names:
                if ref_theory.dict[int_name].type == 'teleportToExit' and int_name[1] == self.name:
                    stype = ref_theory.dict[self.name].params['stype']
                    teleported_stype = (int_name[0], stype)
                    if teleported_stype in params:
                        sampled_idx = params.index(teleported_stype)
                        if probs[sampled_idx] == 0:
                            sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(probs)), p=probs)
        teleported, exit_obj = params[sampled_idx]
        new_vgdl_lines = [VGDLLine(self.name, 'Portal', {'stype': exit_obj, 'singleton': self.singleton(ref_theory)}),
                          VGDLLine((teleported, self.name), 'teleportToExit', )]
        return new_vgdl_lines, np.log(1 / len(self.game_names))

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        if len(new_names) > 0:
            for new_name in self.game_names:
                if new_name is not self.name and AVATAR_NAME in new_name:
                    for new_name2 in self.game_names:
                        if new_name2 is not self.name and new_name2 != new_name and AVATAR_NAME not in new_name2 and new_name2 != 'wall':
                            if (new_name, new_name2) not in self.teleported_exits.keys():
                                self.teleported_exits[(new_name, new_name2)] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.teleported_exits.items():
            teleported, exit = k
            if theory_pairs[(teleported, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.teleported_exits[k]


class Flicker(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Flicker'
        self.move = 0
        self.flicker_lives = []
        self.flicker_birth = dict()

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive and mvt != (0, 0):  # no mvt
            self.move += 1
        elif prev_pos is None and pos is not None:
            self.flicker_birth[obj_id] = i_abs_step
        elif just_died:
            age = i_abs_step - self.flicker_birth.get(obj_id, 0)
            self.flicker_lives.append(age)

    def get_score(self):
        if self.move > 0:
            return 0
        elif len(self.flicker_lives) == 0 or np.max(self.flicker_lives) > 30:
            return low_not_impossible
        elif len(self.flicker_lives) < 2:
            return 0.1
        else:
            values, counts = np.unique(self.flicker_lives, return_counts=True)
            counts = counts.astype(float) / len(self.flicker_lives)
            score = np.max(counts)
            if score == 1:
                score = 5
            elif score > 0.7 and np.max(self.flicker_lives) == values[np.argmax(counts)]:
                score = 3
            return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        if len(self.flicker_lives) == 0:
            limit = 5
        else:
            limit = np.max(self.flicker_lives)
        prob = 1 / 100  # low prior on any particular value for 'limit'
        return [VGDLLine(self.name, 'Flicker', {'limit': limit, 'singleton': self.singleton(ref_theory)})], np.log(prob)
    def update_with_new_names(self, new_names):
        pass

    def update_with_no_int(self, theory_pairs):
        pass


class ResourcePack(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'ResourcePack'
        self.resources_max = None
        self.collectors = set()
        self.avatar_name = None

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # prob = 1
        # sample extra parameters and related interactions
        new_vgdl_lines = [VGDLLine(self.name, 'ResourcePack', {'limit': self.resources_max[self.name], 'singleton': self.singleton(ref_theory)})]
        for collector in self.collectors:
            new_vgdl_lines.append(VGDLLine((self.name, collector), 'addResource', {'resource': self.name}))
            assert collector != self.name
            new_vgdl_lines.append(VGDLLine((collector, self.name), 'noInteraction'))
        if self.avatar_name not in self.collectors:
            avatar_collects = None
            if ref_theory:
                interaction = ref_theory.dict.get((self.name, self.avatar_name))
                if interaction and interaction.type == 'addResource':
                    avatar_collects = True
                else:
                    assert False
                    # avatar_collects = False
            new_vgdl_lines.append(VGDLLine((self.name, self.avatar_name), 'addResource', {'resource': self.name}))
            assert self.name != self.avatar_name
            new_vgdl_lines.append(VGDLLine((self.avatar_name, self.name), 'noInteraction'))
            # prob *= 0.5
        return new_vgdl_lines, np.log(1)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if self.resources_max is None:
            self.resources_max = obj['resources_max'][0]
        if just_died:
            collectors_candidates = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=0)
            for collector in collectors_candidates:
                if obj_episode[collector]['resources'][i_step][self.name] > obj_episode[collector]['resources'][i_step - 1][self.name]:
                    self.collectors.add(obj_episode[collector]['name'])

    def update_with_new_names(self, new_names):
        for new_name in new_names:
            if AVATAR_NAME in new_name:
                self.avatar_name = new_name

    def update_with_no_int(self, theory_pairs):
        pass
    def get_score(self):
        return 1


class SpawnPoint(ObjectType):
    """
    
    """
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'SpawnPoint'
        self.name = name
        self.move = 0
        self.spawned = dict()  # track nb of times each obj appeared to be spawned onto self.name
        self.spawned_times = dict()  # track when they were spawned
        self.spawned_totals = dict()
        self.ep_len = dict()  # track episode length

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        for key in self.spawned_times.keys():
            if i_ep not in self.spawned_times[key].keys():
                self.spawned_times[key][i_ep] = dict()
        if i_ep not in self.ep_len.keys():
            self.ep_len[i_ep] = i_abs_step
        else:
            self.ep_len[i_ep] = max(self.ep_len[i_ep], i_abs_step)

        if still_alive:
            if mvt != (0, 0):
                self.move += 1
            else:
                # did something just got born there?
                objs_here = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=0)
                for obj_id2 in objs_here:
                    if self.name == 'hotdoghole':
                        stop = 1
                    if obj_episode[obj_id2]['pos'][i_step - 1] is None:
                        assert obj_episode[obj_id2]['pos'][i_step] is not None
                        obj_name = obj_episode[obj_id2]['name']
                        if obj_name in self.spawned.keys():
                            self.spawned[obj_name] += 1
                            if obj_id not in self.spawned_times[obj_name][i_ep].keys():
                                self.spawned_times[obj_name][i_ep][obj_id] = []
                            self.spawned_times[obj_name][i_ep][obj_id].append(i_abs_step)

        if just_died:
            for obj_name in self.spawned_times.keys():
                if obj_name != self.name:
                    if i_ep in self.spawned_times[obj_name].keys():
                        if obj_id in self.spawned_times[obj_name][i_ep].keys():
                            self.spawned_totals[obj_name].append(len(self.spawned_times[obj_name][i_ep][obj_id]))
                        else:
                            self.spawned_totals[obj_name].append(0)
                    else:
                        self.spawned_totals[obj_name].append(0)


    def get_score(self):
        if self.move > 0:
            return 0
        elif sum(self.spawned.values()) == 0:
            return low_not_impossible
        else:
            return 2

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        # sample spawned object name
        params = list(self.spawned.keys())
        if len(params) == 0:
            # No spawn targets available - return empty result
            return [], 0.0
        scores = np.array(list(self.spawned.values()))
        if np.sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            scores = self.add_bonus_for_robustness(scores)
            if not self.params['exp_params']['use_data_proposal']:
                scores[np.where(scores > 0)] = 1
            probs = scores / sum(scores)

        sampled_idx = None
        if ref_theory:
            if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'SpawnPoint':
                stype = ref_theory.dict[self.name].params['stype']
                if stype in params:
                    sampled_idx = params.index(stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(probs)), p=probs)
        spawned = params[sampled_idx]

        # what is the total before death?
        # if len(self.spawned_totals[spawned]) == 0:  # we haven't seen the obj die yet
        #     n_spawns = []
        #     for spawned_per_episode in self.spawned_times[spawned].values():
        #         for spawned_per_obj in spawned_per_episode.values():
        #             n_spawns.append(len(spawned_per_obj))
        #     if len(n_spawns) == 0:
        #         total = 5
        #     else:
        #         total = 2 * np.max(n_spawns)  # assume twice the highest number of spawns
        # else:
        #     total = np.max(self.spawned_totals[spawned])
        #     if total == 0:
        #         total = 5
        if self.spawned[spawned] == 0:
            spawned_prob = 0.1
            new_vgdl_lines = [VGDLLine(self.name, 'SpawnPoint', {'stype': spawned, 'prob': float(f"{spawned_prob:.3f}"), 'singleton': self.singleton(ref_theory)})]
        else:
            probs = []
            diffs = []
            for i_ep in range(len(self.ep_len)):
                if i_ep in self.spawned_times[spawned].keys():
                    probs += [len(val) / self.ep_len[i_ep] for val in self.spawned_times[spawned][i_ep].values()]
                    diffs += sum([list(np.diff(val)) for val in self.spawned_times[spawned][i_ep].values()], [])
            if len(diffs) > 0 and len(set(diffs)) == 1:
                cooldown = list(set(diffs))[0]
                new_vgdl_lines = [VGDLLine(self.name, 'SpawnPoint', {'stype': spawned, 'cooldown': cooldown, 'singleton': self.singleton(ref_theory)})]
            elif len(probs) > 0 and float(f"{np.mean(probs):.3f}") > 0:
                new_vgdl_lines = [VGDLLine(self.name, 'SpawnPoint', {'stype': spawned, 'prob': float(f"{np.mean(probs):.3f}"), 'singleton': self.singleton(ref_theory)})]
            else:
                new_vgdl_lines = [VGDLLine(self.name, 'Immovable', {'singleton': self.singleton(ref_theory)})]
        return new_vgdl_lines, np.log(1)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.spawned.keys() and new_name != self.name and AVATAR_NAME not in new_name and new_name != 'wall':
                self.spawned[new_name] = 0
                self.spawned_times[new_name] = dict()
                self.spawned_totals[new_name] = []

    def update_with_no_int(self, theory_pairs):
        pass


class Passive(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Passive'
        self.plausible_move = 0
        self.implausible_move = 0
        self.pushers = dict()  # list possible pushers
        self.blockers = dict()  # list possible blockers

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    
    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                # maybe nothing pushed it, or it was blocked
                # can we find one blocker?
                # a blocker doesn't collide now, but is less than two steps away
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    # is there another object on the other side that has not moved?
                    vec = pos - np.array(obj_episode[b]['pos'][i_step])
                    pusher_pos = pos + vec
                    pusher_candidates = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", None, pusher_pos, radius=1)
                    for pusher in pusher_candidates:
                        # the pusher should not have moved in the last step
                        if obj_episode[pusher]['mov'][i_step] is not None and np.sum(np.abs(obj_episode[pusher]['mov'][i_step])) == 0:
                            if AVATAR_NAME in obj_episode[pusher]['name']:
                                sign_vec = vec.copy()
                                if np.abs(sign_vec[0]) < 1: sign_vec[0] = 0
                                if np.abs(sign_vec[1]) < 1: sign_vec[1] = 0
                                if avatar_int_to_dir.get(actions[i_step-1]) == inv_dir_dict.get(tuple(np.sign(-sign_vec)), None):
                                    # here we're sure the avatar tried to push
                                    blocker = obj_episode[b]['name']
                                    if blocker in self.blockers.keys():
                                        self.blockers[blocker] += 5
                                        break
                            blocker = obj_episode[b]['name']
                            if blocker in self.blockers.keys():
                                self.blockers[blocker] += 0.5


            else:
                # it moved, did something push it?
                pusher_candidates = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=1)
                if len(pusher_candidates) > 0:
                    prev_pusher_pos = prev_pos + (prev_pos - pos)
                    prev_pusher_candidates = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, np.array(prev_pusher_pos), radius=1)
                    pusher_candidates = set(pusher_candidates).intersection(set(prev_pusher_candidates))
                    if 'wall' in pusher_candidates: pusher_candidates.remove('wall')
                    valid_pusher_candidate = False
                    for pusher in pusher_candidates:
                        pusher_name = obj_episode[pusher]['name']
                        if pusher_name in self.pushers.keys():
                            valid_pusher_candidate = True
                            self.pushers[pusher_name] += 1
                    if valid_pusher_candidate:
                        self.plausible_move += 1
                else:
                    self.implausible_move += 1

    def get_score(self):
        if self.implausible_move > 0:
            return 0
        elif len(self.pushers) == 0:
            return 0
        elif sum(self.pushers.values()) == 0:
            return 0.1
        else:
            if self.count == 0:
                ratio_mvt = 0
            else:
                ratio_mvt = len(self.mvts_mag) / self.count * self.cooldown
            if ratio_mvt < 0.2:
                return 3
            else:
                return 1

    def get_params_and_probs_from_interaction_data(self):
        params = list(self.pushers.keys())
        scores = np.array(list(self.pushers.values()))
        if np.sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            scores = self.add_bonus_for_robustness(scores)
            if not self.params['exp_params']['use_data_proposal']:
                scores[np.where(scores> 0)] = 1
            probs = scores / sum(scores)
        return params, probs




    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        new_vgdl_lines = [VGDLLine(self.name, 'Passive', {'singleton': False, 'speed': self.speed})]

        pusher_names = list(self.pushers.keys())
        pusher_scores = np.array(list(self.pushers.values()))

        # get avatar name
        avatar_names = [name for name in self.pushers.keys() if AVATAR_NAME in name]
        assert len(avatar_names) <= 1
        # if len(avatar_names) > 0:
        #     pusher_scores[pusher_names.index(avatar_names[0])] = max(1, max(pusher_scores))

        if sum(np.array(pusher_scores) > 0) > 1:
            several_pusher_possible = True
        else:
            several_pusher_possible = False

        prior_prob = 1 / len(pusher_scores)
        if sum(pusher_scores) == 0:
            new_vgdl_lines = [VGDLLine(self.name, 'Immovable', {'singleton': self.singleton(ref_theory)})]
        else:
            pusher_scores = self.add_bonus_for_robustness(pusher_scores)

            # prevent sampling pushers if other interactions sampled for that pair (eg step back)
            for i_k, k in enumerate(list(self.pushers.keys())):
                if (self.name, k) in int_keys_taken:
                    pusher_scores[i_k] = 0

            if sum(pusher_scores) == 0:
                new_vgdl_lines = [VGDLLine(self.name, 'Immovable', {'singleton': self.singleton(ref_theory)})]
            else:
                probs = pusher_scores / np.sum(pusher_scores)

                pushers = []
                if ref_theory:
                    for int_name in ref_theory.int_names:
                        if ref_theory.dict[int_name].type == 'bounceForward' and int_name[0] == self.name:
                            pusher = int_name[1]
                            if pusher in self.pushers.keys():
                                sampled_idx = list(self.pushers.keys()).index(pusher)
                                if probs[sampled_idx] > 0:
                                    pushers.append(pusher)

                if len(pushers) == 0:
                    n_pushers = 1 if not several_pusher_possible else np.random.choice([1, 2])
                    sampled_idx = np.random.choice(range(len(pusher_scores)), p=probs, size=n_pushers, replace=False)
                    pushers = [list(self.pushers.keys())[idx] for idx in sampled_idx]
                for pusher in pushers:
                    assert (pusher, self.name) not in int_keys_taken
                    assert (self.name, pusher) not in int_keys_taken
                    new_vgdl_lines.append(VGDLLine((self.name, pusher), 'bounceForward'))
                    if self.name != pusher:
                        new_vgdl_lines.append(VGDLLine((pusher, self.name), 'noInteraction'))

        return new_vgdl_lines, np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.pushers.keys() and new_name != 'wall':
                self.pushers[new_name] = 0
            if new_name not in self.blockers.keys():
                self.blockers[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)

        for k in keys_to_remove:
            del self.blockers[k]
        keys_to_remove = []
        for k, v in self.pushers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.pushers[k]


class Missile(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Missile'
        self.blockers = dict()
        self.episode_starts = dict()  # track obj creation for each object and episode
        self.first_dirs = dict(UP=0, DOWN=0, RIGHT=0, LEFT=0)
        self.turn_around_objs = dict()
        self.directions_per_obj = dict()
        self.mvts_mag_per_obj = dict()

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            mvt = obj['mov'][i_step]
            if mvt == (0, 0):  # no mvt
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.blockers.keys():
                        self.blockers[blocker] += 1
            elif mvt == (0, 1) and just_changed_direction_left_right(obj_episode, obj_id, i_step):
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.turn_around_objs.keys():
                        self.turn_around_objs[blocker] += 1
            else:
                if still_alive and mvt != (0, 0):  # no mvt
                    if obj_id + str(i_ep) not in self.mvts_mag_per_obj.keys():
                        self.mvts_mag_per_obj[obj_id + str(i_ep)] = []
                    self.mvts_mag_per_obj[obj_id + str(i_ep)].append(np.sum(np.abs(mvt)))

                obj_dir = inv_dir_dict[tuple(np.sign(mvt))]
                if i_ep not in self.episode_starts.keys():
                    self.episode_starts[i_ep] = set()
                if obj_id not in self.episode_starts[i_ep]:
                    self.episode_starts[i_ep].add(obj_id)
                    self.first_dirs[obj_dir] += 1
                if obj_id + str(i_ep) not in self.directions_per_obj.keys():
                    self.directions_per_obj[obj_id + str(i_ep)] = dict(UP=0, DOWN=0, RIGHT=0, LEFT=0)
                self.directions_per_obj[obj_id + str(i_ep)][obj_dir] += 1

    def get_score(self):

        if self.count == 0:
            ratio_mvt = 0
        else:
            ratio_mvt = len(self.mvts_mag) / self.count * self.cooldown

        if len(self.mvts_mag) == 0:
            score = low_not_impossible
        else:
            obj_ids = self.directions_per_obj.keys()
            # could be turning around
            double_keys = [('UP', 'DOWN'), ('LEFT', 'RIGHT')]
            two_dir_scores = [np.max(np.array([self.directions_per_obj[obj_id][keys[0]] + self.directions_per_obj[obj_id][keys[1]] for keys in double_keys]) / len(
                self.mvts_mag_per_obj[obj_id])) for obj_id in obj_ids]
            two_dir_score = np.mean(two_dir_scores)
            two_dir_score = max(low_not_impossible, (two_dir_score - 0.5) * 2)
            self.two_dirs_max = two_dir_score

            # could be one dir only
            keys = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            one_dir_scores = [np.max(np.array([self.directions_per_obj[obj_id][key] for key in keys]) / len(self.mvts_mag_per_obj[obj_id])) for obj_id in obj_ids]
            one_dir_score = np.mean(one_dir_scores)
            one_dir_score = max(low_not_impossible, (one_dir_score - 0.25) / 0.75)
            self.one_dir_max = one_dir_score
            if ratio_mvt < 0.2:
                score = 0.1
            else:
                score = max(one_dir_score, two_dir_score)
                if score == 1:
                    score = 5
        return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        if sum(self.first_dirs.values()) == 0:
            orientation = 'UP'
        else:
            scores = np.array(list(self.first_dirs.values()))
            probs = scores / sum(scores)
            orientation = None
            if ref_theory:
                if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'Missile':
                    orientation = ref_theory.dict.get(self.name).params['orientation']
            if orientation is None:
                orientation = list(self.first_dirs.keys())[np.random.choice(range(len(scores)), p=probs)]

        new_vgdl_lines = [VGDLLine(self.name, 'Missile', {'orientation': orientation, 'singleton': self.singleton(ref_theory), 'speed': self.speed, 'cooldown': self.cooldown})]
        data_prob = 1
        prior_prob = 1
        # does it switch direction?
        if len(self.mvts_mag) > 0:

            if ref_theory:
                switches = False
                for int_name in ref_theory.int_names:
                    if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                        switches = True
                        break
                # data_prob = 1
            else:
                switch_prob = self.two_dirs_max / (self.two_dirs_max + self.one_dir_max)
                switches = np.random.rand() < switch_prob
            if len(self.turn_around_objs) > 0:
                prior_prob *= self.prior_prob_low
                blocked_scores = np.array(list(self.turn_around_objs.values()))
                for i_k, k in enumerate(list(self.turn_around_objs.keys())):
                    if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                        blocked_scores[i_k] = 0
                blocked_probs = blocked_scores / sum(blocked_scores)
                sampled_idx = None
                if ref_theory:
                    for int_name in ref_theory.int_names:
                        if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                            blocker = int_name[1]
                            if blocker in self.turn_around_objs.keys():
                                sampled_idx = list(self.turn_around_objs).index(blocker)
                                if blocked_probs[sampled_idx] == 0:
                                    sampled_idx = None
                if sampled_idx is None:
                    sampled_idx = np.random.choice(range(len(self.turn_around_objs)), p=blocked_probs)
                blocker = list(self.turn_around_objs.keys())[sampled_idx]
                assert (self.name, blocker) not in int_keys_taken
                assert (blocker, self.name) not in int_keys_taken
                new_vgdl_lines.append(VGDLLine((self.name, blocker), 'turnAround'))
                if blocker != self.name:
                    new_vgdl_lines.append(VGDLLine((blocker, self.name), 'noInteraction'))
            elif switches:
                prior_prob *= self.prior_prob_low
                blocked_scores = np.array(list(self.blockers.values()))
                for i_k, k in enumerate(list(self.blockers.keys())):
                    if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                        blocked_scores[i_k] = 0

                if sum(blocked_scores) > 0:
                    blocked_scores = self.add_bonus_for_robustness(blocked_scores)
                    for i_k, k in enumerate(list(self.blockers.keys())):
                        if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                            blocked_scores[i_k] = 0
                    blocked_probs = blocked_scores / sum(blocked_scores)

                    sampled_idx = None
                    if ref_theory:
                        for int_name in ref_theory.int_names:
                            if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                                blocker = int_name[1]
                                if blocker in self.blockers.keys():
                                    sampled_idx = list(self.blockers).index(blocker)
                                    if blocked_probs[sampled_idx] == 0:
                                        sampled_idx = None
                    if sampled_idx is None:
                        sampled_idx = np.random.choice(range(len(self.blockers)), p=blocked_probs)

                    blocker = list(self.blockers.keys())[sampled_idx]
                    assert (self.name, blocker) not in int_keys_taken
                    assert (blocker, self.name) not in int_keys_taken
                    new_vgdl_lines.append(VGDLLine((self.name, blocker), 'reverseDirection'))
                    if blocker != self.name:
                        new_vgdl_lines.append(VGDLLine((blocker, self.name), 'noInteraction'))
            else:
                prior_prob *= (1 - self.prior_prob_low)
        return new_vgdl_lines, np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys():
                self.blockers[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]


class Bomber(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Bomber'
        self.blockers = dict()
        self.episode_starts = dict()  # track obj creation for each object and episode
        self.first_dirs = dict(UP=0, DOWN=0, RIGHT=0, LEFT=0)
        self.directions_per_obj = dict()
        self.mvts_mag_per_obj = dict()
        self.turn_around_objs = dict()

        self.spawned = dict()  # track nb of times each obj appeared to be spawned onto self.name
        self.spawned_times = dict()  # track when they were spawned
        self.spawned_totals = dict()
        self.ep_len = dict()  # track episode length

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        for key in self.spawned_times.keys():
            if i_ep not in self.spawned_times[key].keys():
                self.spawned_times[key][i_ep] = dict()
        if i_ep not in self.ep_len.keys():
            self.ep_len[i_ep] = i_abs_step
        else:
            self.ep_len[i_ep] = max(self.ep_len[i_ep], i_abs_step)

        if still_alive:
            mvt = obj['mov'][i_step]
            if mvt == (0, 0):  # no mvt
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.blockers.keys():
                        self.blockers[blocker] += 1
            elif mvt == (0, 1) and just_changed_direction_left_right(obj_episode, obj_id, i_step):
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.turn_around_objs.keys():
                        self.turn_around_objs[blocker] += 1
            else:
                if obj_id + str(i_ep) not in self.mvts_mag_per_obj.keys():
                    self.mvts_mag_per_obj[obj_id + str(i_ep)] = []
                self.mvts_mag_per_obj[obj_id + str(i_ep)].append(np.sum(np.abs(mvt)))

                obj_dir = inv_dir_dict[tuple(np.sign(mvt))]
                if i_ep not in self.episode_starts.keys():
                    self.episode_starts[i_ep] = set()
                if obj_id not in self.episode_starts[i_ep]:
                    self.episode_starts[i_ep].add(obj_id)
                    self.first_dirs[obj_dir] += 1
                if obj_id + str(i_ep) not in self.directions_per_obj.keys():
                    self.directions_per_obj[obj_id + str(i_ep)] = dict(UP=0, DOWN=0, RIGHT=0, LEFT=0)
                self.directions_per_obj[obj_id + str(i_ep)][obj_dir] += 1

            # did something just got born there?
            objs_here = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=0)
            for obj_id2 in objs_here:
                if obj_episode[obj_id2]['pos'][i_step - 1] is None:
                    assert obj_episode[obj_id2]['pos'][i_step] is not None
                    if name == 'alien':
                        stop = 1
                    obj_name = obj_episode[obj_id2]['name']
                    if obj_name in self.spawned.keys():
                        self.spawned[obj_name] += 1
                        if obj_id not in self.spawned_times[obj_name][i_ep].keys():
                            self.spawned_times[obj_name][i_ep][obj_id] = []
                        self.spawned_times[obj_name][i_ep][obj_id].append(i_abs_step)

        if just_died:
            for obj_name in self.spawned_times.keys():
                if obj_name != self.name:
                    if i_ep in self.spawned_times[obj_name].keys():
                        if obj_id in self.spawned_times[obj_name][i_ep].keys():
                            self.spawned_totals[obj_name].append(len(self.spawned_times[obj_name][i_ep][obj_id]))
                        else:
                            self.spawned_totals[obj_name].append(0)
                    else:
                        self.spawned_totals[obj_name].append(0)


    def get_score(self):
        if self.count == 0:
            ratio_mvt = 0
        else:
            ratio_mvt = len(self.mvts_mag) / self.count * self.cooldown

        if len(self.mvts_mag) == 0 or sum(self.spawned.values()) == 0:
            score = low_not_impossible
            self.two_dirs_max = 0
            self.one_dir_max = 1
        else:
            obj_ids = self.directions_per_obj.keys()
            # could be turning around
            double_keys = [('UP', 'DOWN'), ('LEFT', 'RIGHT')]
            two_dir_scores = [np.max(np.array([self.directions_per_obj[obj_id][keys[0]] + self.directions_per_obj[obj_id][keys[1]] for keys in double_keys]) / len(
                self.mvts_mag_per_obj[obj_id])) for obj_id in obj_ids]
            two_dir_score = np.mean(two_dir_scores)
            two_dir_score = max(low_not_impossible, (two_dir_score - 0.5) * 2)
            self.two_dirs_max = two_dir_score

            # could be one dir only
            keys = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            one_dir_scores = [np.max(np.array([self.directions_per_obj[obj_id][key] for key in keys]) / len(self.mvts_mag_per_obj[obj_id])) for obj_id in obj_ids]
            one_dir_score = np.mean(one_dir_scores)
            one_dir_score = max(low_not_impossible, (one_dir_score - 0.25) / 0.75)
            self.one_dir_max = one_dir_score
            if ratio_mvt < 0.2:
                score = 0.1
            else:
                score = max(one_dir_score, two_dir_score)
                if score == 1:
                    score = 5
        return score


    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        if self.name == 'alien':
            stop = 1
        if sum(self.first_dirs.values()) == 0:
            orientation = 'UP'
        else:
            scores = np.array(list(self.first_dirs.values()))
            probs = scores / sum(scores)
            orientation = None
            if ref_theory:
                if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'Bomber':
                    orientation = ref_theory.dict.get(self.name).params['orientation']
            if orientation is None:
                orientation = list(self.first_dirs.keys())[np.random.choice(range(len(scores)), p=probs)]

        # sample extra parameters and related interactions
        # sample spawned object name
        params = list(self.spawned.keys())
        if len(params) == 0:
            # No spawn targets available - return empty result
            return [], 0.0
        scores = np.array(list(self.spawned.values()))
        if np.sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            scores = self.add_bonus_for_robustness(scores)
            if not self.params['exp_params']['use_data_proposal']:
                scores[np.where(scores > 0)] = 1
            probs = scores / sum(scores)

        sampled_idx = None
        if ref_theory:
            if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'Bomber':
                stype = ref_theory.dict[self.name].params['stype']
                if stype in params:
                    sampled_idx = params.index(stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(probs)), p=probs)
        spawned = params[sampled_idx]

        # what is the total before death?
        # if len(self.spawned_totals[spawned]) == 0:  # we haven't seen the obj die yet
        #     n_spawns = []
        #     for spawned_per_episode in self.spawned_times[spawned].values():
        #         for spawned_per_obj in spawned_per_episode.values():
        #             n_spawns.append(len(spawned_per_obj))
        #     if len(n_spawns) == 0:
        #         total = 5
        #     else:
        #         total = 2 * np.max(n_spawns)  # assume twice the highest number of spawns
        # else:
        #     total = np.max(self.spawned_totals[spawned])
        #     if total == 0:
        #         total = 5
        if self.spawned[spawned] == 0:
            spawned_prob = 0.1
        else:
            probs = []
            for i_ep in range(len(self.ep_len)):
                if i_ep in self.spawned_times[spawned].keys():
                    probs += [len(val) / self.ep_len[i_ep] for val in self.spawned_times[spawned][i_ep].values() if len(val) > 0]
            if len(probs) == 0:
                spawned_prob = None
            else:
                spawned_prob = np.mean(probs)

        if spawned_prob is not None and float(f"{spawned_prob:.3f}") > 0:
            # prob = float(f"{spawned_prob:.2f}")
            # print(f'prob: {prob}')
            new_vgdl_lines = [VGDLLine(self.name, 'Bomber', {'orientation': orientation, 'singleton': False, 'speed': self.speed, 'cooldown': self.cooldown,
                                                         'stype': spawned, 'prob': float(f"{spawned_prob:.3f}")})]
        else:
            new_vgdl_lines = [VGDLLine(self.name, 'Missile', {'orientation': orientation, 'singleton': False, 'speed': self.speed, 'cooldown': self.cooldown})]

        data_prob = 1
        prior_prob = 1
        # does it switch direction?
        if len(self.mvts_mag) > 0:
            if ref_theory:
                switches = False
                for int_name in ref_theory.int_names:
                    if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                        switches = True
                        break
                # data_prob = 1
            else:
                switch_prob = self.two_dirs_max / (self.two_dirs_max + self.one_dir_max)
                switches = np.random.rand() < switch_prob

            if len(self.turn_around_objs) > 0:
                blocked_scores = np.array(list(self.turn_around_objs.values()))
                for i_k, k in enumerate(list(self.turn_around_objs.keys())):
                    if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                        blocked_scores[i_k] = 0
                if sum(blocked_scores) > 0:
                    blocked_probs = blocked_scores / sum(blocked_scores)
                    sampled_idx = None
                    if ref_theory:
                        for int_name in ref_theory.int_names:
                            if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                                blocker = int_name[1]
                                if blocker in self.turn_around_objs.keys():
                                    sampled_idx = list(self.turn_around_objs).index(blocker)
                                    if blocked_probs[sampled_idx] == 0:
                                        sampled_idx = None
                    if sampled_idx is None:
                        sampled_idx = np.random.choice(range(len(self.turn_around_objs)), p=blocked_probs)
                    blocker = list(self.turn_around_objs.keys())[sampled_idx]
                    assert (self.name, blocker) not in int_keys_taken
                    assert (blocker, self.name) not in int_keys_taken
                    new_vgdl_lines.append(VGDLLine((self.name, blocker), 'turnAround'))
                    if blocker != self.name:
                        new_vgdl_lines.append(VGDLLine((blocker, self.name), 'noInteraction'))
            elif switches:
                prior_prob *= self.prior_prob_low
                blocked_scores = np.array(list(self.blockers.values()))
                for i_k, k in enumerate(list(self.blockers.keys())):
                    if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                        blocked_scores[i_k] = 0

                if sum(blocked_scores) > 0:
                    blocked_scores = self.add_bonus_for_robustness(blocked_scores)
                    for i_k, k in enumerate(list(self.blockers.keys())):
                        if (self.name, k) in int_keys_taken or (k, self.name) in int_keys_taken:
                            blocked_scores[i_k] = 0
                    blocked_probs = blocked_scores / sum(blocked_scores)

                    sampled_idx = None
                    if ref_theory:
                        for int_name in ref_theory.int_names:
                            if ref_theory.dict[int_name].type == 'reverseDirection' and int_name[0] == self.name:
                                blocker = int_name[1]
                                if blocker in self.blockers.keys():
                                    sampled_idx = list(self.blockers).index(blocker)
                                    if blocked_probs[sampled_idx] == 0:
                                        sampled_idx = None
                    if sampled_idx is None:
                        sampled_idx = np.random.choice(range(len(self.blockers)), p=blocked_probs)

                    blocker = list(self.blockers.keys())[sampled_idx]
                    assert (self.name, blocker) not in int_keys_taken
                    assert (blocker, self.name) not in int_keys_taken
                    new_vgdl_lines.append(VGDLLine((self.name, blocker), 'reverseDirection'))
                    if blocker != self.name:
                        new_vgdl_lines.append(VGDLLine((blocker, self.name), 'noInteraction'))
            else:
                prior_prob *= (1 - self.prior_prob_low)
        return new_vgdl_lines, np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys():
                self.blockers[new_name] = 0
                self.turn_around_objs[new_name] = 0
            if new_name not in self.spawned.keys() and new_name != self.name and AVATAR_NAME not in new_name and new_name != 'wall':
                self.spawned[new_name] = 0
                self.spawned_times[new_name] = dict()
                self.spawned_totals[new_name] = []

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]


class RandomNPC(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'RandomNPC'
        self.blockers = dict()
        self.directions_per_obj = dict()
        self.mvts_mag_per_obj = dict()

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                # maybe nothing pushed it, or it was blocked
                # can we find one blocker?
                # a blocker doesn't collide now, but is less than two step away
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.blockers.keys():
                        self.blockers[blocker] += 1
            else:
                if still_alive and mvt != (0, 0):  # no mvt
                    if obj_id + str(i_ep) not in self.mvts_mag_per_obj.keys():
                        self.mvts_mag_per_obj[obj_id + str(i_ep)] = []
                    self.mvts_mag_per_obj[obj_id + str(i_ep)].append(np.sum(np.abs(mvt)))

                sign_mvt = (np.sign(mvt[0]), np.sign(mvt[1]))
                if obj_id + str(i_ep) not in self.directions_per_obj.keys():
                    self.directions_per_obj[obj_id + str(i_ep)] = dict(UP=0, DOWN=0, RIGHT=0, LEFT=0)
                self.directions_per_obj[obj_id + str(i_ep)][inv_dir_dict[sign_mvt]] += 1


    def get_score(self):
        if self.count == 0:
            ratio_mvt = 0
        else:
            ratio_mvt = len(self.mvts_mag) / self.count * self.cooldown
        if len(self.mvts_mag) == 0:
            score = low_not_impossible
        elif ratio_mvt < 0.2:
            score = 0.1
        else:
            obj_ids = self.mvts_mag_per_obj.keys()
            keys = ['UP', 'DOWN', 'LEFT', 'RIGHT']
            prob_dir = np.array([np.array([self.directions_per_obj[obj_id][key] for key in keys]) / len(self.mvts_mag_per_obj[obj_id]) for obj_id in obj_ids])
            prob_dir = np.clip(prob_dir, 0.00001, 1)
            entropies = -np.sum(np.array([p * np.log(p) for p in prob_dir.T]), axis=0)
            entropy = np.mean(entropies)
            # entropy = -np.sum([p * np.log(p) for p in prob_dir])
            score = max(min(1, (entropy / 1.38)), low_not_impossible)
        return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        new_vgdl_lines = [VGDLLine(self.name, 'RandomNPC', {'speed': self.speed, 'singleton': self.singleton(ref_theory), 'cooldown': self.cooldown})]
        return new_vgdl_lines, np.log(1)
    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys():
                self.blockers[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]

class Chaser(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'Chaser'
        self.blockers = dict()
        self.targets = dict()
        self.target_counts = dict()

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                # maybe nothing pushed it, or it was blocked
                # can we find one blocker?
                # a blocker doesn't collide now, but is less than two step away
                colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                possible_blockers = set(close) - set(colliding_objects)
                for b in possible_blockers:
                    blocker = obj_episode[b]['name']
                    if blocker in self.blockers.keys():
                        self.blockers[blocker] += 1
            else:
                for target in self.targets.keys():
                    # find closest target object
                    # the actual target is the position of the object after movement, before effects (unobserved)
                    # if object doesn't move then it's their position at i_step - 1
                    # but if they move: it's their position at i_step, unless they died, which usually happens when the chaser reaches the target

                    closest_poss = find_closest_target_in_map(state_episode[i_step-1], state_episode[i_step], target, prev_pos, ord=2)
                    if closest_poss is not None:
                        # did the object get closer?
                        diffs = []
                        for target_pos in closest_poss:
                            if tuple(target_pos) == tuple(prev_pos):
                                # could be just pushing
                                continue
                            old_dist = np.linalg.norm(target_pos - np.array(prev_pos), ord=1)
                            new_dist = np.linalg.norm(target_pos - np.array(pos), ord=1)
                            if old_dist - new_dist > 0:
                                diffs.append('closer')
                            elif old_dist - new_dist < 0:
                                diffs.append('further')
                        diffs = list(set(diffs))
                        if len(diffs) == 1:
                            self.target_counts[target] += 1
                            if diffs[0] == 'closer':
                                self.targets[target][0] += 1
                            else:
                                self.targets[target][1] += 1

    def get_score(self):
        if self.name == 'shark':
            stop = 1
        if self.count == 0:
            ratio_mvt = 0
        else:
            ratio_mvt = len(self.mvts_mag) / self.count * self.cooldown
        if len(self.mvts_mag) == 0:
            score = low_not_impossible
        elif ratio_mvt < 0.2:
            score = 0.1
        else:
            chased = list(self.targets.keys())
            scores = []
            for k in chased:
                if self.target_counts[k] == 0:
                    scores.append(0)
                else:
                    ratio = max(self.targets[k]) / self.target_counts[k]
                    if ratio > 0.9 and self.target_counts[k] > 30:
                        scores.append(5)
                    else:
                        scores.append((max(self.targets[k]) / self.target_counts[k] - 0.5) * 2)
            score = np.max(scores)
        return max(score, low_not_impossible)

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        if self.name == 'shark':
            stop = 1
        # sample extra parameters and related interactions
        chased = list(self.targets.keys())
        scores = []
        for k in chased:
            if self.target_counts[k] == 0:
                scores.append(0)
            else:
                ratio = max(self.targets[k]) / self.target_counts[k]
                if ratio > 0.9:
                    scores.append(5)
                else:
                    scores.append((max(self.targets[k]) / self.target_counts[k] - 0.5) * 2)
        scores = np.array(scores)
        if sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            scores = self.add_bonus_for_robustness(scores)
            probs = scores / sum(scores)

        sampled_idx = None
        if ref_theory:
            if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'Chaser':
                stype = ref_theory.dict[self.name].params['stype']
                if stype in self.targets.keys():
                    sampled_idx = list(self.targets).index(stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(chased)), p=probs)

        chased_obj = chased[sampled_idx]
        fleeing = self.targets[chased_obj][0] < self.targets[chased_obj][1]
        new_vgdl_lines = [VGDLLine(self.name, 'Chaser', {'stype': chased_obj, 'fleeing': fleeing, 'singleton': self.singleton(ref_theory), 'speed': self.speed, 'cooldown': self.cooldown})]
        return new_vgdl_lines, np.log(1)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys():
                self.blockers[new_name] = 0
            if new_name not in self.targets.keys() and new_name != self.name and new_name != 'wall':
                self.targets[new_name] = [0, 0]
                self.target_counts[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]


class MovingAvatar(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'MovingAvatar'
        self.blockers = dict()
        self.game_names = set()
        self.expected = 0
        self.unexpected = 0
        self.action_id_mapping = {0: np.array([0, -1]),
                                  1: np.array([0, 1]),
                                  2: np.array([-1, 0]),
                                  3: np.array([1, 0]),
                                  4: np.array([0, 0]),
                                  5: np.array([0, 0]),
                                  -1: np.array([0, 0])}

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                if np.sum(np.abs(self.action_id_mapping[actions[i_step - 1]])) == 0:
                    self.expected += 1
                else:
                    # maybe nothing pushed it, or it was blocked
                    # can we find one blocker?
                    # a blocker doesn't collide now, but is less than two step away
                    colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                    close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                    possible_blockers = set(close) - set(colliding_objects)
                    for b in possible_blockers:
                        blocker = obj_episode[b]['name']
                        if blocker in self.blockers.keys():
                            self.blockers[blocker] += 1
                    if len(possible_blockers) > 0:
                        self.expected += 1
                    else:
                        self.unexpected -= 1
            else:
                expected_mvt = self.action_id_mapping[actions[i_step - 1]]
                mvt = (np.sign(mvt[0]), np.sign(mvt[1]))
                if tuple(expected_mvt) == mvt:
                    self.expected += 1
                else:
                    self.unexpected += 1

    def get_score(self):
        score = (self.expected - self.unexpected) / self.count
        return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        new_vgdl_lines = [VGDLLine(self.name, 'MovingAvatar', {'singleton': self.singleton(ref_theory), 'speed': self.speed})]
        return new_vgdl_lines, np.log(1)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys() and new_name != self.name and AVATAR_NAME not in new_name:
                self.blockers[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]

class FlakAvatar(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'FlakAvatar'
        self.expected = 0
        self.unexpected = 0
        self.blockers = dict()
        self.spawned = dict()
        self.action_id_mapping = {0: np.array([0, 0]),
                                  1: np.array([0, 0]),
                                  2: np.array([-1, 0]),
                                  3: np.array([1, 0]),
                                  4: np.array([0, 0]),
                                  5: np.array([0, 0]),
                                  -1: np.array([0, 0])}
        self.spawning_action = 5
        self.n_spawn_attempts = 0

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                if np.sum(np.abs(self.action_id_mapping[actions[i_step - 1]])) == 0:
                    self.expected += 1
                else:
                    # maybe nothing pushed it, or it was blocked
                    # can we find one blocker?
                    # a blocker doesn't collide now, but is less than two step away
                    colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                    close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                    possible_blockers = set(close) - set(colliding_objects)
                    for b in possible_blockers:
                        blocker = obj_episode[b]['name']
                        if blocker in self.blockers.keys():
                            self.blockers[blocker] += 1
                    if len(possible_blockers) > 0:
                        self.expected += 1
                    else:
                        self.unexpected -= 1
            else:
                expected_mvt = self.action_id_mapping[actions[i_step - 1]]
                mvt = (np.sign(mvt[0]), np.sign(mvt[1]))
                if tuple(expected_mvt) == mvt:
                    self.expected += 1
                else:
                    self.unexpected += 1

            if actions[i_step - 1] == self.spawning_action:
                self.n_spawn_attempts += 1
                spawned_candidates = []
                spawned_candidates += get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, np.array(prev_pos), 0.)
                spawned_found = False
                for spawned_candidate in spawned_candidates:
                    if obj_episode[spawned_candidate]['pos'][i_step - 1] is None:
                        spawned_name = obj_episode[spawned_candidate]['name']
                        self.spawned[spawned_name] += 1
                        spawned_found = True
                if spawned_found:
                    self.expected += 1
                else:
                    self.unexpected += 1

    def get_score(self):
        score = (self.expected - self.unexpected) / self.count
        if self.n_spawn_attempts > 0:
            spawn_factor = np.sum(list(self.spawned.values())) / self.n_spawn_attempts * 5
        else:
            spawn_factor = 0.5
        score = score * spawn_factor
        return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        spawned_candidates = [k for k in self.spawned.keys() if AVATAR_NAME not in k]
        spawned_scores = np.array([self.spawned[k] for k in spawned_candidates])
        if sum(spawned_scores) == 0:
            probs = np.ones(len(spawned_scores)) / len(spawned_scores)
        else:
            spawned_scores = self.add_bonus_for_robustness(spawned_scores)
            probs = spawned_scores / sum(spawned_scores)

        sampled_idx = None
        if ref_theory:
            if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'ShootAvatar':
                stype = ref_theory.dict[self.name].params['stype']
                if stype in self.spawned.keys():
                    sampled_idx = list(self.spawned).index(stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
                    # else:
                    #     data_prob = 1
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(spawned_scores)), p=probs)

        prior_prob = 1
        spawned = spawned_candidates[sampled_idx]
        new_vgdl_lines = [VGDLLine(self.name, 'FlakAvatar', {'stype': spawned, 'singleton': True, 'speed': self.speed})]
        return new_vgdl_lines, np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys() and new_name != self.name and AVATAR_NAME not in new_name:
                self.blockers[new_name] = 0
            if new_name not in self.spawned.keys() and new_name != self.name and AVATAR_NAME not in new_name and new_name != 'wall':
                self.spawned[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]



class ShootAvatar(ObjectType):
    def __init__(self, name, params):
        super().__init__(name, params)
        self.obj_type = 'ShootAvatar'
        self.expected = 0
        self.unexpected = 0
        self.blockers = dict()
        self.spawned = dict()
        self.action_id_mapping = {0: np.array([0, -1]),
                                  1: np.array([0, 1]),
                                  2: np.array([-1, 0]),
                                  3: np.array([1, 0]),
                                  4: np.array([0, 0]),
                                  5: np.array([0, 0]),
                                  -1: np.array([0, 0])}
        self.spawning_action = 5
        self.n_spawn_attempts = 0

    def get_str(self, param=None):
        return format_prompt([self.name], self.obj_type, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

        if still_alive:
            if mvt == (0, 0):  # no mvt
                if np.sum(np.abs(self.action_id_mapping[actions[i_step - 1]])) == 0:
                    self.expected += 1
                else:
                    # maybe nothing pushed it, or it was blocked
                    # can we find one blocker?
                    # a blocker doesn't collide now, but is less than two step away
                    colliding_objects = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
                    close = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=2)
                    possible_blockers = set(close) - set(colliding_objects)
                    for b in possible_blockers:
                        blocker = obj_episode[b]['name']
                        if blocker in self.blockers.keys():
                            self.blockers[blocker] += 1
                    if len(possible_blockers) > 0:
                        self.expected += 1
                    else:
                        self.unexpected -= 1
            else:
                expected_mvt = self.action_id_mapping[actions[i_step - 1]]
                mvt = (np.sign(mvt[0]), np.sign(mvt[1]))
                if tuple(expected_mvt) == mvt:
                    self.expected += 1
                else:
                    self.unexpected += 1

            if actions[i_step - 1] == self.spawning_action:
                orientation = None
                for i in range(i_step - 2, 0, -1):
                    if np.sum(np.abs(self.action_id_mapping[actions[i]])) != 0:
                        orientation = self.action_id_mapping[actions[i]]
                        break
                self.n_spawn_attempts += 1
                spawned_candidates = []
                if orientation is None:
                    for orientation in [np.array((0, 1)), np.array((0, -1)), np.array((1, 0)), np.array((-1, 0))]:
                        spawned_pos = prev_pos + orientation
                        spawned_candidates += get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, np.array(spawned_pos), 0.)
                else:
                    spawned_pos = prev_pos + orientation
                    spawned_candidates += get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, np.array(spawned_pos), 0.)
                spawned_found = False
                for spawned_candidate in spawned_candidates:
                    if obj_episode[spawned_candidate]['pos'][i_step - 1] is None:
                        spawned_name = obj_episode[spawned_candidate]['name']
                        self.spawned[spawned_name] += 1
                        spawned_found = True
                if spawned_found:
                    self.expected += 1
                else:
                    self.unexpected += 1

    def get_score(self):
        score = (self.expected - self.unexpected) / self.count
        if self.n_spawn_attempts > 0:
            spawn_factor = np.sum(list(self.spawned.values())) / self.n_spawn_attempts * 5
        else:
            spawn_factor = 0.5
        score = score * spawn_factor
        return score

    def sample(self, comm_engine, int_keys_taken, linguistic_data, ref_theory):
        # sample extra parameters and related interactions
        spawned_candidates = [k for k in self.spawned.keys() if AVATAR_NAME not in k]
        spawned_scores = np.array([self.spawned[k] for k in spawned_candidates])
        if sum(spawned_scores) == 0:
            probs = np.ones(len(spawned_scores)) / len(spawned_scores)
        else:
            spawned_scores = self.add_bonus_for_robustness(spawned_scores)
            probs = spawned_scores / sum(spawned_scores)

        sampled_idx = None
        if ref_theory:
            if self.name in ref_theory.obj_names and ref_theory.dict[self.name].type == 'ShootAvatar':
                stype = ref_theory.dict[self.name].params['stype']
                if stype in self.spawned.keys():
                    sampled_idx = list(self.spawned).index(stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
                    # else:
                    #     data_prob = 1
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(spawned_scores)), p=probs)

        prior_prob = 1
        spawned = spawned_candidates[sampled_idx]
        new_vgdl_lines = [VGDLLine(self.name, 'ShootAvatar', {'stype': spawned, 'singleton': self.singleton(ref_theory), 'speed': self.speed})]
        return new_vgdl_lines, np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for new_name in new_names:
            if new_name not in self.blockers.keys() and new_name != self.name and AVATAR_NAME not in new_name:
                self.blockers[new_name] = 0
            if new_name not in self.spawned.keys() and new_name != self.name and AVATAR_NAME not in new_name and new_name != 'wall':
                self.spawned[new_name] = 0

    def update_with_no_int(self, theory_pairs):
        keys_to_remove = []
        for k, v in self.blockers.items():
            if theory_pairs[(k, self.name)].is_no_int():
                keys_to_remove.append(k)
        for k in keys_to_remove:
            del self.blockers[k]


possible_object_types = [Immovable, Flicker, SpawnPoint, ResourcePack, Passive, Missile, Bomber, Chaser, RandomNPC, Portal]
possible_avatar_types = [MovingAvatar, ShootAvatar, FlakAvatar]