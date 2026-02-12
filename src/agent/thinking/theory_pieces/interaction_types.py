from abc import abstractmethod

import numpy as np

from src.utils import AVATAR_NAME, get_neighbors, format_prompt, compute_probability_and_confidence
from src.game.rules import VGDLLine

low_not_impossible = 0.1

class InteractionType:
    def __init__(self, names, params):
        self.names = names
        self.params = params
        self.prior_prob_low = params['agent']['thinker']['prior_prob_low']
        self.name1, self.name2 = self.names
        self.count = 0
        self.deaths = 0
        self.killer_is_plausible = 0
        self.only_killer_candidate_around = 0
        self.game_names = set()
        colors = params['true_game_info']['colors']
        self.colors = dict(zip(colors.keys(), [v.lower() for v in colors.values()]))
        self.linguistic_scores = dict()
        self.beta_softmax_lang_proposal = params['agent']['thinker']['beta_softmax_lang_proposal']


    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        self.count += 1
        name = obj['name']
        obj_id = obj['obj_id']
        mvt = obj['mov'][i_step]
        prev_pos, pos = obj['pos'][i_step - 1], obj['pos'][i_step]
        still_alive = prev_pos is not None and pos is not None
        just_died = prev_pos is not None and pos is None
        if just_died:
            self.deaths += 1
        if pos: pos = np.array(pos)
        if prev_pos: prev_pos = np.array(prev_pos)
        return name, obj_id, prev_pos, pos, mvt, still_alive, just_died

    def get_score(self, vgdl_lines=None):
        invalid = False
        if self.name1 == 'wall':
            invalid = True
        if vgdl_lines is not None:
            vgdl_line = vgdl_lines.dict.get(self.names, None)
            if vgdl_line is not None:
                # this slot is already filled
                invalid = True
            reverse_vgdl_line = vgdl_lines.dict.get((self.name2, self.name1), None)
            if reverse_vgdl_line is not None and reverse_vgdl_line.type in ['stepBack', 'bounceForward', 'noInteraction']:
                # the reverse slot is filled by an interaction that prevents any interaction in this slot
                invalid = True
            obj1 = vgdl_lines.dict.get(self.name1)
            obj2 = vgdl_lines.dict.get(self.name2)
            if not obj1.moves() and not obj2.moves():
                avatar_obj = vgdl_lines.dict.get(self.avatar_name)
                stype = avatar_obj.params.get('stype')
                if stype not in [obj1.name, obj2.name]:
                    invalid = True
        return invalid

    @abstractmethod
    def sample(self, comm_engine, linguistic_data, ref_theory):
        pass

    def get_linguistic_probs(self, comm_engine, linguistic_data, params):
        if len(linguistic_data) == 0 or len(params) == 0:
            return None, None, None
        linguistic_data = tuple(linguistic_data)
        loglikes = []
        uncertain_str = "I don't know / something else"
        candidates = [self.get_str(param=p) for p in params] + [uncertain_str]
        for p in params:
            prompt_to_eval = self.get_str(param=p)
            if linguistic_data not in self.linguistic_scores.keys():
                self.linguistic_scores[linguistic_data] = dict()
            if p not in self.linguistic_scores[linguistic_data].keys():
                self.linguistic_scores[linguistic_data][p] = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates, self.names)
            loglike = self.linguistic_scores[linguistic_data][p]
            loglikes.append(loglike)
        if uncertain_str not in self.linguistic_scores[linguistic_data].keys():
            unsure_loglike = comm_engine.get_linguistic_score(uncertain_str, linguistic_data, self.game_names, candidates, self.names)
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

    def update_with_new_names(self, new_names):
        self.game_names = self.game_names.union(new_names)
        self.avatar_names = [name for name in self.game_names if AVATAR_NAME in name]
        assert len(self.avatar_names) == 1
        self.avatar_name = self.avatar_names[0]

    def add_bonus_for_robustness(self, scores):
        # add small score such that no option has probability 0
        min_non_zero = np.min(scores[np.where(scores!=0)])
        return scores + low_not_impossible * min(5, min_non_zero)

class NoInteraction(InteractionType):
    def __init__(self, names, params):
        super().__init__(names, params)
        self.int_name = 'noInteraction'
        self.steps_in_contact = []  # track instances of consecutive steps where objects where in contact
        self.contacts = dict()
        self.kills = False
        self.guaranteed = False

    def get_str(self, param=None):
        return format_prompt(self.names, self.int_name, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
        if pos is not None:
            contact = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, pos, radius=1)
            for obj_contact_id in contact:
                if i_step == 0 or (obj_episode[obj_id]['pos'][i_step-1] is not None and obj_episode[obj_contact_id]['pos'][i_step-1] is not None):
                    obj_contact_name = obj_episode[obj_contact_id]['name']
                    if obj_contact_name == self.name2:
                        if i_ep not in self.contacts.keys():
                            self.contacts[i_ep] = dict()
                        if (obj_id, obj_contact_id) not in self.contacts[i_ep].keys():
                            self.contacts[i_ep][(obj_id, obj_contact_id)] = []
                        self.contacts[i_ep][(obj_id, obj_contact_id)].append(i_abs_step)

    def get_score(self, vgdl_lines=None):
        # if you find at least one extended contact (more than one step), this is evidence for no interactions
        if self.guaranteed or (len(self.contact_list) > 0 and np.max(self.contact_list) > 1):
            score = 5
        else:
            score = 1
        return score

    def sample(self, comm_engine, linguistic_data, ref_theory):
        # add no interactions between these objects
        prob = 1
        if self.guaranteed or (len(self.contact_list) > 0 and np.max(self.contact_list) > 0):
            return [VGDLLine(self.names, 'noInteraction'), VGDLLine((self.name2, self.name1), 'noInteraction')], np.log(prob)
        elif len(self.contact_list) > 0:
            # we're unsure here so let's not block the reverse option
            return [VGDLLine(self.names, 'noInteraction')], np.log(prob)
        else:
            return [], np.log(prob)

    @property
    def contact_list(self):
        contact_list = []
        for contacts_per_episode in self.contacts.values():
            for contacts_per_obj_pair in contacts_per_episode.values():
                if len(contacts_per_obj_pair) == 1:
                    contact_list.append(1)
                elif len(contacts_per_obj_pair) > 1:
                    diffs = np.diff(contacts_per_obj_pair)
                    contact_len = 2
                    for i_d, d in enumerate(diffs):
                        assert d != 0
                        if d == 1:
                            contact_len += 1
                        if d > 1 or i_d == len(diffs) - 1:
                            contact_list.append(contact_len)
                            contact_len = 0
        return contact_list

class KillSprite(InteractionType):
    def __init__(self, names, params):
        super().__init__(names, params)
        self.int_name = 'killSprite'
        self.kills = True

    def get_str(self, param=None):
        return format_prompt(self.names, self.int_name, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
        if just_died:
            possible_killers = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=3)
            still_around = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=2)

            candidate_killers = []
            for killer_id in set(possible_killers + still_around):
                # if killer still around or is dead, then it is valid
                if killer_id in still_around or obj_episode[killer_id]['pos'][i_step] is None:
                    candidate_killers.append(obj_episode[killer_id]['name'])

            if self.name2 in candidate_killers:
                self.killer_is_plausible += 1
                if len(set(candidate_killers)) == 1:
                    self.only_killer_candidate_around += 1

    def get_score(self, vgdl_lines=None):
        invalid = super().get_score(vgdl_lines)
        if invalid:
            return 0
        # measures the percentage of death explained by this interaction
        # if self.name1 == self.avatar_name:
        #     score = 1
        # elif self.deaths == 0:
        if self.deaths == 0:
            score = low_not_impossible
        else:
            ratio = self.killer_is_plausible / self.deaths
            if ratio > 0.25 and self.deaths > 1 or self.only_killer_candidate_around > 0:
                score = 2
            elif ratio > 0:
                score = 1
            else:
                score = low_not_impossible
        return score

    def sample(self, comm_engine, linguistic_data, ref_theory):
        prob = 1
        return [VGDLLine(self.names, 'killSprite')], np.log(prob)

class TransformTo(InteractionType):
    def __init__(self, names, params):
        super().__init__(names, params)
        self.int_name = 'transformTo'
        self.transformed = dict()
        self.kills = True

    def get_str(self, param=None):
        return format_prompt(self.names, self.int_name, param, self.colors)


    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
        if just_died:
            possible_killers = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=3)
            still_around = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=2)

            # for this to work we need a possible killer and a possible transformed id
            candidate_killers = []
            candidate_transformed = []
            for killer_id in set(possible_killers + still_around):
                # if obj_episode[killer_id]['name'] == self.name2:
                # if killer still around or is dead, then it is valid
                if killer_id in still_around or obj_episode[killer_id]['pos'][i_step] is None:
                    # if a new object just appeared
                    for transformed_id in still_around:
                        if transformed_id != killer_id and transformed_id != obj_id and obj_episode[transformed_id]['pos'][i_step - 1] is None:
                            # this object just got born
                            assert obj_episode[transformed_id]['pos'][i_step] is not None
                            if obj_episode[transformed_id]['name'] != self.name1:
                                candidate_killers.append(obj_episode[killer_id]['name'])
                                candidate_transformed.append(obj_episode[transformed_id]['name'])

            if self.name2 in candidate_killers:
                self.killer_is_plausible += 1
                if len(set(candidate_killers)) == 1:
                    self.only_killer_candidate_around += 1
                for killer, transformed in zip(candidate_killers, candidate_transformed):
                    if killer == self.name2:
                        self.transformed[transformed] += 1


    def get_score(self, vgdl_lines=None):
        invalid = super().get_score(vgdl_lines)
        if invalid:
            return 0

        # measures the percentage of death explained by this interaction
        if self.name1 == self.avatar_name or self.name2 == 'wall' or self.name1 == 'wall':
            score = 0
        elif self.deaths == 0:
            score = low_not_impossible
        else:
            ratio = self.killer_is_plausible / self.deaths
            if ratio > 0.9 and self.deaths > 1 or self.only_killer_candidate_around > 0:
                score = 5
            elif ratio > 0.25 and self.deaths > 1:
                score = 3
            elif ratio > 0:
                score = 2
            else:
                score = low_not_impossible

        return score


    def get_params_and_probs_from_interaction_data(self):
        params = list(self.transformed.keys())
        scores = np.array(list(self.transformed.values()))
        if np.sum(scores) == 0:
            probs = np.ones(len(scores)) / len(scores)
        else:
            scores = self.add_bonus_for_robustness(scores)
            if not self.params['exp_params']['use_data_proposal']:
                scores[np.where(scores > 0)] = 1
            probs = scores / sum(scores)
        return params, probs
    
    def sample(self, comm_engine, linguistic_data, ref_theory):
        params, probs = self.get_params_and_probs(comm_engine, linguistic_data)

        if len(params) == 0:
            # No transform targets available - return empty result
            return [], 0.0

        sampled_idx = None
        if ref_theory:
            # sample param from ref_theory preferentially
            if self.names in ref_theory.int_names and ref_theory.dict[self.names].type == 'transformTo':
                ref_stype = ref_theory.dict[self.names].params['stype']
                if ref_stype in params:
                    sampled_idx = params.index(ref_stype)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(probs)), p=probs)
        # non_small_probs = sum(probs>(1/(2*len(params))))
        # prior_prob = 1 / max(1, non_small_probs)
        prior_prob = 1 / len(params)
        stype = params[sampled_idx]
        return [VGDLLine(self.names, 'transformTo', {'stype': stype})], np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        for name in new_names:
            if name not in self.transformed.keys() and name != self.name1 and AVATAR_NAME not in name and name != 'wall':
                self.transformed[name] = 0


class RemoveResource(InteractionType):
    def __init__(self, names, params):
        super().__init__(names, params)
        self.int_name = 'removeResource'
        self.lost_resources = dict()
        self.resources_max = None
        self.kills = True

    def get_str(self, param=None):
        return format_prompt(self.names, self.int_name, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        if self.resources_max is None:
            self.resources_max = obj['resources_max'][0]
            for k in self.resources_max.keys():
                self.lost_resources[k] = 0
        name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
        if self.resources_max is None:
            self.resources_max = obj['resources_max'][0]
        if just_died:
            possible_killers = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=3)
            still_around = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=2)

            candidate_killers = []
            candidate_resource = []
            for killer_id in set(possible_killers + still_around):
                # if killer still around or is dead, then it is valid
                if killer_id in still_around:
                    # check that it lost a resource
                    prev_resources, new_resources = obj_episode[killer_id]['resources'][i_step - 1], obj_episode[killer_id]['resources'][i_step]
                    resource_keys = obj_episode[killer_id]['resources'][i_step].keys()
                    for k in resource_keys:
                        if new_resources[k] < prev_resources[k]:
                            candidate_killers.append(obj_episode[killer_id]['name'])
                            candidate_resource.append(k)
                elif obj_episode[killer_id]['pos'][i_step] is None:
                    candidate_killers.append(obj_episode[killer_id]['name'])
                    candidate_resource.append(None)

            if self.name2 in candidate_killers:
                self.killer_is_plausible += 1
                if len(set(candidate_killers)) == 1:
                    self.only_killer_candidate_around += 1
                for killer, resource in zip(candidate_killers, candidate_resource):
                    if killer == self.name2 and resource is not None:
                        self.lost_resources[resource] += 1


    def get_score(self, vgdl_lines=None):
        invalid = super().get_score(vgdl_lines)
        if invalid or (not vgdl_lines is None and not (vgdl_lines.dict[self.name1].moves() or vgdl_lines.dict[self.name2].moves())):
            return 0
        # measures the percentage of death explained by this interaction
        resource_keys = [] if self.resources_max is None else list(self.resources_max.keys())
        if len(resource_keys) == 0 or self.name1 in resource_keys or self.name1 == AVATAR_NAME or self.name2 in resource_keys or AVATAR_NAME not in self.name2:
            score = 0
        elif self.deaths == 0:
            score = low_not_impossible
        else:
            ratio = self.killer_is_plausible / self.deaths
            if ratio > 0.9 and self.deaths > 1 or self.only_killer_candidate_around > 0:
                score = 5
            elif ratio > 0.25 and self.deaths > 1:
                score = 3
            elif ratio > 0:
                score = 2
            else:
                score = low_not_impossible
        return score

    def sample(self, comm_engine, linguistic_data, ref_theory):
        # sample stype
        scores = np.array(list(self.lost_resources.values()))
        if sum(scores) == 0:
            probs = np.ones(len(self.lost_resources)) / len(self.lost_resources)
        else:
            scores = self.add_bonus_for_robustness(scores)
            probs = scores / sum(scores)

        sampled_idx = None
        if ref_theory:
            # sample param from ref_theory preferentially
            if self.names in ref_theory.int_names  and 'resource' in ref_theory.dict[self.names].params:
                ref_resource = ref_theory.dict[self.names].params['resource']
                if ref_resource in list(self.lost_resources.keys()):
                    sampled_idx = list(self.lost_resources.keys()).index(ref_resource)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
            sampled_idx = np.random.choice(range(len(scores)), p=probs)
    
        prior_prob = 1 / len(self.lost_resources)
        resource = list(self.lost_resources.keys())[sampled_idx]
        return [VGDLLine(self.names, 'removeResource', {'resource': resource})], np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        if self.resources_max is not None:
            for name in self.game_names:
                if name in self.resources_max.keys() and name not in self.names:
                    if name not in self.lost_resources.keys():
                        self.lost_resources[name] = 0


class KillIfHasLess(InteractionType):
    def __init__(self, names, params):
        super().__init__(names, params)
        self.int_name = 'killIfHasLess'
        self.resources = dict()
        self.resources_max = None
        self.kills = True

    def get_str(self, param=None):
        return format_prompt(self.names, self.int_name, param, self.colors)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        if self.resources_max is None:
            self.resources_max = obj['resources_max'][0]
            for k in self.resources_max.keys():
                self.resources[k] = 0
        if len(self.resources_max.keys()) > 0:
            name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

            if just_died:
                possible_killers = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=3)
                still_around = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=2)
                resources_below_limit = [r_key for r_key, r_count in obj['resources'][i_step - 1].items() if r_count <= 0]
                candidate_killers = []
                candidate_resource = []
                if len(resources_below_limit) > 0:
                    for killer_id in set(possible_killers + still_around):
                        # if killer still around or is dead, then it is valid
                        if killer_id in still_around or obj_episode[killer_id]['pos'][i_step] is None:
                            for resource in resources_below_limit:
                                self.resources[resource] += 1
                                candidate_killers.append(obj_episode[killer_id]['name'])
                                candidate_resource.append(resource)

                if self.name2 in candidate_killers:
                    self.killer_is_plausible += 1
                    if len(set(candidate_killers)) == 1:
                        self.only_killer_candidate_around += 1
                    for killer, resource in zip(candidate_killers, candidate_resource):
                        if killer == self.name2 and resource is not None:
                            self.resources[resource] += 1


    def get_score(self, vgdl_lines=None):
        invalid = super().get_score(vgdl_lines)
        if invalid or (not vgdl_lines is None and not (vgdl_lines.dict[self.name1].moves() or vgdl_lines.dict[self.name2].moves())):
            return 0
        # measures the percentage of death explained by this interaction
        if self.resources_max is None or len(self.resources_max.keys()) == 0 or self.name1 != self.avatar_name:
            # this cannot happen in games without resources
            score = 0
        # elif self.name1 == self.avatar_name:
        #     score = 1
        elif self.deaths == 0:
            score = low_not_impossible
        else:
            ratio = self.killer_is_plausible / self.deaths
            if ratio > 0.9 and self.deaths > 1 or self.only_killer_candidate_around > 0:
                score = 5
            elif ratio > 0.25 and self.deaths > 1:
                score = 3
            elif ratio > 0:
                score = 2
            else:
                score = low_not_impossible
        return score

    def sample(self, comm_engine, linguistic_data, ref_theory):
        # sample stype
        scores = np.array(list(self.resources.values()))
        if sum(scores) == 0:
            probs = np.ones(len(self.resources)) / len(self.resources)
        else:
            scores = self.add_bonus_for_robustness(scores)
            probs = scores / sum(scores)

        sampled_idx = None
        if ref_theory:
            # sample param from ref_theory preferentially
            if self.names in ref_theory.int_names and 'resource' in ref_theory.dict[self.names].params:
                ref_resource = ref_theory.dict[self.names].params['resource']
                if ref_resource in list(self.resources.keys()):
                    sampled_idx = list(self.resources.keys()).index(ref_resource)
                    if probs[sampled_idx] == 0:
                        sampled_idx = None
        if sampled_idx is None:
           sampled_idx = np.random.choice(range(len(scores)), p=probs)
        prior_prob = 1 / len(self.resources)
        resource = list(self.resources.keys())[sampled_idx]
        return [VGDLLine(self.names, 'killIfHasLess', {'resource': resource, 'limit': 0})], np.log(prior_prob)

    def update_with_new_names(self, new_names):
        super().update_with_new_names(new_names)
        if self.resources_max is not None:
            for name in self.game_names:
                if name in self.resources_max.keys():
                    if name not in self.resources.keys():
                        self.resources[name] = 0


# class KillIfFromAbove(InteractionType):
#     def __init__(self, names, params):
#         super().__init__(names, params)
#         self.int_name = 'killIfFromAbove'
#         self.kills = True
#
#     def get_str(self, param=None):
#         if add_intro:
#             vgdl_prompt = "Interactions:\n"
#         else:
#             vgdl_prompt = ""
#         vgdl_prompt += "> obj_{self.colors[self.name1]}, obj_{self.colors[self.name2]}:"
#         vgdl_answer = f" killIfFromAbove"
#         return vgdl_prompt, vgdl_answer
#
#     def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
#         name, obj_id, prev_pos, pos, mvt, still_alive, just_died = super().learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
#         if just_died:
#             possible_killers = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "prev", obj_id, prev_pos, radius=3)
#             still_around = get_neighbors(nn_episode, i_ep, i_abs_step, i_step, "current", obj_id, prev_pos, radius=2)
#
#             candidate_killers = []
#             for killer_id in set(possible_killers + still_around):
#                 # if killer still around or is dead, then it is valid
#                 if killer_id in still_around or obj_episode[killer_id]['pos'][i_step] is None:
#                     if obj_episode[killer_id]['pos'][i_step - 1] is None:
#                         continue
#                     from_above = obj_episode[killer_id]['pos'][i_step - 1][1] < prev_pos[1]
#                     if from_above:
#                         candidate_killers.append(obj_episode[killer_id]['name'])
#
#             if self.name2 in candidate_killers:
#                 self.killer_is_plausible += 1
#                 if len(set(candidate_killers)) == 1:
#                     self.only_killer_candidate_around += 1
#
#
#     def get_score(self, vgdl_lines=None):
#         invalid = super().get_score(vgdl_lines)
#         if invalid or (not vgdl_lines is None and not (vgdl_lines.dict[self.name1].moves() or vgdl_lines.dict[self.name2].moves())):
#             return 0
#         # measures the percentage of death explained by this interaction
#         if self.deaths == 0:
#             score = low_not_impossible
#         else:
#             ratio = self.killer_is_plausible / self.deaths
#             if ratio > 0.25 and self.deaths > 1 or self.only_killer_candidate_around > 0:
#                 score = 2
#             elif ratio > 0:
#                 score = 1
#             else:
#                 score = low_not_impossible
#         return score
#
#     def sample(self, comm_engine, linguistic_data, ref_theory):
#         prob = 1
#         return [VGDLLine(self.names, 'killIfFromAbove')], np.log(prob)

# other interactions are generated with object types: bounceForward (with Passive objects), reverseDirection (with Missile objects),
# addResource (with ResourcePack objects), stepBack (with moving object types), teleportToExit (with Portal objects).
possible_interaction_types = [NoInteraction, KillSprite, TransformTo, RemoveResource, KillIfHasLess]#, KillIfFromAbove] # KillIfOtherHasMore
possible_reflexive_interaction_types = [NoInteraction]  # interactions that can apply on a pair of the same object

