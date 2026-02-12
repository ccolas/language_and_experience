import numpy as np

from src.agent.thinking.theory_pieces.object_types import *
from src.agent.thinking.theory_pieces.interaction_types import *
from src.agent.thinking.theory_pieces.terminations import KillAll, Timeout
from src.game.rules import VGDLLines
from src.utils import AVATAR_NAME, hashdict, normalize_log_probs, compute_probability_and_confidence
from mpi4py import MPI

class ObjectConstraint:
    """
    For each object name look at possible types
    """

    def __init__(self, obj_name, params, is_resource=False):
        self.obj_name = obj_name
        self.params = params
        self.is_wall = self.obj_name == 'wall'
        self.is_avatar = AVATAR_NAME in self.obj_name
        self.is_resource = is_resource
        self.game_names = set()

        # Initialize possible object types based on (observed) supertype
        if self.is_wall:
            self.possible_values = [Immovable(obj_name, self.params)]
        elif self.is_avatar:
            self.possible_values = [obj_type(obj_name, self.params) for obj_type in possible_avatar_types]
        elif self.is_resource:
            self.possible_values = [ResourcePack(obj_name, self.params)]
        else:
            self.possible_values = [obj_type(obj_name, self.params) for obj_type in possible_object_types if obj_type.__name__ != 'ResourcePack']  # an object could
            # be any object
            # type

        self.prior_probs = np.ones(len(self.possible_values)) / len(self.possible_values)
        self.possible_value_names = [value.__class__.__name__ for value in self.possible_values]
        self.steps = 0
        # self.linguistic_scores = dict()
        self.beta_softmax_lang_proposal = params['agent']['thinker']['beta_softmax_lang_proposal']


    def update_with_new_names(self, new_names):
        self.game_names = self.game_names.union(set(new_names))
        for value in self.possible_values:
            value.update_with_new_names(new_names)

    def update_with_no_int(self, theory_pairs):
        # remove possibility for interactions between pairs of objects we've seen co-occur
        for value in self.possible_values:
            value.update_with_no_int(theory_pairs)

    # Ask each of possible options to 
    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        self.steps += 1

        # use current transition to update prior on each possible object type values
        for value in self.possible_values:
            value.learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

    def get_params_linguistic_probs(self,  comm_engine, linguistic_data):
        all_probs, all_confidence, all_params = [], [], []
        for value in self.possible_values:
            params, _ = value.get_params_and_probs_from_interaction_data()
            probs, confidence, params = value.get_linguistic_probs(comm_engine, linguistic_data, params)
            all_probs.append(probs)
            all_confidence.append(confidence)
            all_params.append(params)
        return all_probs, all_confidence, all_params

    def get_linguistic_probs(self, comm_engine, interaction_probs, linguistic_data):
        # return None
        if len(linguistic_data) == 0:
            assert False
            return None, None
        linguistic_data = tuple(linguistic_data)
        # if linguistic_data not in self.linguistic_scores.keys():
        #     self.linguistic_scores[linguistic_data] = dict()

        one_possible_value = sum(interaction_probs > 0) == 1
        if len(self.possible_values) == 1:
            sampling_probabilities = np.array([1])
            confidence = 1
        elif one_possible_value:
            sampling_probabilities = interaction_probs.copy()
            confidence = 1
        else:
            loglikes = []
            valid_values = [value for value, int_prob in zip(self.possible_values, interaction_probs) if int_prob > 0]
            uncertain_str = "I don't know"
            candidates = [value.get_str() for value in valid_values] + [uncertain_str]
            for interaction_prob, value, value_name in zip(interaction_probs, self.possible_values, self.possible_value_names):
                if interaction_prob == 0:
                    # something makes this value impossible
                    loglikes.append(-np.inf)
                else:
                    prompt_to_eval = value.get_str()
                    # if value_name not in self.linguistic_scores[linguistic_data].keys():
                    score = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates, self.obj_name)
                        # self.linguistic_scores[linguistic_data][value_name] = score
                    # else:
                    #     score = self.linguistic_scores[linguistic_data][value_name]
                    loglikes.append(score)
            # if uncertain_str not in self.linguistic_scores[linguistic_data].keys():
            unsure_loglike = comm_engine.get_linguistic_score(uncertain_str, linguistic_data, self.game_names, candidates, self.obj_name)
                # self.linguistic_scores[linguistic_data][uncertain_str] = unsure_loglike
            # else:
            #     unsure_loglike = self.linguistic_scores[linguistic_data][uncertain_str]

            # loglike_max = np.max(all_loglikes)  # For numerical stability
            # exp_scores = np.exp((all_loglikes - loglike_max) / params.temperature)
            # probs = exp_scores / np.sum(exp_scores)

            sampling_probabilities, confidence = compute_probability_and_confidence(loglikes, unsure_loglike)

        return sampling_probabilities, confidence


    def get_interaction_probs(self):
        # compute score from bottom-up process and normalize them to get probabilities
        scores = np.array([value.get_score() for value in self.possible_values])
        if not self.params['exp_params']['use_data_proposal']:
            scores[np.where(scores>0)] = 1
        probs = scores / np.sum(scores)
        return probs

    # 
    def get_probs(self, comm_engine, linguistic_data=[]):
        interaction_probs = self.get_interaction_probs()
        confidence = None
        if self.params['exp_params']['use_language_proposal'] and len(linguistic_data) > 0:
            linguistic_probs, confidence = self.get_linguistic_probs(comm_engine, interaction_probs, linguistic_data)
            if confidence < 0.5:
                probs = interaction_probs
            else:
                probs = (interaction_probs + 2 * confidence * linguistic_probs) / (1 + 2 * confidence)
        else:
            probs = interaction_probs
            linguistic_probs = np.zeros(len(probs))
        return interaction_probs, linguistic_probs, probs, confidence

    # Test whether 'parent' theory piece is still viable, if not resample 
    def sample(self, comm_engine, int_keys_taken=[], ref_theory=None, linguistic_data=[]):
        # sample obj types with probabilities coming from bottom-up process
        assert len(self.possible_values) > 0

        # first get probability of general object types
        int_probs, ling_probs, probs, _ = self.get_probs(comm_engine, linguistic_data)

        value_idx = None

        # try to sample the value from the ref_theory if it has p>0
        if ref_theory and self.obj_name in ref_theory.obj_names and ref_theory.dict[self.obj_name].type in self.possible_value_names:
            value_idx = self.possible_value_names.index(ref_theory.dict[self.obj_name].type)
            if probs[value_idx] == 0:
                value_idx = None

        # if not possible, sample a new value given the probabilities
        if value_idx is None:
            value_idx = np.random.choice(range(len(self.possible_values)), p=probs)

        prior_logprob = np.log(self.prior_probs[value_idx])
        value = self.possible_values[value_idx]

        # Sample parameters and additional relations with other objects using bottom-up process (type/class-specific sampling)
        vgdl_lines, sub_prior_logprob = value.sample(comm_engine, int_keys_taken, linguistic_data, ref_theory)
        prior_logprob += sub_prior_logprob

        # create the sampled vgdl lines
        vgdl_lines = VGDLLines(from_list=vgdl_lines)
        return vgdl_lines, prior_logprob


class ObjectPairConstraint:
    def __init__(self, obj_names, params):
        self.obj_names = obj_names
        self.params = params
        self.prob_no_int = params['agent']['thinker']['prior_prob_no_int']
        self.prior_prob_low = params['agent']['thinker']['prior_prob_low']

        self.obj_name1, self.obj_name2 = obj_names
        self.possible_values = dict()
        self.game_names = set()
        self.possible_values = []

        if self.obj_name1 == self.obj_name2:
            if AVATAR_NAME in self.obj_name1:
                self.possible_values = [NoInteraction(obj_names, self.params)]
                self.prior_probs = [1]
                self.possible_values[0].guaranteed = True
            else:
                self.possible_values = [int_type(obj_names, self.params) for int_type in possible_reflexive_interaction_types]
                if len(self.possible_values) == 1:
                    assert self.possible_values[0].__class__.__name__ == 'NoInteraction'
                    self.prior_probs = np.array([1.])
                else:
                    self.prior_probs = np.ones(len(self.possible_values)) * (1 - self.prob_no_int)/ (len(self.possible_values) - 1)
                    no_int_idx = [value.__class__.__name__ == 'NoInteraction' for value in self.possible_values].index(True)
                    self.prior_probs[no_int_idx] = self.prob_no_int
                    self.possible_values[0].guaranteed = True
        else:
            self.possible_values = [int_type(obj_names, self.params) for int_type in possible_interaction_types]
            if len(self.possible_values) == 1:
                assert False
                assert self.possible_values[0].__class__.__name__ == 'NoInteraction'
                self.prior_probs = np.array([1.])
            else:
                self.prior_probs = np.ones(len(self.possible_values)) * (1 - self.prob_no_int) / (len(self.possible_values) - 1)
                no_int_idx = [value.__class__.__name__ == 'NoInteraction' for value in self.possible_values].index(True)
                self.prior_probs[no_int_idx] = self.prob_no_int
        np.testing.assert_approx_equal(np.sum(self.prior_probs), 1)
        self.possible_value_names = [value.int_name for value in self.possible_values]
        self.steps = 0  # steps learned from
        # self.linguistic_scores = dict()
        self.beta_softmax_lang_proposal = params['agent']['thinker']['beta_softmax_lang_proposal']


    def update_with_new_names(self, new_names):
        self.game_names = self.game_names.union(new_names)
        for value in self.possible_values:
            value.update_with_new_names(new_names)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode):
        self.steps += 1
        # use current transition to update prior on each possible object type values
        for value in self.possible_values:
            value.learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)

    def get_params_linguistic_probs(self,  comm_engine, linguistic_data):
        all_probs, all_confidence, all_params = [], [], []
        for value in self.possible_values:
            params, _ = value.get_params_and_probs_from_interaction_data()
            probs, confidence, params = value.get_linguistic_probs(comm_engine, linguistic_data, params)
            all_probs.append(probs)
            all_confidence.append(confidence)
            all_params.append(params)
        return all_probs, all_confidence, all_params

    def get_linguistic_probs(self, comm_engine, interaction_probs, linguistic_data):
        # return None
        if len(linguistic_data) == 0:
            assert False
            return None, None
        linguistic_data = tuple(linguistic_data)
        # if linguistic_data not in self.linguistic_scores.keys():
        #     self.linguistic_scores[linguistic_data] = dict()
        loglikes = []
        valid_values = [value for value, int_prob in zip(self.possible_values, interaction_probs) if int_prob > 0]
        uncertain_str = "I don't know / something else"
        candidates = [value.get_str() for value in valid_values] + [uncertain_str]
        one_possible_value = sum(interaction_probs > 0) == 1
        if len(self.possible_values) == 1:
            sampling_probabilities = np.array([1])
            confidence = 1
        elif one_possible_value:
            sampling_probabilities = interaction_probs.copy()
            confidence = 1
        else:
            for interaction_prob, value, value_name in zip(interaction_probs, self.possible_values, self.possible_value_names):
                if interaction_prob == 0:
                    # something makes this value impossible
                    loglikes.append(-np.inf)
                else:
                    prompt_to_eval = value.get_str()
                    # if value_name not in self.linguistic_scores[linguistic_data].keys():
                    score = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates, self.obj_names)
                        # self.linguistic_scores[linguistic_data][value_name] = score
                    # else:
                    #     score = self.linguistic_scores[linguistic_data][value_name]
                    loglikes.append(score)
            # if uncertain_str not in self.linguistic_scores[linguistic_data].keys():
            unsure_loglike = comm_engine.get_linguistic_score(uncertain_str, linguistic_data, self.game_names, candidates, self.obj_names)
                # self.linguistic_scores[linguistic_data][uncertain_str] = unsure_loglike
            # else:
            #     unsure_loglike = self.linguistic_scores[linguistic_data][uncertain_str]

            sampling_probabilities, confidence = compute_probability_and_confidence(loglikes, unsure_loglike)


        return sampling_probabilities, confidence

    def get_interaction_probs(self, vgdl_lines, except_no_int=False):
        # compute score from bottom-up process and normalize them to get probabilities
        scores = np.array([value.get_score(vgdl_lines) for value in self.possible_values])
        if except_no_int:
            scores[self.no_int_index] = 0
        # compute sampling probabilities
        if scores[self.possible_value_names.index('noInteraction')] == 5:
            # if we saw object not interact for several timesteps, they don't interact
            assert self.possible_value_names.index('noInteraction') == 0
            scores[1:] = [0] * (len(scores) - 1)
        assert sum(scores) > 0
        if not self.params['exp_params']['use_data_proposal']:
            if scores[self.no_int_index] > 0:
                scores[np.where(scores > 0)] = 1
                scores[self.no_int_index] = 3
            else:
                scores[np.where(scores > 0)] = 1
        probs = scores / np.sum(scores)
        return probs

    def get_probs(self, comm_engine, theory_piece_obj1, linguistic_data=[], vgdl_lines=None, except_no_int=False):
        # compute prob of a step back interaction
        if vgdl_lines is None:
            if AVATAR_NAME in self.obj_name1:
                selected_value = 'MovingAvatar'
            else:
                selected_value = 'RandomNPC'
        else:
            selected_value = vgdl_lines.dict[self.obj_names[0]].type
        if selected_value not in theory_piece_obj1.possible_value_names:
            # the object doesn't move
            stepback_prob = 0
        else:
            selected_idx = theory_piece_obj1.possible_value_names.index(selected_value)
            stepback_prob = theory_piece_obj1.possible_values[selected_idx].get_prob_stepback(self.obj_names[1], vgdl_lines)

        confidence = None
        interaction_probs = self.get_interaction_probs(vgdl_lines, except_no_int)
        if self.params['exp_params']['use_language_proposal'] and len(linguistic_data) > 0:
            linguistic_probs, confidence = self.get_linguistic_probs(comm_engine, interaction_probs, linguistic_data)
            if confidence < 0.5:
                probs = interaction_probs
            else:
                probs = (interaction_probs + 2 * confidence * linguistic_probs) / (1 + 2* confidence)
        else:
            probs = interaction_probs
            linguistic_probs = np.zeros(len(interaction_probs))
        return stepback_prob, interaction_probs, linguistic_probs, probs, confidence


    def evidence_for_kill(self):
        return any([value.get_score() > low_not_impossible for value in self.possible_values if value.kills])

    def sample_stepback(self, stepback_prob, ref_theory, ignore_parent):
        # sample a step back interaction or not
        if ref_theory and not ignore_parent:# and np.random.rand() < 0.75: #TODO: removed this for tests but?
            vgdl_line = ref_theory.dict.get(self.obj_names)
            block = vgdl_line and vgdl_line.type == 'stepBack' and stepback_prob > 0
        else:
            # make sure it's compatible with existing vgdl lines
            block = np.random.rand() < stepback_prob
        prior_prob = self.prior_prob_low if block else (1 - self.prior_prob_low)
        if self.obj_name2 == 'wall':
            prior_prob = 0.9 if block else 0.1
        if block:
            # if ref_theory: print(f'block in parent: {self.obj_names}', {stepback_prob})
            if self.obj_name1 == self.obj_name2:
                new_vgdl_lines = [VGDLLine(self.obj_names, 'stepBack')]
            else:
                new_vgdl_lines = [VGDLLine(self.obj_names, 'stepBack'),
                                  VGDLLine((self.obj_name2, self.obj_name1), 'noInteraction')]
        else:
            # if ref_theory: print(f'no block in parent: {self.obj_names}, {stepback_prob}')
            new_vgdl_lines = []
        return new_vgdl_lines, np.log(prior_prob)

    def sample(self, comm_engine, termination_pieces, theory_piece_obj1, linguistic_data, vgdl_lines, except_no_int=False, ref_theory=None, ignore_parent=False):
        assert len(self.possible_values) > 0
        stepback_prob, interaction_probs, linguistic_probs, probs, _ = self.get_probs(comm_engine, theory_piece_obj1, linguistic_data, vgdl_lines, except_no_int)

        # should it be a stepback?
        new_vgdl_lines, prior_logprob = self.sample_stepback(stepback_prob, ref_theory, ignore_parent)

        if len(new_vgdl_lines) == 0:
            # sample another interaction?
            value_idx = None
            if ref_theory and not ignore_parent:
                value_idx = self.no_int_index
                line = ref_theory.dict.get(self.obj_names)
                if line and line.type in self.possible_value_names:
                    value_idx = self.possible_value_names.index(line.type)
                # if the choice made in ref theory has 0 probability, sample something else
                if probs[value_idx] == 0:
                    value_idx = None

            if value_idx is None:
                value_idx = np.random.choice(range(len(self.possible_values)), p=probs)
            # if self.obj_name2 == 'wall':
            #     # if it's not a step back, it's a killing int
            #     if 'killSprite' in self.possible_value_names:
            #         value_idx = self.possible_value_names.index('killSprite')
            prior_logprob += np.log(self.prior_probs[value_idx])
            value = self.possible_values[value_idx]

            # sample parameters and additional relations with other objects using bottom-up process
            new_vgdl_lines, sub_prior_logprob = value.sample(comm_engine, linguistic_data, ref_theory)
            prior_logprob += sub_prior_logprob
        if len(new_vgdl_lines) == 1 and new_vgdl_lines[0].type == 'noInteraction':
            new_vgdl_lines = []


        # sample rewards?
        for new_vgdl_line in new_vgdl_lines:
            if new_vgdl_line.kills_a():
                # resample score change only if the interaction stayed the same
                resample_reward = False
                if ref_theory and new_vgdl_line.name == self.obj_names:
                    parent_line = ref_theory.dict.get(self.obj_names)
                    if parent_line and parent_line.type == new_vgdl_line.type and parent_line.params.get('stype') == new_vgdl_line.params.get('stype'):
                        resample_reward = True  # interaction did not change
                r = self.sample_reward(termination_pieces, ref_theory, vgdl_lines, resample_reward)
                if r != 0:
                    # print('\n\nSCORE CHANGE', str(vgdl_line), self.obj_names)
                    new_vgdl_line.params['scoreChange'] = r
                    prior_logprob += np.log(0.1)
                else:
                    prior_logprob += np.log(0.9)

        new_vgdl_lines = VGDLLines(from_list=new_vgdl_lines)
        return new_vgdl_lines, prior_logprob

    def sample_reward(self, termination_pieces, ref_theory, vgdl_lines, resample_reward):
        r = None
        probs = termination_pieces.prob_reward(self.obj_names, vgdl_lines)  # positive, negative, zero
        if ref_theory and not resample_reward:
            interaction = ref_theory.dict.get(self.obj_names)
            if interaction:
                if 'scoreChange' in interaction.params.keys():
                    r = interaction.params['scoreChange']
                    if r == 1 and probs[0] > 0:
                        r = 1
                    elif r == -1 and probs[1] > 0:
                        r = -1
                    else:
                        assert False
                else:
                    assert probs[2] > 0
                    r = 0
            else:
                # if we use to have no interaction there, let's keep it to 0 scoreChange, and this will be updated later
                r = 0

        if r is None:
            idx = np.random.choice(range(3), p=probs)
            r = [1, -1, 0][idx]
        return r


    def is_no_int_valid(self):
        assert False
        params, probs = self.get_probs()
        if probs[self.no_int_index] > 0:
            # return it's prior and data logprobs
            return True, np.log(self.prior_probs[self.no_int_index])
        else:
            return False, None
    def is_no_int(self):
        return self.possible_values[self.no_int_index].get_score() == 5

    @property
    def no_int_index(self):
        return self.possible_value_names.index('noInteraction')

class Terminations:
    def __init__(self, params):
        self.params = params
        self.prior_prob_termination = params['agent']['thinker']['prior_prob_low']
        self.possible_values = []
        self.possible_value_names = []
        self.steps = 0  # steps learned from
        # self.linguistic_scores = dict()
        self.game_names = set()
        self.prior_timeout = 0.5
        self.timeout_score_win = np.nan
        self.timeout_score_lose = np.nan
        self.language_win_proposals = None
        self.language_lose_proposals = None
        self.beta_softmax_lang_proposal = params['agent']['thinker']['beta_softmax_lang_proposal']


    def update_with_new_names(self, new_names):
        self.game_names = self.game_names.union(new_names)
        for name in new_names:
            self.possible_values.append(KillAll(name, self.params))
            self.possible_value_names.append(name)
        self.timeout = Timeout(self.params)
        self.prior_probs = np.ones(len(self.possible_values)) / len(self.possible_values)

    def learn_from(self, i_ep, i_step, i_abs_step, actions, obj, traj_episode, obj_episode, nn_episode):
        self.steps += 1
        # use current transition to update prior on each possible object type values
        for value in self.possible_values:
            if value.name == obj['name']:
                value.learn_from(i_ep, i_step, i_abs_step, actions, obj, traj_episode, obj_episode, nn_episode)

    def get_linguistic_probs(self, comm_engine, interaction_win_probs, interaction_lose_probs, linguistic_data):
        # return None, None
        if len(linguistic_data) == 0:
            return None, None, None, None
        linguistic_data = tuple(linguistic_data)
        # if linguistic_data not in self.linguistic_scores.keys():
        #     self.linguistic_scores[linguistic_data] = dict()

        win_scores, lose_scores = [], []
        win_confidences, lose_confidences = [], []
        for win_prob, lose_prob, value, value_name in zip(interaction_win_probs, interaction_lose_probs, self.possible_values, self.possible_value_names):
            if win_prob == 0:
                # something makes this value impossible
                win_scores.append(0)
                win_confidences.append(1)
            else:
                prompts_to_eval = [f"To win you need to reach/touch {value.colors[value_name]} objects",
                                   f"To win you need to kill all {value.colors[value_name]} objects",
                                   f"To win you don't need to kill all {value.colors[value_name]} objects",
                                   f"To win I don't know if you need to kill or reach {value.colors[value_name]} objects"]
                key = value_name + '_win'
                # if key not in self.linguistic_scores[linguistic_data].keys():
                loglikes = []
                for prompt_to_eval in prompts_to_eval:
                    loglike = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates=prompts_to_eval, key=f"win_{value_name}")
                    loglikes.append(loglike)
                    # self.linguistic_scores[linguistic_data][key] = loglikes
                # else:
                #     loglikes = self.linguistic_scores[linguistic_data][key]

                unsure_loglike = loglikes[-1]
                loglikes = [max(loglikes[:2])] + [loglikes[2]]

                sampling_probs, confidence = compute_probability_and_confidence(loglikes, unsure_loglike)
                win_confidences.append(confidence)
                win_scores.append(sampling_probs[0])
                # win_scores.append(sampling_probs)

            if lose_prob == 0:
                # something makes this value impossible
                lose_scores.append(0)
                lose_confidences.append(1)
            else:
                prompts_to_eval = [f"You lose if all {value.colors[value_name]} objects die or disappear",
                                   f"You don't lose if all {value.colors[value_name]} objects die or disappear",
                                   f"I don't know if you would lose if all {value.colors[value_name]} objects die or disappear"]
                key = value_name + '_lose'
                # if key not in self.linguistic_scores[linguistic_data].keys():
                loglikes = []
                for prompt_to_eval in prompts_to_eval:
                    loglike = comm_engine.get_linguistic_score(prompt_to_eval, linguistic_data, self.game_names, candidates=prompts_to_eval, key=f"lose_{value_name}")
                    loglikes.append(loglike)
                    # self.linguistic_scores[linguistic_data][key] = loglikes
                # else:
                #     loglikes = self.linguistic_scores[linguistic_data][key]
                unsure_loglike = loglikes[-1]
                loglikes = loglikes[:2]
                sampling_probs, confidence = compute_probability_and_confidence(loglikes, unsure_loglike)
                lose_confidences.append(confidence)
                lose_scores.append(sampling_probs[0])
                # lose_scores.append(sampling_probs)

        return np.array(win_scores), np.array(lose_scores), np.array(win_confidences), np.array(lose_confidences)

    def get_interaction_probs(self, vgdl_lines):
        # compute score from bottom-up process and normalize them to get probabilities
        # if vgdl_lines is not None:
            # can_be_killed_by_avatar = self.can_be_killed_by_avatar(vgdl_lines)
            # can_be_killed = self.can_be_killed(vgdl_lines)
        win_scores = []
        lose_scores = []
        for i_val, value in enumerate(self.possible_values):
            win_score, lose_score = value.get_scores()
            if not self.params['exp_params']['use_data_proposal']:
                if win_score > 0:
                    win_score = 0.33
                if lose_score > 0:
                    lose_score = 0.33
            win_scores.append(win_score)
            lose_scores.append(lose_score)
        return np.array(win_scores), np.array(lose_scores)

    def get_probs(self, comm_engine, linguistic_data=[], vgdl_lines=None):
        interaction_win_probs, interaction_lose_probs = self.get_interaction_probs(vgdl_lines)
        win_confidences = [None] * len(interaction_win_probs)
        lose_confidences = [None] * len(interaction_lose_probs)
        if self.params['exp_params']['use_language_proposal'] and len(linguistic_data) > 0:
            linguistic_win_probs, linguistic_lose_probs, win_confidences, lose_confidences = self.get_linguistic_probs(comm_engine, interaction_win_probs,
                                                                                                                       interaction_lose_probs, linguistic_data)

            win_probs = np.zeros(len(interaction_win_probs))
            lose_probs = np.zeros(len(interaction_lose_probs))
            for i in range(len(interaction_win_probs)):
                if win_confidences[i] > 0.5:
                    win_probs[i] = (interaction_win_probs[i] + 2 * win_confidences[i] * linguistic_win_probs[i]) / (1 + 2 * win_confidences[i])
                else:
                    win_probs[i] = interaction_win_probs[i]
                if lose_confidences[i] > 0.5:
                    lose_probs[i] = (interaction_lose_probs[i] + 2 * lose_confidences[i] * linguistic_lose_probs[i]) / (1 + 2 * lose_confidences[i])
                else:
                    lose_probs[i] = interaction_lose_probs[i]
        else:
            win_probs = interaction_win_probs
            lose_probs = interaction_lose_probs
            linguistic_win_probs = np.zeros(len(win_probs))
            linguistic_lose_probs = np.zeros(len(lose_probs))

        return interaction_win_probs, interaction_lose_probs, linguistic_win_probs, linguistic_lose_probs, win_probs, lose_probs, win_confidences, lose_confidences


    def sample(self, comm_engine, vgdl_lines=None, ref_theory=None, parent_theory=None, linguistic_data=[]):
        terminations = dict(win=[], lose=[])
        prior_prob = 1  # tracks sampling probability

        int_win_probs, int_lose_probs, ling_win_probs, ling_lose_probs, win_probs, lose_probs, _, _ = self.get_probs(comm_engine, linguistic_data, vgdl_lines)
        for value, win_score, lose_score in zip(self.possible_values, win_probs, lose_probs):
            if AVATAR_NAME in value.name:
                terminations['lose'].append(value.name)
            else:
                if ref_theory:
                    if (value.name in ref_theory['win'] and win_score > 0) or win_score == 1:
                        add_to_win = True
                    else:
                        add_to_win = False
                    if (value.name in ref_theory['lose'] and lose_score > 0) or lose_score == 1:
                        add_to_lose = True
                    else:
                        add_to_lose = False
                elif parent_theory:
                    if np.random.rand() < 0.5:
                        if (value.name in parent_theory['win'] and win_score > 0) or win_score == 1:
                            add_to_win = True
                        else:
                            add_to_win = False
                    else:
                        add_to_win = np.random.rand() < win_score
                    if np.random.rand() < 0.5:
                        if (value.name in parent_theory['lose'] and lose_score > 0) or lose_score == 1 and not add_to_win:
                            add_to_lose = True
                        else:
                            add_to_lose = False
                    else:
                        add_to_lose = not add_to_win and np.random.rand() < lose_score
                else:
                    add_to_win = np.random.rand() < win_score
                    add_to_lose = not add_to_win and np.random.rand() < lose_score
                if add_to_win:
                    terminations['win'].append(value.name)
                    prior_prob *= self.prior_prob_termination
                else:
                    prior_prob *= 1 - self.prior_prob_termination
                if add_to_lose:
                    terminations['lose'].append(value.name)
                    prior_prob *= self.prior_prob_termination
                else:
                    prior_prob *= 1 - self.prior_prob_termination
        terminations['win'] = tuple(terminations['win'])
        terminations['lose'] = tuple(terminations['lose'])
        assert prior_prob > 0
        return hashdict(terminations), np.log(prior_prob)

    def prob_reward(self, names, vgdl_lines=None):
        for value in self.possible_values:
            if value.name == names[0]:
                return value.prob_reward(names, vgdl_lines)
        assert False

    def can_be_killed_by_avatar(self, vgdl_lines):
        # things the agent can control are:
        # - things it can kill
        # - things the things it can push or spawn can kill
        avatar_name = [name for name in vgdl_lines.obj_names if AVATAR_NAME in name]
        assert len(avatar_name) == 1
        avatar_name = avatar_name[0]

        can_be_killed = set()
        controlled_objs  = {avatar_name}
        if 'stype' in vgdl_lines.dict[avatar_name].params.keys():
            controlled_objs.add(vgdl_lines.dict[avatar_name].params['stype'])
        for int in vgdl_lines.int.dict.values():
            if int.name[1] == 'avatar' and int.type == 'bounceForward':
                controlled_objs.add(int.name[0])
        for int in vgdl_lines.int.dict.values():
            if int.kills_a() and int.name[1] in controlled_objs:
                can_be_killed.add(int.name[0])
        for obj in vgdl_lines.obj.dict.values():
            if obj.type == 'SpawnPoint':
                can_be_killed.add(obj.name)
        return can_be_killed

    def can_be_killed(self, vgdl_lines):

        can_be_killed = set()
        for int in vgdl_lines.int.dict.values():
            if int.kills_a():
                can_be_killed.add(int.name[0])
        return can_be_killed