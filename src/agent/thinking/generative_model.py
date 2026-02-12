# import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pickle

from src.agent.thinking.theory_pieces.basic_constraints import  ObjectConstraint, ObjectPairConstraint, Terminations
from src.game.rules import VGDLLines, Rules, VGDLLine, Particles
from src.game.game import Game
from src.utils import hashdict


class GenerativeModel:
    """
    Proposes VGDL games
    """

    def __init__(self, params, datastore, comm_engine):
        self.params = params
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.llm_params = params['agent']['thinker']['llm_params']
        self.verbose = params['verbose'] if self.rank == 0 else False
        self.formats = params['true_game_info']['formats']
        self.datastore = datastore
        self.comm_engine = comm_engine
        self.last_n_steps = 0  # keep track of number of steps saved to know when to udpate the generative model

        self.init_names = None
        self.theory_pieces = dict(objects=dict(), object_pairs=dict(), terminations=Terminations(self.params))  # tracks theory pieces for all object names and
        # object pairs

    # Getting interaction data 
    def update(self):
        updated = False
        latest_interaction_data = None
        if self.rank == 0:
            if self.datastore.n_steps > self.last_n_steps:
                latest_interaction_data = self.datastore.get_latest_sensorimotor_data()
                self.last_n_steps = self.datastore.n_steps
        if self.size > 1:
            latest_interaction_data = self.comm.bcast(latest_interaction_data, root=0)
        if latest_interaction_data:
            # update generative model (spread across workers)
            self.update_from_sensory_data(latest_interaction_data)
            updated = True
        return updated

    # Creates and updates theory pieces
    def update_from_sensory_data(self, latest_data):
        # this ingests new sensory data and updates probabilities over theory elements
        episodes, stepss, new_trajs, new_objs, new_nn, new_events = latest_data
        # create theory pieces
        self.create_theory_pieces(stepss, new_trajs)
        # update theory pieces points
        self.update_theory_pieces(episodes, stepss, new_trajs, new_objs, new_nn)

    def update_from_linguistic_data(self, linguistic_data):
        # this should update linguistic probabilities over theory elements
        updated_with_language = self.comm_engine.update_proposal_distribution(self.theory_pieces, linguistic_data)
        return updated_with_language

    # samples a theory from the prior using probabilities from linguistic data and sensory data
    def sample(self, n=1, particles=None, ref_theory=None):
        assert ref_theory is None
        # self.print_prior()

        # this should output a Game object (?)

        # - presampled_theory is a dict containing theory pieces sampled by the mutation function
        # - if a ref_theory is provided, we must stay as close as possible from it
        # self.print_prior()
        # assert False
        # stop = 1
        theories = []
        if particles is None:
            particles = [None for _ in range(n)]
        for _, base_theory in enumerate(particles):
            if base_theory is None:
                base_vgdl_lines = None
                base_terminations = None
            else:
                base_vgdl_lines = base_theory.vgdl_lines
                base_terminations = base_theory.terminations

            # sample object types
            vgdl_lines, prior_logprob = self.sample_objects(ref_theory=base_vgdl_lines)

            # sample interactions
            vgdl_lines, sub_prior_logprob = self.sample_interactions(vgdl_lines, ref_theory=base_vgdl_lines)
            prior_logprob += sub_prior_logprob

            # sample terminations
            terminations, sub_prior_logprob = self.sample_terminations(vgdl_lines, ref_theory=base_terminations)
            prior_logprob += sub_prior_logprob

            # build new theory
            vgdl_obj = hashdict(dict(obj=vgdl_lines.obj, names=tuple(self.names), int=vgdl_lines.int, terminations=terminations))
            theory = Rules(self.params, vgdl_obj)
            theory.logprior = prior_logprob
            theories.append(theory)

        return Particles(theories)

    # sample object types
    def sample_objects(self, vgdl_lines=None, ref_theory=None):
        # vgdl lines may contain presampled vgdl lines (e.g. from earlier sampling steps)
        # ref_theory: is provided, try to stay as close as possible to that theory while staying compatible with vgdl_lines
        if vgdl_lines is None:
            vgdl_lines = VGDLLines()
        prior_logprob = 0.
        pieces = list(self.theory_pieces['objects'].values())
        np.random.shuffle(pieces)  # shuffle to increase diversity and prevent order issues
        for obj_line in pieces:
            obj_line.update_with_no_int(theory_pairs=self.theory_pieces['object_pairs'])
            if obj_line.obj_name not in vgdl_lines.obj_names:
                int_keys_taken = vgdl_lines.int_names
                new_vgdl_lines, sub_prior_logprob = obj_line.sample(self.comm_engine, int_keys_taken=int_keys_taken, ref_theory=ref_theory, linguistic_data=self.linguistic_data)
                vgdl_lines += new_vgdl_lines
                prior_logprob += sub_prior_logprob

        # if sampled pushing interaction between objects that don't move, remove that
        for obj_name, obj_line in vgdl_lines.obj.dict.items():
            if obj_line.type == 'Passive':
                pushers = []
                for int_names, int_line in vgdl_lines.int.dict.items():
                    if int_names[0] == obj_name and int_line.type == 'bounceForward':
                        pushers.append(int_names[1])
                non_static_pusher = False
                for pusher in pushers:
                    if not vgdl_lines.obj.dict[pusher].moves():
                        del vgdl_lines.int.dict[(obj_name, pusher)]
                    else:
                        non_static_pusher = True
                if not non_static_pusher:
                    vgdl_lines.obj.dict[obj_name].type = 'Immovable'
        return vgdl_lines, prior_logprob

    # sample interaction types
    def sample_interactions(self, vgdl_lines, ref_theory=None, key_to_skip=None):
        # vgdl lines may contain presampled vgdl lines (eg from earlier sampling steps)
        # ref_theory: is provided, try to stay as close as possible to that theory while staying compatible with vgdl_lines
        pieces = list(self.theory_pieces['object_pairs'].values())
        np.random.shuffle(pieces)  # shuffle to increase diversity and prevent order issues
        prior_logprob = 0.
        for obj_pair_line in pieces:
            if obj_pair_line.obj_names not in vgdl_lines.int_names and obj_pair_line.obj_names != key_to_skip:
                new_vgdl_lines, sub_prior_logprob = obj_pair_line.sample(self.comm_engine, theory_piece_obj1=self.theory_pieces['objects'][obj_pair_line.obj_names[0]], vgdl_lines=vgdl_lines,
                                                                         termination_pieces=self.theory_pieces['terminations'], ref_theory=ref_theory,
                                                                         linguistic_data=self.linguistic_data)
                prior_logprob += sub_prior_logprob
                vgdl_lines += new_vgdl_lines
            elif obj_pair_line.obj_names != key_to_skip:
                # here the slot was filled by object sampling, so we need to add the prior corresponding to sampling an empty interaction
                prior_logprob += np.log(obj_pair_line.prior_probs[obj_pair_line.no_int_index])

        return vgdl_lines, prior_logprob

    # sample terminations
    def sample_terminations(self, vgdl_lines, ref_theory=None, parent_theory=None):
        # vgdl lines may contain presampled vgdl lines (eg from earlier sampling steps)
        # ref_theory: is provided, try to stay as close as possible to that theory while staying compatible with vgdl_lines
        terminations, prior_logprob = self.theory_pieces['terminations'].sample(self.comm_engine, vgdl_lines=vgdl_lines, ref_theory=ref_theory,
                                                                                parent_theory=parent_theory, linguistic_data=self.linguistic_data)
        return terminations, prior_logprob


    def mutate_obj(self, p_dict, parent_vgdl_lines, parent_terminations):
        mutation_str = ""
        logp_mutated_given_current = np.log(p_dict['obj'])
        candidate_names = parent_vgdl_lines.obj_names
        if 'wall' in candidate_names: candidate_names.remove('wall')

        i_trial = 0
        while i_trial < 50:
            i_trial += 1
            # sample a slot
            name = np.random.choice(candidate_names)
            # resample a value
            mutated_vgdl_lines, prior_logprob = self.theory_pieces['objects'][name].sample(self.comm_engine, linguistic_data=self.linguistic_data)
            # print(prior_logprob, prior_logprob - 2.0794415416798357)
            # is it new?
            mutation_found = mutated_vgdl_lines not in parent_vgdl_lines
            if mutation_found:
                break

        if mutation_found:
            mutated_str = ', '.join([str(value) for value in mutated_vgdl_lines.dict.values()])
            mutation_str = f"mutation: obj {name} from {parent_vgdl_lines.dict[name].type} to {mutated_str}"

            # sample object types
            mutated_vgdl_lines, sub_prior_logprob = self.sample_objects(vgdl_lines=mutated_vgdl_lines, ref_theory=parent_vgdl_lines)
            prior_logprob += sub_prior_logprob
            # sample interactions
            mutated_vgdl_lines, sub_prior_logprob = self.sample_interactions(mutated_vgdl_lines, ref_theory=parent_vgdl_lines)
            prior_logprob += sub_prior_logprob
            # sample terminations
            mutated_terminations, sub_prior_logprob = self.sample_terminations(mutated_vgdl_lines, ref_theory=parent_terminations)
            prior_logprob += sub_prior_logprob
        else:
            mutated_terminations = None

        return mutation_found, mutation_str, mutated_vgdl_lines, mutated_terminations, prior_logprob


    def mutate_int(self, p_dict, parent_vgdl_lines, parent_terminations):

        # resample objects similar to parents
        mutated_vgdl_lines, prior_logprob = self.sample_objects(ref_theory=parent_vgdl_lines)

        candidate_int_names = []
        for int_names in sorted(self.theory_pieces['object_pairs'].keys()):
            # make sure they can have an interaction
            existing_value = mutated_vgdl_lines.dict.get(int_names)
            # skip slots that are already enforced by the object sampling
            if existing_value and existing_value.type in ['noInteraction', 'reverseDirection', 'bounceForward', 'addResource', 'teleportToExit']:
                continue
             # skip slots that must be noInteraction
            no_int_index = self.theory_pieces['object_pairs'][int_names].no_int_index
            probs = self.theory_pieces['object_pairs'][int_names].get_probs(self.comm_engine, self.theory_pieces['objects'][int_names[0]], vgdl_lines=mutated_vgdl_lines,
                                                                            linguistic_data=self.linguistic_data)
            if probs[no_int_index] == 1:
                continue
            # remaining values have not been enforced yet and have multiple possible values
            candidate_int_names.append(int_names)

        # pick an int_names
        i_trial = 0
        while i_trial < 50:
            i_trial += 1
            sampled_int_names =  candidate_int_names[np.random.choice(range(len(candidate_int_names)))]
            mutated_lines, sub_prior_logprob = self.theory_pieces['object_pairs'][sampled_int_names].sample(self.comm_engine, theory_piece_obj1=self.theory_pieces['objects'][sampled_int_names[0]],
                                                                                                            vgdl_lines=mutated_vgdl_lines,
                                                                                                            ref_theory=parent_vgdl_lines,
                                                                                                            ignore_parent=True,
                                                                                                            termination_pieces=self.theory_pieces['terminations'],
                                                                                                            linguistic_data=self.linguistic_data )
            # is it new?
            if len(mutated_lines.dict) == 0 or len(mutated_lines.dict) == 1 and mutated_lines.dict[sampled_int_names].type == 'noInteraction':  # sampled no int
                key_to_skip = sampled_int_names
                parent_line = parent_vgdl_lines.dict.get(sampled_int_names)
                if parent_line is None or parent_line.type == 'noInteraction':
                    mutation_found = False
                    mutation_str = ""
                else:
                    mutation_found = True
                    mutation_str = f"mutation: change int {sampled_int_names} from {parent_line} to noInt"
            else:
                key_to_skip = sampled_int_names
                mutation_found = mutated_lines not in parent_vgdl_lines
                parent_line = parent_vgdl_lines.dict.get(sampled_int_names)
                new_line = mutated_lines.dict.get(sampled_int_names)
                mutation_str = f"mutation: change int {sampled_int_names} from {parent_line} to {new_line}"
            if mutation_found:
                break

        if mutation_found:
            mutated_vgdl_lines += mutated_lines
            prior_logprob += sub_prior_logprob
            # sample remaining interactions
            mutated_vgdl_lines, sub_prior_logprob = self.sample_interactions(mutated_vgdl_lines, ref_theory=parent_vgdl_lines, key_to_skip=key_to_skip)
            prior_logprob += sub_prior_logprob
            # sample terminations
            mutated_terminations, sub_prior_logprob = self.sample_terminations(mutated_vgdl_lines, ref_theory=parent_terminations)
            prior_logprob += sub_prior_logprob
            # if 'scoreChange' in mutation_str:
            #     print(mutation_str)
            #     removed, added = [], []
            #     for key, value in parent_vgdl_lines.dict.items():
            #         if value not in mutated_vgdl_lines:
            #             removed.append(value)
            #     for key, value in mutated_vgdl_lines.dict.items():
            #         if value not in parent_vgdl_lines:
            #             added.append(value)
            #     print('added', added)
            #     print('removed', removed)
            #     stop = 1
        else:
            mutated_terminations = None

        return mutation_found, mutation_str, mutated_vgdl_lines, mutated_terminations, prior_logprob
    

    def mutate_terminations(self, p_dict, parent_vgdl_lines, parent_terminations):
        mutation_str = ""
        # sample object types (same as parent)
        mutated_vgdl_lines, prior_logprob = self.sample_objects(ref_theory=parent_vgdl_lines)
        # sample interactions
        mutated_vgdl_lines, sub_prior_logprob = self.sample_interactions(mutated_vgdl_lines, ref_theory=parent_vgdl_lines)
        prior_logprob += sub_prior_logprob

        # resample terminations
        i_trial = 0
        while i_trial < 10:
            i_trial += 1
            mutated_terminations, sub_prior_logprob = self.sample_terminations(vgdl_lines=mutated_vgdl_lines, parent_theory=parent_terminations)
            mutation_found = mutated_terminations != parent_terminations
            if mutation_found:
                break

        if mutation_found:
            mutation_str = f"replace terminations (from {parent_terminations} to {mutated_terminations})"
            prior_logprob += sub_prior_logprob
        return mutation_found, mutation_str, mutated_vgdl_lines, mutated_terminations, prior_logprob


    def mutate(self, theories):
        # to mutate a theory:
        # - enforce a piece of the theory
        # - then resample other pieces to be as close as possible from the parent theory
        # - the piece to sample is either and obj (p=0.3) or an interaction (p=0.7)
        # - if it's an interaction it's either adding one (p=0.5) or removing one (p=0.5)
        mutated_theories = []
        for theory in theories:
            og_parent_vgdl_lines = theory.vgdl_lines
            og_parent_terminations = theory.terminations
            parent_vgdl_lines = og_parent_vgdl_lines
            parent_terminations = og_parent_terminations
            mutation_strs = []

            # perform several mutations
            mutation_count = 0
            n_mutations = 1#np.random.choice([1, 2, 3])
            while mutation_count < n_mutations:
                p_dict = dict(obj=0.2, int=0.6, termination=0.2)
                mutation_type = np.random.choice(list(p_dict.keys()), p=list(p_dict.values()))
                if mutation_type == 'obj':
                    out  = self.mutate_obj(p_dict, parent_vgdl_lines, parent_terminations)
                elif mutation_type == 'int':
                    out = self.mutate_int(p_dict, parent_vgdl_lines, parent_terminations)
                elif mutation_type == 'termination':
                    out = self.mutate_terminations(p_dict, parent_vgdl_lines, parent_terminations)
                else: raise ValueError
                mutation_found, mutation_str, mutated_vgdl_lines, mutated_terminations, prior_logprob = out
                if mutated_vgdl_lines == og_parent_vgdl_lines and mutated_terminations == og_parent_terminations:
                    mutation_found = False
                if mutation_found:
                    # print(mutation_str)
                    mutation_count += 1
                    mutation_strs.append(mutation_str)
                    # if not tmp_vgdl_lines == parent_vgdl_lines:
                    #     stop = 1
                    # assert tmp_terminations == parent_terminations
                    if not (mutated_vgdl_lines != parent_vgdl_lines or mutated_terminations != parent_terminations):
                        stop = 1
                    parent_vgdl_lines = mutated_vgdl_lines
                    parent_terminations = mutated_terminations
                else:
                    stop = 1
            # build the mutate theory
            vgdl_obj = hashdict(dict(obj=mutated_vgdl_lines.obj, names=tuple(self.names), int=mutated_vgdl_lines.int, terminations=mutated_terminations))
            mutated_theory = Rules(self.params, vgdl_obj)
            removed, added = [], []
            for key, value in theory.vgdl_lines.dict.items():
                if value not in mutated_theory.vgdl_lines:
                    removed.append(value)
            for key, value in mutated_theory.vgdl_lines.dict.items():
                if value not in theory.vgdl_lines:
                    added.append(value)
            # print(removed)
            # print(added)
            stop = 1

            mutated_theory.mutation_strs = mutation_strs
            mutated_theory.logprior = prior_logprob
            # mutated_theory = Game(self.params, rules=rules)
            mutated_theories.append(mutated_theory)
        return Particles(mutated_theories)

    def create_theory_pieces(self, stepss, traj_episodes):
        # creates new theory pieces if needed, for both objects and object pairs
        old_names = self.names

        keys_to_udpate = []
        # Sequence of game states
        for steps, traj_episode in zip(stepss, traj_episodes):
            assert len(steps) == len(traj_episode['state'])
            for i_step, step in zip(steps, traj_episode['state']):
                names_in_this_step = set()
                for i in range(len(step)):
                    for j in range(len(step[0])):
                        objs = step[i][j]  # objects in the cell
                        for obj in objs:
                            name = obj['name']
                            if name not in self.names:
                                self.theory_pieces['objects'][name] = ObjectConstraint(name, self.params)  # add new object slot
                                keys_to_udpate.append(name)
                            names_in_this_step.add(name)
                            for r_name in obj['resources'].keys():
                                if r_name not in self.theory_pieces['objects'].keys():
                                    self.theory_pieces['objects'][r_name] = ObjectConstraint(r_name, self.params, is_resource=True)  # add new object slot
                                    keys_to_udpate.append(name)
                                elif not self.theory_pieces['objects'][r_name].is_resource:
                                    self.theory_pieces['objects'][r_name].is_resource = True

                # track names of objects present in first frame
                if i_step == 0:
                    if self.init_names is None:
                        self.init_names = names_in_this_step
                    else:
                        self.init_names = self.init_names.intersection(names_in_this_step)

        # #TODO remove this -- adding all names from the start to get prior from language messages
        # true_names = list(self.params['true_game_info']['colors'].keys())
        # for name in true_names:
        #     if name not in self.names:
        #         self.theory_pieces['objects'][name] = ObjectConstraint(name, self.params)  # add new object slot
        #         keys_to_udpate.append(name)


        # For each new object, create a new pair constraint with existing objects
        for name1 in self.names:
            for name2 in self.names:
                if (name1, name2) not in self.theory_pieces['object_pairs'].keys():
                    self.theory_pieces['object_pairs'][(name1, name2)] = ObjectPairConstraint((name1, name2), self.params)
                    keys_to_udpate.append((name1, name2))

        new_names = self.names - old_names
        for obj_key, obj_constraint in self.theory_pieces['objects'].items():
            if obj_key in keys_to_udpate:
                obj_constraint.update_with_new_names(self.names)
            else:
                obj_constraint.update_with_new_names(new_names)
        for obj_pair_key, obj_pair_constraint in self.theory_pieces['object_pairs'].items():
            if obj_pair_key in keys_to_udpate:
                obj_pair_constraint.update_with_new_names(self.names)
            else:
                obj_pair_constraint.update_with_new_names(new_names)

        # Integrate into potential termination conditions        
        if len(new_names) > 0:
            self.theory_pieces['terminations'].update_with_new_names(new_names)

            # Redo prompt to language model with new object names
            if self.rank == 0 and self.comm_engine is not None:
                self.comm_engine.reset_cache_proposal()  # new names change the proposal prompt, so proposals should be recomputed


    def update_theory_pieces(self, episodes, stepss, traj_episodes, obj_episodes, nn_episodes):
        # scatter work across workers
        names = sorted(self.theory_pieces['objects'].keys())
        n_names_per_rank = len(names) // self.size
        rest_names = len(names) % self.size
        names_here = names[self.rank * n_names_per_rank: (self.rank + 1) * n_names_per_rank]
        if self.rank < rest_names:
            names_here.append(names[self.size * n_names_per_rank + self.rank])

        # int_names
        int_names = sorted(self.theory_pieces['object_pairs'].keys())
        n_int_names_per_rank = len(int_names) // self.size
        rest_int_names = len(int_names) % self.size
        int_names_here = int_names[self.rank * n_int_names_per_rank: (self.rank + 1) * n_int_names_per_rank]
        if self.rank < rest_int_names:
            int_names_here.append(int_names[self.size * n_int_names_per_rank + self.rank])

        # Looking over data and different object constraints 
        for i_ep, steps, traj_episode, obj_episode, nn_episode in zip(episodes, stepss, traj_episodes, obj_episodes, nn_episodes):
            state_episode = traj_episode['state']
            actions = traj_episode['action']
            if self.verbose: print(f"          > integrating episode #{i_ep} from step {steps[0]} ({len(steps)} steps)")
            for i_step, i_abs_step in zip(range(len(steps)), steps):
                for obj in obj_episode.values():
                    if i_step > 0:
                        if obj['pos'][i_step - 1] or obj['pos'][i_step]:  # only learn from valid transitions (obj was there or is there now)
                            name = obj['name']
                            # learn about object type
                            if name in names_here:
                                self.theory_pieces['objects'][name].learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
                            # learn about all pairs of objects
                            for names in int_names_here:
                                if names[0] == name:
                                    self.theory_pieces['object_pairs'][names].learn_from(i_ep, i_step, i_abs_step, actions, obj, state_episode, obj_episode, nn_episode)
                    # if i_step == 98 and obj['name'] == 'quiet':
                    #     stop = 1
                    if obj['pos'][i_step] or obj['pos'][i_step - 1]:
                        if self.rank == 0:
                            self.theory_pieces['terminations'].learn_from(i_ep, i_step, i_abs_step, actions, obj, traj_episode, obj_episode, nn_episode)

        # gather results from workers
        if self.size > 1:
            if self.rank > 0:
                # delete keys not handled here to facilitate the merge
                keys_to_delete = []
                for k, v in self.theory_pieces['objects'].items():
                    if k not in names_here:
                        keys_to_delete.append(k)
                for k in keys_to_delete:
                    del self.theory_pieces['objects'][k]
                keys_to_delete = []
                for k, v in self.theory_pieces['object_pairs'].items():
                    if k not in int_names_here:
                        keys_to_delete.append(k)
                for k in keys_to_delete:
                    del self.theory_pieces['object_pairs'][k]
            objects = self.comm.gather(self.theory_pieces['objects'], root=0)
            object_pairs = self.comm.gather(self.theory_pieces['object_pairs'], root=0)
            if self.rank == 0:
                self.theory_pieces['objects'] = self.merge_dicts(objects)
                self.theory_pieces['object_pairs'] = self.merge_dicts(object_pairs)
            self.theory_pieces = self.comm.bcast(self.theory_pieces, root=0)


    def merge_dicts(self, dicts):
        output_dict = dicts[0]
        for this_dict in dicts[1:]:
            output_dict.update(this_dict)
        return output_dict


    def print_prior(self, key=None, to_print=True):
        s = ""
        data = dict()
        if key is None:

            s += 'Objects:\n'
            for obj_name, obj in self.theory_pieces['objects'].items():
                data[obj_name] = dict()
                if key is None or key == obj_name:
                    s += f'  > {obj_name}\n'
                    int_probs, ling_probs, probs, confidence = obj.get_probs(self.comm_engine, linguistic_data=self.linguistic_data)
                    all_param_probs, all_param_confidence, all_param_params = obj.get_params_linguistic_probs(self.comm_engine, self.linguistic_data)
                    s += f"    > lang confidence: {confidence}\n"
                    for v, int_p, ling_p, p, param_probs, param_confidence, param_params in zip(obj.possible_values, int_probs, ling_probs, probs, all_param_probs,
                                                                                                all_param_confidence, all_param_params):
                        data[obj_name][v.obj_type] = (int_p, ling_p, p, param_probs, param_confidence, param_params, confidence)
                        if p > 0:
                            s += f"    > {v.obj_type}: global:{p:.3f}, int:{int_p:.3f}, ling:{ling_p:.3f}\n"
                            if param_params is not None:
                                s += f"      > param confidence: {param_confidence}\n"
                                for prob, param in zip(param_probs, param_params):
                                    s += f"      > param: {param}: {prob:.3f}"
                                s += '\n'


            s += 'Interactions:\n'
            for int_name, interaction in self.theory_pieces['object_pairs'].items():
                data[int_name] = dict()
                if key is None or key == int_name:
                    s += f'  > {int_name}\n'
                    stepback_prob, int_probs, ling_probs, probs, confidence = interaction.get_probs(self.comm_engine, theory_piece_obj1=self.theory_pieces['objects'][int_name[0]],
                                                                                                      linguistic_data=self.linguistic_data)
                    all_param_probs, all_param_confidence, all_param_params = interaction.get_params_linguistic_probs(self.comm_engine, self.linguistic_data)
                    s += f"    > lang confidence: {confidence}\n"
                    for v, int_p, ling_p, p, param_probs, param_confidence, param_params in zip(interaction.possible_values, int_probs, ling_probs, probs, all_param_probs,
                                                                                                all_param_confidence, all_param_params):
                        data[int_name][v.int_name] = (int_p, ling_p, p, param_probs, param_confidence, param_params, confidence)
                        if p > 0:
                            s += f"    > {v.int_name}: global:{p:.3f}, int:{int_p:.3f}, ling:{ling_p:.3f}\n"
                            if param_params is not None:
                                s += f"      > param confidence: {param_confidence}\n"
                                for prob, param in zip(param_probs, param_params):
                                    s += f"      > param: {param}: {prob:.3f}"
                                s += '\n'
                    s += f"    > prob stepback: {stepback_prob}\n"
                    prob_reward = self.theory_pieces['terminations'].prob_reward(int_name)
                    s += f'    > prob rewards (1, -1, 0): :{prob_reward}\n'

            s += 'Terminations:\n'
            terminations = self.theory_pieces['terminations']
            data['win'] = dict()
            data['lose'] = dict()
            int_win_probs, int_lose_probs, ling_win_probs, ling_lose_probs, win_probs, lose_probs, win_confidences, lose_confidenecs = terminations.get_probs(self.comm_engine,
                                                                                                                                                              linguistic_data=self.linguistic_data)
            for i_val, value in enumerate(terminations.possible_values):
                data['win'][value.name] = (int_win_probs[i_val], ling_win_probs[i_val], win_probs[i_val], win_confidences[i_val])
                data['lose'][value.name] = (int_lose_probs[i_val], ling_lose_probs[i_val], lose_probs[i_val], lose_confidenecs[i_val])
                s += (f'    > {value.name}\n'
                      f'       - win: global:{win_probs[i_val]:.3f}, int:{int_win_probs[i_val]:.3f}, ling:{ling_win_probs[i_val]:.3f}, conf: {win_confidences[i_val]}\n'
                      f'       - lose: global:{lose_probs[i_val]:.3f}, int:{int_lose_probs[i_val]:.3f}, ling:{ling_lose_probs[i_val]:.3f}, conf: {lose_confidenecs[i_val]}\n')
                      # f'       - win: global:{win_probs[i_val]:.3f}, int:{int_win_probs[i_val]:.3f}, ling:{ling_win_probs[i_val]:.3f}, conf: {win_confidences[i_val]}\n'
                      # f'       - lose: global:{lose_probs[i_val]:.3f}, int:{int_lose_probs[i_val]:.3f}, ling:{ling_lose_probs[i_val]:.3f}, conf: {lose_confidenecs[i_val]}\n')
            s += f'    > timeout win: {terminations.timeout_score_win:.3f}\n'
            s += f'    > timeout lose: {terminations.timeout_score_lose:.3f}\n'

        else: raise ValueError
        with open(self.params['exp_path'] + 'prior.txt', 'w') as f:
            f.write(s)
        with open(self.params['exp_path'] + 'prior_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        # plt.figure(figsize=(10, 7))
        # for key in data.keys():
        #     if 'obj_' in key:
        #         color = 'r'
        #     elif 'int_' in key:
        #         color = 'b'
        #     else:
        #         continue
        #         color = 'g'
        #     values = np.array(list(data[key].values()))
        #     plt.scatter(values[:, 0], values[:, 1], color=color, s=50, alpha=0.3)
        # plt.savefig(self.params['exp_path'] + 'plot_proposal.png')
        # plt.close()
        if to_print:
            print(s)
        return s, data

    @property
    def linguistic_data(self):
        return self.datastore.linguistic_data

    @property
    def names(self):
        return set(sorted(self.theory_pieces['objects'].keys()))
    
    