import os

# import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pickle
import tracemalloc
import time
import psutil

from src.game.game import Game
from src.agent.thinking.generative_model import GenerativeModel
from src.agent.thinking.likelihood import LikelihoodComputer, accept_or_reject
from src.game.rules import Particles
from src.utils import pickle_save

COLORS = [[0, 0.447, 0.7410], [0.85, 0.325, 0.098], [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.494, 0.1844, 0.556], [0, 0.447, 0.7410], [0.3010, 0.745, 0.933], [0.85, 0.325, 0.098],
          [0.466, 0.674, 0.188], [0.929, 0.694, 0.125],
          [0.3010, 0.745, 0.933], [0.635, 0.078, 0.184]]


class OracleThinker:
    """
    Has the correct theory, doesn't really think
    """
    def __init__(self, params, game, datastore, comm_engine):
        self.params = params
        self.time_tracker = params['time_tracker']
        self.true_game = Game(self.params, vgdl_script=self.params['true_game_info']['vgdl_script'])
        self.generative_model = None
        self.mcmc_process = None
        self.n_particles = params['agent']['thinker']['n_particles']
        self.particles = Particles([self.true_game.rules for _ in range(self.n_particles)])
        self.idx_rules_to_plan = 0

    def think(self, step_info, n_steps=1, print_shift=0):
        pass

    def log(self):
        return dict(thinker={})

    def dump_data(self, life_step_tracker):
        pass


class SMCThinker:
    """
    Propose theories using a generative model and does inference
    """
    def __init__(self, params, true_game, datastore, comm_engine):

        self.params = params
        self.true_game = true_game
        self.datastore = datastore

        self.comm_engine = comm_engine
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.size

        self.generative_model = GenerativeModel(self.params, self.datastore, self.comm_engine)

        # inference parameters
        self.strategy = params['agent']['thinker']['alg']
        self.n_particles = params['agent']['thinker']['n_particles']
        self.n_mcmc_steps = params['agent']['thinker']['n_mcmc_steps']
        self.n_simulations_likelihood = params['agent']['thinker']['n_simulations_likelihood']
        self.llm_params = params['agent']['thinker']['llm_params']
        self.verbose = params['verbose'] if self.rank == 0 else False
        self.time_tracker = params['time_tracker']
        self.data_path = params['exp_path'] + 'dumps/'
        os.makedirs(self.data_path, exist_ok=True)

        # inference variables
        self.i_inference_step = 1
        self.n_theories_evaluated = 0
        self.particles = None
        self.best_theory = None  # TODO how do we know in SMC?
        self.last_n_steps = 0  # keep track of number of steps saved to know when to udpate the generative model
        self.best_ever = None
        self.history = []
        self.idx_rules_to_plan = None

        self.likelihood_computer = LikelihoodComputer(self.comm_engine, self.params, self.time_tracker)

    # 
    def setup_proposal(self, type):
        if type == 'prior':
            for obj_piece in self.generative_model.theory_pieces['objects'].values():
                for value in obj_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = False
                    value.params['exp_params']['use_language_proposal'] = False
                obj_piece.params['exp_params']['use_data_proposal'] = False
                obj_piece.params['exp_params']['use_language_proposal'] = False
            for int_piece in self.generative_model.theory_pieces['object_pairs'].values():
                for value in int_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = False
                    value.params['exp_params']['use_language_proposal'] = False
                int_piece.params['exp_params']['use_data_proposal'] = False
                int_piece.params['exp_params']['use_language_proposal'] = False
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_data_proposal'] = False
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_language_proposal'] = False
        elif type == 'data_biased':
            for obj_piece in self.generative_model.theory_pieces['objects'].values():
                for value in obj_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = True
                    value.params['exp_params']['use_language_proposal'] = False
                obj_piece.params['exp_params']['use_data_proposal'] = True
                obj_piece.params['exp_params']['use_language_proposal'] = False
            for int_piece in self.generative_model.theory_pieces['object_pairs'].values():
                for value in int_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = True
                    value.params['exp_params']['use_language_proposal'] = False
                int_piece.params['exp_params']['use_data_proposal'] = True
                int_piece.params['exp_params']['use_language_proposal'] = False
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_data_proposal'] = True
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_language_proposal'] = False
        elif type == 'data_lang_biased':
            for obj_piece in self.generative_model.theory_pieces['objects'].values():
                for value in obj_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = True
                    value.params['exp_params']['use_language_proposal'] = True
                obj_piece.params['exp_params']['use_data_proposal'] = True
                obj_piece.params['exp_params']['use_language_proposal'] = True
            for int_piece in self.generative_model.theory_pieces['object_pairs'].values():
                for value in int_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = True
                    value.params['exp_params']['use_language_proposal'] = True
                int_piece.params['exp_params']['use_data_proposal'] = True
                int_piece.params['exp_params']['use_language_proposal'] = True
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_data_proposal'] = True
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_language_proposal'] = True
        elif type == 'lang_biased':
            for obj_piece in self.generative_model.theory_pieces['objects'].values():
                for value in obj_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = False
                    value.params['exp_params']['use_language_proposal'] = True
                obj_piece.params['exp_params']['use_data_proposal'] = False
                obj_piece.params['exp_params']['use_language_proposal'] = True
            for int_piece in self.generative_model.theory_pieces['object_pairs'].values():
                for value in int_piece.possible_values:
                    value.params['exp_params']['use_data_proposal'] = False
                    value.params['exp_params']['use_language_proposal'] = True
                int_piece.params['exp_params']['use_data_proposal'] = False
                int_piece.params['exp_params']['use_language_proposal'] = True
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_data_proposal'] = False
            self.generative_model.theory_pieces['terminations'].params['exp_params']['use_language_proposal'] = True
        else:
            raise ValueError

    # Infer the most likely game descriptions
    def think(self, step_info, n_steps=1, print_shift=0):
        
        if self.i_inference_step == 1:
            self.loglike_study = dict(interaction=[self.true_game.rules.interaction_loglikelihood], 
                                      language=[self.true_game.rules.language_loglikelihood],
                                      rules=[self.true_game.rules], 
                                      hash=[hash(self.true_game.rules)], 
                                      accept_info=[])

        if self.verbose: print(f'{" " * print_shift}> agent thinks: running {n_steps} steps of inference')
        if self.verbose: print(f'{" " * print_shift}  > updating generative model')

        # Update generative model (prior)
        self.time_tracker.tic('thinker_update_prior')
        if self.verbose: print(f"{' ' * print_shift}  > updating proposal distribution from new data")
        updated = self.generative_model.update()
        if updated:
            self.best_ever = None
        self.time_tracker.toc('thinker_update_prior')

        # Sample from datastore (?)
        self.time_tracker.tic('thinker_sample_data')
        if self.rank == 0:
            self.generative_model.print_prior(to_print=False)
            self.data = self.datastore.sample_data_for_likelihood()
            if self.verbose:  print(f'{" " * print_shift}  > using {len(self.data[-2])} datapoints to compute likelihood')
            # TODO should we replace this by uniform novlety-based sampling?
        else:
            self.data = None
        if self.size > 1:
            self.data = self.comm.bcast(self.data, root=0)
        if self.rank in [0, 1]:
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            print(f"\nMEM USAGE {self.rank}: {mem_info.rss / 1024 / 1024:.2f} MB", )
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')[:10]
        # print(f"Top 10 memory allocations (rank {self.rank}:")
        # for stat in top_stats:
        #     print(stat)
        self.time_tracker.toc('thinker_sample_data')

        # Sample theories (particles) from generative model
        if self.verbose: print(f'{" " * print_shift}  > sampling or extending particles')
        self.time_tracker.tic('thinker_sample_prior')
        if self.rank == 0:
            # this will extend particles if needed
            self.particles = self.generative_model.sample(n=self.n_particles, particles=self.particles)
        else: self.particles = None
        self.time_tracker.toc('thinker_sample_prior')
        # loglike_tracker = []

        # Run SMC
        for i_smc in range(n_steps):
            t_init_2 = time.time()

            if self.verbose: print(f'{" " * print_shift}  > smc step #{i_smc+1}/{n_steps} (global step #{self.i_inference_step})')

            # Compute likelihood for each particle
            self.time_tracker.tic('thinker_compute_likelihood')
            self.particles = self.likelihood_computer.compute_likelihood(self.particles, self.generative_model, self.data)
            if self.params['exp_params']['use_language_likelihood'] and self.rank == 0 and self.params['exp_params']['use_oracle_data']:
            #     self.loglike_study['interaction'] += self.particles.interaction_loglikelihoods.copy()
            #     self.loglike_study['language'] += self.particles.language_loglikelihoods.copy()
            #     self.loglike_study['hash'] += [hash(particle) for particle in self.particles]
            #     self.loglike_study['rules'] += self.particles.copy()
                self.n_theories_evaluated += self.n_particles
            self.time_tracker.toc('thinker_compute_likelihood')

            # Resample
            if self.rank == 0:
                # resample particles in proportion to posterior
                new_indexes = np.random.choice(range(self.n_particles), p=self.particles.posteriors, replace=True, size=self.n_particles)
                # print(f'{" " * print_shift}      > prior logprobs: {array2str(self.particles.logpriors)}')
                # print(f'{" " * print_shift}      > loglikelihoods: {array2str(self.particles.loglikelihoods)}')
                # print(f'{" " * print_shift}      > logjoints: {array2str(self.particles.logjoints)}')
                # print(f'{" " * print_shift}      > posterior probs: {array2str(self.particles.posteriors)}')
                # print(f'{" " * print_shift}      > new indexes: {new_indexes}')
                self.particles.resample(new_indexes)
            self.track_best(step_info, print_shift)
            if self.verbose: print(f'{" " * print_shift}    > best logjoint: {np.max(self.particles.logjoints):.3f}, best loglike: {np.max(self.particles.loglikelihoods):.3f}')

            # MCMC rejuvenation
            for i_mcmc in range(self.n_mcmc_steps):
                t_init_1 = time.time()
                if self.verbose: print(f'{" " * print_shift}    > mcmc rejuvenation step #{i_mcmc+1}/{self.n_mcmc_steps}')

                self.time_tracker.tic('thinker_mutate')
                if self.rank == 0:
                    name = "mutate"
                    # for p in self.particles:
                    #     print(p.logprior)
                    mutated_particles  = self.generative_model.mutate(self.particles)
                    # for p, mp in zip(self.particles, mutated_particles):
                    #     print(p.vgdl_lines.dict['box'], mp.vgdl_lines.dict['box'], p.logprior, mp.logprior, mp.logprior - p.logprior)
                    self.n_theories_evaluated += len(mutated_particles)

                else:
                    mutated_particles = None
                self.time_tracker.toc('thinker_mutate')

                # compute loglikelihoods
                self.time_tracker.tic('thinker_compute_likelihood')
                mutated_particles = self.likelihood_computer.compute_likelihood(mutated_particles, self.generative_model, self.data)
                # if self.params['exp_params']['use_language_likelihood'] and self.rank == 0 and self.params['exp_params']['use_oracle_data']:
                    # self.loglike_study['interaction'] += self.particles.interaction_loglikelihoods.copy()
                    # self.loglike_study['language'] += self.particles.language_loglikelihoods.copy()
                    # self.loglike_study['rules'] += self.particles.copy()
                    # self.loglike_study['hash'] += [hash(particle) for particle in self.particles]
                self.time_tracker.toc('thinker_compute_likelihood')

                # decide whether to accept or reject mutations
                if self.rank == 0:
                    # ratios, accepts = [], []
                    # ratio_int_like, ratio_lang_like, ratio_prior, ratio_prop = [], [], [], []
                    accepts = []
                    for particle, mutated_particle in zip(self.particles, mutated_particles):
                        linguistic_data = self.data[-1]
                        accept_info = accept_or_reject(particle, mutated_particle, self.generative_model, linguistic_data)
                        self.loglike_study['accept_info'].append(accept_info)
                        # ratios.append(accept_info['ratio'])
                        # ratio_int_like.append(accept_info['ratio_int_loglikelihood'])
                        # ratio_lang_like.append(accept_info['ratio_lang_loglikelihood'])
                        # ratio_prior.append(accept_info['ratio_logprior'])
                        accepts.append(accept_info['accept'])
                        # loglike_tracker.append((particle.logjoint, mutated_particle.logjoint)
                    # update the set of particles
                    self.particles.replace_with(mutated_particles, accepts)
                    # track new best
                    self.track_best(step_info, print_shift)

                    # replace worse particle by the best
                    self.particles.keep_best(self.best_ever)
                    if self.verbose:
                        print(f'{" " * print_shift}    > best logjoint: {np.max(self.particles.logjoints):.3f}, best loglike: {np.max(self.particles.loglikelihoods):.3f}')
                    # with open(self.params['exp_path'] + 'likelihood_data.pkl', 'wb') as f:
                    #     pickle.dump(self.loglike_study, f)
                if self.verbose:
                    print(f'{" " * print_shift}    > mcmc step ran in {time.time() - t_init_1:.2f} ')
            if self.verbose:
                print(f'{" " * print_shift}    > whole step ran in {time.time() - t_init_2:.2f} ')

            self.idx_rules_to_plan = self.particles.sample()
            self.append_history(step_info)
            if self.params['exp_params']['use_oracle_data']:
                self.dump_data(dict(gen=0, life=0, n_levels_solved=0))
            self.i_inference_step += 1
            if self.verbose:
                print(f'{" " * print_shift}    > new best theory:\n{self.best_ever}')
                # print(self.best_ever.prompt)

            # if self.rank == 0 and self.params['exp_params']['use_language_likelihood'] and self.params['exp_params']['use_oracle_data']:
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(13, 10))
            #     plt.scatter(self.loglike_study['interaction'][0], self.loglike_study['language'][0], s=100, color='r', zorder=3)
            #     plt.scatter(self.loglike_study['interaction'], self.loglike_study['language'], s=70, alpha=0.3)
            #     plt.savefig(self.params['exp_path'] + f'like_{self.i_inference_step}.png')
            #     plt.xlabel('loglikelihood of interaction data')
            #     plt.ylabel('loglikelihood of description')
            #     plt.close()
                # with open(self.params['exp_path'] + 'likelihood_data.pkl', 'wb') as f:
                #     pickle.dump(self.loglike_study, f)

            #TODO remove that bit
            # if self.rank == 0:
            #     with open(self.params['exp_path'] + 'likelihood_data.pkl', 'wb') as f:
            #         pickle.dump(self.loglike_study, f)
        # assert False

    def track_best(self, step_info, print_shift=0):
        if self.rank == 0:
            if self.best_ever is None or np.max(self.particles.logjoints) >= self.best_ever.logjoint and self.best_ever != self.particles[self.particles.best_index]:
                self.best_ever = self.particles.copy_best(criterion='logjoint')
                if self.generative_model.comm_engine is not None:
                    self.generative_model.comm_engine.best_theory = self.best_ever


                # description = self.comm_engine.generate_description(self.best_ever)
                # print('Best game description: ', description)
                # if len(step_info) > 0:
                #     file_name = f"gen_{step_info['gen']}_life_{step_info['life']}_lvl_{step_info['lvl']}_envstep_{step_info['env_step']}"
                    # with open(self.params['exp_path'] + f'theories_and_descriptions/{file_name}.txt', 'w') as f:
                    #     f.write(self.best_ever.vgdl_script + '\n\n' + description)
                # if self.params['exp_params']['use_oracle_data']:
                # if self.verbose:
                #     print(f'{" " * print_shift}    > new best theory:\n{self.best_ever}')
                #     print(self.best_ever.prompt)
                


            save_path = self.params['exp_path'] + f'best_theory/'
            os.makedirs(save_path, exist_ok=True)
            best_game = Game(self.params, rules=self.best_ever)

            # if 'sam' in best_game.rules.obj.obj_names and best_game.rules.obj.params('sam')['singleton'] is False:
            #     print(best_game.rules)
            best_game.save_to(save_path)

    def append_history(self, step_info):
        if self.rank == 0:
            to_save = step_info.copy()
            with open(self.params['exp_path'] + f'prior_gen_{self.params["current_gen"]}.txt', 'w') as f:
                f.write(self.generative_model.print_prior(to_print=False)[0])
            if self.i_inference_step % 10 == 0:
                to_save.update(dict(smc_step=self.i_inference_step,
                                    mcmc_step=self.i_inference_step * self.n_mcmc_steps,
                                    # proposal_distribution=self.generative_model.theory_pieces,
                                    proposal_distribution_str=self.generative_model.print_prior(to_print=False)[0],
                                    vgdl_scripts=[particle.vgdl_script for particle in self.particles],
                                    best_particle=self.particles[self.idx_rules_to_plan] if self.idx_rules_to_plan is not None else None,
                                    idx_rules_to_plan=self.idx_rules_to_plan,
                                    loglikelihood=self.particles.loglikelihoods,
                                    loglikelihood_lang=self.particles.language_loglikelihoods,
                                    loglikelihood_int=self.particles.interaction_loglikelihoods,
                                    logprior=self.particles.logpriors,
                                    logjoint=self.particles.logjoints,
                                    ground_truth_loglikelihood=self.true_game.rules.loglikelihood,
                                    ground_truth_loglikelihood_lang=self.true_game.rules.language_loglikelihood,
                                    ground_truth_loglikelihood_int=self.true_game.rules.interaction_loglikelihood,
                                    ground_truth_logprior=self.true_game.rules.logprior,
                                    ground_truth_logjoint=self.true_game.rules.logjoint,
                                    n_theories_evaluated=self.n_theories_evaluated
                                    ))
                self.history.append(to_save)

    def dump_data(self, life_step_tracker):
        if self.rank == 0:
            name = f'thinking_output_generation_{life_step_tracker["gen"]}_life_{life_step_tracker["life"]}_lvl_solved_{life_step_tracker["n_levels_solved"]}.pkl'
            pickle_save(self.history, self.data_path + name)
            self.history = []

    def log(self):
        with open(self.params['exp_path'] + 'particle_history.pkl', 'wb') as f:
            pickle.dump(self.history, f)
        return dict(thinker=dict(i_smc_steps=self.i_inference_step,
                                 i_mcmc_steps=self.i_inference_step * self.n_mcmc_steps,
                                 n_theories_evaluated=self.n_theories_evaluated))
            
    #
    def test(self, print_shift=0):

        # update prior
        if self.i_inference_step == 1:
            self.loglike_study = dict(interaction=[self.true_game.rules.interaction_loglikelihood], language=[self.true_game.rules.language_loglikelihood],
                                      rules=[self.true_game.rules], hash=[hash(self.true_game.rules)], proposal_type=['true_theory'], accept_info=[])

        self.time_tracker.tic('thinker_update_prior')

        if self.verbose: print(f"{' ' * print_shift}  > updating proposal distribution from new data")
        updated = self.generative_model.update()
        if updated:
            self.best_ever = None
        self.time_tracker.toc('thinker_update_prior')

        self.time_tracker.tic('thinker_sample_data')
        if self.rank == 0:
            self.data = self.datastore.sample_data_for_likelihood()
            if self.verbose:  print(f'{" " * print_shift}  > using {len(self.data[-2])} datapoints to compute likelihood')
            # TODO should we replace this by uniform novelty-based sampling?
        else:
            self.data = None
        if self.size > 1:
            self.data = self.comm.bcast(self.data, root=0)
        self.time_tracker.toc('thinker_sample_data')

        n_particles = 100
        prop_types = ['lang_biased'] #['prior', 'data_biased', 'lang_biased', 'data_lang_biased']
        for prop_type in prop_types:

            self.setup_proposal(prop_type)
            if self.verbose: print(f'{" " * print_shift}  > sampling or extending particles for type {prop_type}')

            self.time_tracker.tic('thinker_sample_prior')
            if self.rank == 0:
                # this will extend particles if needed
                self.particles = self.generative_model.sample(n=n_particles, particles=None)
            else:
                self.particles = None
            self.time_tracker.toc('thinker_sample_prior')

            # compute likelihoods
            self.time_tracker.tic('thinker_compute_likelihood')
            if self.verbose: print(f'{" " * print_shift}  > compute likelihood')

            self.particles = self.likelihood_computer.compute_likelihood(self.particles, self.generative_model, self.data)
            # self.loglike_study['interaction'] += self.particles.interaction_loglikelihoods.copy()
            # self.loglike_study['language'] += self.particles.language_loglikelihoods.copy()
            # self.loglike_study['hash'] += [hash(particle) for particle in self.particles]
            # self.loglike_study['rules'] += self.particles.copy()
            # self.loglike_study['proposal_type'] += [prop_type] * len(self.particles)
            self.time_tracker.toc('thinker_compute_likelihood')
            if self.verbose: print(f'{" " * print_shift}  > mutating')
            for i_mut in range(5):
                if self.rank in [0, 1]:
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    print(f"\nMEM USAGE {self.rank}: {mem_info.rss / 1024 / 1024:.2f} MB", )
                if self.rank == 0:
                    mutated_particles = self.generative_model.mutate(self.particles)
                else:
                    mutated_particles = None

                # compute loglikelihoods
                if self.verbose: print(f'{" " * print_shift}  > compute likelihood of mutation {i_mut}')
                self.time_tracker.tic('thinker_compute_likelihood')
                mutated_particles = self.likelihood_computer.compute_likelihood(mutated_particles, self.generative_model, self.data)
                self.time_tracker.toc('thinker_compute_likelihood')

                # decide whether to accept or reject mutations
                if self.rank == 0:
                    for particle, mutated_particle in zip(self.particles, mutated_particles):
                        linguistic_data = self.data[-1]
                        accept_info = accept_or_reject(particle, mutated_particle, self.generative_model, linguistic_data)
                        self.loglike_study['accept_info'].append(accept_info)
                with open(self.params['exp_path'] + 'likelihood_data.pkl', 'wb') as f:
                    pickle.dump(self.loglike_study, f)

            if self.verbose: print(f'{" " * print_shift}  > plotting for type {prop_type}')

            # if self.rank == 0 and self.params['exp_params']['use_language_likelihood'] and self.params['exp_params']['use_oracle_data']:
            #     import matplotlib.pyplot as plt
            #     plt.figure(figsize=(13, 10))
            #     plt.scatter(self.loglike_study['interaction'][0], self.loglike_study['language'][0], s=100, color='r', zorder=3, )
            #
            #     plt.scatter(self.loglike_study['interaction'][-n_particles:], self.loglike_study['language'][-n_particles:], s=70, alpha=0.3, color=[COLORS[prop_types.index(stype)] for stype in
            #                                                                                                                    self.loglike_study['proposal_type'][-n_particles:]])
            #     plt.xlabel('loglikelihood of interaction data')
            #     plt.ylabel('loglikelihood of description')
            #     plt.title(prop_type)
            #     plt.savefig(self.params['exp_path'] + f'like_{prop_type}.png')
            #     plt.close()
            #
            #     plt.figure(figsize=(13, 10))
            #     plt.scatter(self.loglike_study['interaction'][0], self.loglike_study['language'][0], s=100, color='r', zorder=3, )
            #
            #     plt.scatter(self.loglike_study['interaction'][1:], self.loglike_study['language'][1:], s=70, alpha=0.3, color=[COLORS[prop_types.index(stype)] for stype in
            #                                                                                                             self.loglike_study['proposal_type'][1:]])
            #     plt.xlabel('loglikelihood of interaction data')
            #     plt.ylabel('loglikelihood of description')
            #     plt.savefig(self.params['exp_path'] + f'like_all.png')
            #
            #     plt.close()
            #     with open(self.params['exp_path'] + 'likelihood_data.pkl', 'wb') as f:
            #         pickle.dump(self.loglike_study, f)
        stop = 1

