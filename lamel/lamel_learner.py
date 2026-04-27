import numpy as np
import pandas as pd
# import sys
import os
import glob
import random
import datetime
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.sparse import csr_matrix, vstack

# Cheminformatics
# from rdkit import Chem

# Custom
# from minervachem import fingerprinters as fp
# from minervachem import transformers as tf
# from minervachem.plotting import parity_plot_train_test
from minervachem.regressors import HierarchicalResidualModel

from dataset_loader import LoadedDataset
from lamel_utilities import *



class MetaLearner:
    def __init__(self, database_path, max_subgraph_size=5, n_shots=20, random_state=16, epsilon_par=1.0,
                 epsilon_perp=1.0, hierarchical=False, working_directory=None, make_log=False, a_range='a1'):
        self.dataset = LoadedDataset()
        if isinstance(database_path, str):
            self.dataset.load(database_path)
        elif isinstance(database_path, list):
            self.dataset.load(database_path[0])
            for path_td in database_path[1:]:
                self.dataset.load(path_td)
        else:
            raise TypeError('Database_path must be a string or a list')

        self.max_subgraph_size = max_subgraph_size
        self.n_shots = n_shots
        self.random_state = random_state
        self.epsilon_par = epsilon_par
        self.epsilon_perp = epsilon_perp
        self.hierarchical = hierarchical
        self.working_directory = working_directory if working_directory is not None else os.getcwd()
        self.make_log = make_log
        self.alpha_range = a_range

        self.fingerprints = self.dataset.get_fingerprint_descriptors(self.max_subgraph_size)
        # self.task_names = self.dataset.tasks
        self.new_task_name = None
        self.old_task_names = []
        self.n_ots = len(self.old_task_names)
        print("[!!!] Warning: currently we are cross-validating Ridge model for alpha parameter only [!!!]")

    def set_task_names(self, new_task, old_tasks=None):
        self.new_task_name = new_task
        if old_tasks is not None:
            self.old_task_names = [pr for pr in old_tasks if pr != self.new_task_name]
        else:
            print("Old task names not set. All tasks will be treated as old tasks.")
            self.old_task_names = [pr for pr in self.dataset.tasks if pr != self.new_task_name]
        self.n_ots = len(self.old_task_names)

    def regenerate_fingerprints(self):
        self.fingerprints = self.dataset.get_fingerprint_descriptors(self.max_subgraph_size)

    def process_old_tasks(self, new_task=None, old_tasks=None):
        if new_task is None:
            assert self.new_task_name is not None, "New task name must be set"
            new_task = self.new_task_name

        if old_tasks is None:
            old_tasks = self.old_task_names

        n_ots = len(old_tasks)
        rstate_addons = np.arange(0, n_ots) + 2  # makes sure that each task is split with different random seed
        old_tasks_sets = [split_data_per_task(self.fingerprints,
                                              self.dataset.data[old_tasks[i]].to_numpy(),
                                              rstate=self.random_state + rstate_addons[i],
                                              make_log=self.make_log)
                          for i in range(n_ots)]
        return old_tasks_sets

    def process_new_task(self, new_task=None):
        if new_task is None:
            assert self.new_task_name is not None, "New task name must be set"
            new_task = self.new_task_name
        # assert self.n_ots > 0, "Number of old tasks must be greater than 0"

        new_task_sets = split_data_per_task(self.fingerprints, self.dataset.data[new_task].to_numpy(),
                                            rstate=self.random_state + 123, make_log=self.make_log)
        random.seed(self.random_state)

        try:
            new_task_ids = random.sample(range(len(new_task_sets[1])), self.n_shots)
        except ValueError:
            print(f'You ask for too many samples! \n '
                  f'The number of shots will be downgraded to the size of the set: {len(new_task_sets[1])}')
            n_shots = len(new_task_sets[1])
            new_task_ids = random.sample(range(len(new_task_sets[1])), n_shots)
        new_task_x = vstack([new_task_sets[0].getrow(i) for i in new_task_ids])
        new_task_y = new_task_sets[1][new_task_ids]
        return [new_task_x, new_task_y], new_task_sets

    # def run_default_ridge_regressions(self, input_sets, hierarchical=False):
    #     return run_default_ridge_regressions(input_sets, hierarchical)

    def calculate_residuals(self, model_vectors, t_bar, new_task_nshots):
        task_residuals = []
        for task_id in range(len(model_vectors)):
            theta_temp = model_vectors[task_id] - t_bar
            e_temp = np.concatenate(np.array(np.dot(new_task_nshots[0].todense(), theta_temp)))
            task_residuals.append(e_temp)
        e_residuals = np.array(task_residuals).T
        # The residuals of the new task are using real values instead of predictions
        residuals_for_new_task = new_task_nshots[1] - np.concatenate(
            np.array(np.dot(new_task_nshots[0].todense(), t_bar)))
        return e_residuals, residuals_for_new_task

    def solve_parallel(self, model_vectors, t_bar, new_task_nshots, print_alpha=True):
        eres, ress_nt = self.calculate_residuals(model_vectors, t_bar, new_task_nshots)
        p_par, _ = cross_validate_parameters(eres, ress_nt, alpha_range=self.alpha_range)
        model_par = Ridge(**p_par)
        if self.hierarchical:
            hmodel_par = HierarchicalResidualModel(regressor=model_par, verbose=1)
            hmodel_par.fit(eres, ress_nt)
            c_is = np.copy(hmodel_par.coef_)
            if print_alpha:
                print(f'Alpha for parallel solution: {hmodel_par.alpha}')
        else:
            model_par.fit(eres, ress_nt)
            c_is = np.copy(model_par.coef_)
            if print_alpha:
                print(f'Alpha for parallel solution: {model_par.alpha}')
        c_is = c_is * self.epsilon_par
        t_is = np.dot(c_is, np.asarray(model_vectors) - t_bar)
        return t_is, model_par.alpha

    def solve_perpendicular(self, new_task_nshots, t_bar, t_par, print_alpha=True):
        perp_par, _ = cross_validate_parameters(new_task_nshots[0], new_task_nshots[1], alpha_range=self.alpha_range)
        model_perp = Ridge(**perp_par)
        tt = t_bar + t_par
        if self.hierarchical: # NOT WORKING AS OF NOW
            hmodel_perp = HierarchicalResidualModel(regressor=model_perp, verbose=1)
            fit_with_prior(hmodel_perp, np.asarray(new_task_nshots[0].todense()), new_task_nshots[1],
                           prior=tt, a_range=self.alpha_range)
            theta_task = np.copy(hmodel_perp.coef_)
        else:
            theta_task, alpha_perp = fit_with_prior(model_perp, np.asarray(new_task_nshots[0].todense()),
                                                    new_task_nshots[1],
                                                    prior=tt, a_range=self.alpha_range,
                                                    reval_final=True) 
        t_perp = theta_task - tt
        theta_task = theta_task * self.epsilon_perp

        if print_alpha:
            print(f'Alpha for perp: {alpha_perp}')
        return t_perp, theta_task, alpha_perp

    @staticmethod
    def collect_cossim_measures_for_thetas(theta_bar, theta_is, theta_perp, theta_task, print_on=False):
        """
            A function to collect cossim measures for different thetas involved in the meta learning.
            :param theta_bar: list. Centroid vector for all old tasks.
            :param theta_is: list. Parallel part of the meta learning solution.
            :param theta_perp: list. Perpendicular part of the meta learning solution.
            :param theta_task: list. Task specific meta learning solution.
            :param print_on: bool. Whether to print the cos' or not. Default is False.
            :return: dict. Dictionary of cossim measures for different thetas.
            """
        cossim_dict = {'theta_bar:theta_bar': cosine_measure(theta_bar, theta_bar),
                       'theta_bar:theta_is': cosine_measure(theta_bar, theta_is),
                       'theta_bar:theta_perp': cosine_measure(theta_bar, theta_perp),
                       'theta_bar:theta_task': cosine_measure(theta_bar, theta_task),
                       'theta_is:theta_is': cosine_measure(theta_is, theta_is),
                       'theta_is:theta_perp': cosine_measure(theta_is, theta_perp),
                       'theta_is:theta_task': cosine_measure(theta_is, theta_task),
                       'theta_perp:theta_perp': cosine_measure(theta_perp, theta_perp),
                       'theta_perp:theta_task': cosine_measure(theta_perp, theta_task),
                       'theta_task:theta_task': cosine_measure(theta_task, theta_task)}
        if print_on:
            print(f"__Cosine similarity measures__ \n"
                  f"Theta_bar & theta_is: {cossim_dict['theta_bar:theta_is']}, \n"
                  f"Theta_is & theta_perp: {cossim_dict['theta_is:theta_perp']}, \n"
                  f"Theta_bar & theta_task: {cossim_dict['theta_bar:theta_task']}, \n")
        return cossim_dict

    def collect_non_meta_results_nshots(self, new_task_nshots, new_task_sets):
        """
        A function to run the default linear regression on the n shots of the new task.
        :param new_task_nshots: list of lists. X and Y values for the nshots from the training set of the new task.
        :param new_task_sets: list of lists. Train and test sets for the new task.
        :param hierarchical: bool. Whether to use hierarchical or not. Default is False (classical ridge solution).
        :return: dict. Dictionary of mae, rmse, r2 for non-meta solution and dictionary for non-meta solution
        model parameters.
        """
        nmpars_nshots, _ = cross_validate_parameters(new_task_nshots[0], new_task_nshots[1],
                                                     alpha_range=self.alpha_range)
        nmmodel_nshots = Ridge(**nmpars_nshots)
        if self.hierarchical:
            hmodel_nshots = HierarchicalResidualModel(regressor=nmmodel_nshots, verbose=1)
            hmodel_nshots.fit(new_task_nshots[0], new_task_nshots[1])
            predicted = hmodel_nshots.predict(new_task_sets[2])
        else:
            nmmodel_nshots.fit(new_task_nshots[0], new_task_nshots[1])
            predicted = nmmodel_nshots.predict(new_task_sets[2])

        nm_mae = round(mean_absolute_error(new_task_sets[3], predicted), 2)
        nm_rmse = round(np.sqrt(mean_squared_error(new_task_sets[3], predicted)), 2)
        nm_r2 = round(r2_score(new_task_sets[3], predicted), 4)
        nonmeta_errors = {'mae': nm_mae,
                          'rmse': nm_rmse,
                          'r2': nm_r2}
        return nonmeta_errors, nmpars_nshots

    def log_results_dict(self, nm_parameters, parallel_alpha, perp_alpha, meta_errors, nonmeta_errors, suffix_notes='',
                         ext='txt'):
        """
        A function to log the results of the meta learning.
        :param nm_parameters:
        :param parallel_alpha:
        :param perp_alpha:
        :param meta_errors:
        :param nonmeta_errors:
        :param suffix_notes:
        :return:
        """
        results_dictionary = dict({'property': self.new_task_name,
                                   'support_tasks': tuple(self.old_task_names),
                                   'max_subgraph_size': self.max_subgraph_size,
                                   'solver': 'sparse_cg',
                                   'alpha_for_new_task': nm_parameters['alpha'],
                                   'alpha_par': parallel_alpha,
                                   'alpha_perp': perp_alpha,
                                   'epsilon_par': self.epsilon_par,
                                   'epsilon_perp': self.epsilon_perp,
                                   'mae': meta_errors['mae'],
                                   'rmse': meta_errors['rmse'],
                                   'r_2': meta_errors['r2'],
                                   'nonmeta_mae': nonmeta_errors['mae'],
                                   'nonmeta_rmse': nonmeta_errors['rmse'],
                                   'nonmeta_r_2': nonmeta_errors['r2'],
                                   'n_shots': self.n_shots,
                                   'notes': f'{suffix_notes}'})
        if ext:
            save_dict_to_file(results_dictionary, self.working_directory, file_prefix=f'{suffix_notes}_metalearner',
                              extension=ext)
        return results_dictionary

    def save_metalearner_results_to_file(self, results, results_filename):
        """
        Saves a dictionary as a Pandas DataFrame to a pickle file, with the filename containing the current date.
        If the file already exists, a consecutive number is added to the filename.

        Args:
            results (dict): The results dictionary to be saved.
            results_filename (str): The prefix for the filename.
        """
        today = datetime.date.today().strftime('%m%d%Y')
        filename = f"{results_filename}_{today}.pkl"
        file_path = os.path.join(self.working_directory, filename)

        if os.path.exists(file_path):
            i = 1
            while os.path.exists(file_path):
                filename = f"{results_filename}_{today}_{i}.pkl"
                file_path = os.path.join(self.working_directory, filename)
                i += 1

        # Convert the dictionary to a Pandas DataFrame and save it as a pickle file
        df = pd.DataFrame.from_dict([results])
        df.to_pickle(file_path)
        return

    def single_layer_model(self, new_task=None, old_tasks=None, task_vectors=None, print_ot_evaluations=False,
                           print_cossim=False, printmetares=True, check_vec_pca=False,
                           save_res_to=None, resnotes=''):

        if not isinstance(self.fingerprints, csr_matrix):
            raise ValueError('Something went wrong. Fingerprints must be a csr_matrix')

        if new_task is None:
            new_task = self.new_task_name
        if old_tasks is None:
            old_tasks = self.old_task_names

        print(f"***** Number of support tasks used in single layer model: {len(old_tasks)} *****")

        old_tasks_material = self.process_old_tasks(new_task=new_task, old_tasks=old_tasks)
        new_nshots, new_sets = self.process_new_task(new_task=new_task)

        if task_vectors:
            parvecs_old_tasks_material = [task_vectors[task] for task in old_tasks]
        else:
            parvecs_old_tasks_material, params_old_tasks_material = run_default_ridge_regressions(old_tasks_material,
                                                                                                  a_range=self.alpha_range)
            if print_ot_evaluations:
                print_evaluations(params_old_tasks_material, parvecs_old_tasks_material, old_tasks_material)

        if check_vec_pca:
            ev_range = [0.95, 0.99, 0.999]
            ncomps = []
            for evar in ev_range:
                pca_ncomp = PCA(svd_solver='full', n_components=evar)
                pca_ncomp.fit(np.array(parvecs_old_tasks_material))
                ncomps.append(pca_ncomp.n_components_)
            print(
                f"{ncomps} principal components explain {ev_range} variance")

        # Calculating theta_bar
        theta_bar = centroid_by_points(parvecs_old_tasks_material)
        # Calculating theta_parallel
        theta_parallel, parallel_alpha = self.solve_parallel(parvecs_old_tasks_material, theta_bar, new_nshots)
        # Calculating theta_perpendicular
        theta_perpendicular, theta_task, perpendicular_alpha = self.solve_perpendicular(new_nshots, theta_bar,
                                                                                        theta_parallel)

        self.collect_cossim_measures_for_thetas(theta_bar, theta_parallel, theta_perpendicular, theta_task,
                                                print_on=print_cossim)
        nm_results, nmparameters = self.collect_non_meta_results_nshots(new_nshots, new_sets)
        meta_results = evaluate_regression_model(nmparameters, theta_task, new_sets[2], new_sets[3],
                                                 print_on=printmetares)
        results = self.log_results_dict(nmparameters, parallel_alpha, perpendicular_alpha,
                                        meta_results, nm_results, suffix_notes=resnotes, ext=save_res_to)
        # return theta_task, results # 12/12 working on epsilon
        return theta_task, results, theta_perpendicular, theta_bar

    def single_layer_model_orthomode(self, new_task=None, old_tasks=None, task_vectors=None,
                                     theta_bar_calc=True,
                                     print_ot_evaluations=False,
                                     print_cossim=False, printmetares=True,
                                     save_res_to=None, resnotes=''):

        if not isinstance(self.fingerprints, csr_matrix):
            raise ValueError('Something went wrong. Fingerprints must be a csr_matrix')

        if new_task is None:
            new_task = self.new_task_name
        if old_tasks is None:
            old_tasks = self.old_task_names

        print(f"***** Number of support tasks used in single layer model: {len(old_tasks)} *****")

        old_tasks_material = self.process_old_tasks(new_task=new_task, old_tasks=old_tasks)
        new_nshots, new_sets = self.process_new_task(new_task=new_task)

        if task_vectors:
            parvecs_old_tasks_material = [task_vectors[task] for task in old_tasks]
        else:
            parvecs_old_tasks_material, params_old_tasks_material = run_default_ridge_regressions(old_tasks_material,
                                                                                                  a_range=self.alpha_range)
            if print_ot_evaluations:
                print_evaluations(params_old_tasks_material, parvecs_old_tasks_material, old_tasks_material)

        # new step - orthonormalization of the set of parvecs for old tasks
        parvecs_ortho = gram_schmidt(np.array(parvecs_old_tasks_material))

        # Calculating theta_bar
        if theta_bar_calc:
            theta_bar = centroid_by_points(parvecs_ortho)
        else:
            theta_bar = np.zeros(self.fingerprints.shape[1])
        # Calculating theta_parallel
        theta_parallel, parallel_alpha = self.solve_parallel(parvecs_ortho, theta_bar, new_nshots)
        # Calculating theta_perpendicular
        theta_perpendicular, theta_task, perpendicular_alpha = self.solve_perpendicular(new_nshots, theta_bar,
                                                                                        theta_parallel)

        self.collect_cossim_measures_for_thetas(theta_bar, theta_parallel, theta_perpendicular, theta_task,
                                                print_on=print_cossim)
        nm_results, nmparameters = self.collect_non_meta_results_nshots(new_nshots, new_sets)
        meta_results = evaluate_regression_model(nmparameters, theta_task, new_sets[2], new_sets[3],
                                                 print_on=printmetares)
        results = self.log_results_dict(nmparameters, parallel_alpha, perpendicular_alpha,
                                        meta_results, nm_results, suffix_notes=resnotes, ext=save_res_to)
        # return theta_task, results # 12/12 working on epsilon
        return theta_task, results, theta_perpendicular, theta_bar

    # 03/18/25 addition
    def single_layer_pcamodel_fps(self, new_task=None, old_tasks=None, task_vectors=None,
                                  expl_variance=0.95,
                                  print_ot_evaluations=False,
                                  print_cossim=False, printmetares=True,
                                  save_res_to=None, resnotes=''):

        if not isinstance(self.fingerprints, csr_matrix):
            raise ValueError('Something went wrong. Fingerprints must be a csr_matrix')

        pca_ncomp = PCA(svd_solver='full', n_components=expl_variance)
        reduced_fps = pca_ncomp.fit_transform(self.fingerprints.toarray())
        print(f"Shape of the reshaped fingerprints: {reduced_fps.shape}")
        print(
            f"There are {pca_ncomp.n_components_} principal components to explain {pca_ncomp.n_components} variance")
        reduced_fps_sparse = csr_matrix(reduced_fps)
        self.fingerprints = reduced_fps_sparse

        if new_task is None:
            new_task = self.new_task_name
        if old_tasks is None:
            old_tasks = self.old_task_names

        print(f"***** Number of support tasks used in single layer model: {len(old_tasks)} *****")

        old_tasks_material = self.process_old_tasks(new_task=new_task, old_tasks=old_tasks)
        new_nshots, new_sets = self.process_new_task(new_task=new_task)

        if task_vectors:
            parvecs_old_tasks_material = [task_vectors[task] for task in old_tasks]
        else:
            parvecs_old_tasks_material, params_old_tasks_material = run_default_ridge_regressions(old_tasks_material,
                                                                                                  a_range=self.alpha_range)
            if print_ot_evaluations:
                print_evaluations(params_old_tasks_material, parvecs_old_tasks_material, old_tasks_material)

        # Calculating theta_bar
        theta_bar = centroid_by_points(parvecs_old_tasks_material)
        # Calculating theta_parallel
        theta_parallel, parallel_alpha = self.solve_parallel(parvecs_old_tasks_material, theta_bar, new_nshots)
        # Calculating theta_perpendicular
        theta_perpendicular, theta_task, perpendicular_alpha = self.solve_perpendicular(new_nshots, theta_bar,
                                                                                        theta_parallel)

        self.collect_cossim_measures_for_thetas(theta_bar, theta_parallel, theta_perpendicular, theta_task,
                                                print_on=print_cossim)
        nm_results, nmparameters = self.collect_non_meta_results_nshots(new_nshots, new_sets)

        meta_results = evaluate_regression_model(nmparameters, theta_task, new_sets[2], new_sets[3],
                                                 print_on=printmetares)
        results = self.log_results_dict(nmparameters, parallel_alpha, perpendicular_alpha,
                                        meta_results, nm_results, suffix_notes=resnotes, ext=save_res_to)
        reduced_results = [theta_task, results, theta_perpendicular, theta_bar]

        theta_task_rec = pca_ncomp.inverse_transform(theta_task)
        meta_results_rec = evaluate_regression_model(nmparameters, theta_task_rec,
                                                     pca_ncomp.inverse_transform(new_sets[2]), new_sets[3],
                                                     print_on=printmetares)
        results_rec = self.log_results_dict(nmparameters, parallel_alpha, perpendicular_alpha,
                                            meta_results_rec, nm_results, suffix_notes=resnotes, ext=save_res_to)
        reconstructed_results = [theta_task_rec, results_rec, pca_ncomp.inverse_transform(theta_perpendicular),
                                 pca_ncomp.inverse_transform(theta_bar)]
        return reduced_results, reconstructed_results


    def multi_layer_model(self, n_layers=2, dumpfreq=1, more_notes='',  # Do not change dumpfreq!!!! IF YOU skip layers, meta updates will be skipped too -> something to fix later
                          save_to_file=True, res_file_name='meta_results_multi',
                          learning_rate=1.0):
        if not isinstance(self.fingerprints, csr_matrix):
            raise ValueError('Something went wrong. Fingerprints must be a csr_matrix')
        assert self.new_task_name, "New task name cannot be empty"

        layers = list(range(n_layers))
        layers_to_save = layers[::dumpfreq]
        if layers[-1] not in layers_to_save:
            layers_to_save.append(layers[-1])

        t_spec_vectors = {}
        layer_metrics = {
            'MAE': [],
            'RMSE': [],
            'R_2': []
        }

        for layer in layers:
            print(f'[***] LAYER {layer} [***]')
            # here let's separate the "creation of the meta vectors for the old tasks" and "actually caclulating for the task at hand"
            # creation of the meta vectors for the old tasks
            if layer == 0:
            # let's create true 0 layer support vectors - just linear regressions (pure, no meta)
                old_tasks_data = self.process_old_tasks(new_task=self.new_task_name, old_tasks=self.old_task_names)
                orig_parvecs_old_tasks, _ = run_default_ridge_regressions(old_tasks_data, a_range=self.alpha_range)
                t_spec_vectors = {k: v for k, v in zip(self.old_task_names, orig_parvecs_old_tasks)}
                # for task in self.old_task_names:
                    # t_spec_vectors[task] = t_spec_parvector
            else:
                t_spec_vectors_temp = {}
                for task in self.old_task_names:
                    t_spec_parvector, t_spec_res, t_spec_perp, _ = self.single_layer_model(
                        new_task=task,
                        old_tasks=[pr for pr in self.old_task_names if pr != task],
                        task_vectors=t_spec_vectors,
                        resnotes=f'layer{layer}{more_notes}',
                        printmetares=False, save_res_to=None)
                    # t_spec_vectors_temp[task] = t_spec_vectors[task] + learning_rate*(t_spec_parvector - t_spec_vectors[task])
                    t_spec_vectors_temp[task] = t_spec_parvector
                t_spec_vectors = t_spec_vectors_temp

            # saving results
            if layer in layers_to_save: # TODO change to all layers to save/calc
                print(f'[__] Target Task: {self.new_task_name} | Layer {layer} is being saved [___]')

                temp_theta_task, temp_task_res, temp_task_perp, temp_task_bar = self.single_layer_model(
                    resnotes='', save_res_to=None)
                tt_shared_temp = temp_theta_task - temp_task_perp
                if layer == 0:
                    tt_shared = tt_shared_temp
                for sample_task in t_spec_vectors.values():
                    tt_shared = tt_shared - np.asarray(sample_task) * learning_rate

                new_nshots, new_sets = self.process_new_task(new_task=self.new_task_name)
                new_theta_perpendicular, new_theta_task, new_perpendicular_alpha = self.solve_perpendicular(
                    new_nshots,
                    temp_task_bar,
                    tt_shared-temp_task_bar)
                nm_results, nmparameters = self.collect_non_meta_results_nshots(new_nshots, new_sets)
                meta_results = evaluate_regression_model(nmparameters, new_theta_task, new_sets[2], new_sets[3],
                                                         print_on=False)
                temp_task_res.update({
                    'rmse': meta_results['rmse'],
                    'mae': meta_results['mae'],
                    'r_2': meta_results['r2'],
                    'notes': f'{more_notes}_layer{layer}',
                })

                if save_to_file:
                    self.save_metalearner_results_to_file(temp_task_res,
                                                          f'{res_file_name}_{self.new_task_name}_l{layer}')
                else:
                    print(temp_task_res)
                layer_metrics['MAE'].append(temp_task_res['mae'])
                layer_metrics['RMSE'].append(temp_task_res['rmse'])
                layer_metrics['R_2'].append(temp_task_res['r_2'])

        return temp_theta_task, temp_task_res, layer_metrics

    def scan_monolayers(self, res_file_name='test',
                        rnotes='test',
                        task_range=('test_task',),
                        old_tasks=None,
                        mss_range=(5, 7, 9),
                        nshots_range=(20, 50, 100),
                        rss_range=tuple(np.arange(12, 32, 1)),
                        clean_when_done=True,
                        regenerate_fp=True,
                        ortonormalize_rvectors=False,
                        with_theta_bar=False):
        for task_name in task_range:
            self.set_task_names(task_name, old_tasks=old_tasks) # old tasks will be all but the new task if None
            for size in mss_range:
                self.max_subgraph_size = size
                if regenerate_fp:
                    self.regenerate_fingerprints()
                for n in nshots_range:
                    self.n_shots = n
                    for sd in rss_range:
                        self.random_state = int(sd)
                        try:
                            if ortonormalize_rvectors:
                                _, rd, _, _ = self.single_layer_model_orthomode(resnotes=f'{rnotes}_rs{int(sd)}',
                                                                                save_res_to=None,
                                                                                theta_bar_calc=with_theta_bar)
                            else:
                                _, rd, _, _ = self.single_layer_model(resnotes=f'{rnotes}_rs{int(sd)}',
                                                                      save_res_to=None)
                            self.save_metalearner_results_to_file(rd,
                                                                  f'{res_file_name}_{task_name}_monoscan')
                        except Exception as e:
                            print(f"Skipping {int(sd)} due to exception: {str(e)}")
                            results_if_error = dict({'property': f'{task_name}',
                                                     'support_tasks': tuple(self.old_task_names),
                                                     'max_subgraph_size': size,
                                                     'solver': 'sparse_cg',
                                                     'alpha_for_new_task': 0.0,
                                                     'alpha_par': 0.0,
                                                     'alpha_perp': 0.0,
                                                     'mae': 0.0,
                                                     'rmse': 0.0,
                                                     'r_2': 0.0,
                                                     'nonmeta_mae': 0.0,
                                                     'nonmeta_mse': 0.0,
                                                     'nonmeta_r_2': 0.0,
                                                     'n_shots': n,
                                                     'notes': f'{rnotes}_rs{int(sd)}_error_msg_{str(e)}'})
                            self.save_metalearner_results_to_file(results_if_error,
                                                                  f'{res_file_name}_{task_name}_monoscan_err')
                        continue
            # now let's combine all results from this scan into a single pkl (per task)
            search_pattern = os.path.join(self.working_directory, f"{res_file_name}_{task_name}_monoscan_*.pkl")
            single_files = glob.glob(search_pattern)
            dfs = [pd.read_pickle(file) for file in single_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            today = datetime.date.today().strftime('%m%d%Y')
            scan_task_output_path = os.path.join(self.working_directory, f"{res_file_name}_{task_name}_monoscanall_{today}.pkl")
            combined_df.to_pickle(scan_task_output_path)
            print(f"Combined scanned results for task {task_name} saved!")
            if clean_when_done:
                for file in single_files:
                    os.remove(file)

    def scan_multilayers(self, res_file_name='test',
                         rnotes='test',
                         number_of_layers=2,
                         task_range=('test_task',),
                         mss_range=(5, 7, 9),
                         nshots_range=(20, 50, 100),
                         rss_range=tuple(np.arange(12, 32, 1)),
                         clean_when_done=True):
        for task_name in task_range:
            self.set_task_names(task_name, old_tasks=None) # old tasks will be all but the new task
            for size in mss_range:
                self.max_subgraph_size = size
                self.regenerate_fingerprints()
                for n in nshots_range:
                    self.n_shots = n
                    for sd in rss_range:
                        self.random_state = int(sd)
                        try:
                            _, rd = self.multi_layer_model(more_notes=f'{rnotes}_rs{int(sd)}', n_layers=number_of_layers)
                            self.save_metalearner_results_to_file(rd,
                                                                  f'{res_file_name}_{task_name}_multiscan')
                        except Exception as e:
                            print(f"Skipping {int(sd)} due to exception: {str(e)}")
                            results_if_error = dict({'property': f'{task_name}',
                                                     'max_subgraph_size': size,
                                                     'solver': 'sparse_cg',
                                                     'alpha_for_new_task': 0.0,
                                                     'alpha_par': 0.0,
                                                     'alpha_perp': 0.0,
                                                     'epsilon_par': 0.0,
                                                     'epsilon_perp': 0.0,
                                                     'mae': 0.0,
                                                     'rmse': 0.0,
                                                     'r_2': 0.0,
                                                     'nonmeta_mae': 0.0,
                                                     'nonmeta_rmse': 0.0,
                                                     'nonmeta_r_2': 0.0,
                                                     'n_shots': n,
                                                     'notes': f'{rnotes}_rs{int(sd)}_error_msg_{str(e)}'})
                            self.save_metalearner_results_to_file(results_if_error,
                                                                  f'{res_file_name}_{task_name}_multiscan_err')
                        continue
            # now let's combine all results from this scan into a single pkl (per task)
            search_pattern = os.path.join(self.working_directory, f"{res_file_name}_{task_name}_multiscan_*.pkl")
            single_files = glob.glob(search_pattern)
            dfs = [pd.read_pickle(file) for file in single_files]
            combined_df = pd.concat(dfs, ignore_index=True)
            today = datetime.date.today().strftime('%m%d%Y')
            scan_task_output_path = os.path.join(self.working_directory,
                                                 f"{res_file_name}_{task_name}_multiscanall_{today}.pkl")
            combined_df.to_pickle(scan_task_output_path)
            print(f"Combined scanned results for task {task_name} saved!")
            if clean_when_done:
                for file in single_files:
                    os.remove(file)
