#!/usr/bin/env python3
from copy import deepcopy
from types import MethodDescriptorType
from builtins import print
import os, time
import copy
import pickle
import random
from unittest import result
import scipy
from pathlib import Path

#from corati.inference import sample_particles, infer_parameters
from corati.design import Exp_designer_single, get_design_bounds
from corati.inference import infer_parameters
from corati.model_selection import sample_model_instances, instantiate_participant_model, \
	train_model_selection, assign_parameter_sample, update_model_prior

from corati.vizualization import plot_participant_convergence, plot_correlation, \
	plot_convergence, plot_model_marginals, plot_model_results, plot_designs, plot_verification, \
		plot_behavioral_fit, plot_behavioral_convergence, plot_model_convergence, create_plots_for_paper

# -------------------------------------



def generate_synthetic_participants( n, task ):
	'''
	Generate n synthetic participants which are sampled from model.
	A scalar parameter value will be sampled from all parameters which are represented as distributions with the exception of the parameter named in hyper.exception.
	The purpose of the exception is to permit the synthetic participants to be trained on a distribution of values for one parameter
	The exception should be extended to a list of parameters.
	'''

	output_dir = task.hyper['output_dir']
	result = []

	for p in range(1,n+1):
		new = instantiate_participant_model(task)

		new.pars = new.sample_model_parameters( new.pars, exclude_pars = new.hyper['design_parameters'])
		new.hyper['output_dir'] = f'{output_dir}participant_{p}/'

		train_op = getattr(new, "train", None)
		if train_op is not None:
			new.train()

		print(new.pars)
		#os.makedirs( new.hyper['output_dir'], exist_ok=True )
		result.append(new)

	os.makedirs(output_dir, exist_ok=True )
	with open(f'{output_dir}synthetic_participants', "wb") as fp:
		pickle.dump( result, fp )
	return result


# -------------------------------------

def train_multiple_models( n, model ):
	'''
	Train multiple ensemble models and store them separately.
	'''

	output_dir = model.hyper['output_dir']
	result = []
	for p in range(1,n+1):
		new = copy.deepcopy(model)
		new.hyper['output_dir'] = f'{output_dir}model_{p}/'
		new.train()
		print(new.pars)
		#os.makedirs( new.hyper['output_dir'], exist_ok=True )
		result.append(new)

	with open(f'{output_dir}trained_models', "wb") as fp:
		pickle.dump( result, fp )
	return result

# -------------------------------------

import numpy as np
import pandas as pd

def corati( task, n_participants ):
	'''
	Conduct model selection (or/and parameter inference) for stored participants
	and store preformance information for plotting and evaluating the experiments. 
	'''
	# we assume that models are already trained and synthetic participants generated.
	# load the pretrained synthetic participants.
	output_dir = task.hyper['output_dir']
	with open(f'{output_dir}synthetic_participants', "rb") as fp:
		participants = pickle.load(fp)
	
	# initialize variables for recodring performance
	resulting_posteriors, all_estimates, all_ground_truth, all_model_choices, true_models = [], [], [], [], []
	rand_model_verifs, true_model_verifs, est_model_verifs = [], [], []
	model_variance, times, dists = [], [], []
	model_posteriors, all_designs, fitness_dists, true_models, estimate_trajectories, posterior_trajectories = [], [], [], [], [], [] 
	participant_shift = task.hyper['participant_shift']
	
	# for each participant, we do: 1) select design; 2) conduct an experiment; and 3) update model parameters.
	for p in range(participant_shift-1, n_participants+participant_shift-1): # list([7]): #
		print(f'\n\nAnalyse participant {p+1} of {n_participants+participant_shift-1}: {participants[p].name}, ' + \
			str([participants[p].pars[key]['value'] for key in participants[p].parameters_of_interest]))
		
		# record the start time of the algorithm
		start = time.time() 
		
		# conduct full model selection for one participant
		posterior, estimate, model_choices, estimate_log, model_post, designs, fitness_dist, particles_history = design_conduct_infer( task, participants[p], participants) 
		
		# the rest of the loop is processing and storing performance information:
		fitness_dists.append(fitness_dist)
		all_designs.append(designs)
		if len(task.model_priors) > 1:
			model_posteriors.append(model_post)

		# verify the estimated model against the model with random parameters
		estimated_model = task.instantiate_model(model_choices[-1])
		estimated_pars = copy.deepcopy( estimated_model.pars)
		for key in estimated_model.parameters_of_interest:
			estimated_pars[key]['value'] = estimate[key]
		rand_model_verif, true_model_verif, est_model_verif = verify_model(task, participants[p], estimated_pars, model_choices[-1])
		rand_model_verifs.append(rand_model_verif)
		true_model_verifs.append(true_model_verif)
		est_model_verifs.append(est_model_verif)
		
		# store the history of estimates and particles (beliefs about parameters and models)
		temp_trajectory = []
		for model_choice, est in zip(model_choices, estimate_log):
			temp_model_pars = task.instantiate_model(model_choice).parameters_of_interest
			temp_trajectory.append([temp_model_pars, [est[key] for key in temp_model_pars]] )
		estimate_trajectories.append(temp_trajectory)
		posterior_trajectories.append(particles_history)
		
		# store inference information
		resulting_posteriors.append( posterior )
		parameter_names = [x for x in list(posterior[model_choices[-1]].keys()) if x not in task.design_parameters ]
		model_variance.append( [posterior[model_choices[-1]][parameter]['value'].var() for parameter in parameter_names ]) # what type is it? Pandas.Dataframe
		all_estimates.append( [estimate[key] for key in estimated_model.parameters_of_interest ] )
		all_ground_truth.append( [participants[p].parameters_of_interest, [participants[p].pars[key]['value'] for key in participants[p].parameters_of_interest]] )
		all_model_choices.append( model_choices)
		true_models.append( participants[p].name)
		times.append((time.time() - start) / 60.)

		# preparing the loss in case of model selection
		temp_dist = list()
		for model_choice, estimate in zip(model_choices, estimate_log):
			if model_choice != true_models[-1]:
				# assign loss to 1. if the estimated model is wrong (this value is not used anywhere, but here to keep the loss variable as a float array)
				temp_dist.append(1.)
			else:
				# calculate Euclidian distance between estimated vs true parameters
				bounds = [ participants[p].pars[key]['bounds'] for key in participants[p].parameters_of_interest ]
				cur_estimate = [estimate[key] for key in participants[p].parameters_of_interest] 
				distance = np.linalg.norm((np.asarray(cur_estimate) - np.asarray(all_ground_truth[-1][1]))/ (np.max(bounds) - np.min(bounds)) )
				temp_dist.append(distance)
		
		# making sure that the loss array is consistent with the allocated design budget
		if len(temp_dist) != task.hyper['corati_budget']:
			temp_dist =  temp_dist + [None] * ( task.hyper['corati_budget'] - len(dists)) 
		dists.append(temp_dist)
	
	# plot this only when there is one single model
	if len(task.model_priors.keys()) == 1:
		bounds = [ participants[-1].pars[key]['bounds'] for key in participants[-1].parameters_of_interest ]
		# plot_correlation(all_estimates, all_ground_truth, dists, output_dir, participants[-1].parameters_of_interest, bounds) 
	else:
		true_models = [participant.name for participant in participants]
		# plot_model_results([model_post[-1] for model_post in model_posteriors], true_models, output_dir)
	
	# store information about design parameters
	if participants[-1].hyper['design_parameters']:
		design_name = participants[-1].hyper['design_parameters'][-1]
		design_bounds = participants[-1].pars[design_name]['bounds']
		# plot_designs(all_designs, design_bounds, output_dir)
		
	# plot_convergence( dists, output_dir)
	# plot_behavioral_fit(rand_model_verifs, true_model_verifs, est_model_verifs, output_dir)
	# plot_behavioral_convergence(fitness_dists, output_dir)
	print('\nEstimates: ', all_estimates, '\nGround truth: ', all_ground_truth)

	# save performance data for vizualization
	save_file_name = task.hyper['mat_file_name'] + '-' + str(task.hyper['participant_shift']) \
		+ '-' + str(n_participants+participant_shift) + '.mat'
	mdict = {
		'rand_model_fitness': rand_model_verifs, 
		'true_model_fitness': true_model_verifs,
		'est_model_fitness':  est_model_verifs,
		'fitness_trajectories': fitness_dists,
		'dist_trajectories':	dists,
		'true_models': 			true_models,	
		'model_posterior_keys': list(task.model_priors.keys()) ,
		'model_posterior': 		[[list( x.values() ) for x in y] for y in model_posteriors ],
		'model_variance': 		model_variance,
		'all_designs':			all_designs,
		'final_estimates':		all_estimates,
		'all_estimates': 		estimate_trajectories, 
		'all_ground_truth':		all_ground_truth,
		'all_model_choices':		all_model_choices,
		'posterior_trajectories':	posterior_trajectories,
		'time': 			times
		}
	scipy.io.savemat(save_file_name, mdict=mdict)
	return resulting_posteriors


import torch
from corati.inference import init_minebed, find_kde_peak


def design_conduct_infer( task, participant, participants):
	'''
	Conduct model selection for a single participant: 1) select design; 2) conduct an experiment; and 3) update model parameters.
	'''
	# get initial particle set from priors
	model_posterior = task.model_priors 
	
	# we allocate 5000 particles for each model
	num_particles = 5000. * len(model_posterior)
	
	# get initial particle set from priors
	parameter_particles = task.sample_particles(N=num_particles, plot_dir=participant.hyper['output_dir'] )
	# plot_model_marginals(model_posterior, participant.hyper['output_dir'], 0)

	# if MINEBED method is chosen, train a MI surrogate for each of the models 
	if task.hyper['minebed'] == True:
		numpy_prior, minebed_solvers = {}, {}

		for key in list(model_posterior.keys()):
			model = task.instantiate_model(key)
			minebed_solvers[key] = init_minebed(model, task, parameter_particles, key)
			print('Finished training the MINE surrogate for the model: ', key)

	# assign uniform weights to samples from the prior
	weights = dict()
	for model_name in task.model_priors:
		weights[model_name] = [ 1. / num_particles ] * int(num_particles / len(model_posterior))
	
	# initialize variables for performance metrics
	model_posteriors, model_choices, estimates, designs, fitness_dist, particles_history = [], [], [], [], [], []
	participant.hyper['n_prediction_trials'] = task.hyper['n_prediction_trials']
	budget = task.hyper['corati_budget']
	
	# proceed with model selection until simulation budget is exhausted
	for i in range(budget):
		print(f'\nIteration {i} ===')
		
		# 1: select the design for the experiment
		if task.hyper['minebed'] == True:
			model_choice = random.choices(list(model_posterior.keys()), weights=model_posterior.values(), k=1)[0] # max( model_posterior, key=model_posterior.get )
			expt = minebed_solvers[model_choice].d_opt
			minebed_solvers[model_choice].train_final_model(n_epoch=1000) # , batch_size=DATASIZE)
		elif task.hyper['random_design'] == False:
			expt = design_experiment( task, parameter_particles, model_posterior, weights) #, participant)
		else:
			# TODO: the random designs for the risky choice and sample examples, not automatic
			expt = np.random.uniform(0, 1., size=4) if len(task.design_parameters) == 4 else [np.random.uniform(0, 100.)]
		designs.append(expt)
		
		# 2: collect the data at the selected design location expt
		data = conduct_experiment( participant, expt )
		
		# corrupt observed data with noise for testing purposes
		# k = 0.4 # specify the percentage of noise
		# data = corrupt_data(participants, participant, data, k, expt, random=False)
		
		# 3: update current beliefs about the models and parameters
		estimate = dict()
		for key in list(model_posterior.keys()):
			# skip the updates if a model has a very low probability for computational reasons
			if model_posterior[key] < 0.02:
				first_key = list(parameter_particles[key].keys())[0]
				N = len( parameter_particles[key][first_key]['value'])
				weights[key] = [0] * N
				continue
			
			# prepare and conduct inference for one model
			model = task.instantiate_model(key)
			
			# if MINEBED method is chosen, it has a separate API for parameter inference
			if task.hyper['minebed'] == True:
				params = parameter_particles[key]
				numpy_prior[key] = np.array([params[key]['value'].to_numpy() for key in sorted(params.keys())]).transpose()

				# prepare parameters
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
				X = torch.tensor(numpy_prior[key], dtype=torch.float, device=device)
				
				par_stds = []
				for par_key in sorted(model.parameters_of_interest):
					par_std =  (parameter_particles[key][par_key]['bounds'][1] - parameter_particles[key][par_key]['bounds'][0]) / 1e5
					par_stds.append(par_std)
					
				# add small random noise
				X = X + torch.normal( mean=0., std=torch.Tensor(par_stds))
				
				# prepare responses
				Y = torch.cat(len(X)*[ torch.tensor(model.get_summary_statistics(data)) ]).type(dtype=torch.float)
				Y = torch.unsqueeze(Y, 1)
				Y.to(device)
				X.to(device)
				
				# assign weights to parameter-response pairs
				weights[key] = minebed_solvers[key].model(X, Y).data.numpy().reshape(-1) + 1e-10
				weights[key][weights[key] < 0] = 1e-10
				n_weights = weights[key] / np.sum(weights[key])
				N = len(n_weights)
				
				# use weights to resample particles
				resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
				resampled_theta = copy.deepcopy(X[resample_index,:])
				theta_posterior = pd.DataFrame.from_records(resampled_theta, columns =sorted(params.keys()))
				
				# estimate parameters
				minloc = find_kde_peak(theta_posterior)
				estimate[key] = {key : x for x, key in zip(minloc, sorted(model.parameters_of_interest))}
			else:
				# if true likelihood is given use it
				true_lik = task.get_likelihood if task.hyper['true_likelihood'] else None
				
				# assign budget to 1 if true likelihood is used (one simulation is required to get the format of responses automatically)
				budget = 1 if task.hyper['true_likelihood'] else task.hyper['inference_budget']
				print(key, parameter_particles[key])
				
				# run parameter inference procedure (BOLFI is used if true likelihood is not used) 
				parameter_particles[key], estimate[key], weights[key] = infer_parameters( data, model, parameter_particles[key], expt, participant, 
											true_lik=true_lik, budget=budget, it=i+1) # Alex prior=prior[cur_model_choice]
		
		# if we have more than one model, then we also do model selection		
		if len(model_posterior) > 1:
			# resample beliefs according to their likelihood approximations (weights)
			parameter_particles, model_posterior, done, weights = resample_particles(parameter_particles, weights)
			
			# if believes converged, apply the decision rule to select the model
			if not done:
				model_posteriors.append( model_posterior )
				model_sel_rule = task.hyper['model_sel_rule'] 
				
				if model_sel_rule == 'map':
					# MAP estimation
					model_choice = max( model_posterior, key=model_posterior.get )
				elif model_sel_rule == 'bic':
					# BIC estimation: BIC = -2 * LL + log(N) * k, where N is the number of examples in the training dataset, and k is the number of parameters in the model
					bic_score = dict()
					for key in model_posterior:
						num_of_pars = len(task.instantiate_model(key).parameters_of_interest)
						bic_score[key] = -2 * np.log(model_posterior[key]) + np.log(i) * num_of_pars
					model_choice = min( bic_score, key=bic_score.get ) 
				
				# store selected model
				model_choices.append( model_choice )
				# plot_model_marginals( model_posterior, participant.hyper['output_dir'], i+1 )
			else:
				break
		else: 
			parameter_particles, _, done = resample_particles(parameter_particles, weights)
		
		# to save memory, we store posteriors only for the steps that we will use for vizualization
		if i in [0, 1, 3, 19, 99] and task.hyper['record_posterior']:
			print('Record history')
			particle_set = copy.deepcopy(parameter_particles)
			formatted_history = []
			for model_key in particle_set:
				for param_key in particle_set[model_key]:
					formatted_history.append([param_key, particle_set[model_key][param_key]['value'].to_numpy()])
			particles_history.append(formatted_history)
			
		# if MINEBED is chosen, retrain the method for each model with the new beliefs
		if task.hyper['minebed'] == True:
			for key in list(model_posterior.keys()):
				# skip training if the model is very unlikely
				if model_posterior[key] < 0.02:
					continue
				else:
					model = task.instantiate_model(key)
					minebed_solvers[key] = init_minebed(model, task, parameter_particles, key)
		
		# evaluate the estimated model vs random model and vs true model
		estimated_model = task.instantiate_model(model_choice)
		estimated_pars = copy.deepcopy( estimated_model.pars)
		for key in estimated_model.parameters_of_interest:
			estimated_pars[key]['value'] = estimate[model_choice][key]		
		random_par_dist, true_par_dist, sel_par_dist = verify_model(task, participant, estimated_pars, model_choice)
		
		# store evaluation
		fitness_dist.append([random_par_dist, true_par_dist, sel_par_dist])
		estimates.append( estimate[model_choice] )
			
	print('Model choices: ', model_choices)
	print('Model posteriors: ', model_posteriors)
	# plot_participant_convergence(estimates, participant)

	if len(model_posterior) > 1:
		# plot_model_convergence(model_posteriors, participant)
		pass

	return parameter_particles, estimates[-1], model_choices, estimates, model_posteriors, designs, fitness_dist, particles_history


# -------------------------------------

def corrupt_data(participants, true_model, data, k, expt, random=False):
	'''
	Corrupts data with noise, where k regulates how much data needs to be corrupted.
	WARNING: this function was used in early stages of development and was not tested 
	for cases other than signal_detection.
	'''
	print(f'Corrupt {k*100}% of data...')
	trial_num = len(data)
	corrupt_num = int(trial_num * k)
	corrupt_data = []

	if random:
		# generates random data according to the response space bounds
		rng = np.random.default_rng()
		steps = rng.integers(1, 11, size=(corrupt_num, 1)).flatten()
		correct = rng.integers(0, 2, size=(corrupt_num, 1)).flatten()
		done = np.array([[True]] * corrupt_num).flatten()
		corrupt_data = pd.DataFrame(data=np.array([steps, correct, done]).transpose(), columns=['steps', 'correct', 'done'])
	else:	
		# generate noise from alternative participant models
		for participant in participants:
			if participant != true_model:
				participant.hyper['n_prediction_trials'] = corrupt_num
				corrupt_data_from_model = conduct_experiment( participant, expt ) 
				corrupt_data.append(corrupt_data_from_model)
		corrupt_data = pd.concat(corrupt_data).sample(n=corrupt_num)
		
	# apply corruption
	data = data.sample(frac=(1 - k))
	data = pd.concat([data, corrupt_data]).sample(frac=1.)
	return data


# -------------------------------------

def resample_particles(parameter_particles, weights):
	'''
	Resample particles in parameter_particles according to weights
	'''
	done = False
	index_bounds = dict()
	concat_weights = []
	total_size = 0

	# concatanate weights, determine index bounds for each model
	model_posterior = dict()
	model_names = list(weights.keys())
	cur_index = 0
	for model_name in model_names:
		prev_index = cur_index
		cur_index = prev_index + len(weights[model_name])
		# model_posterior[model_name] = np.sum(weights[model_name])
		index_bounds[model_name] = (prev_index, cur_index-1)

		total_size += len(weights[model_name])
		concat_weights = np.hstack((concat_weights, weights[model_name]))

	# normalize weights, sample indexes
	concat_weights = concat_weights.flatten()
	if np.sum(concat_weights) == 0:
		concat_weights += 1e-10
		done = True
	n_weights = concat_weights / np.sum(concat_weights)
	resample_indexes = np.random.choice(total_size, size=total_size, replace=True, p=n_weights)
	
	for model_name in model_names:
		# leave only indexes inside the model bounds
		resample_indexes_for_model = resample_indexes[[ np.where( (resample_indexes >= index_bounds[model_name][0]) & \
			(resample_indexes <= index_bounds[model_name][1]))]]  - index_bounds[model_name][0]
		resample_indexes_for_model = resample_indexes_for_model.flatten() 

		# the model posterior is proprtional to the model entries in the joint posterior
		model_posterior[model_name] = len(resample_indexes_for_model)
		weights[model_name] = np.array(weights[model_name])[resample_indexes_for_model.astype(int)]

		# for each parameter resample particles
		for model_key in list(parameter_particles[model_name].keys()):
			if isinstance(parameter_particles[model_name][model_key]['value'], pd.Series):
				parameter_particles[model_name][model_key]['value'] = parameter_particles[model_name][model_key]['value'].take(resample_indexes_for_model) # .reset_index()
				print(model_name, len(parameter_particles[model_name][model_key]['value']))
	
	# normalize the model posterior
	normalizing_constant = sum(list(model_posterior.values()))
	if normalizing_constant == 0:
		normalizing_constant += 1e-10
	for model_name in model_names:
		model_posterior[model_name] /= normalizing_constant
	
	print('Model posterior: ', model_posterior)
	return parameter_particles, model_posterior, done, weights


# -------------------------------------

def design_experiment( task, parameter_particles, model_priors, weights=None, participant=None):
	'''
	Run design optimization procedure given current beliefs
	'''
	# get hyperparameters for initializing design optimization
	init_sample = task.hyper['init_design_samples']
	max_iter = task.hyper['max_design_iterations']
	batch_size = task.hyper['design_batch_size']
	num_models = task.hyper['num_design_models']
	
	# if participant if given, run additional sensitivity analysis
	if participant is None:
		models = sample_model_instances( task, parameter_particles, model_priors, num_models)
	else: 
		keys = list(model_priors.keys())
		probs = list(model_priors.values())

		models = []
		for _ in range(num_models):
			models.append(copy.deepcopy(participant))
		
		for i in range(num_models):
			model_name = np.random.choice(keys, 1, p=probs)[0]
			models[i].pars['signal_level']['value'] = (np.random.uniform, [0., 4.])

			if i != -1:
				models[i].pars = assign_parameter_sample(models[i], parameter_particles[model_name])

	print('Designing expereiments...')
	bounds = get_design_bounds(models[0])
	
	# conduct design optimization according with the specified method
	if task.hyper['ado']:
		print('Using ADO...')
		new_parameter_particles = deepcopy(parameter_particles)
		new_model_priors = deepcopy(model_priors)
		for model_name in model_priors:
			if model_priors[model_name] == 0:
				del new_model_priors[model_name]
			for par_name in parameter_particles[model_name]:
				if par_name in task.design_parameters:
					del new_parameter_particles[model_name][par_name]

		designer = Exp_designer_single( init_sample, max_iter, batch_size, bounds, \
			ado=True, likelihood_func=task.get_likelihood, instantiate_func=task.instantiate_model, \
				model_priors=new_model_priors, parameter_particles=new_parameter_particles, weights=weights)
	else:
		print('Using the default method...')
		designer = Exp_designer_single( init_sample, max_iter, batch_size, bounds)
		
	# extract the design location
	expt = designer.design_experiment( models ) 
	return expt



# -------------------------------------

def conduct_experiment( model, expt, verbose=True ):
	'''
	Conduct the experiments using the synthetic participant (model) at the design location expt
	'''
	if verbose:
		print(f'Conduct experiment {expt}.')
	output_dir = model.hyper['output_dir']
	# set parameters for expt
	keys = model.hyper['design_parameters'] 
	
	# set model parameters
	trace = model.predict_one(keys, expt) 
	return trace
	

# -------------------------------------

def verify( task, n_participants ):
	'''
	Evaluate the prior predictive for the task and stored participants
	'''
	# load synthetic participants
	output_dir = task.hyper['output_dir']
	with open(f'{output_dir}synthetic_participants', "rb") as fp:
		participants = pickle.load(fp)
		
	# sample designs from the proposal distirbution
	expts = [] # np.linspace(0, 4, 10).tolist()
	design_names = participants[-1].hyper['design_parameters']
	for _ in range(100): 
		design_values = []
		for name in design_names:
			func = participants[-1].pars[name]['value'][0]
			inputs = participants[-1].pars[name]['value'][1]
			design_values.append(func(*inputs))
		expts.append(design_values)
	
	random_par_dist = []
	random_beh_dist = []
	
	# evaluate all synthetic participants
	for i, participant in zip(range(len(participants)), participants):
		true_parameters = [participant.pars[key]['value'] for key in participant.parameters_of_interest]
		bounds = [ participant.pars[key]['bounds'] for key in participant.parameters_of_interest ]
		
		# evaluate for all sampled designs		
		for expt in expts: 
			# instantiate a model and sample parameters
			model_name = participant.name
			random_par_model = task.instantiate_model(model_name)
			random_par_model.pars = random_par_model.pars_sample 
		
			# get participant data
			data_part = conduct_experiment( participant, expt, False )
			data_part = participant.get_summary_statistics(data_part)
			
			# get data from random model	
			random_par_data = conduct_experiment( random_par_model, expt, False )
			random_par_data = random_par_model.get_summary_statistics(random_par_data) 
			
			# calculate behavioural fitness
			random_beh_dist.append(np.linalg.norm(np.asarray(data_part)-np.asarray(random_par_data)))
			
			# calculate parameter distance
			random_estimate = [random_par_model.pars[key]['value'] for key in participant.parameters_of_interest] 
			random_par_dist.append( np.linalg.norm((np.asarray(random_estimate) - np.asarray(true_parameters)) / (np.max(bounds) - np.min(bounds))) )
				
	print(f'Behavioural distance: {np.mean(random_beh_dist)} $pm$ {np.std(random_beh_dist)} ' )
	print(f'Parameter distance: {np.mean(random_par_dist)} $pm$ {np.std(random_par_dist)} ' )


# -------------------------------------

def verify_model(task, participant, pars=None, chosen_model=None):
	'''
	Verify the estimated model vs random model and vs ground-truth model
	'''
	steps=1 # task.hyper['verification_budget']
	random_par_dist = []
	true_par_dist = []
	sel_par_dist = []
	true_model = participant.name
	true_parameters = [participant.pars[key]['value'] for key in participant.parameters_of_interest]
	chosen_model = chosen_model if chosen_model else participant.name
	model = task.instantiate_model(participant.name)
	
	# sample designs from the proposal distirbution
	expts = [] 
	design_names = model.hyper['design_parameters']
	for _ in range(steps): 
		design_values = []
		for name in design_names:
			func = model.pars[name]['value'][0]
			inputs = model.pars[name]['value'][1]
			design_values.append(func(*inputs))
		expts.append(design_values)
	
	print(f'\nVerifying the chosen model "{chosen_model}" compared to the true model "{true_model}" with parameters {true_parameters}')
	
	# steps samples are used for verification
	for _ in range(steps):
		# prepare the next design to conduct experiment with the synthetic participant model
		expt = expts.pop()
		data_part = conduct_experiment( participant, expt, False )
		data_part = participant.get_summary_statistics(data_part)

		# conduct experiments with parameters sampled from the prior
		random_model_name = np.random.choice(list(task.model_priors.keys()), 1, p=list(task.model_priors.values()))[0]
		random_model = task.instantiate_model(random_model_name)
		random_model.pars = random_model.pars_sample 
		random_par_data = conduct_experiment( random_model, expt, False )
		random_par_data = random_model.get_summary_statistics(random_par_data) 
		random_par_dist.append(np.linalg.norm(np.asarray(data_part)-np.asarray(random_par_data)))
		
		# conduct experimetns with true parameters of the synthetic participant
		true_par_model = task.instantiate_model(true_model)
		true_par_model.update_model_parameters( copy.deepcopy(participant.pars) )
		true_par_data = conduct_experiment( true_par_model, expt, False )
		true_par_data = true_par_model.get_summary_statistics(true_par_data)
		true_par_dist.append(np.linalg.norm(np.asarray(data_part)-np.asarray(true_par_data)))
		
		# if parameters are provided, use them to conduct experiments
		if pars is not None:
			sel_par_model = task.instantiate_model(chosen_model) 
			sel_par_model.update_model_parameters(pars)
			sel_par_data = conduct_experiment( sel_par_model, expt, False )
			sel_par_data = sel_par_model.get_summary_statistics(sel_par_data)
			sel_par_dist.append(np.linalg.norm(np.asarray(data_part)-np.asarray(sel_par_data)))

	print(f'Mean distance for participant (random_par): {np.mean(random_par_dist)}' )
	print(f'Mean distance for participant (true_par): {np.mean(true_par_dist)}' )

	if pars is not None:
		print(f'Mean distance for participant (sel_par): {np.mean(sel_par_dist)}' )
	# plot_verification(random_par_dist, true_par_dist, sel_par_dist, participant.hyper['output_dir'])
	return np.mean(random_par_dist), np.mean(true_par_dist), np.mean(sel_par_dist)



def verify_saved_models(task, all_dict):
	'''
	Calculate behavioral fitness error for the stored estimated models
	'''
	# load synthetic participants
	output_dir = task.hyper['output_dir']
	with open(f'{output_dir}synthetic_participants', "rb") as fp:
		participants = pickle.load(fp)
		
	# sample designs from the proposal distirbution
	expts = [] 
	model = participants[0]
	design_names = model.hyper['design_parameters']
	for _ in range(100): 
		design_values = []
		for name in design_names:
			func = model.pars[name]['value'][0]
			inputs = model.pars[name]['value'][1]
			design_values.append(func(*inputs))
		expts.append(design_values)
	
	trials = 100
	random_par_dist, true_par_dist, sel_par_dist = [], [], []
	table_ticks = [0, 1, 3, 19, 99]
	
	# for all stored estimated models
	for filename in all_dict:
		print(filename)
		if filename != 'minebed_bic': # and filename != 'sp' :
			continue
			
		# prepare estimated models and their parameters
		all_model_choices = all_dict[filename]['all_model_choices'][1:]
		all_estimates = all_dict[filename]['all_estimates'][1:]
		
		# prepare participants (make sure that their number is equal to the number of estimated models
		participants = participants[:np.shape(all_model_choices)[0]]
		
		# for all participants do
		sel_par_dist = []
		for i, participant in zip(range(len(participants)), participants):
			# prepare the next design to conduct experiment for the synthetic participant model
			sel_par_dist.append([])
			for expt in expts: 
				# get observed data
				participant.hyper['n_prediction_trials'] = trials
				participant.max_episode_steps = 10
				data_part = conduct_experiment( participant, expt, False )
				data_part = participant.get_summary_statistics(data_part)
					.
				j = -1
				sel_par_dist[-1].append([])
				for model_choice, estimate in zip(all_model_choices[i], all_estimates[i]):
					j += 1
					if not(j in table_ticks):
						continue
					
					# get data from the estimated model
					sel_par_model = task.instantiate_model(model_choice.strip())	
					keys, values =  np.array(estimate)[0].flatten(), np.array(estimate)[1].flatten()
					pars = { key.strip(): {'value': float(value)} for key, value in zip(keys, values)}
					sel_par_model.update_model_parameters(pars, keys=pars.keys())
					sel_par_model.hyper['n_prediction_trials'] = trials
					sel_par_model.max_episode_steps = 10
					
					sel_par_data = conduct_experiment( sel_par_model, expt, False )
					sel_par_data = sel_par_model.get_summary_statistics(sel_par_data)
					sel_par_dist[-1][-1].append(np.linalg.norm(np.asarray(data_part)-np.asarray(sel_par_data)))
			sel_par_dist[-1] = np.mean(sel_par_dist[-1], axis=0)
				
		print('Mean distance for participant:', np.shape(sel_par_dist))
		mean, std = np.mean(sel_par_dist, axis=0), np.std(sel_par_dist, axis=0)
		print( [r"{mean:.2f} $\pm$ {std:.2f}".format(mean=m, std=s) for item_id, m, s in zip(range(len(mean)), mean, std)])
		print(f'Mean distance for participant: {np.mean(sel_par_dist)} $pm$ {np.std(sel_par_dist)} ' )

# -------------------------------------

def switch(model, x='empty'):
	'''
	switch() is the main function for corati. 
	It takes a model object as a parameter. The model parameters are described in the model class.

	Multiple workflows are possible with corati. Usually the workflow is likely to involved something like the following steps:
		1. define a model (an instance of the model class)
		2. use switch to explore() the model and check that it is behaving as expected given random action selections. If the behaviour is not what is wanted then go back to step 1.
		3. train the model with parameter n_training_timesteps set to something small (e.g. 1e4). If training fails with an error then go back to 1.
		4. train on a larger number of training timesteps (e.g. 1e6). If there is a crash or max_episode_steps is exceeded to often then return to 1.
		5. examine the file learning_curve.png. If it has not converged then return to 4 and adjust hyper parameters accordingly or return to 1.
		6. call predict(). This will make use of the learned policy to generate predictions.csv.
		7. Inspect the predictions.csv file to make sure that the output is as required. 
		8. 
	switch() accepts the following user input:

	RET - explore() the model. This is used fro debugging. It provides an interactive report on the state of the model after each step in a single episode with random action generation. (It does not use the trained policy.) Replacing model's get_info function can be used to see different information.
	t - train() the model. Output is a a set of files that include a plot of the learning curve, the learned policy and a dump of the parameter values.
	p - predict(). Once the model has been trained (above) it can be used to generate predictions. Predictions are stepwise traces of behaviour stored in the file predictions.csv. Each row of the file corresponds to an action taken by the agent. The columns are determined by the model method get_info().
	l - plot(). calls a user defined plot() method that typically will take data from predictions.csv
	tpl - can be called to run the above three methods in sequence.
	o - open the plots (if they are in the correct folder).
	g - generate_synthetic_participants(). N participant models are trained and their policies and learning curves stored in folders 1...N. 
	c - corati(). This calls the corati main loop with the assumption that the model has been already been trained (command t) and that there are participant files (e.g.  generated with g). For each participant, the main loop designs an experiment, conducts the experiment, and makes inferences to adjust the model parameters. 
	q - quit.
	'''
	n_participants = model.hyper['n_participants']
	start_time = time.time()

	# This is main() for model files to call.
	if x == 'empty':
		input_flag = True
	else:
		input_flag = False

	while True:
		print("")
		print("")
		print(f'RET  	explore')
		print(f't 		train')
		print(f'v		verify')
		print(f'm		train_multiple_models')
		print(f'p 		predict')
		print(f'l 		plot')
		print(f'tpl		train predict plot')
		print(f'o 		open plots')
		print(f'g 		generate synthethetic participants')
		print(f'c 		corati')
		print(f'q 		quit')
		print("")

		if input_flag == True:
			x = input('Next? ')

		if x == 't':
			model.train()
		elif x == 'p':
			model.predict()
		elif x == 'tpl':
			model.train()
			model.predict()
			model.plot()
		elif x == 'm':
			train_multiple_models(10, model)
		elif x == 'l':
			methods = ['sp', 'true_lik', 'sp_rand', 'true_lik_rand', 'ado', 'minebed', 'sp_bic', 'minebed_bic'] 

			# load saved models and generate plots / evaluate models
			cwd = os.path.dirname(os.path.realpath(__file__))
			paths = Path('').glob('*.mat')
			sorted_paths = sorted(paths, key = lambda x: int(os.path.basename(x).split("-")[1]))
			all_files = {}
			
			# aggregate all stored information about performance of models in one dictionary
			vstack_keys = ['fitness_trajectories', 'model_posterior', 'dist_trajectories', 'all_designs', 'all_model_choices']
			for filename in sorted_paths:
				print(filename)
				f = scipy.io.loadmat(filename)
				for meth in methods:
					if meth + '-' in str(filename): 
						if meth in all_files:
							for key in all_files[meth]:
								# print(filename, meth, key, np.shape(all_files[meth][key]), np.shape(f[key]))
								if key == 'model_posterior_keys' or  key == 'model_variance':
									continue
								elif key in vstack_keys:
									all_files[meth][key] = np.vstack((all_files[meth][key], f[key])) # np.concatenate((all_files[meth][key], f[key]), axis=1)
								elif key == 'true_models':
									pos = str(filename).split('.')[0].split('-')[1:3]
									# print(pos)
									temp = f[key][int(pos[0]):int(pos[1])]
									all_files[meth][key] = np.hstack((all_files[meth][key], temp)) 
								else:
									if key == 'all_estimates' or key == 'final_estimates' or key == 'all_ground_truth':
										all_files[meth][key] = all_files[meth][key] if type(all_files[meth][key]) == list else all_files[meth][key].tolist()
										f[key] = f[key].tolist()
										all_files[meth][key] += f[key]
									else:
										all_files[meth][key] = np.hstack((all_files[meth][key], f[key]))
									# print( np.shape(all_files[meth][key]), np.shape(f[key]))
						else:
							all_files[meth] = f
							del all_files[meth]['__header__']
							del all_files[meth]['__version__']
							del all_files[meth]['__globals__']
			print(Path('').cwd())
			plot = False
			# either plot or calculate behavioural fitness
			if plot:
				create_plots_for_paper(all_files, output_dir=str(Path('').cwd()) + '/' )
			else:
				verify_saved_models(model, all_files)
		elif x == 'g':
			generate_synthetic_participants( n_participants, model )
		elif x == 'c':
			corati(model, n_participants)
			print(f'Running time: { (time.time() - start_time) / 3600. } hours' )
			start_time = time.time()
		elif x == 'o':
			output_dir = model.hyper['output_dir']
			os.system(f'open {output_dir}model/png/*.png')
		elif x == 'v':
			verify(model, n_participants)
		elif x == 'q':
			input_flag = False
		else:
			model.explore()

		if input_flag == False:
			break

		
