#!/usr/bin/env python3
from builtins import float
import numpy as np
import math
import pandas as pd
import matplotlib.backends.backend_pdf
from scipy.stats import norm
import os

from copy import deepcopy
from corati import cognitive_theory as ct
from corati import synthetic_psychologist as sp
from corati.base import BaseTask, BaseModel
from corati import test
# -------------------------------------


THIS_FILE = __file__


# -------------------------------------

class ProbabilityRatio(BaseModel):
	def __init__(self) -> None:
		hit_bounds = [1,7]
		signal_level_bounds = [0,4]
		observ_limit_bounds = [2, 10]
		sigma_bounds = [0.1,1]
		
		lower_thr_bounds = [0, 5]
		thr_gap_bounds = [0, 5]
		self.pars = {		
			'sensor_noise': 		{'value':(np.random.uniform, sigma_bounds), 'bounds':sigma_bounds, 'levels':1},
			'hit': 					{'value':(np.random.uniform, hit_bounds), 'bounds':hit_bounds, 'levels':5},
			'lower_thr':			{'value':(np.random.uniform, lower_thr_bounds), 'bounds': lower_thr_bounds, 'levels': 5},
			'thr_gap':				{'value':(np.random.uniform, thr_gap_bounds), 'bounds': thr_gap_bounds, 'levels': 5},
			'miss': 				{'value':-1, 'bounds':[-1,-1], 'levels':1 },
			'false_alarm': 			{'value':-1, 'bounds':[-1,-1], 'levels':1 },
			'correct_rejection': 	{'value':2, 'bounds':[2,2], 'levels':1 },
			'observ_limit':			{'value': (np.random.uniform, observ_limit_bounds), 'bounds': observ_limit_bounds,  'levels':20 },
			'signal_level':		 	{'value':(np.random.uniform, signal_level_bounds), 'bounds':signal_level_bounds, 'levels':20},
			'p_signal':			 	{'value':0.5, 'bounds':[0.5,0.5], 'levels':1 }
		}

		file_dir, _ = os.path.split(os.path.realpath(__file__))
		self.hyper = {
			'output_dir':  			file_dir + '/output/',
			'design_parameters': 	['observ_limit', 'signal_level'],
			'n_prediction_trials':	1
		}

		self.max_episode_steps = 50
		
		# This is an important distinction compared to the original experiments;
		# we do not allow inference on two parameters, which makes inference model misspecified
		self.parameters_of_interest = ['hit', 'sensor_noise', 'lower_thr', 'thr_gap']
		self.name = 'ProbabilityRatio'
		self.misspecify = False
		super().__init__()


	def predict_one(self, keys=None, v=None):
		if keys is not None and v is not None: 
			i = 0
			for k in keys:
				self.pars[k]['value'] = v[i]
				i += 1

		num_trials_per_d = self.hyper['n_prediction_trials']
		experiment_outcomes = []

		# parameters of the model
		sensor_noise = self.pars['sensor_noise']['value']
		max_episode_steps = self.pars['observ_limit']['value']
		hit = self.pars['hit']['value']
		lower_thr = self.pars['lower_thr']['value']
		upper_thr = self.pars['lower_thr']['value'] + self.pars['thr_gap']['value']

		# design parameter
		signal_level = self.pars['signal_level']['value']

		p_signal = self.pars['p_signal']['value']
		correct_rejection = self.pars['correct_rejection']['value']
		false_alarm = self.pars['false_alarm']['value']
		miss = self.pars['miss']['value']
		for _ in range(num_trials_per_d):
			present = np.random.choice([0,1], p=[(1-p_signal), p_signal] )
			signal = present * signal_level
			ratio_product = 1

			for i in range(round(max_episode_steps)):
				base_rate = (1 - p_signal) / p_signal
				criterion = base_rate * (correct_rejection+false_alarm)/(hit+miss)

				ratio = (norm(loc=signal, scale=sensor_noise).cdf(criterion) + 1e-10) / (norm(loc=0, scale=sensor_noise).cdf(criterion) + 1e-10)
				ratio_product *= ratio

				if lower_thr > ratio_product:
					# the decision is 'target not present'
					result = 1 if present == 0 else 0
				elif ratio_product > upper_thr:
					# the decision is 'target present'
					result = 1 if present == 1 else 0
				elif i == round(max_episode_steps) - 1:
					result = 0
				else: 
					continue
				experiment_outcomes.append(result)
				break

		return pd.DataFrame(experiment_outcomes, columns=['outcome'])


	def get_summary_statistics(self, data):
		if 'done' in data.columns:
			data_done = data[ data['done']==True ]

			s3 = data_done['steps'].mean() / 5.
			s13 = data_done['correct'].mean()

			result = [s13] # [TPs / 5., TNs / 5., FPs, FNs]
			temp = []
			for res in result:
				res = 0 if math.isnan(res) else res
				res = 1. if math.isinf(res) else res
				temp.append(res)
			result = temp
			# print('\t= Summaries: ', result, len(data_done.index)) 
			return result
		else:
			mean = data['outcome'].mean(axis=0) # np.mean(outcome, axis=0)
			return [mean] # [mean] # , var]



class SignalDetectionTask(BaseTask):
	def __init__(self):
		file_dir, _ = os.path.split(os.path.realpath(__file__))

		self.hyper = {
			'n_participants':			5, #? 30
			'output_dir':  				file_dir + '/output/', ## #?
			'inference_budget':			100,
			'corati_budget':			20, 
			'model_selection_budget':   1000,
            'verification_budget':      1000,
			'max_design_iterations':	5,
			'num_design_models':		10,
			'init_design_samples': 		10,
			'design_batch_size': 		1,
			'n_prediction_trials': 		1,
			'mat_file_name':            'true_lik',
            'random_design':            False,
            'true_likelihood':          False,
            'ado':                      False,
            'minebed':                  False,
            'output_dim':               1,
            'participant_shift':        1,
            'model_sel_rule':		'map',
            'record_posterior':		True,
            'misspecify':		0,
            'observ_noise':		0
		}
		self.model_priors = {'ProbabilityRatio': 1. }
		self.design_parameters = ['observ_limit', 'signal_level']


	def instantiate_model(self, model_name):
		if model_name == 'ProbabilityRatio':
			model = ProbabilityRatio()
			model.hyper['n_prediction_trials'] = self.hyper['n_prediction_trials']
			model.parameters_of_interest = model.parameters_of_interest if self.hyper['misspecify'] == 0 else model.parameters_of_interest[:self.hyper['misspecify']]
		model.max_episode_steps = 10
		
		pars = len(model.parameters_of_interest)
		# if some of the parameters are excluded from inference, replace them with their expected value
		if type(model.pars['hit']['value']) is tuple and pars == 0:
			model.pars['hit']['value'] = 4
		if type(model.pars['sensor_noise']['value']) is tuple and pars <= 1:
			model.pars['sensor_noise']['value'] = 0.55	
		if type(model.pars['lower_thr']['value']) is tuple and pars <= 2:
			model.pars['lower_thr']['value'] = 0.25
		if type(model.pars['thr_gap']['value']) is tuple and pars <= 3:
			model.pars['thr_gap']['value'] = 0.25
		
		return model
		

	def plot( model ):
		output_dir = model.hyper['output_dir']
		pdf = matplotlib.backends.backend_pdf.PdfPages(f'{output_dir}figures.pdf')
		data = pd.read_csv(f'{output_dir}predictions.csv')
		sp.plot_boiler( pdf, model )

		data_done = data[ data['done']==True ]
		sp.plot_vic( pdf, data_done, model.hyper, 'correct', 'signal_level', 'hit', np.mean, 'lower right', 'proportion correct (all episodes)' )
		sp.plot_vic( pdf, data_done, model.hyper, 'reward', 'signal_level', 'hit', np.mean, 'lower right', 'mean reward (all episodes)' )

		data_present = data_done[ data_done['present']==1 ]
		#data_choice = data_present[ ((data_present['action']==0) | (data_present['action']==1)) ]
		#print("check that no actions > 1:")
		#print(not any(data_choice['action']>1))		
		sp.plot_vic( pdf, data_present, model.hyper, 'correct', 'signal_level', 'hit', np.mean, 'lower right', 'proportion correct when target present' )
		sp.plot_vic( pdf, data_present, model.hyper, 'correct', 'signal_level', 'sensor_noise', np.mean, 'lower right', 'proportion correct when target present')
		sp.plot_vic( pdf, data_present, model.hyper, 'steps', 'signal_level', 'hit', np.mean, 'upper right', 'steps taken when target present')
		sp.plot_vic( pdf, data_present, model.hyper, 'steps', 'signal_level', 'sensor_noise', np.mean, 'upper right', 'steps taken when target present')
		#plt.savefig(f'{output_dir}model/png/figures')

		# plot ROC curve
		true_positive = (data_done['present']==1) & (data_done['action']==1)
		false_positive = (data_done['present']==1) & (data_done['action']==0)
		roc_data = {'TP':true_positive, 'FP':false_positive, 'hit':data_done['hit']}
		roc = pd.DataFrame(roc_data)
		sp.plot_vic( pdf, roc_data, model.hyper, 'TP', 'FP', 'hit', np.mean, 'lower right', 'ROC curve' )
		
		pdf.close()

	
	def plot_participants( model ):
		n_participants = model.hyper['n_participants']
		with open(f'{output_dir}synthetic_participants', "rb") as fp:
			participants = pickle.load(fp)

		for p in range(1,n+1):
			plot(model)

# -------------------------------------

def local_switch(task):
	print(f't test_ensemble_synthetic')
	print(f'p plot_ensemble_synthetic')
	x = input('Command? ')

	if x == 't':
		test.test_ensemble_synthetic(task)
	elif x == 'p':
		test.plot_ensemble_synthetic(task)
	#sp.plot_learning_curve(model.hyper['output_dir'])


import time
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--start')
	parser.add_argument('--x')
	parser.add_argument('--randomd')
	parser.add_argument('--truelik')
	parser.add_argument('--ado')
	parser.add_argument('--minebed')
	parser.add_argument('--rule')
	parser.add_argument('--misspecify')
	parser.add_argument('--noise')
	args = parser.parse_args()

	task=SignalDetectionTask()

	task.hyper['random_design'] = args.randomd=='True' if args.randomd else task.hyper['random_design']
	task.hyper['true_likelihood'] = args.truelik=='True' if args.truelik else task.hyper['true_likelihood']
	task.hyper['ado'] = args.ado=='True' if args.ado else task.hyper['ado']
	task.hyper['minebed'] = args.minebed=='True' if args.ado else task.hyper['minebed']
	task.hyper['model_sel_rule'] = args.rule if args.rule else task.hyper['model_sel_rule']
	task.hyper['misspecify'] = int(args.misspecify) if args.misspecify else task.hyper['misspecify']
	task.hyper['observ_noise'] = float(args.noise) if args.noise else task.hyper['observ_noise']
    
	if task.hyper['true_likelihood'] == True:
		task.hyper['mat_file_name'] = 'true_lik'

		if task.hyper['ado'] == True:
			task.hyper['mat_file_name'] = 'ado'
	elif task.hyper['minebed'] == True:
		task.hyper['mat_file_name'] = 'minebed'
		task.hyper['n_participants'] = 1
	else:
		task.hyper['mat_file_name'] = 'sp' 
	
	if task.hyper['model_sel_rule'] != 'map':
		task.hyper['mat_file_name'] += '_' + task.hyper['model_sel_rule']
	
	if task.hyper['random_design'] == True:
		task.hyper['mat_file_name'] += '_rand'
	task.hyper['participant_shift'] = int(args.start) * task.hyper['n_participants'] if args.start else task.hyper['participant_shift']
    
	x = str(args.x) if args.x else 'empty'
	
	if x == 'g':
		task.hyper['n_participants'] = 100
		
	time.sleep(task.hyper['participant_shift'])
	sp.switch(task, x)
