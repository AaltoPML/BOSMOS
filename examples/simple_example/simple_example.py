import pandas as pd
import numpy as np
import os
from scipy.stats import norm

from corati import synthetic_psychologist as sp

from corati.base import BaseTask, BaseModel
# -------------------------------------

THIS_FILE = __file__


class NormModel(BaseModel):
    def __init__(self):
        file_dir, _ = os.path.split(os.path.realpath(__file__))
        self.hyper = {
            'output_dir':  			file_dir + '/output/',
            'design_parameters':    ['design'],
            'n_prediction_trials':  10
        }
        super().__init__()


    def predict_one(self, keys=None, v=None):
        if keys is not None and v is not None: 
            i = 0
            for k in keys:
                self.pars[k]['value'] = v[i]
                i += 1

        num_trials_per_d = self.hyper['n_prediction_trials']
        experiment_outcome = []
        
        design = self.pars['design']['value']
        for i in range(num_trials_per_d):
            if self.name == 'pos_mean':
                mean = self.pars['mean_pos']['value']
                experiment_outcome.append( np.random.normal( mean, design )) 
            elif self.name == 'neg_mean':
                mean = self.pars['mean_neg']['value']
                experiment_outcome.append( np.random.normal( mean, design ))
            elif self.name == 'pos_var':
                var = self.pars['var_pos']['value']
                experiment_outcome.append( np.random.normal( design, var ))
            elif self.name == 'neg_var':
                var = self.pars['var_neg']['value']
                experiment_outcome.append( np.random.normal( -design, var ))

        return pd.DataFrame(experiment_outcome, columns=['outcome'])


    def get_summary_statistics(self, outcome):
        mean = outcome['outcome'].mean(axis=0) # np.mean(outcome, axis=0)
        # var = outcome['outcome'].var(axis=0)
        return [mean]



class PositiveMeanNormModel(NormModel):
    def __init__(self):
        mean_bounds = [0, 5]
        design_bounds = [0.001, 5]

        self.pars = {
            'mean_pos': {'value':(np.random.uniform, mean_bounds), 'bounds': mean_bounds, 'levels': 5},
            'design': {'value':(np.random.uniform, design_bounds), 'bounds': design_bounds, 'levels': 5}
        }
        
        self.name = 'pos_mean'
        self.parameters_of_interest = ['mean_pos']
        super().__init__()


class NegativeMeanNormModel(NormModel):
    def __init__(self):
        mean_bounds = [-5, 0]
        design_bounds = [0.001, 5]

        self.pars = {
            'mean_neg': {'value':(np.random.uniform, mean_bounds), 'bounds': mean_bounds, 'levels': 5},
            'design': {'value':(np.random.uniform, design_bounds), 'bounds': design_bounds, 'levels': 5}
        }
        
        self.name = 'neg_mean'
        self.parameters_of_interest = ['mean_neg']
        super().__init__()


class PositiveVarNormModel(NormModel):
    def __init__(self):
        var_bounds = [0.001, 2]
        design_bounds = [0, 5]

        self.pars = {
            'var_pos': {'value':(np.random.uniform, var_bounds), 'bounds': var_bounds, 'levels': 5},
            'design': {'value':(np.random.uniform, design_bounds), 'bounds': design_bounds, 'levels': 5}
        }
        
        self.name = 'pos_var'
        self.parameters_of_interest = ['var_pos']
        super().__init__()


class NegativeVarNormModel(NormModel):
     def __init__(self):
        var_bounds = [0.001, 2]
        design_bounds = [0, 5]

        self.pars = {
            'var_neg': {'value':(np.random.uniform, var_bounds), 'bounds': var_bounds, 'levels': 5},
            'design': {'value':(np.random.uniform, design_bounds), 'bounds': design_bounds, 'levels': 5}
        }
        
        self.name = 'neg_var'
        self.parameters_of_interest = ['var_neg']
        super().__init__()


class SimpleTask(BaseTask):
    def __init__(self):
        self.true_model = None
        self.model_priors = {'pos_mean': .5, 'neg_mean': .5} #, 'pow': 0.5 }  {'exp': 0.5, 'pow': 0.5 } 
        # self.parameters_of_interest = ['a_exp', 'b_exp', 'a_pow', 'b_exp']
        self.design_parameters = ['design']

        file_dir, _ = os.path.split(os.path.realpath(__file__))

        self.hyper = {
            'n_participants':		    5,
            'output_dir':  			    file_dir + '/output/',
            'inference_budget':		    100,
			'corati_budget':		    20,
            'model_selection_budget':   1000,
            'verification_budget':      1000,
			'max_design_iterations':    5,
			'num_design_models':	    10,
			'init_design_samples': 	    10,
			'design_batch_size': 	    1,
			'n_prediction_trials': 	    1,
			'mat_file_name':            'sp',
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
        print(THIS_FILE)
        

    def instantiate_model(self, model_name):
        if model_name == 'pos_mean':
            model = PositiveMeanNormModel()
        elif model_name == 'neg_mean':
            model = NegativeMeanNormModel()
        elif model_name == 'pos_var':
            model = PositiveVarNormModel()
        elif model_name == 'neg_var':
            model = NegativeVarNormModel()
        model.hyper['n_prediction_trials'] = self.hyper['n_prediction_trials']
        model.parameters_of_interest.sort()
        return model


    def get_likelihood(self, data, model, pars, expt):
        design = expt[0]
        likelihoods = []
        for par in pars:
            cur_par = par[0] # np.clip(par[0], [0], [1])[0]
            if model.name == 'pos_mean':
                likelihood = norm.pdf(data, cur_par, design)
            elif model.name == 'neg_mean':
                likelihood = norm.pdf(data, cur_par, design)
            elif model.name == 'pos_var':
                likelihood = norm.pdf(data, design, cur_par)
            elif model.name == 'neg_var':
                likelihood = norm.pdf(data, -design, cur_par)
            
            if np.isnan(likelihood):
            	print(data, design, cur_par)
            	
            likelihoods.append(likelihood)
        return np.array(likelihoods).flatten()


    def train():
        pass




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
    args = parser.parse_args()

    task=SimpleTask()

    task.hyper['random_design'] = args.randomd=='True' if args.randomd else task.hyper['random_design']
    task.hyper['true_likelihood'] = args.truelik=='True' if args.truelik else task.hyper['true_likelihood']
    task.hyper['ado'] = args.ado=='True' if args.ado else task.hyper['ado']
    task.hyper['minebed'] = args.minebed=='True' if args.ado else task.hyper['minebed']
    task.hyper['model_sel_rule'] = args.rule if args.rule else task.hyper['model_sel_rule']
    
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
