from ast import parse
import pandas as pd
import numpy as np
import os
from scipy.stats import binom
import argparse

from corati import synthetic_psychologist as sp

from corati.base import BaseTask, BaseModel
# -------------------------------------

THIS_FILE = __file__


class MemoryModel(BaseModel):
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

        num_bernoulli_trials_per_d = self.hyper['n_prediction_trials']
        experiment_outcome = []
        
        design = self.pars['design']['value']
        for i in range(num_bernoulli_trials_per_d):
            if self.name == 'exp':
                a = self.pars['a_exp']['value']
                b = self.pars['b_exp']['value']
                experiment_outcome.append( np.random.binomial( 1, np.clip(a * np.exp(-b*(design)), [0], [1])) )
            elif self.name == 'pow':
                a = self.pars['a_pow']['value']
                b = self.pars['b_pow']['value']
                experiment_outcome.append( np.random.binomial( 1, np.clip(a * ((design+1.0)**(-b)), [0], [1])) )
        return pd.DataFrame(experiment_outcome, columns=['outcome'])


    def get_summary_statistics(self, outcome):
        mean = outcome['outcome'].mean(axis=0) # np.mean(outcome, axis=0)
        return [mean] # , var]



class PowModel(MemoryModel):
    def __init__(self):
        a_params = [2, 1]
        pow_b_params = [1, 4]
        desgin_bound = [0., 100.]

        self.pars = {
            'a_pow': {'value':(np.random.beta, a_params), 'bounds': [0, 1], 'levels': 5},
            'b_pow': {'value':(np.random.beta, pow_b_params), 'bounds': [0, 1], 'levels': 5},
            'design': {'value':(np.random.uniform, desgin_bound), 'bounds': desgin_bound, 'levels': 5}
        }
        
        self.name = 'pow'
        self.parameters_of_interest = ['a_pow', 'b_pow']
        super().__init__()



class ExpModel(MemoryModel):
    def __init__(self):
        a_params = [2, 1]
        exp_b_params = [1, 8]
        desgin_bound = [0., 100.]

        self.pars = {
            'a_exp': {'value':(np.random.beta, a_params), 'bounds': [0, 1], 'levels': 5},
            'b_exp': {'value':(np.random.beta, exp_b_params), 'bounds': [0, 1], 'levels': 5},
            'design': {'value':(np.random.uniform, desgin_bound), 'bounds': desgin_bound, 'levels': 5}
        }
        
        self.name = 'exp'
        self.parameters_of_interest = ['a_exp', 'b_exp']
        super().__init__()


class MemoryTask(BaseTask):
    def __init__(self):
        self.true_model = None
        self.model_priors = {'pow': .5, 'exp': .5} #, 'pow': 0.5 }  {'exp': 0.5, 'pow': 0.5 } 
        # self.parameters_of_interest = ['a_exp', 'b_exp', 'a_pow', 'b_exp']
        self.design_parameters = ['design']

        file_dir, _ = os.path.split(os.path.realpath(__file__))

        self.hyper = {
            'n_participants':		    5,
            'output_dir':  			    file_dir + '/output/',
            'inference_budget':		    100,
			'corati_budget':		    100,
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
            'record_posterior':	True
		}
        print(THIS_FILE)
        

    def instantiate_model(self, model_name):
        if model_name == 'pow':
            model = PowModel()
        elif model_name == 'exp':
            model = ExpModel()
        model.hyper['n_prediction_trials'] = self.hyper['n_prediction_trials']
        model.parameters_of_interest.sort()
        return model


    def sample_model_parameters(self, model, size=1):
        items = [model + '_a', model + '_b', 'design']
        res = []
        for item in items: 
            temp = self.pars[item]['value'][0]( *self.pars[item]['value'][1], size=size )
            res.append(temp)
        a, b, design = res[0], res[1], res[2]
        return a, b, design


    def get_likelihood(self, data, model, pars, expt):
        pred_trials = self.hyper['n_prediction_trials']
        positives = model.get_summary_statistics(data)[0] * pred_trials
        negatives = pred_trials - positives
        design = expt[0]
        likelihoods = []
        for par in pars:
            a = par[0] # np.clip(par[0], [0], [1])[0]
            b = par[1] # np.max([par[1], 1e-7])
            # print(a, b, design)
            if model.name == 'exp':
                likelihood = binom.pmf(0, 1, a * np.exp(-b*(design)))**negatives * binom.pmf(1, 1, a * np.exp(-b*(design)))**positives
            elif model.name == 'pow':
                likelihood = binom.pmf(0, 1, a * ((design+1.0)**(-b)))**negatives * binom.pmf(1, 1, a * ((design+1.0)**(-b)))**positives

            if (np.isnan(likelihood).any()):
                print(likelihood, a, b, design)
            likelihoods.append(likelihood)
        return np.array(likelihoods).flatten()


    def train():
        pass

import time

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

    task=MemoryTask()

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
    time.sleep(task.hyper['participant_shift'])
    sp.switch(task, x)
