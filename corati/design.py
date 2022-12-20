from doctest import OutputChecker
from builtins import print
import torch

from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.optim.optimize import optimize_acqf

from scipy.special import entr
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from corati.vizualization import show_design_GP
import scipy

def get_design_bounds(model):
    '''
    Return design parameter bounds for a specific task
    '''
    result = list()
    design_names = model.hyper['design_parameters']
    for name in design_names:
        result.append( model.pars[name]['bounds'])
    result = np.array(result).transpose() # reshape((2, -1))
    return result


class Exp_designer_single():
    def __init__(self, init_sample, max_iter, batch_size, bounds, ado=False, **kwargs):
        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        # self.global_bounds = torch.tensor([[self.light[0]], [self.light[-1]]], **self.tkwargs)
        self.global_bounds = torch.tensor(bounds, **self.tkwargs)
        self.NUM_RESTARTS = 20 
        self.RAW_SAMPLES = 1024
        self.MC_SAMPLES = 128
        
        self.INIT_SAMPLE = init_sample  # how many initial random sampling points
        self.MAX_ITER = max_iter        # how many iterations
        self.BATCH_SIZE = batch_size    # in each iterations, how many samples

        self.kde = KernelDensity(kernel="gaussian")
        self.outcome_scaler = StandardScaler()
        self.std_scaler = StandardScaler()

        self.ado = ado
        if self.ado:
            self.likelihood_func = kwargs['likelihood_func']
            self.model_priors = kwargs['model_priors']
            self.parameter_particles = kwargs['parameter_particles']
            self.weights = kwargs['weights']
            self.instantiate_func = kwargs['instantiate_func']
            self.summary = False
        else:
            self.summary = True


    def get_objective(self, outcomes, task=None):
        '''
        Calculate the entropy of empirical samples, the probability of samples
        is determined by a kernel density estimator.

        Parameters
        ----------
        outcomes : array_like of generated data

        Return arry_like of entropies
        ''' 
        # score_samples returns log density, but entr takes probabilities
        # True False
        scaled_outcomes = []
        for m in outcomes:
            scaled_outcomes.append(list())
            for d in m:
                scaled_outcomes[-1].append( self.outcome_scaler.transform(d) )
        outcomes = scaled_outcomes
        means = np.mean(outcomes, axis=2) 
        vars = np.var(outcomes, axis=2)

        mean_vars = np.mean(vars, axis=0)
        var_means = np.var(means, axis=0)

        result = var_means - mean_vars
        result = np.sum(result, axis=1)
        result = np.expand_dims(result, axis=1)
        train_obj = torch.tensor(result, dtype=torch.float64)
        return train_obj


    def get_ado_objective(self, outcomes, designs):
        '''
        The utility objective from Cavagnaro et al. 2013:

        U(d) = \sum_{m=1}^K p(m) \sum_y p(y | m, d) \log(\frac{p(y|m,d)}{\sum_{m=1}^K p(m) p(y | m, d)})

        where

        p(y | m, d) = \int_\theta p(y | \theta_m, d) p(\theta_m)
        '''
        num_par_samples = 500
        outcomes = outcomes[:10]

        results = []
        for design in designs:
            # outcomes = np.unique(outcomes) # REMOVE ALL REPETITIONS
            marg_liks = dict()
            # calculate for y: p(y | m, d) = \int_\theta p(y | \theta_m, d) p(\theta_m)
            for m1 in self.model_priors:
                model = self.instantiate_func(m1)
                thetas = pd.concat([x['value'][:num_par_samples] for x in self.parameter_particles[m1].values()], axis=1)
                column_order = [x for x in self.parameter_particles[m1].keys()]
                thetas.columns = column_order
                thetas = thetas.reindex(sorted(thetas.columns), axis=1)
                
                marg_liks[m1] = list()
                ns = np.min( (len(thetas), num_par_samples) )
                for y in outcomes:
                    lik = self.likelihood_func(y, model, thetas.to_numpy(), design)
                    # print(lik)
                    # print(self.weights[m1])
                    marg_liks[m1].append( sum(lik * self.weights[m1][:ns])) # multiply two vectors element-wise

            # calculate for y: \sum_m p(m) p(y | m, d)
            marg_model_liks = []
            for i, y in zip(range(len(outcomes)), outcomes):
                marg_model_lik = 0
                for m1 in self.model_priors:
                    marg_model_lik += self.model_priors[m1] * marg_liks[m1][i]
                marg_model_liks.append(marg_model_lik)

            # calculate: \sum_m p(m) \sum_y p(y | m, d) * log( p(y | m, d) / (\sum_m p(m) p(y | m, d) ) )
            result = 0
            for m1 in self.model_priors:
                temp_sum = 0
                for i, y in zip(range(len(outcomes)), outcomes):
                    temp_sum += marg_liks[m1][i] * np.log( (marg_liks[m1][i] + 1e-10) / (marg_model_liks[i] + 1e-10) )
                
                result += self.model_priors[m1] * temp_sum
            results.append(result)
        results = np.expand_dims(results, axis=1)
        train_obj = torch.tensor(results, dtype=torch.float64)
        return train_obj


    def generate_initial_data(self, models):
        '''
        Sample design from the prior, generate data for each of the models,
        and compute the respective objectives.

        Parameters
        ----------
        models : array_like of Models

        Return array_like for designs, array_like for objectives
        ''' 
        model = models[0]

        # cosider only design parameters
        keys = model.hyper['design_parameters'] 
        
        # sample initial evidence from the design priors
        train_x_flattened = []
        for _ in range(self.INIT_SAMPLE): 
            design_values = []
            for key in keys:
                func = model.pars[key]['value'][0]
                inputs = model.pars[key]['value'][1]
                design_values.append(func(*inputs))

            train_x_flattened.append(design_values)
        train_x = torch.tensor(train_x_flattened, dtype=torch.float64)
                                         
        # run simulations for each of the models
        outcomes = []
        for m in models:
            outcome_m = self.predict(m, train_x_flattened)
            outcomes.append(outcome_m)
        
        # train kernel density estimator with a heuristic choice of bandwidth
        if self.summary: 
            outcomes = np.array(outcomes) # shape: models x design x batch size x dim
            self.outcome_scaler.fit([val for outcome_m in outcomes for batch in outcome_m for val in batch])

        # calculate the objective function
        if self.ado:
            outcomes = [ val for sublist in outcomes for subsublist in sublist for val in subsublist]
            train_obj = self.get_ado_objective(outcomes, train_x_flattened) 
        else:
            train_obj = self.get_objective(outcomes)
        return train_x, train_obj

        
    def initialize_model(self, train_x, train_obj, state_dict=None):
        # define models for objective and constraint
        model = SingleTaskGP(train_x, train_obj).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # load state dict if it is passed
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return mll, model


    def get_parameter_values_from_bayes_optimizer(self,acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
        # optimize
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.global_bounds,
            q=self.BATCH_SIZE,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        # observe new values 
        new_x = candidates.detach()
        return new_x


    def design_experiment(self, models):
        '''
        Choose the experiment via Bayesian Optimization, which should reveal the most information 
        for model selection.

        Parameters
        ----------
        models : array_like of Models

        Return : array_like, design value
        '''
        self.num_of_models = len(models)

        # get initial evidence and initialize the surrogate model for Bayesian Optimization
        train_x_nei, train_obj_nei = self.generate_initial_data(models)

        # standardize the objective
        mean = train_obj_nei.mean()
        std = train_obj_nei.std()
        std = 1 if std == 0 else std
        train_obj_nei = (train_obj_nei - mean) / std

        # print(f'Design experiment starts with training data: {train_x_nei}, {train_obj_nei}')
        mll_nei, model_nei = self.initialize_model(train_x_nei, train_obj_nei)
        
        # start Bayesian optimization iterations
        for iter_count in range(1, self.MAX_ITER+1):
            # fit the models
            fit_gpytorch_model(mll_nei)
            
            # define the qEI and qNEI acquisition modules using a QMC sampler
            qmc_sampler = SobolQMCNormalSampler(num_samples=self.MC_SAMPLES)
            
            # get the next parameter value
            qNEI = qNoisyExpectedImprovement(
                model=model_nei, 
                X_baseline=train_x_nei,
                sampler=qmc_sampler,
            )
            
            # optimize and get new designs to sample
            new_x_nei = self.get_parameter_values_from_bayes_optimizer(qNEI)
            new_x_list = new_x_nei.cpu().numpy().tolist()
            new_x_flattened = new_x_list
            # new_x_flattened = [[val] for sublist in new_x_list for val in sublist] 

            # evaluate the objectives at new design points
            outcomes = []
            for m in models:
                outcome_m = self.predict(m, new_x_flattened)
                outcomes.append(outcome_m)

            if self.ado:
                outcomes = [ val for sublist in outcomes for subsublist in sublist for val in subsublist]
                new_obj_nei = self.get_ado_objective(outcomes, new_x_flattened)
            else:
                outcomes = np.array(outcomes) # shape: models x outcomes x dim
                # outcomes = np.swapaxes(outcomes, 0, 1) # new shape: outcomes x models x dim
                new_obj_nei = self.get_objective(outcomes)

            new_obj_nei = (new_obj_nei - mean) / std

            # update training points    
            train_x_nei = torch.cat([train_x_nei, new_x_nei])
            train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            mll_nei, model_nei = self.initialize_model(
                train_x_nei, 
                train_obj_nei, 
                model_nei.state_dict(),
            )

        best_expt = take_max(train_x_nei, train_obj_nei)
        # show_design_GP(model_nei, train_x_nei, train_obj_nei, [0, 4])        
        return best_expt


    def predict(self, model, designs, size=10):
        '''
        Generates the data for each design point.

        Parameters
        ----------
        model : Model
        designs : array_like

        Return array_like of summaries
        '''
        results = []
        keys = model.hyper['design_parameters']   
        for d in designs:
            temp_res = []
            for _ in range(size):
                data = model.predict_one(keys, d)
                if self.summary:
                    data = model.get_summary_statistics(data)
                temp_res.append(data)
            results.append(temp_res)
            # results.append(np.mean(temp_res, axis=0))
            # std_results.append(np.std(temp_res, axis=0))

        # print( np.array(results))
        return results # np.array(results), np.array(std_results)



def take_max(train_x_nei, train_obj_nei):
    train_x_np = train_x_nei.cpu().numpy()
    train_obj_np = train_obj_nei.cpu().numpy()
    max_index = np.argmax(train_obj_np)
    expt = train_x_np[max_index]
    return (expt)

