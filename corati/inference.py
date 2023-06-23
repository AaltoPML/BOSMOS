from builtins import print
from cmath import nan
from turtle import pos
import elfi, GPy
import numpy as np
import pandas as pd
import copy
import scipy

from elfi.methods.bo.gpy_regression import GPyRegression
from elfi.methods.bo.acquisition import LCBSC, UniformAcquisition
from elfi.methods.bo.utils import minimize

from corati.vizualization import plot_GP_mean, plot_parameter_marginals

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import minebed.static.bed as bed
import minebed.networks as mn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from functools import partial

def init_minebed(model, task, prior, key):
    print('Using MINEBED for likelihood-free inference...')
    bounds = tuple(model.pars[task.design_parameters[-1]]['bounds'])

    numpy_prior = np.array([prior[key][par_key]['value'].to_numpy() for par_key in sorted(prior[key].keys())]).transpose()
    # TODO: observ_limit is one of the signal detection designs, this is not automatic
    if 'observ_limit' in task.design_parameters:
        dom = [{'name': 'design', 'type': 'continuous', 'domain':  model.pars[task.design_parameters[0]]['bounds'], 'dimensionality': 1}, {'name': 'design', 'type': 'continuous', 'domain':  model.pars[task.design_parameters[1]]['bounds'], 'dimensionality': 1}]
    else: 
        dom = [{'name': 'design', 'type': 'continuous', 'domain':  bounds, 'dimensionality': len(task.design_parameters)}]
    
    N_EPOCH = 5000 # risky choice all -- 5000, signal_detection -- 5000, the rest is 7500
    BO_INIT_NUM = task.hyper['init_design_samples']
    BO_MAX_NUM = task.hyper['max_design_iterations']

    net = mn.FullyConnected(var1_dim=len(model.parameters_of_interest), var2_dim=task.hyper['output_dim'], L=2, H=64)
    opt = optim.Adam(net.parameters(), lr=1e-3)
    sch = StepLR(opt, step_size=1000, gamma=0.95)

    def local_simulator(cur_model, d, prior):
        keys = cur_model.hyper['design_parameters'] + cur_model.parameters_of_interest
        d = np.array(d).flatten()
        # print(keys, DATASIZE, cur_model, d, prior)
        result = []
        for sample in prior:
            v = np.concatenate([d, sample]) 
            result.append( cur_model.get_summary_statistics(cur_model.predict_one(keys, v)) )
        result = np.array(result)
        return result

    minebed_solver = bed.GradientFreeBED(model=net, optimizer=opt, scheduler=sch, 
        simulator=partial(local_simulator, model), prior=numpy_prior, domain=dom, n_epoch=N_EPOCH, 
        batch_size = len(numpy_prior), ma_window=100, constraints=None)

    minebed_solver.train(BO_init_num=BO_INIT_NUM, BO_max_num=BO_MAX_NUM, verbosity=False)
    return minebed_solver


class NumpyPrior(elfi.Distribution):
    '''
    Interface for ELFI that allows the usage of numpy.random distributions
    '''
    def rvs(func, inputs, size=1, random_state=None):
        '''
        Parameters
        ----------
        func : numpy.random function
        inputs : list of all arguments for func

        Return array_like (size,) 
        '''
        # TODO: random_state is not used 
        sample = func( *inputs , size=size)
        return sample


class PandasPrior(elfi.Distribution):
    '''
    Interface for ELFI that allows the usage of joint empirical samples (pd.DataFrame) as distributions
    '''
    def rvs(df, size=1, random_state=None):
        '''
        Parameters
        ----------
        df : pd.Dataframe, joint empirical samples 

        Returns array_like (size,) 
        '''
        sample = df.sample(n=size[0], replace=False) 
        return sample


class CustomPrior(elfi.Distribution):
    '''
    Interface for marginal distribution from joint PandasPrior
    '''
    def rvs(sample, key, size=1, random_state=None):
        '''
        Parameters
        ----------
        sample : array_like
        ind : int, index of a parameter that needs sampling

        Return array_like (size,)  
        '''
        return sample[key]


class ElfiSim:
    '''
    Simulator for LFI problem in ELFI, based on Synthetic Psychologist model
    '''
    def __init__(self, model, pars, expt, budget, infer_pars=None):
        '''
        Parameters
        ----------
        model : corati.synthetic_psychologist.Model, trained SP model
        pars : dict, of model parameters, e.g. 
            {'par1': {'value': pd.Series, 'bounds': list, 'levels': int}, ... }
        infer_pars : list, of parameters that require inference
        '''
        self.model = model
        self.params = pars
        self.expt = expt
        self.total_budget = budget
        self.cur_budget = 0

        model.parameters_of_interest.sort()

        # infer only those parameters that have priors
        if infer_pars is None:
            inf_keys = model.parameters_of_interest
        self.keys = []
        self.bounds = {}
        
        for key in inf_keys:
            if isinstance(self.params[key]['value'], pd.Series):
                self.keys.append(key)
                # samples = self.params[key]['value']
                self.bounds[key] = self.params[key]['bounds']

        print('Inference for parameters: ', self.keys)
        self.samples_in_prior = len(self.params[self.keys[0]]['value'])
        self.param_dim = len(self.keys)
        self.stored_summaries = []
        return


    def simulate(self, *par_list, n_obs=1, batch_size=1, random_state=None):
        '''
        Run SP simulations for multiple sets of parameters

        Parameters
        ----------
        par_list : array_like (batch_size, param_dim), list of parameter values
        n_obs : int, how many simulations per one set of parameters is required
        batch_size : int, the amount of different parameter sets in par_list

        Return array_like (batch_size, 1)
        '''
        sim_params = np.transpose(par_list)
        design_keys = self.model.hyper['design_parameters']

        results = []
        for i in range(batch_size):
            result = []
            # udate model parameters
            params = { key : {'value' : val} for key, val in zip(self.keys, sim_params[i]) }
            self.model.update_model_parameters(params, self.keys)

            for j in range(n_obs):   
                trace = self.model.predict_one( design_keys, self.expt)
                result.append(trace)
            results.append(result)
        
        self.stored_summaries.append(self.get_summary(results))
        self.cur_budget += len(results)
        # print(f'Simulated {self.cur_budget} out of {self.total_budget}')
        return results


    def get_summary(self, observations):
        '''
        Create a summary for observations (pandas.dataframe)

        Parameters
        ----------
        observations : array_like (batch_size, 1), where each
            element is a pandas.dataframe

        Return array_like (batch_size, 1)
        '''
        
        results = []
        if len(self.obs_shape) == len(np.shape(observations)):
            observations = [observations]

        for obs in observations:
            result = []
            for j in range(len(obs)):   
                data = obs[j]
                summary = self.model.get_summary_statistics(data)
                result.append(summary)
            results.append(np.mean(result, axis=0))
        return np.array(results)


    def get_model(self, priors, data=None):
        '''
        Create an ELFI graph

        Parameters
        ----------
        priors : dict, of model parameters, e.g. 
            {'par1': {'value': pd.Series, 'bounds': list, 'levels': int}, ... }
        data : sp.predict_one resul, external observations

        Return ElfiModel
        '''
        m = elfi.ElfiModel()        
        params = self.params

        # check if the prior is a particle set
        pd_params = pd.DataFrame()
        empirical_samples = False
        for key in self.keys:
            if isinstance(params[key]['value'], pd.Series):
                pd_params = pd.concat( [pd_params, params[key]['value'].to_frame(name=key)], axis=1)
                empirical_samples = True

        priors = []
        if empirical_samples:
            # use draws from empiricial samples as a prior
            elfi.RandomVariable(PandasPrior, pd_params, model=m, name='pd')
            for key in self.keys:
                elfi.Prior(CustomPrior, m['pd'], key, name=key)
                priors.append(m[key])

        if data is None:
            random_pars = m['pd'].generate()
            data = self.simulate(*random_pars)
            print('True parameters to be inferred: ', random_pars)
        else:
            data = [data]

        self.obs_shape = np.shape(data)
        elfi.Simulator(self.simulate, *priors, observed=data, name='sim')  # enable the simulator for inference
        elfi.Summary(self.get_summary, m['sim'], name='sum') # use summary statistics for the data
        elfi.Distance('euclidean', m['sum'], name='dist') # choose a discrepancy measure between synthetic and observed data
        
        def log_with_check(observatons):
            result = []
            for x in observatons:
                temp = np.log(x) if x > 0 else np.log(1.e-10)
                result.append(temp)
            return np.asarray(result)
        return m


    def add_gaussian_noise(self, particle_set, post, N):
        mean = []
        cov = []
        param_dim = self.param_dim
        # print('Bounds: ', post.model.bounds)
        bounds_min, bounds_max = [], []
        for i in range(param_dim):
            bounds_min.append(post.model.bounds[i][0])
            bounds_max.append(post.model.bounds[i][1])

            par_mean = 0. # (post.model.bounds[i][1] + post.model.bounds[i][0]) / 2.
            par_std = float(post.model.bounds[i][1] - post.model.bounds[i][0]) / 1e5
            cov_row = np.zeros(param_dim)
            cov_row[i] = 1.
            mean.append(par_mean)
            cov.append( par_std * cov_row)

        # resample each particle from a Gaussian in its local proximity
        sample_noise = np.random.multivariate_normal(mean, cov, N)
        sample_noise = sample_noise.reshape(N, param_dim)
        # sample_noise = np.einsum('ijk->ikj', sample_noise)

        new_particle_set = sample_noise + particle_set
        new_particle_set = np.clip(new_particle_set, bounds_min, bounds_max)
        return new_particle_set



def sample_posterior(post, elfi_sim, true_likelihood=None, data=None, model=None, expt=None, x_scaler=None):
    '''
    Sample posterior of the trained surrogate model

    Parameters
    ----------
    post : ElfiPosterior
    elfi_sim : ElfiSim 
    samples_per_particle : int, number of samples 
        per each particle in empirical prior

    Return DataFrame 
    '''
    # retrieve all empirical samples from the prior
    N = elfi_sim.samples_in_prior
    theta = post.prior.rvs(size=N)

    # ensure that it has the right dimensionality
    if theta.ndim == 1:
        theta = theta.reshape(N, -1)
    
    # prepare mean and covariance for resampling
    particles = elfi_sim.add_gaussian_noise( theta, post, N)

    if x_scaler is not None:
        scaled_particles = x_scaler.transform(particles, copy=False )
    else:
        scaled_particles = particles

    if true_likelihood is None:
        print('Using BOSMOS for likelihood-free inference...')
    else:
        print('Using the true likelihood...')

    for i in range(20):    
        if true_likelihood is None:
            weights = post._unnormalized_likelihood(scaled_particles)
        else:
            weights = true_likelihood(data, model, scaled_particles, expt)

        if np.sum(weights) == 0 or np.count_nonzero(weights==0) > 0.9 * len(weights):
            post.threshold = np.quantile(post.model.Y, (i + 1) / 20. )
            # print('WARNING: all samples are filtered out, increasing the threshold to ', post.threshold)
            # print('Adjustment: ', np.mean(post.model.Y),  np.count_nonzero(weights==0), 0.01 * len(weights))
            # print(weights)
        else:
            break

    # if all particles are filtered out, we want to sample them uniformly instead
    if np.sum( weights) != 0: 
        n_weights = weights / np.sum(weights)
        try:
            resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
        except ValueError:
            print('Value Error: ', n_weights)
            n_weights = np.nan_to_num(n_weights)
            resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
        resampled_theta = copy.deepcopy(particles[resample_index,:])
        theta_posterior = pd.DataFrame.from_records(resampled_theta, columns = elfi_sim.keys)
    else:
        # print('\n\n==== The weights are zeros, returning the previous posterior ====\n\n')
        theta_posterior = pd.DataFrame.from_records(particles, columns = elfi_sim.keys)
    particles = pd.DataFrame.from_records(particles, columns=elfi_sim.keys)
    return theta_posterior, particles, weights


def posterior_averaging(post, elfi_sim, it, prev_theta=None):
    N = elfi_sim.samples_in_prior
    theta = post.prior.rvs(size=N)
    weights = post._unnormalized_likelihood(theta)
    n_weights = weights / np.sum(weights)
    resample_index = np.random.choice(N, size=N, replace=True, p=n_weights)
    theta_resampled = theta[resample_index,:]
    theta_df = pd.DataFrame.from_records(theta_resampled, columns=elfi_sim.keys)

    if prev_theta is not None:
        prev_theta_df = pd.DataFrame.from_records( {k : prev_theta[k]['value'] for k in elfi_sim.keys}, columns=elfi_sim.keys)
        merged_theta = pd.concat([prev_theta_df, theta_df]) 
        theta_df = merged_theta.sample(N, weights=[ float(it-1) / float(it)] * N + [ 1. / float(it)] * N)
    return theta_df


def infer(model, pars, expt, budget, data=None, seed=0):
    '''
    Run LFI for the model

    Parameters
    ----------
    model : corati.synthetic_psychologist.Model, trained SP model
    pars : dict, of model parameters, e.g. 
        {'par1': {'value': float or tuple, 'bounds': list, 'levels': int}, ... }
    infer_parameters : list, of parameters that require inference
    data : sp.predict_one result, observations from a target user
    budget : int, simulation budget, the same amount is used as initial evidence
        and for the acquisition in BO
    seed : int, random seed

    Return ElfiSim, ElfiModel, ElfiPosterior
    '''
    # initialize an ELFI model
    elfi_sim = ElfiSim(model=model, pars=copy.deepcopy(pars), expt=expt, budget=budget)
    elfi_model = elfi_sim.get_model(priors=copy.deepcopy(pars), data=data)

    mean_func = GPy.mappings.Constant(elfi_sim.param_dim, 1, 0)

    # Construct a default kernel
    kernel = GPy.kern.RBF(input_dim=elfi_sim.param_dim)
    bias = GPy.kern.Bias(input_dim=elfi_sim.param_dim)
    kernel += bias
    GP_surrogate = GPyRegression(parameter_names=elfi_sim.keys, bounds=elfi_sim.bounds, max_opt_iters=10, \
        mean_function=mean_func, kernel=kernel, noise_var=1.)

    # heuristic for the variance of the acquisition noise
    noise_var = []
    for key in elfi_sim.keys:
        par_var = (elfi_sim.bounds[key][1] - elfi_sim.bounds[key][0]) / 1e3
        noise_var.append(par_var)

    # initialize the acquisition function (Lower Confidence Bound Selection Criteria)
    acq_function = LCBSC(model=GP_surrogate, noise_var=noise_var)

    # initialize Bayesian Optimization in LFI
    bolfi = elfi.BOLFI(elfi_model, 'dist', batch_size=5, initial_evidence=int(budget/2),
                                update_interval=int(budget/2), target_model=GP_surrogate,
                                acquisition_method=acq_function)

    # do inference, extract posterior, the threshold value is a placeholder                    
    post = bolfi.fit(n_evidence=budget, threshold=1, bar=False)
    return elfi_sim, elfi_model, post



from sklearn import preprocessing
from sklearn.neighbors import KernelDensity
import arviz as az
from scipy.stats import expon

def find_kde_peak(df):
    '''
    Use kernel density estimator for the particle set to approximate the posterior and 
    return its peak
    '''
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(df.to_numpy())
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05)
    kde.fit(x_scaled)
    x_min = x_scaled[np.argmax( kde.score_samples(x_scaled))]
    x_min = min_max_scaler.inverse_transform([x_min])[0]
    return x_min

def euclidean_distance(x, y):
    d = np.linalg.norm(x - y) # np.sqrt((x - y)**2, axis=1))
    return d + 1e-10

def jacobian(x, y):
    J = []
    for xi in x:
        d = euclidean_distance(xi, y)
        diff = xi - y + 1e-10
        J.append(np.divide(diff, d)[0])
    return np.array(J).transpose()


def infer_parameters(data, model, theta, expt, participant=None, true_lik=None, budget=20, it=None):
    '''
    Do parameter inference step for the Synthetic Psychologist loop. Return an
    empirical posterior and a point estimate from the empirical sample.

    Parameters
    ----------
    data : sp.predict_one result, observed data from experiments
    model : sp.Model, Synthetic Psychologist ensemble model
    theta : dict, piors for the model

    participant : sp.Model, participant model whose parameters we infer (required for plotting)
    it : int, iteration number (required for plotting)

    Return pd.DataFrame
    '''
    elfi_sim, elfi_model, post = infer(model=copy.deepcopy(model), 
                                    pars=copy.deepcopy(theta), 
                                    expt=expt, budget=budget, data=data)
    
    # unscaled_discr = post.model.Y
    # x_scaler = preprocessing.StandardScaler().fit(post.model.X)
    y_scaler = preprocessing.StandardScaler().fit(post.model.Y)
    # x_scaled = x_scaler.transform(post.model.X)
    y_scaled = y_scaler.transform(post.model.Y)
    # post.model.gp_params['noise_var'] = 0.3
    post.model._init_gp( post.model.X, y_scaled) # y_scaled ) #post.model.Y)
    post.model.max_opt_iters = 100
    post.model.optimize()
    # print(post.model._gp)

    # print('Extracting point estimate...')
    if true_lik is None:
        minloc, minval = minimize(
            post.model.predict_mean,
            post.model.bounds,
            grad=post.model.predictive_gradient_mean,
            n_start_points=300,
            random_state=post.random_state)
        
        post.threshold = np.max([minval, np.min(post.model.Y)])
    print('Extracting the posterior...')

    post_samples, particles, weights = sample_posterior(post, elfi_sim, true_lik, data, model, expt, None)
    minloc = find_kde_peak(post_samples)
    estimate = {key : x for x, key in zip(minloc, elfi_sim.keys)}

    if true_lik is None:
        # calculate the expectation for the the marginal likelihood:
        N = elfi_sim.samples_in_prior
        theta_samples = post.prior.rvs(size=N)
        if theta_samples.ndim == 1:
            theta_samples = theta_samples.reshape(N, -1)

        ms, vs = post.model._gp.predict(theta_samples)
        ys = y_scaler.inverse_transform(ms)
    else:
        ys = []

    plot_theta = copy.deepcopy(theta)
    particles_dict = copy.deepcopy(theta) 
    for key in elfi_sim.keys:
        plot_theta[key]['value'] = post_samples[key]
        particles_dict[key]['value'] = particles[key]

    plot = True
    
    if participant is not None and it is not None and plot is True and model.name == participant.name:
        ground_truth = { key : participant.pars[key]['value'] for key in participant.parameters_of_interest }
        # plot_parameter_marginals(plot_theta, estimate, ground_truth, output_dir=participant.hyper['output_dir'], model_name=model.name, it=model.name + str(it), thr=np.sum(weights)) #np.sum(weights))
    
        if len(elfi_sim.keys) == 1 and plot is True:
            key = elfi_sim.keys[0]
    else:
        # plot_parameter_marginals(plot_theta, estimate, output_dir=participant.hyper['output_dir'], model_name=model.name, it=model.name + str(it), thr=np.sum(weights)) #np.sum(weights))
        pass

    # print('THR: ', post.threshold, np.exp(post.threshold))

    return particles_dict, estimate, weights, ys # expected_value

    
