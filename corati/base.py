
import os
import numpy as np
import time
import copy
import pandas as pd

import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from corati.vizualization import plot_parameter_marginals, plot_learning_curve

# -------------------------------------

precision=np.float32

def instantiate_participant_model(task):
    '''
    Generate participant for a given task.
    Parameters
    ----------
    task : Task
    Return Model
    '''
    keys = list(task.model_priors.keys())
    probs = list(task.model_priors.values())

    model_name = np.random.choice(keys, 1, p=probs)
    model = task.instantiate_model(model_name)
    return model


def sample_model_instances(task, parameter_particles, model_priors, sample_size=1):
    '''
    Sample a set of models from the current particle set of parameters.
    Parameters
    ----------
    task : Task
    parameter_particles : dict
    (optional) sample_size : int
    Return list, of Model instances
    '''
    sampled_models = []
    keys = list(model_priors.keys())
    probs = list(model_priors.values())

    for _ in range(sample_size):
        model_name = np.random.choice(keys, 1, p=probs)[0]
        model = task.instantiate_model(model_name)
        model.pars = assign_parameter_sample(model, parameter_particles[model_name])
        sampled_models.append(model)
    return sampled_models


def assign_parameter_sample(model, parameter_particles):
    '''
    Sample parameters from a parameter particle set and and assign the sampled values 
    to the model
    
    Parameters
    ----------
    model : Model, model for the task
    parameter_particles : dict, of model parameters, e.g. 
        {'par1': {'value': float or tuple, 'bounds': list, 'levels': int}, ... }
    size : how many samples to sample (default 1)
    Return dict (same as model.pars)
    '''
    first_key = list(parameter_particles.keys())[0]
    N = len( parameter_particles[first_key]['value'])
    sample_ind = np.random.randint(0, N)
    for key in parameter_particles.keys():
        if isinstance(parameter_particles[key]['value'], pd.Series):
            model.pars[key]['value'] = parameter_particles[key]['value'].values[sample_ind]
    return model.pars



class BaseModel():
    '''
    Basic class for a functional model, all new models should inherit this class and 
    specify all methods below. 
    '''
    def __init__(self):
        '''
        Example: 

        self.hyper = {
            'output_dir':  			    'sd_output/',   # save directory
            'design_parameters': 	    [], # parameters that are used for designs
			'n_prediction_trials':	    1
        }

        self.pars = {
            'par1':                                 # name of the parameter 
                {'value':(np.random.uniform, [0,1]),# prior 
                'bounds': [0,1],                    # an interval of admissable solutions
                'levels': 5}  # number of discretization levels (required only for RL models)
        }

        self.name = 'string' # name of the model
        self.parameters_of_interest = ['hit', 'sensor_noise', 'lower_thr', 'thr_gap'] # parameters that need to be inferred
        '''
        self.pars_sample = self.sample_model_parameters( self.pars )


    def sample_model_parameters(self, prior, exclude_pars=[]):
        '''
        Sample parameters from the model prior.

        Parameters
        ----------
        prior : dict() (same as self.pars), parameter priors of the model

        Return dict(), parameter values of the model
        '''
        new = copy.deepcopy(prior)
        for p in new:
            if p in exclude_pars:
                continue
            if isinstance(new[p]['value'],tuple):
                func = new[p]['value'][0]
                inputs = new[p]['value'][1]
                new[p]['value'] = func( *inputs )
        return new


    def get_summary_statistics(self, data):
        '''
        Summarize data from self.predict_one (required for inference). 

        Parameters
        ----------
        data : pandas.dataframe, the output of self.predict_one

        Return array_like (1, summary_dimensions) a lower-dimensional representation
        '''
        pass


    def predict_one(self, keys, vals):
        '''
        Generate data from the model (required for inference).

        Parameters
        ----------
        keys : array_like, the list of parameters (e.g design parameters) 
            for the simulation
        vals : array_like, values for the simulation

        Return pandas.dataframe, a single synthetic observation
        '''
        pass


    def update_model_parameters(self, par_list, keys=None):
        '''
        Updates parameter values in the model.
        
        Parameters
        ----------
        par_list : dict, of parameter values
        '''
        # if no keys are provided, try to update all parameters
        if keys is None:
            keys = list(self.pars.keys())

        for key in keys:
            self.pars[key]['value'] = par_list[key]['value']
        return


    def get_info(self):
        '''
        Return a dictionary with the model performance information.
        '''
        return {'empty': None}



class BaseTask():
    '''
    Basic class for tasks, all new tasks should inherit this class and specify all methods below. 
    '''
    def __init__(self):
        '''
        Example:

        self.hyper = {
			'n_participants':			5, 
			'output_dir':  				file_dir + '/output/', ## #?
			'inference_budget':			100,
			'corati_budget':			100, 
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

        self.unified_model = None # a unified (ensemble) model if available
        self.model_priors = None # priors for models (a single element if only one model)
        self.design_parameters = [] # list of design parameters 
        '''
        pass

    
    def instantiate_model(self, model_name):
        '''
        Instantiate the model from its name. 

        Parameters
        ----------
        model_name : str, name of the model

        Return Model instance
        '''
        pass


    def sample_particles(self, N=10000, plot_dir=None ):
        '''
        For each model, sample particles for parameters that have defined priors

        Parameters
        ----------
        pars : dict, of model parameters, e.g. 
            {'par1': {'value': float or tuple, 'bounds': list, 'levels': int}, ... }
        inf_pars : list, names of parameters that need inference
        N : int, number of particles to sample

        Return dict
        '''
        result = dict()
        for model_name in list(self.model_priors.keys()):
            temp_model = self.instantiate_model(model_name)
            pars = temp_model.pars
            inf_pars = temp_model.parameters_of_interest

            model_N = int(self.model_priors[model_name] * N)
            
            # for key in pars.keys():
            for key in inf_pars:
                if isinstance(pars[key]['value'], tuple):
                    func = pars[key]['value'][0]
                    inputs = pars[key]['value'][1]
                    sample = func( *inputs , size=10*model_N)

                    lower_bound = pars[key]['bounds'][0]
                    upper_bound = pars[key]['bounds'][1]
                    sample = sample[(sample > lower_bound) & ( sample < upper_bound)]
                    sample = sample[:model_N]

                    pars[key]['value'] = pd.Series(sample)

                    if len(sample) < model_N:
                        raise ValueError(f'The prior for {key} is out of specified bounds [{lower_bound}, {upper_bound}]')
                    # estimtes.append(pars[key]['value'].mean())
            
            # delete keys that are not parameters of interest
            par_keys = list(pars.keys())
            for key in par_keys:
                if key in inf_pars:
                    continue
                else:
                    del pars[key]
                
            result[model_name] = pars

            # if plot_dir is not None:
            #     plot_parameter_marginals(pars, output_dir=plot_dir, it=0)    

        return result




class PPOModel(BaseModel, gym.Env):
    ''' 
    This model class is a Baselines3 PPO agent.
    '''
    
    def __init__(self):
        super().__init__()

		# for the lifetime of model  keep track of which epsiode each action is associated with.
        self.episode = 0
        self.init_action_space( self.env.action_names )
		# provide the lower and upper bound  pair for each feature.
        self.set_bounds(self.pars, self.hyper)
        self.max_steps_count = 0
	

    def reset(self):
		# reset the cognitive POMDP and setup a new task for the episode.
        self.episode += 1
        self.action = -99
		# sample parameters
        self.pars_sample = self.sample_model_parameters( self.pars )
        # reset the estimation function.
        self.agent.estimation.reset( self.pars_sample )
        # reset the external state
        self.stmls = self.env.reset( self.pars_sample )
        # reset the internal state (does not require an action)
        self.reward, self.rspns, self.internal_state = self.agent.internal.reset( self.stmls, self.pars_sample )
        # generate an observation
        self.obs = self.agent.observation.look( self.internal_state, self.pars_sample )
        # update the belief
        self.belief, self.sigma = self.agent.estimation.update( self.obs, self.nBeliefs)
        
        # remove parameters that should not be an input to the policy network.
        pars_net = copy.deepcopy(self.pars_sample)
        for key in list(pars_net.keys()):
            if key not in self.hyper['policy_input']:
                pars_net.pop(key)
         
        # construct the network input for the ensemble model
        self.pars_input = list(map( lambda x: x['value'], pars_net.values()))
        self.network_input = np.concatenate( (self.belief, self.sigma, self.pars_input), axis=None)
        self.done = False
        self.steps = 0
        assert len(self.network_input)==self.nUpper, "The length of the network_input vector is not the same as the length of its lower and upper bounds."
		# return the initial network input vector
        # print("reset", self.network_input)
        return self.network_input 

    
    def step(self, a):
        '''take a single step within an episode given the action a chosen by the actor. Return the policy network input.'''
        self.action = a
		# generate the new response.
        self.rspns = self.agent.internal.effector( a, self.stmls )
		# update the external state and generate a stimulus
        self.stmls, self.done = self.env.transition( self.rspns )
		# get the percept from the sensor
        self.percept = self.agent.internal.sensor( self.stmls, self.pars_sample)
		# perform the internal state transition and generate a reward.
        self.reward, self.internal_state = self.agent.internal.transition( a, self.percept, self.pars_sample )
		# generate an observation
        self.obs=self.agent.observation.look( self.internal_state, self.pars_sample )
		# update the belief
        self.belief, self.sigma = self.agent.estimation.update( self.obs, self.nBeliefs )
	    # create a vector for the network input
        self.network_input = np.concatenate( (self.belief, self.sigma, self.pars_input ), axis=None)
        info={}
		# check whether max steps reached.
        self.steps += 1

        if type(self.pars['observ_limit']['value']) is tuple:
            max_steps = self.hyper['max_episode_steps']
        else:
            max_steps = round(self.pars['observ_limit']['value'])

        if self.steps == max_steps: 
            self.done=True
            self.max_steps_count += 1
			#print("WARNING: max steps reached.")
		# return the network input
		#print("step", self.network_input)
        return self.network_input, self.reward, self.done, info


    def set_bounds(self, pars, hyper):
		# set the bounds for the neural network input.
		# the observation space here is network_input (should be called belief)
		# gym requires self.observation_space
        belief_bounds = np.array(hyper['belief_bounds'], dtype=precision)
        belief_lowerbound = belief_bounds[:,0]
        belief_upperbound = belief_bounds[:,1]
        self.nBeliefs = len(list(belief_lowerbound))

        sigma_bounds = np.array(hyper['sigma_bounds'], dtype=precision)
        sigma_lowerbound = sigma_bounds[:,0]
        sigma_upperbound = sigma_bounds[:,1]
        self.nSigmas = len(list(sigma_lowerbound))

		# create a copy of the parameters and remove features to be excluded from the state.
        pars_net = copy.deepcopy(self.pars)
        for key in list(pars_net.keys()):
            if key not in self.hyper['policy_input']:
                pars_net.pop(key)
		#pars_net.pop('signal_level')

        pars_bounds = list(map( lambda x: x['bounds'], pars_net.values()))
        pars_bounds = np.array(pars_bounds, dtype=precision)
        pars_lowerbound = pars_bounds[:,0]
        pars_upperbound = pars_bounds[:,1]
        pars_values = list(map( lambda x: x['value'], pars_net.values()))
        assert len(pars_values)==len(pars_bounds), "Length of pars does not match number of upper/lower bounds."

        lowerbound = np.concatenate((belief_lowerbound, sigma_lowerbound, pars_lowerbound), axis=None)
        upperbound = np.concatenate((belief_upperbound, sigma_upperbound, pars_upperbound), axis=None)
        self.nLower = len(lowerbound)
        self.nUpper = len(upperbound)
        assert(self.nLower==self.nUpper)
        self.observation_space = spaces.Box(low=lowerbound, high=upperbound, dtype=precision)
        return lowerbound, upperbound

    def init_action_space(self, action_names ):
		# gym requires a variable named action_space
        self.nActions=len(action_names)
        self.action_space = spaces.Discrete(self.nActions)

	
    def explore(self):
        ''' 
        explore() interactively steps through the model with a random action.
	    Can be used for debugging.
        '''
        print("\n\n")
        nActions = self.nActions
        obs = self.reset()
        info = self.get_info()
        dump(info)
        done = False
        steps = 0
        input(f'{steps}: next step?')
        while not done:
			# pick a random action
            action = np.random.randint(0, nActions, 1)[0]
            obs, reward, done, _ = self.step(action)
            info = self.get_info()
            dump(info)
            steps += 1
            input(f'{steps}: next step?')
        print("Steps = ", steps)


    def train(self):
		# use baselines3 PPO to train the agent
        timesteps = self.hyper['n_training_timesteps']
        output_dir = self.hyper['output_dir']
        start = time.time()
        os.makedirs(output_dir, exist_ok=True )
        os.makedirs(f'{output_dir}model/', exist_ok=True)
        os.makedirs(f'{output_dir}model/png/', exist_ok=True)
        pars_file = self.hyper['pars_file']
        os.system(f'cp {pars_file} {output_dir}/training_parameters.py')
        # store data for the learning curve.
        monitor = Monitor( self, output_dir )
        initial_learning_rate = self.hyper['initial_learning_rate']
        bs = self.hyper['batch_size']
        ppo_actor_critic = PPO( 'MlpPolicy', monitor, verbose=1, batch_size=bs, ent_coef=.1, \
            learning_rate=linear_schedule(initial_learning_rate), clip_range=linear_schedule(0.2)) #entropy_weight or c2 #clip_range=0.1)
		
        # Train the agent
        ppo_actor_critic.learn( total_timesteps=int(timesteps) )
		# save the final trained model
        ppo_actor_critic.save( f'{output_dir}ppo_learned_policy-' + str(self.name) )
		# Plot learning curve
        plot_learning_curve( output_dir, self.name )
        monitor.close()
        del ppo_actor_critic
        end = time.time()
        if self.max_steps_count > 0:
            print(f'WARNING: MAX STEPS REACHED on {self.max_steps_count} episodes. This my make results difficult to interpret.')
        print("Elapsed time, ", round((end-start)/60/60, 2), " hours.")

    def predict(self):
        # generate predictions for each of the models in the parameter distribution defined by the model.

        def round3(x):
            return round(x,4)

        output_dir = self.hyper['output_dir']
        pars_file = self.hyper['pars_file']
        os.system(f'cp {pars_file} {output_dir}/predict_parameters.py')
        keys, parameters = generate_parameter_space(self.pars)
		# remove previous prediction file.
        outfile = f'{output_dir}predictions.csv'
        if os.path.isfile(outfile): os.remove(outfile)
        l = len(parameters)
        i = 1
        print(f'keys: {keys}')
        for p in parameters:
            print(f'{i} of {l}. Predict for parameters : {list(map(round3,p))}')
            trace = self.predict_one(self, keys, p)

            if not os.path.isfile(outfile):
                # if the fille does not exist then write the header to the new file...
                trace.to_csv( outfile, mode='a', header=self.get_info().keys()  )
            else:
                trace.to_csv( outfile, mode='a', header=False )
            i+=1
        return outfile 

				
    def get_info_for_predict(self):
		# assume that stimulus[0] is feedback
		#if self.done==True and self.stmls[0]>0:
        if (self.done == True) and (self.env.present == self.action):
            correct=1
        else:
            correct=0
        data={
			'episode': 	self.episode,
			'steps': 	self.steps,
			'stimulus': self.stmls,
			'action': 	self.action,
			'correct':	correct,
			'reward':	self.reward,
			'done':		self.done,
			'present':	self.env.present
		}
        return data


    def get_info(self):
		# assume that stimulus[0] is feedback
        if self.done==True and self.stmls[0]>0:
            correct=1
        else:
            correct=0
        data={
			'episode': 	self.episode,
            'steps': 	self.steps,
			'stimulus': self.stmls,
			'action': 	self.action,
			'belief': 	self.belief,
			'correct':	correct,
			'reward':	self.reward,
			'done':		self.done,
			'present':	self.env.present
		}
        keys = self.pars_sample.keys()
        values = map( lambda x: x['value'], self.pars_sample.values())
        parsd = dict(zip(keys,values))
        return( { **data, **parsd } )
        

    def predict_one(self, keys=None, v=None ):
        ''' 
        Make predictions with a trained model. Return a dataframe where each row is a step within an episode
        '''
        if keys is not None and v is not None: 
            i = 0
            for k in keys:
                self.pars[k]['value'] = v[i]
                i += 1
        else:
            keys = ['hit', 'sensor_noise']
            v = [self.pars[k]['value'] for k in keys]
        
        folder = self.hyper['output_dir']
        policy_file = f'{folder}ppo_learned_policy-' + str(self.name)
        policy = PPO.load(policy_file)
        # print('Loaded policy file: ', policy_file)
        trace = []
        eps = 0
        while eps < self.hyper['n_prediction_trials']:
            obs = self.reset()
            d = self.get_info_for_predict()
            x = dict(zip(keys,v))
            trace.append( {**d,**x} )

            done = False
            while not done:
                # use the policy to choose an action and then implement it
                action, _ = policy.predict( obs, deterministic = True )
                obs, reward, done, info = self.step( action )
                d = self.get_info_for_predict()
                x = dict(zip(keys,v))
                trace.append( {**d,**x} )
                # trace.append( self.get_info() )
            eps += 1
        data = pd.DataFrame(trace)
        return data

# -------------------------------------
# code from:
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html?highlight=clip_range#learning-rate-schedule
# Initial learning rate of 0.001
#model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
#model.learn(total_timesteps=20000)
# By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
# progress_remaining = 1.0 - (num_timesteps / total_timesteps)
#model.learn(total_timesteps=10000, reset_num_timesteps=True)

# reduce entropy bonus?

from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule reduces learning rate with training.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


# -------------------------------------

def dump(info):
	# pretty print info to the command line.
	print("")
	print("")
	for k, v in info.items():
		print(f'{k}:  {v}')

# -------------------------------------


def generate_parameter_space(pars):

	def generate_parameter_values(distribution):
		# generate  'num' parameters settings between each pair of 'bounds'.
		# e.g. 'bounds' = [[0,3],[10,20]] and 'num' = [4,2] results in [[0,1,2,3], [10,20]]
		# input 'distribution' is one of theta, phi or payoff.
		keys = list(distribution.keys())
		bounds = map( lambda x: x['bounds'], distribution.values())
		levels = map( lambda x: x['levels'], distribution.values())
		values = list(map( lambda b,s: np.linspace(*b,s), bounds, levels))
		return keys, values

	# generate the space of parameters implimed by the distributions of parameters.
	pars_keys, pars_values = generate_parameter_values(pars)
	# generate the cartesian product of the parameter values for each distribution.
	pars = it.product(*pars_values)
	return pars_keys, list(pars)
