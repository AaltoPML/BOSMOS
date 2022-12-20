import numpy as np


# -------------------------------------
''' Internal Environment
 An environment that is internal to the agents body. 
 Permitting representation of perceived features of the state.
'''

class Internal_env:
	def __init__( self, hyper ):
		self.hyper = hyper

	def reset( self, stmls, pars ):
		percept = self.sensor(stmls, pars)
		self.state = percept
		rspns = -99
		reward = 0.
		return reward, rspns, self.state

	def sensor( self, stmls, pars ):
		# the sensor gathers information from the stimulus.
		noise = pars['sensor_noise']['value']
		percept = stmls + np.random.normal( np.zeros(len(stmls)), noise)
		# no noise on stmls[0] which is for feedback
		if not self.hyper['feedback_noise']: 
			percept[0] = stmls[0]
		return percept

	def effector( self, action, stmls ):
		# default should asd  discrete motor noise.
		rspns = action
		return rspns

	def transition( self, action, percept, pars ):
		# The internal transition must update the internal state and issue a reward.
		# it is important to update state because it is "observed" (unlike the percept)
		# assume that every action has a step cost + a weighted feedback signal from stimulus[0].
		reward = pars['step_cost']['value'] + percept[0] #* pars['hit']['value']
		self.state = percept
		return reward, self.state

# -------------------------------------

class Observation:
	# observe the internal state.

	def __init__(self, hyper):
		self.hyper = hyper

	def look( self, state, pars_sample ):
		# rename this to reflect other perceptual models. 
		obs = state
		return obs

# -------------------------------------

class Naive_Bayes_estimation:
	# Estimation integrates observations to form a representation of the environment.
	# The representation can be a belief or a history.
	def __init__(self, hyper ):
		self.hyper = hyper

	def reset( self, pars ):
		# add a small number to the initial values to prevent divide by zero.
		self.estimate = np.array(self.hyper['prior'])
		self.sigma1 = np.array(self.hyper['prior_sd']) + self.hyper['small number']
		self.sigma2 = np.array(pars['sensor_noise']['value']) + self.hyper['small number']

	def update( self, observation, nBeliefs ):
		self.estimate, self.sigma1 = self.kalman(self.estimate, observation, self.sigma1, self.sigma2)
		assert len(self.estimate)==nBeliefs, "Number of beliefs does not match number of belief upper/lower bounds."
		return self.estimate, self.sigma1

	def kalman(self, z1, z2, sigma1, sigma2):
		#def Kalman_filter( z1, sigma1, z2, sigma2 ):
		# a simple Kalman filter based on Maybeck et al. Chapter 1.
		# the new estimate z3 is the sum of the two previous estimates weighted by confidence.
		w1 = sigma2**2 / (sigma1**2 + sigma2**2)
		w2 = sigma1**2 / (sigma1**2 + sigma2**2)
		z3 = w1*z1 + w2*z2
		sigma3 = np.sqrt( (sigma1**2 * sigma2**2) / (sigma1**2 + sigma2**2) )
		return z3, sigma3

# -------------------------------------
# define an Agent (a cognitive POMDP).

class Agent():

	def __init__(self, hyper):
		self.hyper = hyper
		# create instances of the required cognitive POMDP processes...
		self.internal = Internal_env( hyper )
		self.observation = Observation( hyper )
		self.estimation = Naive_Bayes_estimation( hyper )

# -------------------------------------

if __name__ == "__main__":
	pass
#	raise SystemExit(0)



