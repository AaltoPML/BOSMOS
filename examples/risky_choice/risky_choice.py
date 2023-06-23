from builtins import print
import os
import pandas as pd
import numpy as np
from typing import Union

from corati import synthetic_psychologist as sp

from corati.base import BaseTask, BaseModel
# -------------------------------------

THIS_FILE = __file__


class RiskyChoice(BaseModel):
    """
    Base class for models of DM under risk as implemented in Cavagnaro et al. (2013). The models 
    describe behaviour in a task where an individual has to choose between to lotteries (A and B). 
    Each lottery has three outcomes (high, medium and low), that occur at probabilities ph, pm and 
    pl. Let's assume that outcomes are fixed for now. Then the design space can be defined with four 
    parameters phA, plA, phB and plB.

    MM-triangle: Take we have three outcomes x_l, x_m and x_h with probabilities pl, pm and ph. 
    Different lotteries where the values of x are constant can be represented on MM-triangle using 
    pl and ph. For lottery A -> pl=0.2, ph=0.1, pm=0.7. For lottery B -> pl=0.5, ph=0.4, pm=0.1. 
    
    ph  | ╲
        |   ╲ 
        |     ╲
        |       ╲
        |         ╲ 
    0.4 |       xB  ╲     
        |             ╲
    0.1 |  xA           ╲
        |_________________╲    
           0.2  0.5       pl
    
    Attributes:
    -----------
        hyper : Dict
            Hyperparameters of the BaseModel class.
    
    Methods:
    --------
        predict_one
            Predicts observations.
        get_probabilities
            Extracts probabilities for low and high outcomes in lotteries A and B from design.
        probability_weighting
            Calculates probability weighting of p (only used in OPT and CPT).
        get_choice_probabilities
            Get choice probabilities of choosing option A or B.
    """
    def __init__(self):
        file_dir, _ = os.path.split(os.path.realpath(__file__))
        self.hyper = {
            'output_dir':  			file_dir + '/output/',
            'design_parameters':    ['plA', 'phA', 'plB', 'phB'],
            'n_prediction_trials':  10
        }
        super().__init__()
    

    def predict_one(self, keys=None, v=None):
        """
        Predicts observations.

        Parameters:
        -----------
            keys: list
                Parameter names.
            v : list
                Values of the parameters.
        Returns:
        --------
            pd.DataFrame containing the number of predictions as specified in self.hyper.
        """
        if keys is not None and v is not None: 
            i = 0
            for k in keys:
                self.pars[k]['value'] = v[i]
                i += 1

        num_trials_per_d = self.hyper['n_prediction_trials']
        experiment_outcome = []
        
        plA, phA, plB, phB = self.get_probabilities()
        
        if self.name == "EU":
            # Calculate the slope in the MM triangle
            A,B = self.check_stochastic_dominance(phA=phA, plA=plA, phB=phB, plB=plB)

            if A == None and B == None:
                A, B = self.get_preference(a=self.get_reference(), slope_AB = self.get_slope(phA, plA, phB, plB), pmA=(1-phA-plA), pmB=(1-phB-plB))
        
        elif self.name == "WEU":
            x = self.pars["x"]["value"]
            y = self.pars["y"]["value"]

            A, B = self.get_preference(slope_A = self.get_slope_A(phA=phA, plA=plA, x=x, y=y), slope_B = self.get_slope_B(phB=phB, plB=plB, x=x, y=y))

        elif self.name == "OPT" or self.name == "CPT":
            r=self.pars["r"]["value"]
            v=self.pars["v"]["value"]

            A = self.calculate_utility(ph=phA, pl=plA, r=r, v=v)
            B = self.calculate_utility(ph=phB, pl=plB, r=r, v=v)

        for i in range(num_trials_per_d):
            # Sample a value for parameters
            epsilon = self.pars["epsilon"]["value"]
            choice_probability = self.get_choice_probability(A = A, B=B, epsilon=epsilon)

            choice = np.random.choice(a=[1, 0],p=choice_probability)

            # Append to results
            experiment_outcome.append(choice)

        return pd.DataFrame(experiment_outcome, columns=['outcome'])


    def check_stochastic_dominance(self, phA, plA, phB, plB):

        """
        Check stochastic dominance (for EU). 

        Parameters:
        -----------
            phA : float
                Probability of high outcome in lottery A.
            plA : float
                Probability of low outcome in lottery A.
            phB : float
                Probability of high outcome in lottery B.
            plB : float 
                Probability of low outcome in lottery B.
        Returns:
        --------
            tuple : ordering of the lotteries IF one is stochastically dominated
        """

        # This part checks for stochastic dominance. The predictions should be the same for all models.
        if phA > phB and plA < plB:
            A,B = 1,0
        elif phA > phB and plA == plB:
            A,B = 1,0
        elif phA == phB and plA < plB:
            A, B = 1,0
        elif phB > phA and plB < plA:
            A,B = 0,1
        elif phB > phA and plB == plA:
            A,B = 0,1
        elif phB == phA and plB < plA:
            A,B = 0,1
        else:
            A = None
            B = None

        return A,B


    def get_probabilities(self, design=None):
        """
        Extracts probabilities for low and high outcomes in lotteries A and B from design.

        Attributes:
        ----------
            design: Dict
                A dictionary containing the task design.
        Returns:
        --------
            phA : float
                Probability of high outcome in lottery A.
            plA : float
                Probability of low outcome in lottery A.
            phB : float
                Probability of high outcome in lottery B.
            plB : float 
                Probability of low outcome in lottery B.
        """
        if design is None:
            plA, phA = [self.pars['plA']['value'], self.pars['phA']['value']]
            plB, phB = [self.pars['plB']['value'], self.pars['phB']['value']]
        else:
            plA, phA = design[:2]
            plB, phB = design[2:4]
        
        pmA = 2 - plA - phA
        pmB = 2 - plB - phB
        norm_constant_A = plA + pmA + phA 
        if norm_constant_A == 0:
            plA, phA = 0.3, 0.3
        else:
            plA, phA = plA / norm_constant_A, phA / norm_constant_A

        norm_constant_B = plB + pmB + phB
        if norm_constant_B == 0:
            plB, phB = 0.3, 0.3
        else:
            plB, phB = plB / norm_constant_B, phB / norm_constant_B
        
        return plA, phA, plB, phB
    

    def probability_weighting(self, p, r):

        """
        Calculates probability weighting of p (only used in OPT and CPT).

        Parameters:
        ----------
            p : float (can also be an array)
                The probability for which weighting is calculated.
            r : float
                The probability weighting parameter (also known as gamma in other literature, but following the notation from Cavagnaro, 2013)
        
        Returns:
        --------
            Weighted probability.
        """
        return p**r / (p**r + (1-p)**r)**(1/r)


    def get_choice_probability(self, A, B, epsilon):
        """
        Get choice probabilities of choosing option A or B.

        Arguments:
        ----------
            A : float
                Reference value for lottery A to indicate preference (this is utility for OPT and CPT, or describes the slope of the indifference curves in the other models).
            B : float
                Reference value for lottery B to indicate preference (this is utility for OPT and CPT, or describes the slope of the indifference curves in the other models).
            epsilon : float
                Parameter indicating stochasticity in choice.

        Returns:
        --------
            choice_probability : np.array
                Array indicating probability of choosing lottery A or B.
        """
        
        if A > B:
            choice_probability = np.array([1-epsilon, epsilon])
        elif A < B:
            choice_probability = np.array([epsilon, 1-epsilon])
        else:
            choice_probability = np.array([0.5, 0.5])
        return choice_probability
        

    def get_summary_statistics(self, outcome):
        mean = outcome['outcome'].mean(axis=0) # np.mean(outcome, axis=0)
        return [mean] # [mean] # , var]

    
class EU(RiskyChoice):
    """
    Expected utility.

    Attributes:
    -----------
        name : str
            Name of the class (Expected Utility)
        pars : Dict
            Parameters BaseClass requires.
        parameters_of_interest: list
            Model parameters being optimised?

    Methods:
    --------
        get_slope
            Get slope of A vs. B on the Marschak-Machina triangle.
        get_reference
            Get value of the parameter a, describing slope of the indifference curve.
    """
    def __init__(self):
        self.pars ={'a' : {'value' : (np.random.uniform, [0, 10.]),
                                'bounds' : [0, 10.],
                                'levels' : None
                               },
                    'epsilon' : {'value' : (np.random.uniform, [0, 0.5]),
                                'bounds' : [0, 0.5],
                                'levels' : None
                               },
                    'plA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'plB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}
                    }

        self.name = 'EU'
        self.parameters_of_interest = ['a', 'epsilon']
        super().__init__()
 

    def get_slope(self, phA, plA, phB, plB):
        """
        Get slope of A vs. B on the Marschak-Machina triangle. Note that if pmA > pmB, lottery B is riskier, and vice versa. Note that this only holds if stochastically dominated lotteries are removed.

        Parameters:
        -----------
            phA : float
                Probability of high outcome in lottery A.
            plA : float
                Probability of low outcome in lottery A.
            phB : float
                Probability of high outcome in lottery B.
            plB : float 
                Probability of low outcome in lottery B.

        Returns:
        --------
            Slope of the line connecting A to a riskier lottery B. 
        """

        # Determine which lottery is riskier 

        pmA = 1-plA-phA
        pmB = 1-plB-phB

        if pmA > pmB:
            slope = np.abs(phB-phA) / np.abs(plB-plA)
        else: 
            slope = np.abs(phA-phB) / np.abs(plA-plB)
        return slope


    def get_reference(self):
        """
        Get value of the parameter a, describing slope of the indifference curve.

        Returns:
        --------
            Value of parameter a.
        """
        return(self.pars["a"]["value"])


    def get_preference(self, a, slope_AB, pmA, pmB):
        # B is riskier (slopes calculated accordingly)
        if pmA > pmB:
            # There are some rounding errors, so checking equivalence like this
            if np.around(slope_AB, decimals=5) == np.around(a, decimals=5):
                A = 0.5
                B = 0.5
            elif slope_AB > a:
                A = 0
                B = 1
            elif slope_AB < a:
                A = 1
                B = 0
            else:
                A = 0.5
                B = 0.5

            return A, B
        
        # A is riskier (slopes calculated accordingly)
        # elif pmA < pmB: 
        else:
            # There are some rounding errors, so checking equivalence like this
            if np.around(slope_AB, decimals=5) == np.around(a, decimals=5):
                A = 0.5
                B = 0.5
            elif slope_AB > a:
                A = 1
                B = 0
            elif slope_AB < a:
                A = 0
                B = 1
            else:
                A = 0.5
                B = 0.5

            return A, B
        


class WEU(RiskyChoice):
    """
    Weighted expected utility. 

    Attributes:
    -----------
        name : str
            Name of the class (Expected Utility)
        pars : Dict
            Parameters BaseClass requires.
        parameters_of_interest: list
            Model parameters being optimised?

    Methods:
    --------
        get_slope
            Get slope of line segment connecting (x,y) to A.
        get_reference
            Get slope of line segment connecting (x,y) to B.
    """
    
    def __init__(self):
        # Parameters (x,y) should be outside of the MM-triangle
        self.pars ={'x' : {'value' : (np.random.uniform, [-100, 0.]),
                                'bounds' : [-100, 0.],
                                'levels' : None
                               },
                    'y' : {'value' : (np.random.uniform, [-100, 0.]),
                                'bounds' : [-100, 0.],
                                'levels' : None
                               },
                    'epsilon' : {'value' : (np.random.uniform, [0, 0.5]),
                                'bounds' : [0, 0.5],
                                'levels' : None
                               },
                    'plA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'phA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'plB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'phB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}
                    }
        self.name = 'WEU'
        self.parameters_of_interest = ['x', 'y', 'epsilon']
        super().__init__()
 

    def get_slope_A(self, phA, plA, x, y):
        """
        Get slope of line segment connecting (x,y) to A.

        Parameters:
        -----------
            phA : float
                Probability of high outcome in lottery A.
            plA : float
                Probability of low outcome in lottery A.
            x : float
                Parameter x of the model (point where indifference curves connect outside the MM-triangle).
            y : float
                Parameter y of the model (point where indifference curves connect outside the MM-triangle).
        
        Returns:
        --------
            The slope.
        """
        slope = np.abs(phA-y) / np.abs(plA-x)

        return slope


    def get_slope_B(self, phB, plB, x, y):
        """
        Get slope of line segment connecting (x,y) to B.

        Parameters:
        -----------
            phA : float
                Probability of high outcome in lottery B.
            plA : float
                Probability of low outcome in lottery B.
            x : float
                Parameter x of the model (point where indifference curves connect outside the MM-triangle).
            y : float
                Parameter y of the model (point where indifference curves connect outside the MM-triangle).
        
        Returns:
        --------
            The slope.
        """
        slope = np.abs(phB-y) / np.abs(plB-x)

        return slope


    def get_preference(self, slope_A, slope_B):

        # There are some rounding errors, so checking equivalence like this
        if np.around(slope_A, decimals=5) == np.around(slope_B, decimals=5):
            A = 0.5
            B = 0.5
        elif slope_A > slope_B:
            A = 1
            B = 0
        elif slope_A < slope_B:
            A = 0
            B = 1
        else:
            A = 0.5
            B = 0.5
        
        return A, B



class OPT(RiskyChoice):

    """
    Original prospect theory.

    Attributes:
    -----------
        name : str
            Name of the class (Expected Utility)
        pars : Dict
            Parameters BaseClass requires.
        parameters_of_interest: list
            Model parameters being optimised?

    Methods:
    --------
        calculate_utility
            Calculate utility without rank-normalisation.
    """
    def __init__(self):
        self.name = "OPT"
        self.pars ={'v' : {'value' : (np.random.uniform, [0, 1.]),
                                'bounds' : [0, 1.],
                                'levels' : None
                               },
                    'r' : {'value' : (np.random.uniform, [0.01, 1.]),
                                'bounds' : [0, 1.],
                                'levels' : None
                               },
                    'epsilon' : {'value' : (np.random.uniform, [0, 0.5]),
                                'bounds' : [0, 0.5],
                                'levels' : None
                               },
                    'plA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'plB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}
                    }
        self.parameters_of_interest = ['v', 'r', 'epsilon']
        super().__init__()


    def calculate_utility(self, pl, ph, r, v):

        """
        Calculate utility of a lottery without rank-normalisation.

        Parameters:
        -----------
            pl : float
                Probability of low outcome.
            ph : float
                Probability of high outcome.
            r : float
                Probability weighting parameter (model parameter).
            v : float
                The value of the middle outcome x_m \\in [0,1], assuming that x_l=0 and x_h=1.
        
        Returns:
        --------
            utility : float
                Utility of the given lottery.
        """

        pm = 1 - pl - ph

        if pl == 0:
            utility = self.probability_weighting(ph, r) * 1 + v * (1-self.probability_weighting(ph, r))
        else:
            utility = self.probability_weighting(ph, r) * 1 + v * self.probability_weighting(pm, r)

        return utility



class CPT(RiskyChoice):
    """
    Cumulative prospect theory.

    Attributes:
    -----------
        name : str
            Name of the class (Expected Utility)
        pars : Dict
            Parameters BaseClass requires.
        parameters_of_interest: list
            Model parameters being optimised?

    Methods:
    --------
        calculate_utility
            Calculate utility with rank-normalisation.
    """
    def __init__(self):
        self.name = "CPT"
        self.pars ={'v' : {'value' : (np.random.uniform, [0, 1.]),
                                'bounds' : [0, 1.],
                                'levels' : None
                               },
                    'r' : {'value' : (np.random.uniform, [0.01, 1.]),
                                'bounds' : [0, 1.],
                                'levels' : None
                               },
                    'epsilon' : {'value' : (np.random.uniform, [0, 0.5]),
                                'bounds' : [0, 0.5],
                                'levels' : None
                               },
                    'plA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phA' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}, 
                    'plB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None},
                    'phB' : {'value' : (np.random.uniform, [0, 1.]),
                              'bounds' : [0,1.],
                              'levels' : None}
                    }
        self.parameters_of_interest = ['v', 'r', 'epsilon']
        super().__init__()


    def calculate_utility(self, pl, ph, r, v):

        """
        Calculate utility of a lottery with rank-normalisation.

        Parameters:
        -----------
            pl : float
                Probability of low outcome.
            ph : float
                Probability of high outcome.
            r : float
                Probability weighting parameter (model parameter).
            v : float
                The value of the middle outcome x_m \\in [0,1], assuming that x_l=0 and x_h=1.
        
        Returns:
        --------
            utility : float
                Utility of the given lottery.
        """
        pm = 1 - pl - ph

        utility = self.probability_weighting(ph, r) * 1 + (self.probability_weighting(pm+ph, r)-self.probability_weighting(ph, r))*v

        return utility



class RiskyChoiceTask(BaseTask):
    def __init__(self):
        self.true_model = None
        self.model_priors = {'EU': .25, 'WEU': .25, 'OPT': .25, 'CPT': .25} 
        self.design_parameters = ['plA', 'phA', 'plB', 'phB']

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
        print(THIS_FILE)


    def instantiate_model(self, model_name):
        if model_name == 'EU':
            model = EU()
        elif model_name == 'WEU':
            model = WEU()
        elif model_name == 'OPT':
            model = OPT()
        elif model_name == 'CPT':
            model = CPT()
        model.hyper['n_prediction_trials'] = self.hyper['n_prediction_trials']
        model.parameters_of_interest.sort()
        return model


    def get_likelihood(self, data : np.array, model : Union[EU, WEU, OPT, CPT], pars : list, expt : list) -> np.array:
        """
        Get likelihood of given data.

        Attributes:
        -----------
            data : np.array 
                Data point -- note that this has shape (1,).
            model : Union[EU, WEU, OPT, CPT]
                Model being used for calculating preference.
            pars : list (of dict)
                Parameter values.
            expt : list (of dict)
                Experimental designs.

        Returns:
        --------
            likelihood : np.array
                Likelihood of a data -- note that this has shape (len(pars),)
        """
        likelihoods = []
        
        plA, phA, plB, phB = model.get_probabilities(expt)

        # If lotteries are not stochastically dominated, consider the models.
        for par in pars:
            if model.name == "EU":
                a = par[0] # 0.5 -> value in Cavagnaro (2013)
                epsilon = par[1]
                # Assign a higher value for the lottery that is preferred (comparison used later in getting choice probabilities)
                A, B = model.get_preference(a=a, slope_AB=model.get_slope(phA=phA, plA=plA, phB=phB, plB=plB), pmA=(1-phA-plA), pmB=(1-phB-plB) )

            if model.name == "WEU":
                epsilon = par[0]
                x = par[1]# 10 -> value in Cavagnaro (2013)
                y = par[2] # 10 -> value in Cavagnaro (2013)

                A, B = model.get_preference(slope_A = model.get_slope_A(phA=phA, plA=plA, x=x, y=y),
                                            slope_B = model.get_slope_B(phB=phB, plB=plB, x=x, y=y))

            if model.name == "OPT" or model.name == "CPT":
                epsilon = par[0]
                r = par[1] # 0.901 -> value in Cavagnaro (2013)
                v = par[2] # 0.601 -> value in Cavagnaro (2013)
                A = model.calculate_utility(ph=phA, pl=plA, r=r, v=v)
                B = model.calculate_utility(ph=phB, pl=plB, r=r, v=v)

            choice_probability=model.get_choice_probability(A=A, B=B, epsilon=epsilon)
            likelihood = np.where(data == 1, choice_probability[0], choice_probability[1])
            likelihoods.append(likelihood)

        return np.array(likelihoods).flatten()


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

    task=RiskyChoiceTask()

    task.hyper['random_design'] = args.randomd=='True' if args.randomd else task.hyper['random_design']
    task.hyper['true_likelihood'] = args.truelik=='True' if args.truelik else task.hyper['true_likelihood']
    task.hyper['ado'] = args.ado=='True' if args.ado else task.hyper['ado']
    task.hyper['minebed'] = args.minebed=='True' if args.ado else task.hyper['minebed']
    task.hyper['model_sel_rule'] = args.rule if args.rule else task.hyper['model_sel_rule']
    task.hyper['misspecify'] = int(args.misspecify) if args.misspecify else task.hyper['misspecify']
    task.hyper['observ_noise'] = float(args.noise) if args.noise else task.hyper['observ_noise']
    
    if task.hyper['observ_noise'] > 0:
    	task.hyper['n_prediction_trials'] = 10
    	task.hyper['corati_budget'] = 20
    	
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
