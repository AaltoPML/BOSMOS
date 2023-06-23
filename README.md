# Online Simulator-Based Experimental Design for Cognitive Model Selection

This repository contains all the necessary code to replicate the experiments in our paper. The main algorithms are found in the `corati` folder, while specific applications can be identified within the `examples` directory.

## Installation Guide

Ensure your Python interpreter version is v3.9 or lower.
1. Clone the repository: 
    ```
    git clone <repository_url>
    ```
2. Create a virtual environment:
    ```
    python3 -m venv .env
    ```
3. Activate the virtual environment:
    ```
    source .env/bin/activate
    ```
4. Install dependencies:
    ```
    python3 -m pip install -r requirements.txt
    ```
    or 
    ```
    pip3 install -r requirements.txt
    ```
    If you encounter any errors during installation due to unsupported package versions, remove the problematic package from `requirements.txt` and install it manually. For instance:
    ```
    python3 -m pip install scipy --pre
    ```

5. Install the package:
    ```
    python3 -m pip install -e .
    ```
    > Note: To run the MINEBED method, you need to install it separately from [here](https://github.com/stevenkleinegesse/minebed).

## Quick-Start Guide

You will find four test cases in the `examples` directory, which you can run to replicate the experiments discussed in the paper. The names of the example scripts match their corresponding tasks.

1. `simple_example.py` (Demonstrative Example)
2. `memory.py` (Memory Retention Task)
3. `signal_detection_sequential.py`
4. `risky_choice.py`

To start an experiment, run a test case file. For instance: 
```
python3 signal_detection_sequential.py
```
or
```
python3 memory.py
```
In the interactive contextual menu, you should first generate synthetic participants (`g`), which will store them in corresponding output folders. Afterwards, you can run the main loop (`c`). 

>Note: The signal detection task additionally requires training of the inference model (`t`).

```
RET  	explore
t 		train
v		verify
g 		generate synthetic participants
c 		corati
q 		quit
```

You can also run the experiments using the command line for computational efficiency. For instance:

```
python3 memory.py --start=experiment_id --x=c --randomd=False --truelik=False --ado=False --minebed=True
```

You can execute the specific methods with the following configurations:
1. BOSMOS with  `{--randomd=False --truelik=False --ado=False --minebed=False}`
2. ADO with `{--randomd=False --truelik=True --ado=True --minebed=False}`, 
3. LBIRD with `{--randomd=True --truelik=True --ado=False --minebed=False}` 
4. MINEBED with `{--randomd=False --truelik=False --ado=False --minebed=True}`. 

> Note: The `experiment_id` argument uniquely identifies experiment filenames. Ensure that its value is unique for each batch.

Detailed instructions for each of these steps, including generating synthetic participants, training an inference model (for the signal detection task), running inference, and interpreting and printing results, can be found in the subsequent sections.


## Generating Synthetic Participants

Generate synthetic participants using the argument `--x=g`. To do this, navigate to the folder of the example (`./examples/simple_example`) and execute:

```
python3 simple_example.py --x=g
```
This creates an `output` folder in the example's directory with a `synthetic_participants` file containing the parameters of the generated synthetic participants. The terminal will output the parameters for each participant, one participant per line:


```
{'mean_pos': {'value': 1.9796845556705933, 'bounds': [0, 5], 'levels': 5}, 'design': {'value': (<built-in method uniform of numpy.random.mtrand.RandomState object at 0x7f27f0965840>, [0.001, 5]), 'bounds': [0.001, 5], 'levels': 5}}
...
{'mean_neg': {'value': -4.22267435958699, 'bounds': [-5, 0], 'levels': 5}, 'design': {'value': (<built-in method uniform of numpy.random.mtrand.RandomState object at 0x7f27f0965840>, [0.001, 5]), 'bounds': [0.001, 5], 'levels': 5}}
```

> Note: To speed up this step, you can modify the task hyperparameters `task.hyper` set in their respective test case files. For instance, generating synthetic participants for the signal sequential task can take hours. You can reduce `n_training_timesteps`, `n_prediction_trials` (which may affect the quality of participant models), and `n_participants` hyperparameters (line 399 in `signal_detection_sequential.py`) to speed up the generation of synthetic participants.


## Pre-Training an Inference Model for Signal Detection

Before conducting inference, pre-training the PPO model for the signal detection task is necessary. Use the following command for pre-training:

```
python3 signal_detection_sequential.py --x=t
```

This command will result in the following output:

```
Using cpu device
Wrapping the env in a DummyVecEnv.
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.55     |
|    ep_rew_mean     | 0.378    |
| time/              |          |
|    fps             | 760      |
|    iterations      | 1        |
|    time_elapsed    | 2        |
|    total_timesteps | 2048     |
---------------------------------
```

It will continue showing intermediate training results until the training is finished:

```
------------------------------------------
| rollout/                |              |
|    ep_len_mean          | 1.26         |
|    ep_rew_mean          | 0.695        |
| time/                   |              |
|    fps                  | 606          |
|    iterations           | 5            |
|    time_elapsed         | 16           |
|    total_timesteps      | 10240        |
| train/                  |              |
|    approx_kl            | 0.0046283477 |
|    clip_fraction        | 0.294        |
|    clip_range           | 0.0362       |
|    entropy_loss         | -0.992       |
|    explained_variance   | 0.232        |
|    learning_rate        | 0.000181     |
|    loss                 | 1.78         |
|    n_updates            | 40           |
|    policy_gradient_loss | -0.00716     |
|    value_loss           | 3.8          |
------------------------------------------
```
The trained model is stored in a new folder `signal_detection/sd_output`.

## Running Inference

To run BOSMOS inference on the first five synthetic participants, use:
```
python3 simple_example.py --x=c
```
This command runs BOSMOS with the MAP decision-rule, showing intermediate results and completing with a summary of estimated parameter vectors, their respective ground-truths, and the total running time of the experiment: 
```
Analyse participant 1 of 5: neg_mean, [-4.865634293078703]

Iteration 1/20:

Designing an experiment...
Using the default optimization for the chosen method...
Conducting an experiment with the design vector [0.32700628].
Inference for parameters:  ['mean_pos']
Extracting the posterior...
Using BOSMOS for likelihood-free inference...
Inference for parameters:  ['mean_neg']
Extracting the posterior...
Using BOSMOS for likelihood-free inference...
Estimated parameters: {'pos_mean': {'mean_pos': 0.10597906009086265}, 'neg_mean': {'mean_neg': -4.896966146722128}}
Model posterior:  {'pos_mean': 0.0609, 'neg_mean': 0.9391}
Recording history...
...
Iteration 20/20:

Designing an experiment...
Using the default optimization for the chosen method...
Conducting an experiment with the design vector [0.001].
Inference for parameters:  ['mean_pos']
Extracting the posterior...
Using BOSMOS for likelihood-free inference...
Inference for parameters:  ['mean_neg']
Extracting the posterior...
Using BOSMOS for likelihood-free inference...
Estimated parameters: {'pos_mean': {'mean_pos': 0.0}, 'neg_mean': {'mean_neg': -4.883455563649417}}
Model posterior:  {'pos_mean': 0.0, 'neg_mean': 1.0}
Recording history...
```
In the example above, the participant's model is `neg_mean`, whose ground-truth parameters are `[-4.865634293078703]`. First, the algorithm designs an experiment, collects the data and then conducts likelihood-free inference for all available models, which are `pos_mean` and `neg_mean`. Finally, it shows iterative results for the paramaeter estimates and model posterior, after which it proceeds to the next iteration. The algorithm finishes with the output of all estimated parameter vectors, their respective ground-truths, and the total time of the experiments:
```
Estimates:  [[-4.883455563649417], [-4.933663326758355], [-2.964718297887275], [-1.3660945583041408], [3.9783437645294777]] 
Ground truth:  [[['mean_neg'], [-4.865634293078703]], [['mean_neg'], [-4.924766168172309]], [['mean_neg'], [-2.9752798145632404]], [['mean_neg'], [-1.357460983651694]], [['mean_pos'], [4.003393993084183]]]
Running time: 0.41316359043121338 hours
```
The results are stored in the `examples/simple_example` folder.

To conduct experiments for subsequent batches of synthetic participants, pass the argument `--start=experiment_id`, where `experiment_id` ranges from 0 to 19 (0 to 100 for MINEBED), for parallelized experiments using a computational cluster:
```
python3 simple_example.py --start=experiment_id --x=c
```
Various methods used in the paper can be executed with their specific configurations (refer to the Quick-Start section), such as the ADO command:
```
python3 simple_example.py --x=c --randomd=False --truelik=True --ado=True --minebed=False
```
which will result in the output using the following format:
```
Analyse participant 1 of 5: neg_mean, [-4.865634293078703]

Iteration 1/20:

Designing an experiment...
Using adaptive design optimization...
Conducting an experiment with the design vector [0.10478985].
Inference for parameters:  ['mean_pos']
We recommend having at least 10 initialization points for the initialization (now 0)
Extracting the posterior...
Using the true likelihood...
...
Estimates:  [[-4.864968934829099], [-4.55117222337257], [-2.974751806655957], [-1.2772441107955494], [4.002353000986053]] 
Ground truth:  [[['mean_neg'], [-4.865634293078703]], [['mean_neg'], [-4.924766168172309]], [['mean_neg'], [-2.9752798145632404]], [['mean_neg'], [-1.357460983651694]], [['mean_pos'], [4.003393993084183]]]
Running time: 0.21159803370634715 hours
```

## Printing the Results
To print the results of the experiments, execute the example script with the argument `--x=v`:
```
python3 simple_example.py --x=v
```
This outputs the performance results for every method used in the experiments (only ADO in the example below): 
```
Performance after 1, 2, 4, 20, 100 design trials:

ado
Parameter estimate error (mean +- std): 
['0.05 \\pm$ 0.06', '0.04 \\pm$ 0.05', '0.03 \\pm$ 0.04', '0.01 \\pm$ 0.01']
Model accuracy: 
pos_mean
[0.98, 0.98, 0.98, 1.00]
neg_mean
[0.96, 1.00, 1.00, 1.00]
Time: 10.46 \pm$ 1.06
...

Behavioural fitness (mean +- std):
ado
['0.03 \\pm$ 0.03', '0.02 \\pm$ 0.02', '0.02 \\pm$ 0.02', '0.01 \\pm$ 0.01']
...
```
The data is subsequently used in the scripts `figures/fig2.py` and `figures/fig3.py` to generate figures for the paper.


## Running Misspecification Experiments

For the misspecification examples from Appendix E, use the `--noise` argument to add a specific percentage of noise. For instance, adding 10% of noise to the risky choice experiment:
```
python3 risky_choice.py --x=c --noise=0.1
```
This automatically adjusts the number of observations to match the experimental setup.
> Note: `--noise=0.01` would not add noise to the observations, but it would adjust the experimental setup to correspond to the 0% noise model we use in Appendix E.

To run the misspecified version of sequential signal detection, use the `signal_detection_sequential.py` file from `signal_detection_misspecified` folder. It only uses the PR model, so pre-training the inference model is not required.

For inference with a variable number of parameters, use the `--misspecify` argument, which ranges from 1-4, indicating the number of parameters in the reduced model. For example, to use the inference model with only two parameters, use:
```
python3 signal_detection_sequential.py --x=c --misspecify=2
```
You can print the results as described in the previous sections.


## Citation
```
@article{aushev2023online,
  title={Online simulator-based experimental design for cognitive model selection},
  author={Aushev, Alexander and Putkonen, Aini and Clarte, Gregoire and Chandramouli, Suyog and Acerbi, Luigi and Kaski, Samuel and Howes, Andrew},
  journal={arXiv preprint arXiv:2303.02227},
  year={2023}
}
```

the arxiv version of the paper: https://arxiv.org/abs/2303.02227