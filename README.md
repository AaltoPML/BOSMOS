# Online simulator-based experimental design for cognitive model selection

This package contains all code required to replicate the experiments in the paper. The main algorithms and specific applications can be found in the 'corati' and 'examples' folders respectively.


## Installation:

Python3 is required, make sure the Python interpreter <= v3.9.

1. Clone the respository with ```git clone``` command;

2. Create the virtual environment:

```python3 -m venv .env```

3. Activate virtual environment:

```source .env/bin/activate```

4. Install dependencies:

```python3 -m. pip install -r requirements.txt```

or 

```pip3 install -r requirements.txt```

NOTE: To run the MINEBED method, you need to install it separately from: https://github.com/stevenkleinegesse/minebed

5. Install the package:

```pip3 install -e .```



## Running the experiments

There are four test cases in the 'examples' folder to run: simple_example\simple_example.py ('demonstative example' in the paper), memory\memory.py (memory retention task), signal_detection\signal_detection_sequential.py and risky_choice\risky_choice.py. To start the experiments, run the test case file, for instance: 

```python3 ./signal_detection_sequential.py```

OR

```python3 ./memory.py```

In the contextual menu, first you need to generate synthetic participants (g) -- it will store them in corresponding output folders. Only then, you can run the main loop (c) (signal detection also requires you to train the unified model (t) ).

Be mindful of the task hyperparameters (task.hyper variables ), which are set in the respective test case files. Generating synthetic participants for the signal sequential task may take up to several hours -- you can reduce n_training_timesteps, n_prediction_trials (these two will hinder the quality of the participant models) and n_participants to speed up this process.

Alternatively, for computational reasons (the scripts can be executed in parallel), you can run the experiments through the command line:

```python3 memory.py --start=experiment_id --x=c --randomd=False --truelik=False --ado=False --minebed=True```

where BOSMOS is executed with the configuration {--randomd=False --truelik=False --ado=False --minebed=False}, ADO with {--randomd=False --truelik=True --ado=True --minebed=False}, EI with {--randomd=True --truelik=True --ado=False --minebed=False) and MINEBED with {--randomd=False --truelik=False --ado=False --minebed=True}. We recommend training synthetic participants with task.hyper['n_participants']=100 in a single process, and then parallelize using batches of 5 (task.hyper['n_participants'] = 5 in test cases files). 

The results can be printed through the command:

```python3 memory.py --x=l```

which may take up to two hours for the signal detection task.

NOTE: ```experiment_id``` automatically adjusts filenames of the experiments, meanning that for each batch its value should be unique.
