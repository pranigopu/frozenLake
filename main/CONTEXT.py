# CONTENTS
# 1. Imports:
#   - Module `Q1_environment`
#   - Library `numpy`: Imported within module `Q1_environment`
#   - Imports needed for `Q5_deepReinforcementLearning.py`
#       - Library `torch`
#       - Class `deque` from module `collections`
# 2. Global constants
#   - `SEED`: Pseudorandom number generator seed
#   - `GAMMA`: Discount factor
#   - `HUGE`: Arbitrarily large integer (practically infinity)
# 3. Method `displayResults`: Policy & state-value display function

#____________________________________________________________
# 1. Imports

# Module `Q1_environment`:
from Q1_environment import *
# NOTE: Library `numpy`: Imported within module `Q1_environment`

# Imports needed for `Q5_deepReinforcementLearning.py`:
# Library `torch`:
import torch
# Class `deque` from module `collections`:
from collections import deque

#____________________________________________________________
# 2. Global constants

# Pseudorandom number generator seed:
SEED = 0
# Discount factor as 0.9 (as instructed for the assignment):
GAMMA = 0.9
# Arbitrarily large integer (practically infinity):
HUGE = 10**100

#____________________________________________________________
# 3. Policy & state-value display function

'''
PURPOSE:
The following function allows the standarised presentation of the results
of various policy-finding functions that simultaneously obtain state values
(i.e. estimate the state value function using an array).
'''

def displayResults(results, labels, env):
    '''
    - `results`
        - Contains k pairs, each pair consisting (in order)
            - The deterministic policy (array of actions to be taken per state)
            - The estimated state values (array of values evaluated per state)
        - The policy and state values are those derived by a specific algorithm
        - Results for each are displayed organisedly for easy testing
    - `labels`
        - Contains the names or labels to be given for each pair of results
        - Generally, it is the name of the algorithm used to derive the results
    - `env` is the instance of the environment model class used for the results
    '''
    # Visualisation of agent's performance:
    for result, label in zip(results, labels):
        print('\n================================================\n')
        print(f'AGENT PERFORMANCE AFTER {label.upper()}\n')
        state = env.reset()
        env.render(policy=result[0], value=result[1])