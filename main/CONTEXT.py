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
# 4. Method `e_greedy`: Epsilon-greedy policy implementation

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

#____________________________________________________________
# 4. Epsilon-greedy policy implementation

# NOTE: The following is generalised for the tabular & non-tabular methods

def e_greedy(q, e, n_actions, random_state, s=None):
    '''
    NOTE ON THE ARGUMENTS:
    - `q`: One of the following...
        - The matrix of action-values for each state
        - The array of action-values for a given state
    - `e`: The exploration factor epsilon
    - `n_actions`: The number of actions an agent can take from any state
    - `random_state`: The set `numpy.random.RandomState` object
    - `s`: The given state

    If `s=None`, `q` is the array of action-values for a given state. Else,
    `q` is the matrix of action-values for each state.
    '''

    # Storing the action-values for the given state `s` (if `s` is given):
    if s != None: q = q[s]

    # The epsilon-greedy policy:
    if random_state.rand() < e:
        return random_state.randint(0,n_actions)
    else:
        # Obtaining all actions that maximise action-value from state `s`:
        best = [a for a in range(n_actions) if np.allclose(np.max(q), q[a])]
        # Breaking the tie (if tie exists) using random selection:
        return random_state.choice(best)
