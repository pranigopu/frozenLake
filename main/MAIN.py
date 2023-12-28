# Importing the necessary context:
from CONTEXT import *
# NOTE: All contents from module `Q1_environment were imported within module `CONTEXT`
from Q2_tabularModelBasedMethods import *
from Q3_tabularModelFreeMethods import *
from Q4_nonTabularModelFreeMethods import *
from Q5_deepReinforcementLearning import *

#____________________________________________________________
# Main function

def main():
    # Preliminary definitions:
    seed = 0
    lake = LAKE['small']
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    gamma = 0.9
    max_episodes = 4000
    H1 = '\n================================================\n'
    H2 = '------------------------------------\n'

    #================================================

    print(f'{H1}Model-based algorithms')
    '''
    ARGUMENTS:
    - Environment
    - Discount factor
    - Error margin for convergence of value function
    - Maximum iterations
    '''

    print(f'{H2}Policy iteration')
    args = [env, gamma, 0.001, 128]

    policy, value, i = policy_iteration(*args)
    env.render(policy, value)

    print(f'{H2}Value iteration')
    policy, value, i = value_iteration(*args)
    env.render(policy, value)

    #================================================

    print(f'{H1}Tabular model-free algorithms')
    '''
    ARGUMENTS:
    - Environment
    - Maximum episodes to iterate over (1 episode ==> play until absorbed)
    - Initial learning rate
    - Discount factor
    - Exploration factor (epsilon)
    - Pseudorandom number generator seed
    '''
    args = [env, max_episodes, 0.5, gamma, 1.0, seed]

    print(f'{H2}SARSA')
    env.resetRandomState()
    policy, value = sarsa(*args)
    env.render(policy, value)

    print(f'{H2}Q-learning')
    env.resetRandomState()
    policy, value = q_learning(*args)
    env.render(policy, value)

    #================================================

    print(f'{H1}Non-tabular model-free algorithms')
    # NOTE: Except for environment, all arguments are the same as before
    args[0] = LinearWrapper(env)

    print(f'{H2}Linear SARSA')
    env.resetRandomState()
    parameters = linear_sarsa(*args)
    policy, value = args[0].decode_policy(parameters)
    args[0].render(policy, value)

    print(f'{H2}Linear Q-learning')
    env.resetRandomState()
    params = linear_q_learning(*args)
    policy, value = args[0].decode_policy(parameters)
    args[0].render(policy, value)

    #================================================

    print(f'{H1}Deep Q-network learning')
    # ARGUMENTS:
    args = [FrozenLakeImageWrapper(env), # Wrapped environment
            max_episodes,                # Maximum episodes
            0.001,                       # Learning rate
            gamma,                       # Discount factor
            0.2,                         # Exploration factor (epsilon)
            32,                          # Batch size (random sample size)
            4,                           # Target update frequency
            256,                         # Replay buffer size
            3,                           # Kernel size
            4,                           # Convolution layer output channels
            8,                           # Fully-connected layer output features
            4]                           # Pseudorandom number generator seed
    
    # NOTE 1: Replay buffer is the dataset from which random samples are drawn
    # NOTE 2: Each tuple in replay buffer denotes an observed state transition

    env.resetRandomState()
    dqn = deep_q_network_learning(*args)
    policy, value = args[0].decode_policy(dqn)
    args[0].render(policy, value)

# Run function of the current file is the file being executed:
if __name__ == '__main__': main()
