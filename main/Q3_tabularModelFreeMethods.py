# Importing the necessary context:
from CONTEXT import *

# CONTENTS:
# 1. Method `sarsa`: SARSA control
# 2. Method `q_learning`: Q-learning
# 3. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code

#____________________________________________________________
# 1. SARSA control

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    NOTE ON THE ARGUMENTS:
    - `env`: Object of the chosen environment class (ex. FrozenLake)
    - `max_episodes`: Upper limit of episodes the agent can go through
    - `eta`:
        - Initial learning rate
        - The learning rate is meant to decrease linearly over episodes
    - `gamma`:
        - Discount factor
        - Not subject to change over time
    - `epsilon`:
        - Initial exploration factor
        - Exploration factor is w.r.t. epsilon-greedy policy
        - Denotes the chance of selecting a random state
        - The exploration factor is meant to decrease linearly over episodes
    - `seed`:
        - Optional seed for pseudorandom number generation
        - By default, it is `None` ==> random seed will be chosen
    '''
    # INITIALISATION
    # Setting random state with `seed` for enabling replicability:
    random_state = np.random.RandomState(seed)

    # Initialising key parameters:
    # 1. Array of linearly decreasing learning rates:
    eta = np.linspace(eta, 0, max_episodes)
    # 2. Array of linearly decreasing exploration factors:
    epsilon = np.linspace(epsilon, 0, max_episodes)
    # 3. Array of state-action values:
    q = np.zeros((env.n_states, env.n_actions))
    '''
    NOTE ON THE NEW `eta` & `epsilon`:
    The above `eta` and `epsilon` are arrays formed by taking the initial
    learning rate and exploration factor given and creating an array of
    linearly decreasing values. Hence:
    - eta[i] is the learning rate for the ith episode
    - epsilon[i] is the exploration factor for the ith episode
    '''

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        s = env.reset()
        # Selecting action `a` for `s` by epsilon-greedy policy based on `q`:
        a = e_greedy(q, epsilon[i], env.n_actions, random_state, s)
        # While the state is not terminal:
        '''
        HOW TO CHECK IF A STATE IS TERMINAL?
        A terminal state is one wherein either the maximum number
        of steps for taking actions is reached or the agent reaches the
        absorbing state or the agent transitions to the absorbing state for any
        action. In this implementation, the check for whether the terminal
        state is reached is handled by the `done` flag of the `env.step`
        function; if `False`, continue, else consider the state as terminal.
        '''
        done = False
        while not done:
            # Next state & reward after taking `a` from `s`:
            next_s, r, done = env.step(a)
            # NOTE: `env.step` automatically updates the state of the agent

            # Selecting action `next_a` for `next_s`:
            # NOTE: Selection is done by epsilon greedy policy based on `q`
            next_a = e_greedy(q, epsilon[i], env.n_actions, random_state, next_s)

            # Updating the action-value for the current state-action pair:
            # USING: Temporal difference for (s,a) with epsilon-greedy policy
            q[s, a] = q[s, a] + eta[i]*(r + gamma*q[next_s, next_a] - q[s, a])

            # Moving to the next state & action pair:
            s, a = next_s, next_a

    #================================================

    # FINAL RESULTS
    # Obtaining the estimated optimal policy
    # NOTE: Policy = The column index (i.e. action) with max value per row
    policy = q.argmax(axis=1)
    # Obtaining the state values w.r.t the above policy:
    # NOTE: Value = Max value per row
    value = q.max(axis=1)
    # Returning the above obtained policy & state value array:
    return policy, value

#____________________________________________________________
# 2. Q-Learning

def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    NOTE ON THE ARGUMENTS:
    Same as for the function `sarsa`.
    '''
    # INITIALISATION
    # Setting random state with `seed` for enabling replicability:
    random_state = np.random.RandomState(seed)

    # Initialising key parameters:
    # 1. Array of linearly decreasing learning rates:
    eta = np.linspace(eta, 0, max_episodes)
    # 2. Array of linearly decreasing exploration factors:
    epsilon = np.linspace(epsilon, 0, max_episodes)
    # 3. Array of state-action values:
    q = np.zeros((env.n_states, env.n_actions))
    '''
    NOTE ON THE NEW `eta` & `epsilon`:
    Check corresponding comment for the function `sarsa`.
    '''

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        s = env.reset()
        # Selecting action `a` for `s` by epsilon-greedy policy based on `q`:
        a = e_greedy(q, epsilon[i], env.n_actions, random_state, s)
        # While the state is not terminal:
        '''
        HOW TO CHECK IF A STATE IS TERMINAL?
        Check corresponding comment for the function `sarsa`.
        '''
        done = False
        while not done:
            # Next state & reward after taking `a` from `s`:
            next_s, r, done = env.step(a)
            # NOTE: `env.step` automatically updates the state of the agent

            # Updating the action-value for the current state-action pair:
            # USING: Temporal difference for (s,a) with greedy policy
            q[s, a] = q[s, a] + eta[i]*(r + gamma*np.max(q[next_s]) - q[s, a])

            # Moving to the next state & action pair:
            s, a = next_s, e_greedy(q, epsilon[i], env.n_actions, random_state, s)

    #================================================

    # FINAL RESULTS
    # Obtaining the estimated optimal policy
    # NOTE: Policy = The column index (i.e. action) with max value per row
    policy = q.argmax(axis=1)
    # Obtaining the state values w.r.t the above policy:
    # NOTE: Value = Max value per row
    value = q.max(axis=1)
    # Returning the above obtained policy & state value array:
    return policy, value

#____________________________________________________________
# 3. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code.

if __name__ == '__main__':
    # Defining the parameters:
    env = FrozenLake(lake=LAKE['small'], slip=0.1, max_steps=None, seed=0)
    # NOTE: Putting `max_steps=None` makes it default to the grid size
    max_episodes = 2000
    eta = 1
    gamma = GAMMA
    epsilon = 1

    # Running the functions:
    SARSA = sarsa(env, max_episodes, eta, gamma, epsilon, 0)
    QLearning = q_learning(env, max_episodes, eta, gamma, epsilon, 0)
    labels = ("sarsa", "q-learning")

    # Displaying results:
    displayResults((SARSA, QLearning), labels, env)
