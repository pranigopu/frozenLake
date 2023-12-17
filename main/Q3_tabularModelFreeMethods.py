# Importing `Q1_environment` module:
from Q1_environment import *

# CONTENTS:
# 1. Method `sarsa`: SARSA control
# 2. Method `q_learning`: Q-learning
# 3. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code.

#____________________________________________________________
# 1. SARSA control

def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    NOTE ON THE ARGUMENTS:
    - `env`: Object of the chosen environment class (ex. FrozenLake)
    - `max_episodes`: Upper limit of episodes the agent can go through
    - `eta`:
        - Array of learning rates w.r.t. episodes
        - `eta[i]` ==> learning rate for the ith episode
        - Meant to decrease linearly over time
    - `gamma`:
        - Discount factor
        - Not subject to change over time
    - `epsilon`:
        - An array of exploration factors w.r.t. episodes
        - `epsilon[i]` ==> exploration factor for the ith episode
        - Exploration factor is w.r.t. epsilon-greedy policy
        - Denotes the chance of selecting a random state
    - `seed`:
        - Optional seed for pseudorandom number generation
        - By default, it is `None` ==> random seed will be chosen
    '''
    # EPSILON-GREEDY POLICY
    # Implementing the epsilon-greedy policy as a lambda function:
    e_greedy = lambda s: {True: np.random.randint(0, env.n_actions),
                          False: policy[s]}[np.random.rand() < epsilon]
    
    #================================================

    # INITIALISATION
    # Choosing a random initial state:
    random_state = np.random.RandomState(seed)
    # Initialising array of learning rates:
    eta = np.linspace(eta, 0, max_episodes)
    # Initialising array of exploration factors:
    epsilon = np.linspace(epsilon, 0, max_episodes)
    # Initialising of state-action values:
    q = np.zeros((env.n_states, env.n_actions))

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        s = env.reset()
        # Selecting action `a` for `s` by epsilon-greedy policy based on `q`:
        a = e_greedy(s)
        # While the state is not terminal:
        '''
        HOW TO CHECK IF A STATE IS TERMINAL?
        A terminal state is one wherein the agent will transition to the
        absorbing state for any action it takes. In our implementation, the
        states are a sequence of integers starting from 0, with the greatest
        integer `env.n_states-1` being reserved for the absorbing state.
        If a state s is terminal, then the probability of transitioning to
        the absorbing state for any action is 1, hence the following condition.
        '''
        while env.p(env.n_states-1, s, a) != 1:
            # Next state & reward after taking `a` from `s`:
            next_s, r = env.draw(s, a)
            # Selecting action `next_a` for `next_s`:
            # NOTE: Selection is done by epsilon greedy policy based on `q`
            next_a = e_greedy(next_s)
            # Updating the action reward for the current state-action pair:
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
    - `env`: Object of the chosen environment class (ex. FrozenLake)
    - `max_episodes`: Upper limit of episodes the agent can go through
    - `eta`:
        - Array of learning rates w.r.t. episodes
        - `eta[i]` ==> learning rate for the ith episode
        - Meant to decrease linearly over time
    - `gamma`:
        - Discount factor
        - Not subject to change over time
    - `epsilon`:
        - An array of exploration factors w.r.t. episodes
        - `epsilon[i]` ==> exploration factor for the ith episode
        - Exploration factor is w.r.t. epsilon-greedy policy
        - Denotes the chance of selecting a random state
    - `seed`:
        - Optional seed for pseudorandom number generation
        - By default, it is `None` ==> random seed will be chosen
    '''
    # EPSILON-GREEDY POLICY
    # Implementing the epsilon-greedy policy as a lambda function:
    e_greedy = lambda s: {True: np.random.randint(0, env.n_actions),
                          False: policy[s]}[np.random.rand() < epsilon]
    
    #================================================

    # INITIALISATION
    # Choosing a random initial state:
    random_state = np.random.RandomState(seed)
    # Initialising array of learning rates:
    eta = np.linspace(eta, 0, max_episodes)
    # Initialising array of exploration factors:
    epsilon = np.linspace(epsilon, 0, max_episodes)
    # Initialising of state-action values:
    q = np.zeros((env.n_states, env.n_actions))

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        s = env.reset()
        # Selecting action `a` for `s` by epsilon-greedy policy based on `q`:
        a = e_greedy(s)
        # While the state is not terminal:
        '''
        HOW TO CHECK IF A STATE IS TERMINAL?
        A terminal state is one wherein the agent will transition to the
        absorbing state for any action it takes. In our implementation, the
        states are a sequence of integers starting from 0, with the greatest
        integer `env.n_states-1` being reserved for the absorbing state.
        If a state s is terminal, then the probability of transitioning to
        the absorbing state for any action is 1, hence the following condition.
        '''
        while env.p(env.n_states-1, s, a) != 1:
            # Next state & reward after taking `a` from `s`:
            next_s, r = env.draw(s, a)
            # Updating the action reward for the current state-action pair:
            # USING: Temporal difference for (s,a) with greedy policy
            q[s, a] = q[s, a] + eta[i]*(r + gamma*np.max(q[next_s]) - q[s, a])
            # Moving to the next state & action pair:
            s = next_s
    
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
    env = FrozenLake(lake['small'], 0.1, 100)
    max_episodes = 2000
    eta = 1
    gamma = 0.9
    epsilon = 1

    # Running the functions:
    SARSA = sarsa(env, max_episodes, eta, gamma, epsilon)
    QLearning = q_learning(env, max_episodes, eta, gamma, epsilon)
    labels = ("sarsa", "q-learning")

    # Displaying results:
    displayResults((SARSA, QLearning), labels, env)