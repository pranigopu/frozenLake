# Importing the necessary context:
from CONTEXT import *

# CONTENTS:
# 1. Class `LinearWrapper`: Wrapping the environment to enable feature mapping
# 2. Method `linear_sarsa`: SARSA with linear action value function approximation
# 3. Method `linear_q_learning`: Q-Learning with linear action value function approximation
# 4. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code

#____________________________________________________________
# 1. Wrapping the environment to enable feature mapping

class LinearWrapper:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_states * self.n_actions

    #================================================

    # Mapping the given state paired with each action to vectors of features:

    def encode_state(self, s):
        # Initialising the feature matrix:
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            # Updating the feature matrix:
            features[a, i] = 1.0
        '''
        EXPECTED RESULT:
        `features` is such that each row i corresponds to an action i, each
        column corresponds to a state-action pair (see implementation notes
        for more clarity on the structure), and for each row i, 1.0 is assigned
        only to those indices that correspond the given state s and action i.
        Hence, each row vector is 1.0 at only one position and 0 in all others.
        '''

        return features

    #================================================

    # Obtaining the policy via decoding the feature matrix:

    def decode_policy(self, theta):
        # Initialising the policy & state value arrays:
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)
        #------------------------------------
        # Decoding the action-value function & obtaining policy & state values:
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)
            policy[s] = np.argmax(q)
            value[s] = np.max(q)
            '''
            NOTE ON THE NATURE OF `q`:
            `q` is calculated each time w.r.t. the given state `s`; do not
            consider it as an array mapping every possible state-action pair to
            a value, but rather an array that, given a state, maps every action
            to a value (effectively mapping the state-action values for the
            particular state)
            '''
        #------------------------------------
        # Returning obtained policy & state values:
        return policy, value

    #================================================

    # Resetting environment & encoding it as feature vector:

    def reset(self):
        return self.encode_state(self.env.reset())

    #================================================

    # Taking a step in environment & encoding next state as feature vector:

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    #================================================

    # Visualising the agent's performance (by inputs or using a policy):

    def render(self, policy=None, value=None):
        self.env.render(policy, value)

#____________________________________________________________
# 2. SARSA with linear action value function approximation

def linear_sarsa(wenv, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    NOTE ON THE ARGUMENTS:
    - `wenv`:
        - Wrapped object of a chosen environment model (ex. FrozenLake)
        - Contains mechanisms to map each state to a feature vector
        - Helpful in estimating action-value function as a linear function
        - Also helps decode feature vectors to estimated optimal policies
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
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(wenv.n_features)

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        features = wenv.reset()
        # NOTE: `features` here represents the initial state
        q = features.dot(theta)
        # NOTE: `q` here is the rewards per action for the initial state
        '''
        NOTE ON THE SHAPE OF `q`:
        `features.dot(theta)` produces `wenv.n_actions` dot products by
        applying dot product between `theta` and each row vector of the matrix
        `features`.
        '''
        a = e_greedy(q, epsilon[i], wenv.n_actions, random_state)

        done = False
        while not done:
            next_features, r, done = wenv.step(a)
            # NOTE: `next_features` here represents the next state reached

            # Obtaining part of the temporal difference of `(features, a)`:
            delta = r - q[a]

            # Selecting action `a` for `next_features`:
            # NOTE: Selection is done by epsilon greedy policy based on `q`
            q = next_features.dot(theta)
            # NOTE: `q` here is the rewards per action for the next state
            next_a = e_greedy(q, epsilon[i], wenv.n_actions, random_state)
            # NOTE: `next_a` is the action taken in the next state

            # Obtaining the full temporal difference of `(features, a)`:
            delta += gamma*q[next_a]

            # Updating model parameters `theta`:
            theta += eta[i]*delta*features[a]
            # `next_features[a]` is feature vector for state `s` & action `a`

            # Moving to the next state & its corresponding action:
            features, a = next_features, next_a

    # Returning the parameter vector `theta`:
    return theta

#____________________________________________________________
# 3. Q-Learning with linear action value function approximation

def linear_q_learning(wenv, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    NOTE ON THE ARGUMENTS:
    Same as for the function `linear_sarsa`.
    '''
    # Setting random state with `seed` for enabling replicability:
    random_state = np.random.RandomState(seed)

    # Initialising key parameters:
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(wenv.n_features)

    #================================================

    # LEARNING LOOP
    for i in range(max_episodes):
        # NOTE: i ==> episode number
        # Beginning at the initial state before each episode:
        features = wenv.reset()
        # NOTE: `features` here represents the initial state
        q = features.dot(theta)
        # NOTE: `q` here is the rewards per action for the initial state
        '''
        NOTE ON THE SHAPE OF `q`:
        See the corresponding comment for the function `linear_sarsa`.
        '''
        a = e_greedy(q, epsilon[i], wenv.n_actions, random_state)

        done = False
        while not done:
            next_features, r, done = wenv.step(a)
            # NOTE: `next_features` here represents the next state reached

            # Obtaining part of the temporal difference of `(features, a)`:
            delta = r - q[a]

            # Selecting action `a` for `next_features`:
            # NOTE: Selection is done by epsilon greedy policy based on `q`
            q = next_features.dot(theta)
            # NOTE: `q` here is the rewards per action for the next state
            max_a = np.argmax(q)
            # NOTE: `max_a` is the action maximising `q` for next state

            # Obtaining the full temporal difference of `(features, a)`:
            delta += gamma*q[max_a]

            # Updating model parameters `theta`:
            theta += eta[i]*delta*features[a]
            # `next_features[a]` is feature vector for state `s` & action `a`

            # Moving to the next state & its corresponding action:
            features, a = next_features, e_greedy(q, epsilon[i], wenv.n_actions, random_state)

    # Returning the parameter vector `theta`:
    return theta

#____________________________________________________________
# 4. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code.

if __name__ == '__main__':
    # Defining the parameters:
    env = FrozenLake(lake=LAKE['small'], slip=0.1, max_steps=None, seed=0)
    # NOTE: Putting `max_steps=None` makes it default to the grid size
    wenv = LinearWrapper(env)
    max_episodes = 2000
    eta = 1
    gamma = GAMMA
    epsilon = 1

    # Running the functions:
    _LSARSA = linear_sarsa(wenv, max_episodes, eta, gamma, epsilon, 0)
    _LQLearning = linear_q_learning(wenv, max_episodes, eta, gamma, epsilon, 0)
    
    LSARSA = wenv.decode_policy(_LSARSA)
    LQLearning = wenv.decode_policy(_LQLearning)
    labels = ("linear sarsa", "linear q-learning")

    # Displaying results:
    displayResults((LSARSA, LQLearning), labels, env)
