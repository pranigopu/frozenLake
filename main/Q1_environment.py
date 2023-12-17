import numpy as np
import contextlib

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try: yield
    finally: np.set_printoptions(**original)

# CONTENTS:
# 1. Class `EnvironmentModel`: General abstract class for framework of environment models
# 2. Class `Environment`: General class for environment model
# 3. Class `FrozenLake`: Frozen lake environment class
# 4. Method `play``: Play-testing
# 5. Method `displayResults`: # Policy & state-value display function
    
# NOTE: `displayResults` will be relevant for other modules where policy & state-value function are obtained.

#____________________________________________________________
# 1. General abstract class for framework of environment models

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.random_state = np.random.RandomState(seed)

    # Function to get the probability of moving to `next_state` from `state` given `action`:
    def p(self, next_state, state, action):
        raise NotImplementedError()

    # Function to get the reward of moving to `next_state` from `state` given `action`:
    def r(self, next_state, state, action):
        raise NotImplementedError()

    # Function to perform the state transition & return resultant reward:
    def draw(self, state, action):
        # Obtaining the probability distribution over state transitions:
        p = [self.p(ns, state, action) for ns in range(self.n_states)]

        # Obtaining the next state & state transition reward:
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward

#____________________________________________________________
# 2. General class for environment model

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)
        # Maximum number of steps an agent can take:
        self.max_steps = max_steps

        # Probability distribution over initial states:
        self.pi = pi
        if self.pi is None:
          # Defaults to uniform distribution:
          self.pi = np.full(n_states, 1./n_states)

        # Initialising the starting state:
        self.reset()

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        # Updating step statistics:
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        # Transitioning the state:
        self.state, reward = self.draw(self.state, action)

        # Returning the new state, reward & whether the agent should stop:
        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()

#____________________________________________________________
# 3. Frozen lake environment class

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # Representations: start:'&', frozen:'.', hole:'#', goal:'$'
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1) # Row major representation

        # Probability of "slipping", i.e. picking a random new direction:
        self.slip = slip

        # Number of states to consider
        n_states = self.lake.size + 1
        # NOTE: To see why 1 was added, see the implementation of function `p`

        # Number of actions possible:
        n_actions = 4 # 0 ==> up, 1 ==> left, 2 ==> down, 3 ==> right

        # pi ==> Probability distribution over initial states
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        # NOTE: In this case, we can only start at the "start" state

        # Setting the absorbing state:
        self.absorbing_state = n_states - 1
        # NOTE `n_states-1` is 1 step out of the range of other valid states

        # Initialising inherited parameters:
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        # NOTE: The starting state is initialised in the above constructor

    #================================================
    
    # Executing state-transition given an action:
    
    def step(self, action):
        # Slipping with `self.slip` chance:
        if np.random.rand() <= self.slip:
            action = np.random.choice(range(self.n_actions))
        '''
        WHAT SLIPPING MEANS:
        To slip here means to randomly pick a new direction which could be the
        same as the intended direction. This is equivalent to randomly picking
        a new action which could be the same as the intended action.

        NOTE: "Slipping" does not consider whether the intended or altered
        direction leads outside the grid or not.
        '''

        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    #================================================
    
    # Probability of transitioning from `state` to `next_state` given `action`:
    
    def p(self, next_state, state, action):
        # CASE 1: At or next to absorbing state
        '''
        CHECKING IF WE NEED TO MOVE TO THE ABSORBING STATE:
        If we are at a hole or a goal, then any action takes us to the
        absorbing state. Furthermore, any action from the absorbing state leads
        to the absorbing state. This is why we added 1 to the number of states
        in the instantiation of the environment; this enables iteration upto
        the absorbing state, allowing us to validate the following.
        '''
        if state == self.absorbing_state:
          if next_state == self.absorbing_state: return 1
          return 0

        if self.lake_flat[state] in "#$":
          # NOTE: '#' ==> hole, '$' ==> goal
          if next_state == self.absorbing_state: return 1
          return 0

        #------------------------------------

        # CASE 2: Not at or next to absorbing state
        nRows, nCols = self.lake.shape # Only `nCols` is used here
        # Checking if `action` leads to `next_state`:
        '''
        HOW ACTIONS ARE APPLIED:
        We apply actions as increments or decrements to the index in the row
        major (i.e. flattened array) representation of the frozen lake:

        0 (up) ==> next = cur-nCols
        1 (left) ==> next = cur-1
        2 (down) ==> next = cur+nCols
        3 (right) ==> next = cur+1

        NOTE: `nCols` is defined at the start of this function. `next` denotes
        the next state the action will lead to (unless the action leads out of
        the grid). `cur` denotes the current state. Both `next` and `cur` must
        be grasped as indices of the row major (flattened array) representation
        of frozen lake.
        ------------------------------------------------
        WHEN AN ACTION LEADS THE AGENT OUTSIDE THE GRID:
        Sometimes, an action would lead the agent outside the grid; in such
        cases, the state remains unchanged. Such cases can be identified as
        follows:

        0 (up): When next < 0
        1 (left): When next % nCols > cur % nCols
        2 (down): When next >= n_states
        3 (right): When next % nCols < cur % nCols

        NOTE: In such a case, 1 is returned only if `next_state` == `state`
        '''
        # Obtaining the next state:
        next = state + {0:-nCols, 1:-1, 2:nCols, 3:1}[action]
        # Checking if action leads out of the grid:
        if {0:next < 0,
            1:next % nCols > state % nCols,
            2:next >= self.n_states,
            3:next % nCols < state % nCols}[action]:
            # If true, no transition; return 1 only if `next_state` == `state`:
            if next_state == state: return 1
            return 0

        # Given the next state obtained is valid, is it the given next state?
        if next == next_state: return 1 # Yes
        return 0                        # No
  
    #================================================
    
    # Reward for transitioning from `state` to `next_state` given `action`:
    
    def r(self, next_state, state, action):
        # Checking for absorbing state, which is out of bounds of the grid:
        if state == self.absorbing_state: return 0
        # NOTE: We don't expect to reach here but are making the code foolproof

        # Reward of 1 is obtained only by taking an action out of a goal:
        if self.lake_flat[state] == '$': return 1

        # In all other cases, no reward is obtained:
        return 0

    #================================================
   
    # Visualising the agent's performance (by inputs or using a policy):

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state: lake[self.state] = '@'
            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']
            print('Lake:')
            print(self.lake)
            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))
            print('Value:')
            with _printoptions(floatmode='fixed', precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

#____________________________________________________________
# 4. Play-testing

# The following function `play` can be used to interactively test the environment...
def play(env):
    actions = ['w', 'a', 's', 'd']
    state = env.reset()
    env.render()
    done = False
    while not done:
        c = input('\nMove: ')
        if c == 'x': break
        if c not in actions:
            raise Exception('Invalid action')
        state, r, done = env.step(actions.index(c))
        env.render()
        print('Reward: {0}.'.format(r))

#================================================

# Defining the arenas for testing, i.e. play-areas (can be used elsewhere too)...
lake = {} # Dictionary to store different lake arenas/play-areas for testing
lake['small'] = [['&', '.', '.', '.'],
                 ['.', '#', '.', '#'],
                 ['.', '.', '.', '#'],
                 ['#', '.', '.', '$']]
                 
lake['big'] = [['&', '.', '.', '.', '.', '.', '.', '.'],
               ['.', '.', '.', '.', '.', '.', '.', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '.'],
               ['.', '.', '.', '.', '.', '#', '.', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '.'],
               ['.', '#', '#', '.', '.', '.', '#', '.'],
               ['.', '#', '.', '.', '#', '.', '#', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '$']]

# Doing the play-test...
# NOTE: It is run only if the module `environment` is run directly & not imported
if __name__ == '__main__': play(FrozenLake(lake=lake['small'], slip=0.1, max_steps=10))

#____________________________________________________________
# 5. Policy & state-value display function

'''
The following function allows the standarised presentation of the results
of various policy-finding functions that simultaneously obtain state values
(i.e. estimate the state value function using an array).
'''

def displayResults(results, labels, env):
    '''
    `results` contains k pairs, each pair consisting (in order)
    (1) the deterministic policy (an array of actions to be taken per state) &
    (2) the estimated state values (an array of values evaluated per state)
    The policy and state values are those derived by a specific algorithm (ex.
    policy iteration). The results for each are displayed in an organised
    format for easy testing.

    `labels` contain the names or labels to be given for each pair of results,
    such as the name of the algorithm used to derive the results.

    `env` is the instance of the environment model class used for the results.
    '''
    #------------------------------------
    '''
    # UNUSED CODE:
    
    # Printing results:
    print("POLICY & STATE VALUES OBTAINED")
    for result, label in zip(results, labels):
        print('------------------------------------')
        print(f'FOR {label.upper()}')
        print(f'Policy:\n{result[0]}\nState values:\n{result[1]}')

    # Visualisation of agent's performance:
    # NOTE: It allows us to directly see the agent following the policy across the grid
    for result, label in zip(results, labels):
        print('\n================================================\n')
        print(f'AGENT PERFORMANCE AFTER {label.upper()}\n')
        state = env.reset()
        done = False
        while not done:
            a = result[0][state]
            state, r, done = env.step(a)
            env.render()
            print(f'Reward: {r}.')
    '''
    #------------------------------------
    # Visualisation of agent's performance:
    for result, label in zip(results, labels):
        print('\n================================================\n')
        print(f'AGENT PERFORMANCE AFTER {label.upper()}\n')
        state = env.reset()
        env.render(policy=result[0], value=result[1])