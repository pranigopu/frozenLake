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
    
#____________________________________________________________
# 1. General abstract class for framework of environment models

class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions
        
        # Setting a `numpy.RandomState` object with given seed:
        # NOTE: This enables the replicability of code during testing
        self.random_state = np.random.RandomState(seed)
        # NEW PARAM: Storing the seed to allow later resetting of random state:
        self.seed = seed

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

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

    #================================================

    # NOTE: The following does note reset the random state object
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)
        return self.state

    #================================================

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
    
    #================================================

    def render(self, policy=None, value=None):
        raise NotImplementedError()

    #================================================

    # NEW FUNCTION: Resetting random state:
    # NOTE: For enabling replicability over multiple attempts while testing
    def resetRandomState(self):
        self.random_state = np.random.RandomState(self.seed)

#____________________________________________________________
# 3. Frozen lake environment class

class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps=None, seed=None):
        # NOTE: `max_steps` should not remain `None`; this is handled later
        '''
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        '''
        # Representations: start:'&', frozen:'.', hole:'#', goal:'$'
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1) # Row major representation

        # Number of states to consider
        n_states = self.lake.size + 1
        # NOTE: To see why 1 was added, see the implementation of function `p`

        # Setting the absorbing state:
        self.absorbing_state = n_states - 1
        # NOTE `n_states-1` is 1 step out of the range of other valid states

        # Number of actions possible:
        n_actions = 4 # 0 ==> up, 1 ==> left, 2 ==> down, 3 ==> right

        #------------------------------------
        # Action step map:
        nr, nc = self.lake.shape # Only `nc` (number of columns) is used here
        '''
        HOW ACTIONS ARE APPLIED:
        We apply actions as increments or decrements to the index in the row
        major (i.e. flattened array) representation of the frozen lake:

        0 (up) ==> next = cur-nc
        1 (left) ==> next = cur-1
        2 (down) ==> next = cur+nc
        3 (right) ==> next = cur+1

        NOTE: `nc` is the number of columns in the lake grid. `next` denotes
        the next state the action will lead to (unless the action leads out of
        the grid). `cur` denotes the current state. Both `next` and `cur` must
        be grasped as indices of the row major (flattened array) representation
        of frozen lake.
        '''
        self.actionStep = [-nc, -1, nc, 1]
        # NOTE: Index corresponds to action

        #------------------------------------
        # Probability distribution over initial states:
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        # NOTE: In this domain, we always start at the 'start', i.e. '&' state

        #------------------------------------
        # Initialising inherited parameters:
        '''
        NOTE ON `max_steps`:
        For the `FrozenLake` constructor, `max_steps` may not be specified.
        But this is a key parameter in the environment, since it determines the
        upper bound on the number of times an agent can interact with the
        environment before giving up (giving up can be necessary to restart
        and try to get a better attempt). Hence, by default, we will assign
        `max_steps` as the number of positions in the lake grid.
        '''
        if max_steps == None: max_steps = self.lake.size
        
        # Calling the parent class' constructor:
        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        # NOTE: The starting state is initialised in the above constructor

        #------------------------------------
        # Probability of slipping:
        '''
        WHAT DOES IT MEAN TO SLIP?
        To slip here means to randomly pick a new direction which could be the
        same as the intended direction. This is equivalent to randomly picking
        a new action which could be the same as the intended action. "Slipping"
        does not consider whether the intended or altered direction leads
        outside the grid or not.
        '''
        self.slip = slip

        #------------------------------------
        # State-transition probability storage:
        self.STP = -np.ones((self.n_states, self.n_states, self.n_actions))
        '''
        IMPLEMENTATION NOTE: Transition probability storage method:
        We store the state transition probabilities for each state-action pair
        in the array `self.STP`. This array is essentially an array of arrays,
        where each array is associated to a possible `next_state`, each
        row of each array is associated to a possible `state`, and each column
        of each row of each array is associated to a possible `action`.

        This array of arrays is initialised with -1's. Hence, seeing an element
        as -1 means the associated transition probability has not been obtained
        yet. If it has been obtained, we simply return the element, else we
        update `self.STP`, thus saving computation for future function calls.
        '''
        # Setting state-transition probabilities by repeated calls to `self.p`:
        for next_state in range(n_states):
            for state in range(n_states):
                for action in range(n_actions):
                    self.p(next_state, state, action)

    #================================================

    # Executing state-transition given an action:

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    #================================================

    # Probability of transitioning from `state` to `next_state` given `action`:

    '''
    WORKING:
    If the state-transition probability is stored in `self.TP`, it simply
    references it and returns the value. Otherwise, it first obtains & stores
    (in `self.STP`) and then returns the probability.
    '''

    def p(self, next_state, state, action):
        # Checking if state-transition probability has already been found:
        stp = self.STP[next_state, state, action]
        if stp != -1: return stp

        # If state-transition not found, find it!
  
        # Initialising 1st dimension for given `state` & `action`:
        self.STP[:, state, action] = np.zeros(self.n_states)

        '''
        FINDING TRANSITION PROBABILITY:
        Where we have certainty of transition...
        In cases where current state is the absorbing state, a hole or goal,
        and where the next state is the absorbing state, the probability of
        transition is 1, i.e. the agent in the current state will certainly
        transition to the absorbing state for any action.

        Where we have certainty of no transition...
        If the next state cannot be reached from the current step in a single
        step, the all transition probabilities are clearly zero. Also, if
        the current state is the absorbing state, a hole or a goal, and if the
        next state is not the absorbing state, the transition probability has
        to be 1 (given NOTE 1).

        What is the probability of transition in other cases?
        ........................
        For convenience, we define:
        - x be the chance of slipping
        - D be the intended state

        Hence, we have that at each time stamp:
        1. There is an overall x chance of slipping
        2. Agent can slip in 1 of 4 directions
        ==> Each time a state is reached by slipping in a certain direction,
        the probability of slipping into that state rises by x/4

        If an action or a slip leads out of the grid, the intended state
        will be equal to the current state, since no transition is intended.
        Generally, slipping in each direction will result in unique states, but
        in corner states, i.e. states at the corners of the grid, there are two
        directions that would lead to the same next state.
        ........................
        Furthermore, we have that at each time stamp:
        1. There is an overall 1-x change of moving to as intended, i.e. to D
        2. This adds to the probability of slipping in that direction too
        ==> Each time a state is reached by intention, the probability of
        moving into that state rises by 1-x, and adds to any existing chance
        of slipping to that state.
        '''

        #------------------------------------

        # CASE 1: Current state is absorbing state, hole or goal
        '''
        CHECKING IF WE NEED TO MOVE TO THE ABSORBING STATE:
        If we are at a hole or a goal, then any action takes us to the
        absorbing state. Furthermore, any action from the absorbing state leads
        to the absorbing state. This is why we added 1 to the number of states
        in the instantiation of the environment; this enables iteration upto
        the absorbing state, allowing us to validate the following.
        '''
        if state == self.absorbing_state or self.lake_flat[state] in '#$':
            # NOTE: '#' ==> hole, '$' ==> goal

            # If next state is absorbing state, probability is 1 for all actions:
            self.STP[self.absorbing_state, state, :] = np.ones(self.n_actions)

            # Returning the probability for the given next state:
            return self.STP[next_state, state, action]

        #------------------------------------

        # CASE 2: Current state is neither absorbing state, hole nor goal
        nr, nc = self.lake.shape # Only `nc` (number of columns) is used here
        '''
        IMPLEMENTATION NOTE 1: When an action leads the agent outside the grid:
        Sometimes, an action would lead the agent outside the grid; in such
        cases, the state remains unchanged. Such cases can be identified as
        follows (given n is the next state, nc is the number of columns in the
        grid, and n_states is the total number of states in the environment):

        0 (up): When n < 0
        1 (left): When n % nc > cur % nc
        2 (down): When n >= n_states
        3 (right): When n % nc < cur % nc

        ------------------------------------

        IMPLEMENTATION NOTE 2: Exact process to get transition probabilities:
        `self.TP[:, state, action]` refers to the transition probabilities to
        each state from state `state` given action `action`. We initialise this
        vector by zeros, then do the following conditional updates:
        - Add `self.slip/4` if `action` does not lead to it intentionally
        - Add `1 - self.slip` if `action` does lead to it intentionally

        This updates the probabilities for all possible states, given the pair
        of state and action (`state`, `action`), saving computation time for
        future function calls.
        '''
        # Frequently reused value:
        sc = state % nc
        # NOTE: sc ==> state column number
        #........................
        # UPDATE LOOP
        for a in range(self.n_actions):
            # Obtaining the next state integer without validation:
            n = state + self.actionStep[a]

            # Checking the out-of-grid condition:
            # NOTE: Index corresponds to action
            if [n < 0,
                n % nc > sc,
                n >= self.n_states,
                n % nc < sc][a]: n = state

            # If intention leads to state `n`:
            if a == action: self.STP[n, state, action] += 1 - self.slip
            # Adding or updating the probability of slipping into `n`:
            self.STP[n, state, action] += self.slip/4

        #------------------------------------

        # Returning obtained transition probability:
        return self.STP[next_state, state, action]

    #================================================

    # Reward for transitioning from `state` to `next_state` given `action`:

    def r(self, next_state, state, action):
        # Checking for absorbing state, which is out of the bounds of the grid:
        if state == self.absorbing_state: return 0
        # NOTE: We don't expect to reach here, just making the code foolproof

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
LAKE = {} # Dictionary to store different lake arenas/play-areas for testing
LAKE['small'] = [['&', '.', '.', '.'],
                 ['.', '#', '.', '#'],
                 ['.', '.', '.', '#'],
                 ['#', '.', '.', '$']]

LAKE['big'] = [['&', '.', '.', '.', '.', '.', '.', '.'],
               ['.', '.', '.', '.', '.', '.', '.', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '.'],
               ['.', '.', '.', '.', '.', '#', '.', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '.'],
               ['.', '#', '#', '.', '.', '.', '#', '.'],
               ['.', '#', '.', '.', '#', '.', '#', '.'],
               ['.', '.', '.', '#', '.', '.', '.', '$']]

# Doing the play-test...
# NOTE: It is run only if the module `environment` is run directly & not imported
if __name__ == '__main__': play(FrozenLake(lake=LAKE['small'], slip=0.1, max_steps=10))