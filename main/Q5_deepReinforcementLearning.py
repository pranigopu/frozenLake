# Importing the necessary context:
from CONTEXT import *

# CONTENTS:
# 1. Class `FrozenLakeImageWrapper`: Wrapping the environment to enable convertion of states to images
# 2. Class `DeepQNetwork`: Neural network implementation
# 3. Class `ReplayBuffer`: Replay buffer implementation
# 4. Method `deep_q_network_learning`: Learning process
# 5. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code.

#____________________________________________________________
# 1. Wrapping the environment to enable convertion of states to images

class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env
        lake = self.env.lake
        # NOTE: The lake grid is converted into an array by the environment
        self.n_actions = self.env.n_actions

        # Obtaining a state image for each state:
        #------------------------------------
        # 1. Shape for each state image:
        self.state_shape = (4, lake.shape[0], lake.shape[1])
        #------------------------------------
        # 2. Obtaining a list of filter arrays:
        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]
        #------------------------------------
        # 3. Obtaining the state image for each state:
        #........................
        # Handling for the absorbing state...
        # a. Channel 1 of the state:
        # NOTE: Absorbing state has no position on the grid, so all zeros
        A = np.zeros(lake.shape)

        # b. Attaching channels 2, 3 & 4, then storing all as an array:
        self.state_image = {env.absorbing_state: np.stack([A] + lake_image)}
        '''
        IMPLEMENTATION NOTE:
        `[A]` is a list containing array A, and `lake_image` is a list
        containing 3 arrays. Using `+` between `[A]` and `lake_image` will
        concatenate the two lists, resulting in a list of 4 arrays.

        `np.stack` joins the above array list into a single array of arrays.
        '''
        #........................
        # Handling for the other states actually present on the grid...
        for state in range(lake.size):
            # a. Channel 1 of the state:
            '''
            NOTE ON CHANNEL 1:
            The 1st channel is the array such that the element is 1 if the
            index matches the state, 0 otherwise. This corresponds to the
            position of the agent if the agent were to be in this state. Hence,
            note that the 1st channel shows not the current position of the
            agent, but its position if it were in this state.
            '''
            # a.1. Initialising it as an array of zeros:
            A = np.zeros(lake.shape)
            # a.2. Assigning the current state's position as 1:
            row = state // lake.shape[0]
            col = state % lake.shape[1]
            A[row, col] = 1.0

            # b. Attaching channels 2, 3 & 4, then storing all as an array:
            self.state_image[state] = np.stack([A] + lake_image)
            '''
            IMPLEMENTATION NOTE:
            Check the implementation note above this loop.
            '''

    #================================================

    # Mapping the given state paired with each action to state image:
    # NOTE: State images were obtained for each state in the class constructor

    def encode_state(self, state):
        return self.state_image[state]

    #================================================

    # Obtaining the policy via decoding neural network's output:
    # 1. Encode states as state images
    # 2. Pass state images as input to the neural network
    # 3. Obtain the action-value function as an output
    # 4. Use the action-value function to obtain the policy & state-values

    def decode_policy(self, dqn):
        # 1. Encode states as state images:
        N = self.env.n_states
        states = np.array([self.encode_state(s) for s in range(N)])

        # 2 & 3: Obtain the action-value function for encoded states:
        q = dqn(states).detach().numpy()
        # NOTE: `torch.no_grad` omitted to avoid import

        # 4. Use the action-value function to obtain the policy & state-values:
        policy = q.argmax(axis=1)
        value = q.max(axis=1)
        return policy, value

    #================================================

    # Resetting environment & encoding it as state image:

    def reset(self):
        return self.encode_state(self.env.reset())

    #================================================

    # Taking a step in environment & encoding next state as state image:

    def step(self, action):
        state, reward, done = self.env.step(action)
        return self.encode_state(state), reward, done

    #================================================

    # Visualising the agent's performance (by inputs or using a policy):

    def render(self, policy=None, value=None):
        self.env.render(policy, value)

#____________________________________________________________
# 2. Neural network implementation

class DeepQNetwork(torch.nn.Module):
    def __init__(self, wenv, learning_rate, kernel_size,
                 conv_out_channels, fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        # Convolutional layer:
        self.conv_layer = torch.nn.Conv2d(in_channels=wenv.state_shape[0], out_channels=conv_out_channels, kernel_size=kernel_size, stride=1)

        # h ==> Number of rows in grid, w ==> Number of columns in grid
        h = wenv.state_shape[1] - kernel_size + 1
        w = wenv.state_shape[2] - kernel_size + 1

        # Fully connected layer:
        self.fc_layer = torch.nn.Linear(in_features=h*w*conv_out_channels, out_features=fc_out_features)

        # Output layer:
        self.output_layer = torch.nn.Linear(in_features=fc_out_features, out_features=wenv.n_actions)

        # Optimiser for gradient descent:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    #================================================

    # Feed-forward function:

    def forward(self, x):
        # Setting the activation function:
        activation = torch.nn.ReLU()

        # Converting inputted array into a tensor:
        y = torch.tensor(x, dtype=torch.float)
        '''
        EXPECTED SHAPE OF THE ABOVE INPUT ARRAY / TENSOR:
        `x.shape` = `y.shape` = (B, 4, h, w), where
        >> B ==> Number of states
        >> 4 ==> Number of channels per state representation
        >> h ==> Number of rows in the playing grid
        >> w ==> Number of columns in the playing grid
        '''
        # Feeding forward the input to convolution layer:
        y = self.conv_layer(y)
        y = activation(y)

        # Flattening `x` before passing it to the fully connected layer:
        y = torch.flatten(y, start_dim=1)
        '''
        NOTE ON FLATTENING:
        We want to flatten each state image representation. Now, a state image
        consists of `conv_out_channels` channels, each of shape `(h, w)`.
        Hence, each state image needs to become a `h*w*conv_out_channels` sized
        tensor.

        Now, `x` holds B states, and before applying ReLU (which does not alter
        the input tensor shape), the state images are arranged in `x` such
        that `x` was an array of arrays, each array being a state image, which
        means the 1st dimension of `x` corresponds to the states. This means
        we want to flatten each state representation while maintaining the
        an array of state representations. Hence, we leave the 1st dimension
        of `x` (i.e. axis 0) and start flattening from the 2nd dimension
        (i.e. axis 1, leading to the argument `start_dim=1`).
        '''

        # Feeding forward the input to fully-connected layer:
        y = self.fc_layer(y)
        y = activation(y)

        # Feeding forward the input to output layer & returning output:
        y = self.output_layer(y)
        return y

    #================================================

    # Single step of training:

    def train_step(self, transitions, gamma, tdqn):
        # TDQN ==> Temporary deep Q-network
        #... explained in comments for function `deep_q_network_learning`
        
        # Organising the transitions data into separate arrays:
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        # Obtaining current action-value estimates:
        q = self(states)
        # NOTE: The above is equivalent to doing `q = self.forward(states)`

        # Obtaining action-values for previously taken actions:
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)
            '''
            EXPLAINING THE ABOVE LINE:
            `tdqn(next_states)` is equivalent to `tdqn.forward(next_states)`,
            and simply applies the forward model of the non-updated model
            (stored in `tdqn`) to the next states, to get an estimate of
            action-values given the previous weights of the model.
            ------------------------------------
            The `.max` function, when applied to a tensor, produces two tensors:
            1. The array of max value(s) along the specified dimension
            2. The dimension-specific indices where the max value(s) were found

            We only want the first of the above two tensors. Hence, we apply
            the subscript `[0]` on `tdqn(next_states).max(dim=1)`, to do
            `tdqn(next_states).max(dim=1)[0]`
            '''

        # Estimating the one-step rewards given the stored rewards:
        target = torch.Tensor(rewards) + gamma*next_q

        # Loss calculation:
        # NOTE 1: The loss is the mean squared error between `q` & `target`
        # NOTE 2: `q - target` is temporal difference for given state-action
        loss = torch.nn.functional.mse_loss(q, target.to(torch.float32))
        # ALTERNATIVE: `loss = torch.mean((q - target)**2)`

        # Performing gradient descent, i.e. optimisation:
        self.optimizer.zero_grad() # Intialising gradient as zero
        loss.backward()            # Computing the current gradient
        self.optimizer.step()      # Performing the optimisation step

#____________________________________________________________
# 3. Replay buffer implementation
'''
ABOUT REPLAY BUFFER:
Replay buffer stores data on state transitions (or simply transitions). A
transition's data consists of a tuple composed of a state, action, reward,
next state, and a flag variable that denotes whether the episode ended at the
next state. The buffer is represented by a Python deque object that
automatically discards the oldest transitions when it reaches capacity. The
replay buffer is vital in utilising previously obtained transitions & observed
rewards. 
'''

class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        # Replay buffer data structure:
        self.buffer = deque(maxlen=buffer_size)

        # Maintaining the given random state for enabling replicability:
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        # Length of the replay buffer:
        N = self.__len__()

        # Randomly sampling `batch_size` buffer indices without replacement:
        I = self.random_state.choice(N, size=batch_size, replace=False)

        # Returning the transitions corresponding to the above indices:
        return [self.buffer[i] for i in I]

#____________________________________________________________
# 4. Learning process
'''
Learning is done by performing gradient descent over the minimum squared error
(MSE) loss function. Instead of obtaining loss with respect to differences from
observed values alone (observed action-values in our case), we obtain the loss
as the mean of the sum of squares of the temporal differences for each
state-action pair. This is done because rewards are very sparse in the given
environment, which means obtaining accurate estimates of action-values from
interaction with the environment alone will take an unfeasibly long time,
prompting us to instead use temporal differences.
'''

def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon, batch_size, target_update_frequency, buffer_size, kernel_size, conv_out_channels, fc_out_features, seed):
    # INITIALISATION

    # Setting random state with given seed for enabling replicability:
    random_state = np.random.RandomState(seed)

    # Initialising replay buffer
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    # Initialising the required deep neural networks:
    args = [env,
            learning_rate,
            kernel_size,
            conv_out_channels,
            fc_out_features,
            seed]
    dqn = DeepQNetwork(*args)
    tdqn = DeepQNetwork(*args)
    '''
    NOTE ON `dqn` & `tdqn`:
    - dqn ==> Deep Q-network
        - The network used to estimate the latest estimates of action-values
        - Updated with every training step
    - tdqn ==> Temporary deep Q-network
        - The network used to estimate action-values based on past weights
        - Updated periodically (we could set update frequency to match `dqp`)
        - Lower update frequency may make the learning process more resistant
          to getting stuck in local optima
    '''

    # Array of linearly decreasing exploration factors:
    epsilon = np.linspace(epsilon, 0, max_episodes)

    #================================================

    # TRAINING LOOP
    for i in range(max_episodes):
        state = env.reset()

        done = False
        while not done:
            # Choosing next action with epsilon-greedy policy:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad(): q = dqn(np.array([state]))[0].numpy()
                qmax = np.max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            # Moving the agent to the next state within the current episode:
            next_state, reward, done = env.step(action)
            # Updating the replay buffer:
            replay_buffer.append((state, action, reward, next_state, done))
            # Updating the state variable:
            state = next_state

            # Once we have enough stored in the replay buffer, draw from it:
            # NOTE: The drawn data is used to train the network
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        #------------------------------------
        # Updating the network weights of `tdqn` as per update frequency:
        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())
            # NOTE: The above copies the parameter values of `dqn` to `tdqn`

    return dqn
#____________________________________________________________
# 5. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code.
if __name__ == '__main__':
    # Defining the parameters:
    env = FrozenLake(lake=LAKE['small'], slip=0.1, max_steps=None, seed=0)
    # NOTE: Putting `max_steps=None` makes it default to the grid size
    wenv = FrozenLakeImageWrapper(env)
    max_episodes = 1000
    learning_rate = 0.01 # Learning rate
    gamma = GAMMA
    epsilon = 0.5
    batch_size = 50
    target_update_frequency = 1
    buffer_size = 500
    kernel_size = 4
    conv_out_channels = 12
    fc_out_features = 12
    seed = 0

    # Running the function:
    DeepQ = deep_q_network_learning(wenv, max_episodes, learning_rate, gamma, epsilon, batch_size, target_update_frequency, buffer_size, kernel_size, conv_out_channels, fc_out_features, seed)

    # Obtaining the policy & state values:
    DeepQ = wenv.decode_policy(DeepQ)
    labels = ("deep q network learning")

    # Displaying results:
    displayResults([DeepQ], labels, env)
