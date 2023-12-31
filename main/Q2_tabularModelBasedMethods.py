# Importing the necessary context:
from CONTEXT import *

# CONTENTS:
# 1. Method `policy_evaluation`
# 2. Method `policy_improvement`
# 3. Method `policy_iteration`
# 4. Method `value_iteration`
# 5. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code

#____________________________________________________________
# 1. Policy evaluation

def policy_evaluation(env, policy, gamma, theta, max_iterations):
    '''
    NOTE ON THE ARGUMENTS:
    `env`: Object of the chosen environment class (ex. `FrozenLake`)
    `policy`: Array giving the probability of taking an action from a state
    `gamma`: Discount factor
    `theta`: Error tolerance level

    FURTHER NOTE ON `policy`:
    We consider the policy to be deterministic, meaning that it maps each state
    to a certain action, rather than each state-action pair to a probability.
    Hence, `policy` is a 1D array with each index corresponding to a state and
    the value at a given index i corresponding to the action to be taken from
    state i. This is equivalent to the policy wherein each state-action pair is
    related to either 0 or 1 so that each state is mapped to 1 only when paired
    with a particular action.
    '''
    # Initialising table of values per state:
    value = np.zeros(env.n_states, dtype=float)
    # Policy evaluation loop:
    for i in range(max_iterations):
        # Flag for indicating convergence of value evaluation:
        flag = 0
        for s in range(env.n_states):
            # NOTE: s ==> state
            # Storing previous value of value function:
            v = value[s]
            #------------------------------------
            # Obtaining current value of value function:
            # NOTE: We iterate through every possible action from state `s`
            value[s] = float(0)

            # Choosing next action `a` based on the policy:
            # NOTE: This only works because the policy is deterministic
            a = policy[s]

            # Considering all states as next states & updating value of each:
            for _s in range(env.n_states):
                # NOTE: _s ==> next state
                value[s] += env.p(_s, s, a)*(env.r(_s, s, a) + gamma*value[_s])
                '''
                NOTE ON ABOVE USED FUNCTIONS:
                `env.p`: Probability of moving `s` to `_s` given action `a`
                `env.r`: Reward of moving `s` to `_s` given action `a`
                '''
            #------------------------------------
            # Obtaining the difference in state value:
            # NOTE: This is why we stored the previous value before in `v`
            if abs(value[s]-v) <= theta: flag += 1

        # If difference in state value <= theta for all states, stop:
        if flag == env.n_states: break

    # FORMAT OF RETURN: state-values, iterations taken to converge
    return value, i

#____________________________________________________________
# 2. Policy improvement

def policy_improvement(env, value, gamma):
    '''
    NOTE ON THE ARGUMENTS:
    `env`: Object of the chosen environment class (ex. `FrozenLake`)
    `value`: Array containing values of each state with respect to some policy
    `gamma`: Discount factor

    NOTE ON POLICY IMPROVEMENT:
    The goal of policy improvement is to improve on the previously used policy
    (which is implicit in the array of state values, which is evaluated with
    respect to some policy). We do this by choosing for each state `s` the
    action `a` such that we maximise the reward of taking `a` from `s`
    (irrespective of policy) then following the previous policy (which is
    implicit in the array of state values).
    '''
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q = np.zeros(env.n_actions, dtype=np.float32)
        for a in range(env.n_actions):
            for _s in range(env.n_states):
                # NOTE: `_s` ==> next state
                # Total reward of taking a from s then following last policy:
                '''
                NOTE ON LAST POLICY:
                The previous policy based on which we are making the current
                improvement is implicit in the array of state values `value`,
                since this array was obtained with respect to some policy.
                '''
                q[a] += env.p(_s, s, a)*(env.r(_s, s, a) + gamma*value[_s])
                '''
                NOTE ON ABOVE USED FUNCTIONS:
                `env.p`: Probability of moving `s` to _s given action `a`
                `env.r`: Reward of moving `s` to `_s` given action `a`
                '''
        # Update policy to maximise the one-step dynamics from `s`:
        policy[s] = np.argmax(q)
    return policy

#____________________________________________________________
# 3. Policy iteration

def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    '''
    NOTE ON THE ARGUMENTS:
    Same as for the function `policy_evaluation`.

    NOTE ON POLICY ITERATION:
    The goal of policy iteration is to perform alternate steps of policy
    evaluation & policy improvement to approach the optimal policy.
    '''
    if policy is None: policy = np.zeros(env.n_states, dtype=int)
    else: policy = np.array(policy, dtype=int)
    # `n` will store the value of iterations to converge:
    n = 0
    # Initialising state values with respect to existing policy:
    value, n = policy_evaluation(env, policy, gamma, theta, max_iterations)
    # Policy iteration loop:
    for i in range(max_iterations):
        new_policy = policy_improvement(env, value, gamma)

        # If policy does not change, it means we have converged to the optimal:
        if all(policy == new_policy): break
        # Else, continue improving the policy:
        policy = new_policy

        # Updating the state-values:
        value, k = policy_evaluation(env, policy, gamma, theta, max_iterations)
        # Updating number of evaluation steps needed:
        n += k

    # FORMAT OF RETURN: policy, state-values, iterations taken to converge
    return policy, value, n

#____________________________________________________________
# 4. Value iteration

def value_iteration(env, gamma, theta, max_iterations, value=None):
    '''
    NOTE ON THE ARGUMENTS:
    Same as for the function `policy_evaluation`.

    NOTE ON POLICY ITERATION:
    The goal of value iteration is to use action-value function estimates to
    directly approach the optimal policy's state values and use this to obtain
    the final (hopefully optimal) policy.
    '''
    if value is None: value = np.zeros(env.n_states)
    else: value = np.array(value, dtype=np.float)
    # Value iteration loop:
    for i in range(max_iterations):
        # Flag for indicating convergence of value evaluation:
        flag = 0
        for s in range(env.n_states):
            q = np.zeros(env.n_actions, dtype=np.float32)
            for a in range(env.n_actions):
                for _s in range(env.n_states):
                    # NOTE: _s ==> next state
                    # Total reward of taking a from s for previous state value:
                    q[a] += env.p(_s, s, a)*(env.r(_s, s, a) + gamma*value[_s])
                    '''
                    NOTE ON ABOVE USED FUNCTIONS:
                    `env.p`: Probability of moving `s` to `_s` given action `a`
                    `env.r`: Reward of moving from `s` to `_s` given action `a`
                    '''
            # Update policy to maximise the one-step dynamics from s:
            v = value[s]
            value[s] = np.max(q)
            if abs(value[s]-v) <= theta: flag += 1

        # If difference in state value <= theta for all states, stop:
        if flag == env.n_states: break

    #================================================

    # Obtaining the (estimated) optimal policy:
    # NOTE: The logic for this is identical to policy improvement
    policy = policy_improvement(env, value, gamma)

    # FORMAT OF RETURN: policy, state-values, iterations taken to converge
    return policy, value, i

#____________________________________________________________
# 5. Code for testing the above functions

# NOTE: The testing code is only run if the current file is executed as the main code

if __name__ == '__main__':
    # Defining the parameters:
    env = FrozenLake(lake=LAKE['small'], slip=0.1, max_steps=None, seed=0)
    # NOTE: Putting `max_steps=None` makes it default to the grid size
    gamma = GAMMA
    theta = 0.001
    max_iterations = 100

    # Running the functions:
    PI = policy_iteration(env, gamma, theta, max_iterations)
    VI = value_iteration(env, gamma, theta, max_iterations)
    labels = ("policy iteration", "value iteration")

    # Displaying results:
    displayResults((PI, VI), labels, env, theta)
