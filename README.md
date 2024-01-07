# _Frozen Lake_:<br>Reinforcement Learning Project

This project was a team project with the following contributors:

- [Malo Hamon a.k.a. nocommentcode](https://github.com/nocommentcode)
- [Pranav Narendra Gopalkrishna a.k.a. Prani Gopu](https://github.com/pranigopu)

This repository contains my personal work with respect to the project. The method of collaboration in this project was to attempt the assignment personally, before coming together to check each one's implementations, decide on the final code and make the report. The main repository of the project (which was used for the final submission and is owned by Malo Hamon) is linked [here](https://github.com/nocommentcode/ecs7002_assignment_2).

The functionality of the implementations in this repository and the main one are similar, if not the same in most cases (although the main repository is more up-to-date). The main difference between this repository and the one used for the final submission is the extensiveness of additional reference material; I have included many comments, notes and demonstrations of parts or aspects of the implementations that were not needed for the final submission but can be quite helpful in understanding the conceptual as well as practical basis of the implementations.

## Acknowledgement

Malo Hamon's code as well as insights helped my own work in the following ways:

- Revealed the necessity for maintaining a random state with a set seed
- Revealed errors in my the state-transition probability calculation method
- Corrected the iterations-to-convergence counting for policy iteration
- Updated the $\epsilon$-greedy policy to break ties between equally good actions during exploitation
- My code for generating the moving averages of discounted rewards per episode was derived from his

## Introduction
This project aims to use reinforcement learning principles to create algorithms to find policies that would help us better navigate the "Frozen Lake" environment defined within this project. In this repository, there are two sets of code that are functionally identical. One set is in the Jupyter Notebook format, meant to visually demonstrate the outputs of the tests as well as present any additional explanations or commentary on the code. The other set is regular Python code, meant to be run as an integrated package. The respective sets of files are stored in the following directories of this repository:

- Jupyter Notebook demonstrations: `frozenLake/demo/`
- Python main implementation codefiles: `frozenLake/main/`

## Question 1: Environment

**File name (without extension)**: `Q1_environment`

<br>Implementation of `FrozenLake` environment, which involves:

- Implementation of state-action transition probability function `p`
- Implementation of state-action transition reward function `r`
- Modication of `step` function to consider the chance of slipping

## Question 2: Tabular Model-based Methods 

**File name (without extension)**: `Q2_tabularModelBasedMethods`

- Policy evaluation function `policy_evaluation`
- Policy improvement function `policy_improvement`
- Policy iteration function `policy_iteration`
- Value iteration function `value_iteration`

## Question 3: Tabular Model-free Methods

**File name (without extension)**: `Q3_tabularModelFreeMethods`

- SARSA control `sarsa` (SARSA $\implies$ State-Action-Reward-State-Action)
- Q-Learning `q_learning` (Q $\implies$ Action-value function)

## Question 4: Non-tabular Model-free Methods

**File name (without extension)**: `Q4_nonTabularModelFreeMethods`

- SARSA with linear approximation of action-value function `linear_sarsa`
- Q-Learning with linear approximation of action-value function `linear_q_learning`

## Question 5: Deep Reinforcement Learning

**File name (without extension)**: `Q5_deepReinforcmentLearning`

- Convolutional neural network to learn action-values
- Replay buffer to store state-transition data & draw samples from for training
- Deep Q-network learning loop `deep_q_network_learning`

## Other implementation notes
### Setting random state for replicability
Sometimes, when you are using randomization in a part of the codebase, you want to get the same result independent of the iteration you are running the code. This enables others to replicate and validate your results, despite the use of pseudorand number generation. The `np.random.RandomState` class allows you to set the same random state in all the NumPy operations involving randomization. In practice, you can pass a particular seed to the aforementioned class' constructor and thereby replicate the same pseudorandom number generations over and over.
<br><br>**NOTE**: The methods available to an `np.random.RandomState` object are exactly all the randomization methods available in NumPy, such as `rand`, `randint`, `choice`, etc.

### Defining the $\epsilon$-greedy policy
The $\epsilon$-greedy (epsilon-greedy) policy is a method of choosing an action from a state such that the probability of choosing a random action is $\epsilon$ while the probability of choosing the action that maximises the action-value function (as far as it has been estimated) from the given state is $1-\epsilon$. $\epsilon$ (epsilon) is called the "exploration factor". $\epsilon$-greedy is a solution to the exploration-exploitation dilemma, wherein the degree of exploration (i.e. expansion of actions unknown potential) is decided by the exploration factor.

#### Breaking ties between reward-maximising actions
When the above policy chooses to exploit rather than explore, it may be the case that there exist multiple actions that (within a tolerance level) maximise the action-value function from a given state. In such a case, exploration can be encouraged even during exploitation by making a random selection from the reward-maximising actions. This approach is _no worse and potentially better_ than simply picking the first action that maximises the action-value function from the given state, since it furthers exploration without reducing exploitation.

#### Implementation details
In practice, in exploitation, we create a list of actions that maximise the action-value for a given state, wherein the comparison of each action's action value to the maximum action-value is made with a tolerance level. This tolerance level is kept at the default values defined in the function `numpy.allclose`. Then, random selection is done on this list of actions to next action action.
