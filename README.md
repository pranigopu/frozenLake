# _Frozen Lake_:<br>Reinforcement Learning Project
This project aims to use reinforcement learning principles to create algorithms to find policies that would help us better navigate the "Frozen Lake" environment defined within this project. In this repository, there are two sets of code that is functionally identical. One set is in the Jupyter Notebook format, meant to visually demonstrate the outputs of the tests as well as present any additional explanations or commentary on the code. The other set is regular Python code, meant to be run as an integrated package. The respective sets of files are stored in the following directories of this repository:

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
- Q-Learning with linear approximation of action-value function `linear_q_learning`<br>
