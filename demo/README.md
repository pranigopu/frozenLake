# DEMO: Frozen Lake

This directory contains Jupyter Notebooks with functionally identical code to the main code. The purpose of the code in this directory is to visually demonstrate the outputs of the tests as well as present any additional explanations or commentary on the code. These files were run on Google Colab, and may have code specific to Google Colab.

## Code differences from main package
The main package contains a file `CONTEXT.py`, which imports `Q1_environment.py`, necessary libraries and modules for each of the `Qi` series files as well as some defined global variables and a function to display the results of a method (namely, render the policy and state-values). This was done to ensure clarity in purpose for each file. In the demo files, however, all the required content for the `Qi` series files were placed within `Q1_environment.ipynb`, because it was simpler to follow when understanding the demos, and because the demonstration of functionalities do not require the same level of organisational clarity as a main package.
