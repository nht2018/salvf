# SALVF: Stochastic Augmented Lagrangian Value Function Method

This repository contains implementations of algorithms from the paper "An Augmented Lagrangian Value Function Method for Lower-level Constrained Stochastic Bilevel Optimization".

The codes are based on https://github.com/Liuyuan999/Penalty_Based_Lagrangian_Bilevel and are released under the Apache License 2.0.

## Overview

This repository implements the Stochastic Augmented Lagrangian Value Function (SALVF) method for solving lower-level constrained stochastic bilevel optimization problems. The method is designed to handle complex optimization problems where the lower-level problem has constraints that must be satisfied.

## Directory Structure

- `SVM/`: Implementation of SALVF method on Support Vector Machine (SVM) problems

  - `algorithms/`: Contains various algorithm implementations:
    - `salvf_cvxpy.py`: SALVF implementation using CVXPY
    - `blooc.py`: BLOOC algorithm implementation
    - `lv_hba.py`: LV-HBA algorithm implementation
    - `gam.py`: GAM algorithm implementation
  - `SVM_Tests_20250125diabetes.ipynb`: Jupyter notebook with experiments on diabetes dataset
  - `SVM_Tests_20250125fourclass.ipynb`: Jupyter notebook with experiments on fourclass dataset
  - `diabete.txt`: Diabetes dataset
  - `fourclass.txt`: Fourclass dataset
  - `utils.py`: Utility functions for data loading and processing
- `weight_decay/`: Implementation of SALVF method on weight decay problems in neural networks

  - Contains various algorithm implementations and experimental results
  - `run.ipynb`: Jupyter notebook with experiments
- `toyexample/`: Toy example demonstrating the SALVF method

  - `toy_example.ipynb`: Jupyter notebook with a simple 2D optimization problem

## Requirements

- Python 3.x
- PyTorch
- CVXPY
- ECOS
- NumPy
- SciPy
- Scikit-learn
- Jupyter
