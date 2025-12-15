# Physics-Informed-Neural-Network (PINN) for solving an ODE
This repository contains an academic project developed during the 3rd semester of a Data Science and Mathematics degree.

The goal of this project is to solve an ordinary differential equation (ODE) using a Physics-Informed Neural Network (PINN), where the governing differential equation is incorporated directly into the loss function of a neural network.

## Problem Description

Traditional neural networks require labeled data to learn a solution. In contrast, PINNs leverage physical laws, expressed as differential equations, to guide the learning process without needing explicit solution data.

In this project, a PINN is trained to approximate the solution of an ODE subject to given initial/boundary conditions.

## Governing Equation

We consider the following ordinary differential equation:

$$
\frac{dy}{dt} = e^{x + y}, \quad y(0) = -1
$$

where $y(t)$ is the unknown function and the initial condition is enforced through the physics-informed loss.

## Analytical Solution

The analytical solution of the differential equation is given by:

$$
y(t) = -\ln\left(e + 1 - e^{t}\right)
$$

This solution is used to compare and validate the approximation obtained by the Physics-Informed Neural Network.


## Methodology

Fully connected neural network implemented in TensorFlow

Physics-informed loss function combining:

Differential equation residual

Initial/boundary condition constraints

Automatic differentiation to compute derivatives

Training using gradient-based optimization

## Results

The PINN solution closely matches the analytical solution of the ODE

Training loss decreases consistently, indicating successful enforcement of the physical constraints

Visual comparison between analytical and PINN-based solutions is provided in the notebook

## Conclusions

Physics-Informed Neural Networks are an effective approach for solving differential equations

The method reduces the need for labeled training data

Performance depends on network architecture and hyperparameter selection

## Notes

This is an academic project, not original research

The full explanation, derivations, and implementation details are included inside the Jupyter notebook

Notebook originally developed and executed in Kaggle

## How to run

This project was developed in Kaggle.
To run locally:

pip install -r requirements.txt
jupyter notebook physics-informed-neural-network-pinn.ipynb
