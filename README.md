Code for NeurIPS 2020 paper titled: Adversarial robustness via robust low rank representations (https://arxiv.org/abs/2007.06555). 

This repository contains the following files:

1. mw.py --- Takes a symmetric matrix M as input and solves the optimization problem max_x x^T M x subject to ||x||_\infty <= 1. This file is used to certify the robustness of a given subspace.

