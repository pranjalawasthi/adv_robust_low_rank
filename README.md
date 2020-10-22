Code for NeurIPS 2020 paper titled: Adversarial robustness via robust low rank representations (https://arxiv.org/abs/2007.06555). 

This repository contains the following files:

1. mw.py --- Takes a symmetric matrix M as input and solves the optimization problem:

	 max_x x^T M x subject to ||x||_\infty <= 1. 
	 
This file is used to certify the robustness of a given subspace.


2. robust_low_rank.py: Takes in a data matrix and computes a robust low dimensional subspace that approximates the data. In order to do this the algorithm uses a combination of the PCA and sparse PCA subroutines.


Dependencies:

Python

scipy

sklearn


The code for robust training and evaluation of neural networks is coming soon.



