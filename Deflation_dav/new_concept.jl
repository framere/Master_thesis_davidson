using LinearAlgebra
using Printf


l = 16 # number of eigenvalues to find
n_aux = l * 16 # number of auxiliary vectors
n_b = 10 #block size
nu_0 = max(1, n_b) # number of eigenvalues to find in each block

nevf = 0 # number of eigenvalues found so far

V_1 = 