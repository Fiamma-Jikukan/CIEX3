# -*- coding: utf-8 -*-
"""
@author: ofersh@telhai.ac.il
(1+1)-Evolution Strategy with the 1/5th success-rule initialized within [lb,ub]**n
The objective function evaluation calls are adjusted to the ObjectiveFunctoin interface.
"""
import numpy as np
import pandas as pd
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc


def OnePlusOneES_Mixed(n, lb, ub, maxEvals, func=lambda x: x.dot(x), fstop=0, seed=None):
    """
    (1+1)-Evolution Strategy for Mixed-Integer Optimization.

    Arguments:
    n -- Total dimension (D). The second half (n//2 to n) are integers.
    lb, ub -- Lower and upper bounds (scalar, e.g., -100, 100).
    maxEvals -- Budget of function evaluations.
    func -- The objective function.
    """
    local_state = np.random.RandomState(seed)

    # 1. Identify Variable Split [cite: 21]
    n_r = n // 2  # Number of continuous variables
    n_z = n - n_r  # Number of integer variables

    # 2. Initialization
    # Initialize random vector in [lb, ub]
    x = local_state.uniform(lb, ub, size=n)

    # Force the integer part to be actual integers initially
    x[n_r:] = np.round(x[n_r:])

    # Evaluate initial point
    # Note: reshape(1,-1) is required by the ObjectiveFunction
    f_curr = func(x.reshape(1, -1))

    # History tracking
    fhistory = [f_curr]
    shistory = []  # Tracks sigma (step size)

    # 3. Algorithm Parameters
    sigma = (ub - lb) / 6.0  # Initial step size for continuous part
    p_mut_int = 1.0 / n_z  # Probability to mutate a specific integer variable

    # 1/5th Rule Parameters
    evalcount = 1
    osuccess = 0
    epoch = 50  # Check success rate every 'epoch' evaluations
    k_sigma = 0.827  # Multiplier for sigma adaptation
    tol = 1e-6

    # 4. Main Loop
    while (evalcount < maxEvals and f_curr > fstop + tol):

        # --- A. MUTATION ---

        # Copy current best to create candidate
        x_trial = np.copy(x)

        # Mechanism 1: Continuous Mutation (Gaussian)
        # Apply normal noise scaled by sigma to the first half
        r_mutation = sigma * local_state.normal(size=n_r)
        x_trial[:n_r] += r_mutation

        # Mechanism 2: Integer Mutation (Geometric Step)
        # We iterate through the integer part.
        # With probability p_mut_int, we change the value.
        # If we change it, we add a step sampled from a Geometric distribution.
        # This allows for local search (+/-1) and occasional long jumps.

        # Generate a mask: True if we should mutate this specific integer
        mutate_mask = local_state.rand(n_z) < p_mut_int

        # If any integers need mutation:
        if np.any(mutate_mask):
            # How many to mutate?
            num_to_mutate = np.sum(mutate_mask)

            # 1. Direction: +1 or -1
            signs = local_state.choice([-1, 1], size=num_to_mutate)

            # 2. Magnitude: Geometric distribution (p=0.5 means avg step is 2)
            # We add 1 to ensure the step is at least 1.
            magnitudes = 1 + local_state.geometric(p=0.5, size=num_to_mutate)

            # Apply changes
            steps = signs * magnitudes
            x_trial[n_r:][mutate_mask] += steps

        # --- B. BOUNDARY HANDLING ---
        # Clip everything to stay within [lb, ub]
        x_trial = np.clip(x_trial, lb, ub)

        # --- C. EVALUATION ---
        f_trial = func(x_trial.reshape(1, -1))
        evalcount += 1

        # --- D. SELECTION & ADAPTATION ---
        if f_trial < f_curr:
            x = x_trial
            f_curr = f_trial
            osuccess += 1

        # 1/5th Success Rule (Affects Continuous Sigma Only)
        if (evalcount % epoch) == 0:
            ps = osuccess / epoch
            if ps < 0.2:
                sigma *= k_sigma  # Decrease step size
            elif ps > 0.2:
                sigma /= k_sigma  # Increase step size
            osuccess = 0

        # Logging
        fhistory.append(f_curr)
        shistory.append(sigma)

    # Return result consistent with the provided interface
    # xmin, fmin, fhistory, shistory
    return x, f_curr, fhistory, shistory


#
"""
The following __main__ function applies the (1+1)-ES to 9 instances of the mixed-integer quadratic function "RotatedEllipse".
The ES does not handle the integer constraint in a particular manner, but lets the objective function evaluation *round* the values 
to the nearest integer. The experimental setup runs the ES NRUNS times on each of the 3 problem instances over 3 dimensions.
"""
if __name__ == "__main__":
    objFunc = "MixedVarsEllipsoid"
    funcName = 'genRotatedHellipse'
    lb, ub = -100, 100
    #
    budget = 1e6
    NRUNS = 30
    results_list = []
    #
    dimension = [10, 30, 80]
    conditioning = [1, 100, 10000]
    #
    for dim in dimension:
        N = dim // 2
        setC(N)
        #
        for c in conditioning:
            print(f"\nRunning: Dim={dim}, Cond={c}")

            # Setup the objective function
            H = eval(f'Efunc.{funcName}')(dim, c)
            f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

            for k in range(NRUNS):
                # Execute (1+1)-ES
                xmin, fmin, fhistory, shistory = OnePlusOneES_Mixed(dim, lb, ub, budget, func=f)

                # Post-process: Round the integer components (the first N variables)
                xx = np.array([xmin[i] if i < N else np.round(xmin[i]) for i in range(len(xmin))])
                fmin_scalar = np.array(fmin).item()
                print(f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {len(fhistory)}")

    # //// EOF ////
