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

def OnePlusOneEvolutionStrategy(n, lb, ub, maxEvals , func=lambda x: x.dot(x), fstop=0, seed = None) :
    local_state = np.random.RandomState(seed)
    fhistory,shistory = [],[]
    xmin = local_state.uniform(size=n)*(ub - lb) + lb
    fmin = func(xmin.reshape(1,-1)) #reshape since it is a singleton and func receives a population in a 2D numpy array
    fhistory.append(fmin)
    sigma = (ub-lb)/6.0
    shistory.append(sigma)
    evalcount,osuccess = 0,0
    tol = 1e-6
    epoch = 50
    k_sigma = 0.827
    while (evalcount < maxEvals and fmin > fstop+tol) :
        x = xmin + sigma*local_state.normal(size=n)
        f_x = func(x.reshape(1,-1)) #reshape since it is a singleton and func receives a population in a 2D numpy array
        evalcount += 1
        if f_x < fmin :
            xmin = np.copy(x)
            fmin = f_x
            osuccess += 1
        if (np.mod(evalcount,epoch)==0) : # 1/5th success-rule every epoch
            ps = osuccess/epoch
            if (ps < 0.2) :
                sigma *= k_sigma
            elif (ps > 0.2) :
                sigma /= k_sigma
            osuccess = 0;
#        
        fhistory.append(fmin)
        shistory.append(sigma)
    return xmin,fmin,fhistory,shistory
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
    best_scores = {}  # Dictionary to store best scores for each (dim, cond) combination

    for dim in dimension:
        N = dim // 2
        setC(N)
        #
        for c in conditioning:
            print(f"\nRunning: Dim={dim}, Cond={c}")
            best_score_key = f"dim{dim}_cond{c}"
            best_scores[best_score_key] = float('inf')  # Initialize with infinity

            # Setup the objective function
            H = eval(f'Efunc.{funcName}')(dim, c)
            f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

            for k in range(NRUNS):
                # Execute (1+1)-ES
                xmin, fmin, fhistory, shistory = OnePlusOneEvolutionStrategy(dim, lb, ub, budget, func=f)

                # Post-process: Round the integer components (the first N variables)
                xx = np.array([xmin[i] if i < N else np.round(xmin[i]) for i in range(len(xmin))])
                fmin_scalar = np.array(fmin).item()

                # Update best score for this (dim, cond) combination
                if fmin_scalar < best_scores[best_score_key]:
                    best_scores[best_score_key] = fmin_scalar

                print(
                    f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {len(fhistory)} | Best so far = {best_scores[best_score_key]:.4e}")

    # Print final best scores for all 9 runs
    print("\n" + "=" * 50)
    print("FINAL BEST SCORES FOR EACH OF THE 9 RUNS:")
    print("=" * 50)
    for key, score in best_scores.items():
        print(f"{key}: {score:.4e}")
    print("=" * 50)

    # //// EOF ////