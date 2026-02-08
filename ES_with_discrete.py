
import numpy as np
import pandas as pd
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc


def mutated_proposal(xmin, sigma, n_r, n_z, local_state):
    # n_r is the number of real variables, n_z is the number of integers
    # Real part: Gaussian mutation
    r_mutation = local_state.normal(0, sigma, size=n_r)

    # Integer part: Discrete mutation (e.g., using a discrete distribution)
    # This ensures we move specifically on the integer grid
    z_mutation = local_state.geometric(p=0.5, size=n_z) - local_state.geometric(p=0.5, size=n_z)

    # Combine them
    return np.concatenate([xmin[:n_r] + r_mutation, xmin[n_r:] + z_mutation])


def OnePlusOneEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, fstop=0, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []

    # Define the split point: half are real (nr), half are integer (nz) [cite: 12, 21]
    nr = n // 2
    nz = n - nr

    # Initialize: Random uniform within bounds
    xmin = local_state.uniform(size=n) * (ub - lb) + lb
    # Ensure the integer half starts as integers
    xmin[nr:] = np.round(xmin[nr:])

    fmin = func(xmin.reshape(1, -1))
    fhistory.append(fmin)

    sigma = (ub - lb) / 6.0
    shistory.append(sigma)
    evalcount, osuccess = 0, 0
    tol, epoch, k_sigma = 1e-6, 50, 0.827

    while (evalcount < maxEvals and fmin > fstop + tol):
        # --- CALL THE DIFFERENTIATED MUTATION HERE ---
        # We pass the split sizes nr and nz to handle them differently [cite: 41]
        x_proposal = mutated_proposal(xmin, sigma, nr, nz, local_state)

        # Evaluate the specific mixed-variable proposal
        f_x = func(x_proposal.reshape(1, -1))
        evalcount += 1

        # Selection step
        if f_x < fmin:
            xmin = np.copy(x_proposal)
            fmin = f_x
            osuccess += 1

        # 1/5th success-rule adaptation
        if (np.mod(evalcount, epoch) == 0):
            ps = osuccess / epoch
            if ps < 0.2:
                sigma *= k_sigma
            elif ps > 0.2:
                sigma /= k_sigma
            osuccess = 0

        fhistory.append(fmin)
        shistory.append(sigma)

    return xmin, fmin, fhistory, shistory


def CommaEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, mu=15, lmbda=100, fstop=0, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []
    nr = n // 2
    nz = n - nr

    # Initialize Population: mu parents [cite: 33, 43]
    # Each row is an individual: [x1, x2, ..., xn]
    population = local_state.uniform(lb, ub, size=(mu, n))
    population[:, nr:] = np.round(population[:, nr:])  # Enforce integer constraints [cite: 20]

    # Initial fitness evaluation
    fitness = func(population)
    evalcount = mu

    sigma = (ub - lb) / 6.0
    tol = 1e-6

    while (evalcount < maxEvals and np.min(fitness) > fstop + tol):
        offspring = []

        # Generate Lambda offspring from Mu parents
        for _ in range(lmbda):
            # Select a random parent
            parent_idx = local_state.randint(mu)
            parent = population[parent_idx]

            # Mutate
            child = mutated_proposal(parent, sigma, nr, nz, local_state)
            offspring.append(child)

        offspring = np.array(offspring)
        offspring_fitness = func(offspring)
        evalcount += lmbda

        # (Mu, Lambda) Selection: Best mu offspring become the next parents
        indices = np.argsort(offspring_fitness)
        best_indices = indices[:mu]

        population = offspring[best_indices]
        fitness = offspring_fitness[best_indices]

        # Basic Step-size adaptation (Success-based or simplified)
        # Note: In (mu, lambda), we often decrease sigma if progress slows
        # or use more complex self-adaptation. For this template, we track best:
        fbest = fitness[0]
        fhistory.append(fbest)
        shistory.append(sigma)

        # Simple heuristic: slowly reduce exploration as evaluations proceed
        # A more robust version would use Rechenberg's rule or CSA.
        if len(fhistory) > 1 and fhistory[-1] >= fhistory[-2]:
            sigma *= 0.95

    return population[0], fitness[0], fhistory, shistory

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
                xmin, fmin, fhistory, shistory = CommaEvolutionStrategy_mixed(dim, lb, ub, budget, func=f)

                # Post-process: Round the integer components (the first N variables)
                xx = np.array([xmin[i] if i < N else np.round(xmin[i]) for i in range(len(xmin))])
                fmin_scalar = np.array(fmin).item()

                # Update best score for this (dim, cond) combination
                if fmin_scalar < best_scores[best_score_key]:
                    best_scores[best_score_key] = fmin_scalar

                print(f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {len(fhistory)} | Best so far = {best_scores[best_score_key]:.4e}")

    # Print final best scores for all 9 runs
    print("\n" + "="*50)
    print("FINAL BEST SCORES FOR EACH OF THE 9 RUNS:")
    print("="*50)
    for key, score in best_scores.items():
        print(f"{key}: {score:.4e}")
    print("="*50)

    # //// EOF ////
