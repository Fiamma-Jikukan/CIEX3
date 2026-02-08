from random import randint

import numpy as np
import pandas as pd
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc


def validate_bounds(vector, lb=-100, ub=100):
    """Enforce search space boundaries."""
    return np.clip(vector, lb, ub)


def handle_real_part(xr, sigma, local_state):
    """Gaussian mutation for continuous variables."""
    r_mutation = xr + sigma * local_state.normal(size=len(xr))
    return validate_bounds(r_mutation)


def handle_discrete_part(xz, mutation_prob, local_state):
    """Geometric mutation for integer variables as per the original logic."""
    # Difference of two geometric variables creates a discrete symmetric distribution
    mutate = local_state.geometric(p=mutation_prob, size=len(xz)) - local_state.geometric(p=mutation_prob, size=len(xz))
    z_mutation = xz + mutate
    return validate_bounds(z_mutation)


def recombine_parents(p1, p2, local_state):
    """Uniform crossover: Select components from parents with equal probability."""
    mask = local_state.rand(len(p1)) < 0.5    # Condition
    return np.where(mask, p1, p2)


def MuLambdaEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, mu=15, lmbda=100, fstop=1e-6, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []

    nr = n // 2

    # Initialize population: mu parents
    population = local_state.uniform(lb, ub, size=(mu, n))
    population[:, nr:] = np.round(population[:, nr:])

    # Initial evaluation
    fitness = func(population)
    evalcount = mu

    # 1/5th rule parameters
    sigma_real = (ub - lb) / 6.0
    mutation_prob = 0.8
    k_sigma = 0.827
    osuccess = 0
    epoch = 50  # Adjust parameters every 'epoch' generations

    fmin = np.min(fitness)
    fhistory.append(fmin)

    gen_count = 0
    while (evalcount < maxEvals and fmin > fstop):
        offspring = []
        gen_count += 1

        for _ in range(lmbda):
            # Selection and Recombination
            idx1, idx2 = local_state.choice(mu, size=2, replace=False)
            child_base = recombine_parents(population[idx1], population[idx2], local_state)

            # Mutation [cite: 41, 42]
            xr_mut = handle_real_part(child_base[:nr], sigma_real, local_state)
            xz_mut = handle_discrete_part(child_base[nr:], mutation_prob, local_state)

            offspring.append(np.concatenate([xr_mut, xz_mut]))

        offspring = np.array(offspring)
        offspring_fitness = func(offspring)
        evalcount += lmbda

        # Identify best offspring
        best_indices = np.argsort(offspring_fitness)
        current_best_f = offspring_fitness[best_indices[0]]

        # Track success for 1/5 rule
        if current_best_f < fmin:
            osuccess += 1
            fmin = current_best_f

        # Selection: Best mu offspring become next parents
        population = offspring[best_indices[:mu]]
        fitness = offspring_fitness[best_indices[:mu]]

        # Adaptation logic based on 1/5th Success Rule
        if gen_count % epoch == 0:
            ps = osuccess / epoch  # Calculate success rate

            if ps > 0.2:
                # Too successful: increase step size to explore further
                sigma_real /= k_sigma
                mutation_prob = np.clip(mutation_prob * k_sigma, 0.01, 0.95)

            elif ps < 0.2:
                # Not successful enough: decrease step size to exploit locally
                sigma_real *= k_sigma
                mutation_prob = np.clip(mutation_prob / k_sigma, 0.01, 0.95)

            osuccess = 0  # Reset counter for next epoch

        fhistory.append(fmin)
        shistory.append(sigma_real)

    return population[0], fmin, fhistory, shistory


def OnePlusOneEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, fstop=0, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, shistory = [], []

    # Define the split point: half are real (nr), half are integer (nz) [cite: 12, 21]
    nr = n // 2
    nz = n - nr

    # Initialize: Random uniform within bounds (for real and integers)
    xr_min = local_state.uniform(size=nr) * (ub - lb) + lb
    xz_min = local_state.randint(lb, ub, size=nz)

    xmin = np.concatenate([xr_min, xz_min])

    fmin = func(xmin.reshape(1, -1))
    fhistory.append(fmin)

    sigma_real = (ub - lb) / 6.0
    mutation_prob = 0.5

    # TODO: add here mutation_prob?
    shistory.append(sigma_real)

    evalcount, osuccess = 0, 0
    tol, epoch, k_sigma = 1e-6, 50, 0.827

    while (evalcount < maxEvals and fmin > fstop + tol):
        xr_min = xmin[:nr]
        xz_min = xmin[nz:]

        x_min = np.concatenate([handle_real_part(xr_min, sigma_real, local_state),
                               handle_discrete_part(xz_min, mutation_prob, local_state)])

        # Evaluate the specific mixed-variable proposal
        f_x = func(x_min.reshape(1, -1))
        evalcount += 1

        # Selection step
        if f_x < fmin:
            xmin = np.copy(x_min)
            fmin = f_x
            osuccess += 1

        # 1/5th success-rule adaptation
        if (np.mod(evalcount, epoch) == 0):
            ps = osuccess / epoch
            if ps < 0.2:
                sigma_real *= k_sigma
                mutation_prob /= k_sigma

            elif ps > 0.2:
                sigma_real /= k_sigma
                mutation_prob *= k_sigma

            mutation_prob = np.clip(mutation_prob, 0.01, 0.95)
            osuccess = 0

        fhistory.append(fmin)
        shistory.append(sigma_real)

    return xmin, fmin, fhistory, shistory


# def CommaEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, mu=15, lmbda=100, fstop=0, seed=None):
#     local_state = np.random.RandomState(seed)
#     fhistory, shistory = [], []
#     nr = n // 2
#     nz = n - nr
#
#     # Initialize Population: mu parents [cite: 33, 43]
#     # Each row is an individual: [x1, x2, ..., xn]
#     population = local_state.uniform(lb, ub, size=(mu, n))
#     population[:, nr:] = np.round(population[:, nr:])  # Enforce integer constraints [cite: 20]
#
#     # Initial fitness evaluation
#     fitness = func(population)
#     evalcount = mu
#
#     sigma = (ub - lb) / 6.0
#     tol = 1e-6
#
#     while (evalcount < maxEvals and np.min(fitness) > fstop + tol):
#         offspring = []
#
#         # Generate Lambda offspring from Mu parents
#         for _ in range(lmbda):
#             # Select a random parent
#             parent_idx = local_state.randint(mu)
#             parent = population[parent_idx]
#
#             # Mutate
#             child = mutated_proposal(parent, sigma, nr, nz, local_state)
#             offspring.append(child)
#
#         offspring = np.array(offspring)
#         offspring_fitness = func(offspring)
#         evalcount += lmbda
#
#         # (Mu, Lambda) Selection: Best mu offspring become the next parents
#         indices = np.argsort(offspring_fitness)
#         best_indices = indices[:mu]
#
#         population = offspring[best_indices]
#         fitness = offspring_fitness[best_indices]
#
#         # Basic Step-size adaptation (Success-based or simplified)
#         # Note: In (mu, lambda), we often decrease sigma if progress slows
#         # or use more complex self-adaptation. For this template, we track best:
#         fbest = fitness[0]
#         fhistory.append(fbest)
#         shistory.append(sigma)
#
#         # Simple heuristic: slowly reduce exploration as evaluations proceed
#         # A more robust version would use Rechenberg's rule or CSA.
#         if len(fhistory) > 1 and fhistory[-1] >= fhistory[-2]:
#             sigma *= 0.95
#
#     return population[0], fitness[0], fhistory, shistory

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
                xmin, fmin, fhistory, shistory = MuLambdaEvolutionStrategy_mixed(dim, lb, ub, budget, func=f)

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
