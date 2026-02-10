from random import randint

import numpy as np
import pandas as pd
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc
import matplotlib.pyplot as plt


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
    initial_sigma = (ub - lb) / 6.0
    restart_threshold = 1e-9  # Threshold to trigger a "Kick" or Restart

    # Initialize population: mu parents [cite: 43]
    population = local_state.uniform(lb, ub, size=(mu, n))
    population[:, nr:] = np.round(population[:, nr:])

    # Initial evaluation
    fitness = func(population)
    evalcount = mu

    sigma_real = initial_sigma
    mutation_prob = 0.8
    k_sigma = 0.827
    osuccess = 0
    epoch = 50

    fmin = np.min(fitness)
    fhistory.append(fmin)

    gen_count = 0
    while (evalcount + lmbda <= maxEvals and fmin > fstop):
        offspring = []
        gen_count += 1

        # --- RESTART / KICK LOGIC ---
        # If sigma becomes too small, the algorithm is stuck in a local minimum.
        # We "kick" the search by resetting sigma and perturbing the population.
        if sigma_real < restart_threshold:
            # Instead of a 10% jump, try a 1% jump or scale it
            sigma_real = initial_sigma * 0.01

            # Focus the kick: Perturb the population around the current best
            # This helps explore the immediate neighborhood of the local minimum
            noise = local_state.normal(0, sigma_real, size=population.shape)
            population = validate_bounds(population + noise)
            population[:, nr:] = np.round(population[:, nr:])

            # Reset success to allow 1/5th rule to re-expand if needed
            osuccess = 0

        for _ in range(lmbda):
            # Selection and Recombination
            idx1, idx2 = local_state.choice(mu, size=2, replace=False)
            child_base = recombine_parents(population[idx1], population[idx2], local_state)

            # Mutation
            xr_mut = handle_real_part(child_base[:nr], sigma_real, local_state)
            xz_mut = handle_discrete_part(child_base[nr:], mutation_prob, local_state)

            offspring.append(np.concatenate([xr_mut, xz_mut]))

        offspring = np.array(offspring)
        offspring_fitness = func(offspring)
        evalcount += lmbda

        # Identify best offspring
        best_indices = np.argsort(offspring_fitness)
        current_best_f = offspring_fitness[best_indices[0]]

        if current_best_f < fmin:
            osuccess += 1
            fmin = current_best_f

        # Selection
        population = offspring[best_indices[:mu]]
        fitness = offspring_fitness[best_indices[:mu]]

        # Adaptation: 1/5th Success Rule
        if gen_count % epoch == 0:
            ps = osuccess / epoch
            if ps > 0.2:
                sigma_real /= k_sigma
                mutation_prob = np.clip(mutation_prob * k_sigma, 0.01, 0.95)
            elif ps < 0.2:
                sigma_real *= k_sigma
                mutation_prob = np.clip(mutation_prob / k_sigma, 0.01, 0.95)
            osuccess = 0

        fhistory.append(fmin)
        shistory.append(sigma_real)

    return population[0], fmin, fhistory, shistory


if __name__ == "__main__":
    objFunc = "MixedVarsEllipsoid"
    funcName = 'genRotatedHellipse'
    lb, ub = -100, 100
    budget = 1e6
    NRUNS = 30

    dimension = [10, 30, 80]
    conditioning = [1, 100, 10000]

    # Storage for final summary
    final_stats = {}

    for dim in dimension:
        N = dim // 2
        setC(N)
        for c in conditioning:
            print(f"\nRunning: Dim={dim}, Cond={c}")
            best_score_key = f"dim{dim}_cond{c}"

            best_scores = {best_score_key: float('inf')}
            all_final_fmin = []
            best_run_history = None

            # Setup Objective Function [cite: 22, 23]
            H = eval(f'Efunc.{funcName}')(dim, c)
            f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

            for k in range(NRUNS):
                # Reset internal evaluation counter if the class supports it
                f.eval_count = 0

                xmin, fmin, fhistory,_ = MuLambdaEvolutionStrategy_mixed(dim, lb, ub, budget, func=f)

                fmin_scalar = np.array(fmin).item()
                all_final_fmin.append(fmin_scalar)

                if fmin_scalar < best_scores[best_score_key]:
                    best_scores[best_score_key] = fmin_scalar
                    best_run_history = fhistory  # Save history of the best run for graphing

                print(f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {f.eval_count} | Best so far = {best_scores[best_score_key]:.4e}")

            # Calculate average for this configuration
            avg_fmin = np.mean(all_final_fmin)
            final_stats[best_score_key] = {"best": best_scores[best_score_key], "avg": avg_fmin}

            # --- Generate Graph for this configuration ---
            plt.figure(figsize=(8, 5))

            # Reconstruct the evaluation counts for the x-axis
            # First point is 'mu', subsequent points are 'mu + lmbda * gen'
            mu_val = 15
            lmbda_val = 100
            eval_counts = [mu_val] + [mu_val + (i + 1) * lmbda_val for i in range(len(best_run_history) - 1)]

            # Plot using the reconstructed x-axis and the 1D history
            plt.semilogy(eval_counts, best_run_history, label='Best Run Convergence')
            plt.title(f"Convergence Plot: Dim {dim}, Cond {c}")
            plt.xlabel("Objective Function Calls")
            plt.ylabel("Best Fitness (fmin)")
            plt.grid(True, which="both", linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # --- Final Summary Print ---
    print("\n" + "=" * 70)
    print(f"{'Configuration':<20} | {'Best Score':<15} | {'Average Score (30 runs)':<15}")
    print("=" * 70)
    for key, val in final_stats.items():
        print(f"{key:<20} | {val['best']:15.4e} | {val['avg']:15.4e}")
    print("=" * 70)