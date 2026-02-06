import numpy as np
import pandas as pd
from MixedVariableObjectiveFunctions import setC
import MixedVariableObjectiveFunctions as f_mixed
import ellipsoidFunctions as Efunc


def gaussian_mutation(real_values, sigma, local_state):
    """Apply Gaussian mutation to real-valued variables."""
    mutation = sigma * local_state.normal(size=len(real_values))
    return real_values + mutation


def geometric_mutation(integer_values, p_mut_int, local_state):
    """Apply geometric mutation to integer variables."""
    n_z = len(integer_values)
    mutated_values = np.copy(integer_values)

    mutate_mask = local_state.rand(n_z) < p_mut_int

    if np.any(mutate_mask):
        num_to_mutate = np.sum(mutate_mask)
        signs = local_state.choice([-1, 1], size=num_to_mutate)
        # magnitude >= 1 ensures we actually jump on the integer grid
        magnitudes = 1 + local_state.geometric(p=0.5, size=num_to_mutate)
        steps = signs * magnitudes
        mutated_values[mutate_mask] += steps

    return mutated_values


def OnePlusOneES_Mixed(n, lb, ub, maxEvals, func=lambda x: x.dot(x), fstop=0, seed=None):
    """(1+1)-Evolution Strategy for Mixed-Integer Optimization."""
    local_state = np.random.RandomState(seed)
    n_r = n // 2  # First half: Continuous 
    n_z = n - n_r  # Second half: Integers 

    x = local_state.uniform(lb, ub, size=n)
    x[n_r:] = np.round(x[n_r:])  # Initialize integer part correctly

    f_curr = func(x.reshape(1, -1))
    fhistory = [f_curr]
    shistory = []

    sigma = (ub - lb) / 6.0
    p_mut_int = 2.0 / n_z
    evalcount = 1
    osuccess = 0
    epoch = 50
    k_sigma = 0.827
    tol = 1e-6

    while (evalcount < maxEvals and f_curr > fstop + tol):
        x_trial = np.copy(x)

        # Split and Mutate
        real_part = x[:n_r]
        integer_part = x[n_r:]

        x_trial[:n_r] = gaussian_mutation(real_part, sigma, local_state)
        x_trial[n_r:] = geometric_mutation(integer_part, p_mut_int, local_state)

        x_trial = np.clip(x_trial, lb, ub)
        f_trial = func(x_trial.reshape(1, -1))
        evalcount += 1

        if f_trial < f_curr:
            x = x_trial
            f_curr = f_trial
            osuccess += 1

        if (evalcount % epoch) == 0:
            ps = osuccess / epoch
            sigma *= (k_sigma if ps < 0.2 else 1 / k_sigma)
            osuccess = 0

        fhistory.append(f_curr)
        shistory.append(sigma)

    return x, f_curr, fhistory, shistory


if __name__ == "__main__":
    objFunc = "MixedVarsEllipsoid"
    funcName = 'genRotatedHellipse'
    lb, ub = -100, 100
    budget = 1e5
    NRUNS = 10  # Required number of independent runs

    # Storage for statistics and best scores
    stats_summary = {}
    best_scores = {}

    dimension = [10, 30, 80]  # Required dimensions
    conditioning = [1, 100, 10000]  # Required conditioning values

    for dim in dimension:
        N = dim // 2
        setC(N)
        for c in conditioning:
            print(f"\nRunning: Dim={dim}, Cond={c}")
            instance_key = f"dim{dim}_cond{c}"
            best_scores[instance_key] = float('inf')

            run_evals = []

            # Setup the objective function [cite: 22, 23, 26]
            H = eval(f'Efunc.{funcName}')(dim, c)
            f = eval(f'f_mixed.{objFunc}')(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

            for k in range(NRUNS):
                # Execute Mixed (1+1)-ES
                xmin, fmin, fhistory, shistory = OnePlusOneES_Mixed(dim, lb, ub, budget, func=f)

                # Post-process: Corrected logic (Indices N to dim are integers) [cite: 14]
                xx = np.array([xmin[i] if i < N else np.round(xmin[i]) for i in range(len(xmin))])

                fmin_scalar = np.array(fmin).item()
                evals_count = len(fhistory)
                run_evals.append(evals_count)

                # Update best score for this instance
                if fmin_scalar < best_scores[instance_key]:
                    best_scores[instance_key] = fmin_scalar

                # Print matches the naive approach style with "Best so far" added
                print(
                    f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {evals_count} | Best so far = {best_scores[instance_key]:.4e}")

            # Store averages for final summary
            stats_summary[instance_key] = {
                'avg_evals': np.mean(run_evals),
                'std_evals': np.std(run_evals),
                'best_score': best_scores[instance_key]
            }

    # Final summary
    print("\n" + "=" * 80)
    print(f"{'Instance':<20} | {'Avg Evals':<15} | {'Std Dev':<15} | {'Best Score':<15}")
    print("-" * 80)
    for key, data in stats_summary.items():
        print(f"{key:<20} | {data['avg_evals']:<15.1f} | {data['std_evals']:<15.1f} | {data['best_score']:<15.4e}")
    print("=" * 80)