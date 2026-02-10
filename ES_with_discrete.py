import numpy as np
import matplotlib.pyplot as plt
from MixedVariableObjectiveFunctions import setC
import ellipsoidFunctions as Efunc
import MixedVariableObjectiveFunctions as f_mixed


def validate_bounds(vector, lb=-100, ub=100):
    """Enforce search space boundaries."""
    return np.clip(vector, lb, ub)


def handle_real_part(xr, sigma, local_state):
    """Gaussian mutation for continuous variables."""
    r_mutated = xr + sigma * local_state.normal(size=len(xr))
    return validate_bounds(r_mutated)


def handle_discrete_part(xz, mutation_prob, local_state):
    """Geometric mutation for integer variables."""
    # sample a symmetric integer step using geometric(p) - geometric(p)
    step = (
        local_state.geometric(p=mutation_prob, size=len(xz))
        - local_state.geometric(p=mutation_prob, size=len(xz))
    )
    # geometric noise to add to the discrete part
    z_mutated = xz + step
    return validate_bounds(z_mutated)


def recombine_parents(p1, p2, local_state):
    """Crossover: select components from parents with equal probability."""
    # boolean mask to choose each coordinate with probability 0.5
    mask = local_state.rand(len(p1)) < 0.5
    # mix coordinates according to mask
    return np.where(mask, p1, p2)


def MuLambdaEvolutionStrategy_mixed(n, lb, ub, maxEvals, func, mu=15, lmbda=100, fstop=1e-6, seed=None):
    local_state = np.random.RandomState(seed)
    fhistory, eval_history = [], []

    # split decision vector: first nr are real, last (n-nr) are discrete
    nr = n // 2

    # set initial real step size relative to the search range
    initial_sigma = (ub - lb) / 6.0
    # if sigma drops below this: trigger a local kick
    restart_threshold = 1e-9

    # initialize mu parents
    population = local_state.uniform(lb, ub, size=(mu, n))
    # enforce integer values on the discrete half in each individual at the population
    population[:, nr:] = np.round(population[:, nr:])

    # evaluate initial population: eval_count in func will be increase in mu (handled by func class)
    fitness = func(population)

    sigma_real = initial_sigma
    mutation_prob = 0.8
    k_sigma = 0.827
    osuccess = 0
    epoch = 50

    # save current global best value
    fmin = np.min(fitness)
    fhistory.append(fmin)
    eval_history.append(func.eval_count)

    gen_count = 0
    while (func.eval_count + lmbda <= maxEvals and fmin > fstop):
        offspring = []
        gen_count += 1

        # create next generation
        for _ in range(lmbda):
            # choose two random distinct parents
            idx1, idx2 = local_state.choice(mu, size=2, replace=False)
            # recombine them to create a child
            child = recombine_parents(population[idx1], population[idx2], local_state)

            # mutate real and discrete parts separately
            xr_mut = handle_real_part(child[:nr], sigma_real, local_state)
            xz_mut = handle_discrete_part(child[nr:], mutation_prob, local_state)

            # assemble child and append to offsprings list
            child_mutated = np.concatenate([xr_mut, xz_mut])
            offspring.append(child_mutated)

        # evaluate offspring
        offspring = np.array(offspring)
        offspring_fitness = func(offspring)

        # sort offspring by fitness (ascending)
        best_indices = np.argsort(offspring_fitness)
        current_best_f = offspring_fitness[best_indices[0]]

        # update global best and success count
        if current_best_f < fmin:
            osuccess += 1
            fmin = current_best_f

        # keep the best mu offspring
        population = offspring[best_indices[:mu]]
        fitness = offspring_fitness[best_indices[:mu]]

        # 1/5 success rule every epoch generations: adapt step size for real and discrete parameters
        if gen_count % epoch == 0:
            ps = osuccess / epoch
            # too many successes -> increase step sizes
            if ps > 0.2:
                sigma_real /= k_sigma    # increase real step size
                mutation_prob = np.clip(mutation_prob * k_sigma, 0.01, 0.95)   # decrease p: larger steps
            # too few successes -> decrease step sizes
            elif ps < 0.2:
                sigma_real *= k_sigma    # decrease real step size
                mutation_prob = np.clip(mutation_prob / k_sigma, 0.01, 0.95)    # increase p: smaller steps

            osuccess = 0

            # kick: local restart around current best parent
            if sigma_real < restart_threshold:
                # reset sigma to be 0.01 of the initial sigma
                sigma_real = initial_sigma * 0.01

                # keeps search in the current region but add noise
                best_idx = np.argmin(fitness)
                x_best = population[best_idx].copy()
                noise = local_state.normal(0, sigma_real, size=population.shape)
                population = validate_bounds(x_best + noise)
                # re-enforce integers on discrete half
                population[:, nr:] = np.round(population[:, nr:])

                # check for new global best
                fitness = func(population)
                current_parent_best = np.min(fitness)

                if current_parent_best < fmin:
                    fmin = current_parent_best
                    osuccess += 1

        # track convergence
        fhistory.append(fmin)
        eval_history.append(func.eval_count)

    # return best parent found, its fitness, and history: for each generation best and calls num
    return population[0], fmin, fhistory, eval_history


if __name__ == "__main__":
    objFunc = "MixedVarsEllipsoid"
    funcName = "genRotatedHellipse"
    lb, ub = -100, 100
    budget = 1e6
    NRUNS = 30

    dimension = [10, 30, 80]
    conditioning = [1, 100, 10000]

    # store best and avg per configuration for final table
    final_stats = {}

    for dim in dimension:
        N = dim // 2
        setC(N)

        for c in conditioning:
            print(f"\nRunning: Dim={dim}, Cond={c}")
            best_score_key = f"dim{dim}_cond{c}"

            # keep the best final value and all finals for average
            best_scores = {best_score_key: float("inf")}
            all_final_fmin = []
            best_run_history = None
            best_run_eval_history = None

            # build objective
            H = eval(f"Efunc.{funcName}")(dim, c)
            f = eval(f"f_mixed.{objFunc}")(d=dim, bid=0, ind=N, H=H, c=c, max_eval=budget)

            for k in range(NRUNS):
                f.eval_count = 0    # reset eval_count
                # run es and collect best-so-far
                xmin, fmin, fhistory, eval_history = MuLambdaEvolutionStrategy_mixed(dim, lb, ub, budget, func=f)

                # normalize to python float
                fmin_scalar = np.array(fmin).item()
                all_final_fmin.append(fmin_scalar)

                # keep the best run and its history
                if fmin_scalar < best_scores[best_score_key]:
                    best_scores[best_score_key] = fmin_scalar
                    best_run_history = fhistory
                    best_run_eval_history = eval_history

                print(
                    f"  Run {k}: fmin = {fmin_scalar:.4e} | Evals = {f.eval_count} "
                    f"| Best so far = {best_scores[best_score_key]:.4e}"
                )

            # compute average for this configuration
            avg_fmin = np.mean(all_final_fmin)
            final_stats[best_score_key] = {"best": best_scores[best_score_key], "avg": avg_fmin}

            # plot convergence for the best run in this configuration
            plt.figure(figsize=(8, 5))
            plt.semilogy(best_run_eval_history, best_run_history, label="Best Run Convergence")
            plt.title(f"Convergence Plot: Dim {dim}, Cond {c}")
            plt.xlabel("Objective Function Calls")
            plt.ylabel("Best Fitness (fmin)")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

    # print final summary table
    print("\n" + "=" * 70)
    print(f"{'Configuration':<20} | {'Best Score':<15} | {'Average Score (30 runs)':<15}")
    print("=" * 70)
    for key, val in final_stats.items():
        print(f"{key:<20} | {val['best']:15.4e} | {val['avg']:15.4e}")
    print("=" * 70)
