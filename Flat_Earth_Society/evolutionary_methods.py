import numpy as np
from evaluate import get_env
from policy import NeuroevoPolicy
import logging
import sys
import cma
import torch

use_tqdm = False
if "tqdm" in sys.modules:
    use_tqdm = True
    from tqdm import tqdm

# Implementation of CMA-ES strategy
def cmaES_strategy(start_x, fitness_func):
    """Input : - starting individual
               - fitness function
    """
    es = cma.CMAEvolutionStrategy(start_x, 0.2, {'popsize': 10, "maxfevals":250}) # {'verbose': -3}
    es.optimize(fitness_func)
    return es.result.xbest

# Function to Create a Learning rate scheduler


def lr_scheduler(curr_g,max_gens=100):
    begin = 0.01
    end = 0.0001
    return np.abs(((begin-end)*((max_gens - curr_g)/max_gens) + 0.5*end*((curr_g)/max_gens)) + 0.05*np.cos(6*np.pi*curr_g/max_gens)) +0.01
# Mu Lambda ES
def mu_lambda(x, fitness, gens=100, lam=10, alpha=0.01, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x)))
        F = np.zeros(lam)
        for i in range(lam):
            ind = x + N[i, :]
            F[i] = fitness(ind)
            if F[i] > f_best:
                f_best = F[i]
                x_best = ind
                print("New best! Fit : " + str(f_best))
        mu_f = np.mean(F)
        std_f = np.std(F)
        A = F
        n_evals += lam
        # logging.info('\t%d\t%d', n_evals, f_best)
        if std_f != 0:
            A = (F - mu_f) / std_f
        alpha = lr_scheduler(g,max_gens=gens)
        x = x - alpha * np.dot(A, N) / lam
        print("curr best, Fit : " + str(f_best) +" , evals: " + str(n_evals) + " and alpha = "+ str(alpha))
    return x_best

def oneplus_lambda(x, fitness, gens, lam, std=0.1, rng=np.random.default_rng()):
    x_best = x
    f_best = -np.Inf
    n_evals = 0
    for g in range(gens):
        N = rng.normal(size=(lam, len(x))) * std
        for i in range(lam):
            ind = x + N[i, :]
            f = fitness(ind)
            if f > f_best:
                print("New best! Fit : " + str(f))
                f_best = f
                x_best = ind
        x = x_best
        n_evals += lam
        logging.info('\t%d\t%d', n_evals, f_best)
    return x_best


###################################
########   Section For GA  ########
###################################

# Function to evaluate population, returns fitness, best fitness and best indiv
def eval_pop(population, fitness_func):
    fitness_pop = np.zeros(len(population))
    fbest = -np.Inf
    xbest = None
    for i in range(len(population)):
        f = fitness_func(population[i])
        fitness_pop[i] = f
        if f > fbest :
            fbest = f
            xbest = population[i]
    return fitness_pop, fbest, xbest

# Function to select best n_elites individuals
def truncation_selection(population, fits, n_elites = 2):
    elites = np.argsort(-fits)[:n_elites]
    return population[elites].tolist(), fits[elites]

# Implementation of a torunamenet selection between 3 individuals of the population
def tournament_selection(population, fits, rng, t_size=3,):
    tournament = rng.choice(len(population), size=t_size)
    ind = tournament[np.argmax(fits[tournament])]
    return population[ind], fits[ind]

# Implementation of our crossover function, (Uniform crossover)
def crossover_uniform(parent1,parent2,rng):
    child = np.copy(parent1)
    for i in range(len(parent1)):
        if rng.uniform(0, 1) < 0.5:
            child[i] = parent2[i]
    return child

# Mutation mutates individual at a certain rate by changing
def mutation(ind,rng, mutation_rate=0.001):
    num_mutations =int(mutation_rate * len(ind))
    mutation_indices = np.array(rng.choice(range(0, len(ind)), num_mutations))
    child = np.copy(ind)
    # Mutation changes a single gene in each offspring randomly.
    for idx in mutation_indices:
        # The random value to be added to the gene.
        random_value = rng.uniform(-1.0, 1.0, 1)
        child[idx] = ind[idx] + random_value
    return child


# Function to initialise the population (adds noise to first individual)
def init_population(s, a, pop_size, rng, std = 0.5):

    pop = []
    # N = rng.normal(size=(pop_size-1, len(x))) * std
    for i in range(pop_size):
        # ind = x + N[i, :]
        x = NeuroevoPolicy(s, a).get_params()
        pop.append(x)#ind)
        print(np.mean(x))
    return np.array(pop)

# Function to compute one step of the genetic algorithm
def ga_step(population, fits, pop_size, mutation_rate, rng,n_elites = 2):
    # Selecting the best parents (Elitism).
    next_pop , _ = truncation_selection(population, fits, n_elites=n_elites)
    # Then filling population by tournament selection, crossover and mutation
    while len(next_pop) < pop_size:
        parent1, _ = tournament_selection(population, fits, rng, t_size=3)
        parent2, _ = tournament_selection(population, fits, rng, t_size=3)
        # Crossover between parents
        child = crossover_uniform(parent1, parent2, rng)
        # Mutating the child
        child = mutation(child, rng, mutation_rate=mutation_rate)
        next_pop.append(child)
    return np.array(next_pop)

def ga(s, a, fitness_func,seed=0, std_init=0.5, n_elites = 2, pop_size = 10, gen = 100, mutation_rate = 0.1,scheduler=False, rng=np.random.default_rng()):
    torch.manual_seed(seed)
    population = init_population(s, a, pop_size,rng,  std=std_init)
    n_evals = 0

    f_best_overall = -np.Inf
    x_best_overall = None
    if use_tqdm :
        for generation in tqdm(range(gen)):
            ###### print("Generation : ", generation)
            fitness_pop, f_best, x_best = eval_pop(population, fitness_func)
            n_evals += len(population)

            # Checking if found a new overall best indiv.

            if f_best > f_best_overall:
                f_best_overall = f_best
                x_best_overall = x_best
                print("New best! Fit : " + str(f_best_overall))

            # If we are using a scheduler instead of a fixed mutation rate, apply it
            if scheduler:
                r = lr_scheduler(generation,max_gens=gen)
            else:
                r = mutation_rate
            # GA step
            population = ga_step(population, fitness_pop, pop_size, r, rng, n_elites=n_elites)

            # if f_best_overall == 0 :
            #     break
            logging.info('\t%d\t%d', n_evals, f_best_overall)
    else:
        for generation in range(gen):
            fitness_pop, f_best, x_best = eval_pop(population, fitness_func)
            n_evals += len(population)

            # Checking if found a new overall best indiv.

            if f_best > f_best_overall:
                f_best_overall = f_best
                x_best_overall = x_best
                print("New best! Fit : " + str(f_best_overall))

            # If we are using a scheduler instead of a fixed mutation rate, apply it
            if scheduler:
                r = lr_scheduler(generation, max_gens=gen)
            else:
                r = mutation_rate
                # GA step
            population = ga_step(population, fitness_pop, pop_size, r, rng, n_elites=n_elites)


            logging.info('\t%d\t%d', n_evals, f_best_overall)
            # if f_best_overall == 0 :
            #     break
    return x_best_overall


##### Specifically for Transfer learning:
# Function to evaluate population, returns fitness, best fitness and best indiv
def eval_pop_transf(s, a, params, population, fitness_func, env):
    fitness_pop = np.zeros(len(population))
    fbest = -np.Inf
    xbest = None
    for i in range(len(population)):
        f = fitness_func(population[i], s, a, env, params)#fitness_func(population[i])
        fitness_pop[i] = f
        if f > fbest :
            fbest = f
            xbest = population[i]
    return fitness_pop, fbest, xbest


def ga_transfer_learning(s, a, fitness_func, seed=0,std_init=0.5, n_elites=2, pop_size=10, gen=100, mutation_rate=0.1, scheduler=False,rng=np.random.default_rng()):
    env, params = get_env("small")
    torch.manual_seed(seed)
    population = init_population(s, a, pop_size, rng, std=std_init)
    n_evals = 0
    switched= False
    f_best_overall = -np.Inf
    x_best_overall = None

    r = mutation_rate
    if use_tqdm:
        for generation in tqdm(range(gen)):
            ###### print("Generation : ", generation)
            fitness_pop, f_best, x_best = eval_pop_transf(s, a, params, population, fitness_func, env)#eval_pop(population, fitness_func)
            n_evals += len(population)

            # Checking if found a new overall best indiv.

            if f_best > f_best_overall:
                f_best_overall = f_best
                x_best_overall = x_best
                print(r)
                print("New best! Fit : " + str(f_best_overall))

            # If we are using a scheduler instead of a fixed mutation rate, apply it
            if scheduler:
                r = lr_scheduler(generation, max_gens=gen)
            else:
                r = mutation_rate
            # GA step
            population = ga_step(population, fitness_pop, pop_size, r, rng, n_elites=n_elites)

            # if f_best_overall == 0 :
            #     break
            logging.info('\t%d\t%d', n_evals, f_best_overall)

            if (f_best > -20 or n_evals > 350) and not switched:
                switched = True
                env, params = get_env("large")
                f_best_overall = -np.Inf
    else:
        for generation in range(gen):
            fitness_pop, f_best, x_best = eval_pop_transf(s, a, params, population, fitness_func, env)#eval_pop(population, fitness_func)
            n_evals += len(population)

            # Checking if found a new overall best indiv.

            if f_best > f_best_overall:
                f_best_overall = f_best
                x_best_overall = x_best
                print("New best! Fit : " + str(f_best_overall))

            # If we are using a scheduler instead of a fixed mutation rate, apply it
            if scheduler:
                r = lr_scheduler(generation, max_gens=gen)
            else:
                r = mutation_rate
                # GA step
            population = ga_step(population, fitness_pop, pop_size, r, rng, n_elites=n_elites)

            logging.info('\t%d\t%d', n_evals, f_best_overall)
            # if f_best_overall == 0 :
            #     break
            if (f_best > -20 or n_evals > 350) and not switched:
                switched= True
                env, params = get_env("large")
                f_best_overall = -np.Inf
    return x_best_overall
#####