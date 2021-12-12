from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from evolutionary_methods import ga, ga_transfer_learning
from argparse import ArgumentParser
import numpy as np
import logging
import sys
use_tqdm = False
if "tqdm" in sys.modules:
    use_tqdm = True
    from tqdm import tqdm



def fitness(x, s, a, env, params):
    policy = NeuroevoPolicy(s, a)
    policy.set_params(x)
    return evaluate(env, params, policy)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', '--env', help='environment', default='small', type=str)
    parser.add_argument('-g', '--gens', help='number of generations', default=100, type=int)
    parser.add_argument('-p', '--pop', help='population size (lambda for the 1+lambda ES)', default=10, type=int)
    parser.add_argument('-s', '--seed', help='seed for evolution', default=0, type=int)
    parser.add_argument('--log', help='log file', default='evolution.log', type=str)
    parser.add_argument('--weights', help='filename to save policy weights', default='weights', type=str)
    args = parser.parse_args()
    logging.basicConfig(filename=args.log, level=logging.DEBUG,
                        format='%(asctime)s %(message)s')

    # starting point
    env, params = get_env(args.env)
    s, a = get_state_action_size(env)
    policy = NeuroevoPolicy(s, a)

    # evolution
    rng = np.random.default_rng(args.seed)
    #start = rng.normal(size=(len(policy.get_params(),)))
    start = policy.get_params()

    def fit(x):
        return fitness(x, s, a, env, params)
    def fit_inv(x):
        return -fitness(x, s, a, env, params)
    print(len(start))
    # x_best = mu_lambda(start, fit, args.gens, args.pop, rng=rng)#cmaES_strategy(start, fit_inv)# oneplus_lambda(start, fit, args.gens, args.pop, rng=rng) #

    # x_best = ga(s,a, fit, std_init=0.5, n_elites=1, pop_size=10, gen=100, mutation_rate=0.1,scheduler=True, rng=rng)# 1/len(start),rng=rng)#
    x_best = ga_transfer_learning(s,a, fitness, std_init=0.5, n_elites=1, pop_size=10, gen=100, mutation_rate=0.1,scheduler=True, rng=rng)
    # Evaluation
    policy.set_params(x_best)
    policy.save(args.weights)
    best_eval = evaluate(env, params, policy)
    print('Best individual: ', x_best[:5])
    print('Fitness: ', best_eval)
