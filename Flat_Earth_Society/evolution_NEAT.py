import os

from evaluate import get_env, get_state_action_size, evaluate
from policy import NeuroevoPolicy
from evolutionary_methods import ga
from argparse import ArgumentParser
import numpy as np
import logging
import sys


use_tqdm = False
if "tqdm" in sys.modules:
    use_tqdm = True
    from tqdm import tqdm

#############################
import torch
import neat.population as pop
from neat.visualize import draw_net
from neat.phenotype.feed_forward import FeedForwardNet

#########


class Config:

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    VERBOSE = True

    NUM_INPUTS = 231
    NUM_OUTPUTS = 5
    USE_BIAS = True

    ACTIVATION = 'relu'
    SCALE_ACTIVATION = 4.9

    FITNESS_THRESHOLD = 0.0

    POPULATION_SIZE = 10
    NUMBER_OF_GENERATIONS = 100
    SPECIATION_THRESHOLD = 3.0

    CONNECTION_MUTATION_RATE = 0.80
    CONNECTION_PERTURBATION_RATE = 0.90
    ADD_NODE_MUTATION_RATE = 0.03
    ADD_CONNECTION_MUTATION_RATE = 0.5

    CROSSOVER_REENABLE_CONNECTION_GENE_RATE = 0.25

    # Top percentage of species to be saved before mating
    PERCENTAGE_TO_SAVE = 0.80

    def fitness_fn(self, genome):

        phenotype = FeedForwardNet(genome, self)
        fitness = evaluate(env, params, phenotype)


        return fitness




def NEAT_evolve():
    neat = pop.Population(Config)
    solution, generation = neat.run()
    phenotype = FeedForwardNet(solution,Config)
    print( evaluate(env, params, phenotype))
    draw_net(solution, view=True, filename='./images/pole-balancing-solution', show_disabled=True)

    return solution


###########################

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

    x_best = NEAT_evolve()
    # Evaluation
    # policy.set_params(x_best)
    # policy.save(args.weights)
    # best_eval = evaluate(env, params, policy)
    # print('Best individual: ', x_best[:5])
    # print('Fitness: ', best_eval)
