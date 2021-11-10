from statistics import mean

import gym
import numpy

from Lab2.Ex2 import Network

# The difference in experiments is related mostly to number of population.
# If we have small population then, we have more impact of fluctuations.
# For example, we may not get optimum, because initially we didn't have suited vectors.

if "__main__" in __name__:
    # Params
    n_episodes = 3
    pop_size = 10
    n_hidden = 5
    n_generations = 100
    gene_range = 0.1
    mut_range = 0.02

    env = gym.make('CartPole-v0')
    threshold = 200

    # env = gym.make("Pendulum-v0")
    # threshold = 0

    network = Network(env, n_hidden, p_variance=gene_range)

    # Creating population. Each gene is tuple (params, index, curr_fitness)
    population = [(Network.gen_rand_params(network.n_inputs, network.n_outputs, n_hidden, network.p_variance), i, 0)
                  for i in range(pop_size)]
    for g in range(n_generations):
        for gene in population:
            params, i, _ = gene

            network.set_params(params)
            curr_fitness = network.evaluate(n_episodes)

            # Save calculated fitness
            population[i] = (params, i, curr_fitness)
        # Sort by fitness
        population.sort(key=lambda x: x[2], reverse=True)

        # Replace worst with the best + mutation
        population = population[:pop_size // 2]
        new_population = []
        for gene in population:
            (w1_noise, b1_noise), (w2_noise, b2_noise) = Network.gen_rand_params(network.n_inputs, network.n_outputs,
                                                                                 n_hidden, mut_range)
            (w1, b1), (w2, b2) = gene[0]
            new_gene = ([(w1 + w1_noise, b1 + b1_noise), (w2 + w2_noise, b2 + b2_noise)], gene[1], gene[2])
            new_population.append(new_gene)

        population.extend(new_population)
        print(f"Generation: {g}, "
              f"Max fitness: {max([x[2][0] if isinstance(x[2], numpy.ndarray) else x[2] for x in population])}, "
              f"Mean: {mean([x[2][0] if isinstance(x[2], numpy.ndarray) else x[2] for x in population])}")
        print("\n")

        if max(population, key=lambda x: x[2])[2] >= threshold:
            break

    # Render the best one
    params, i, _ = population[0]
    curr_fitness = network.evaluate(n_episodes, render=True)
    env.close()
