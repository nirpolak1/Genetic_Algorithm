# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 22:18:33 2020

@author: user
"""

import numpy as np
import geneticalgorithm as ga

#defining a linear cost function:
number_of_weights = 8
cost_function_weights = [-5,-4,2.5,7,-1.3,5.7,9.2,-4.3]

#defining the size of the population:
number_of_individuals = 20
number_of_parents = int(number_of_individuals/2)
population_size = (number_of_individuals, number_of_weights)

#defining the number of generations:
generations = 100

#creating the first generation population:
population = np.random.uniform(-4.0,4.0,population_size)
print(population)

for generation in range(generations):
    print("Generation #", generation)
    #calculating the fitness value of each individual:
    fitness_values = ga.fitness_calculator(cost_function_weights, population)
    #selecting the parents:
    parents = ga.select_parents(population, fitness_values, number_of_parents)
    #creating offsprings:
    offsprings = ga.crossover_mating(parents,((number_of_individuals-number_of_parents),(number_of_weights)))
    #mutating offsprings:
    offsprings_mutated = ga.mutate(offsprings)
    
    #assembling the new population:
    population[0:parents.shape[0],:] = parents
    population[parents.shape[0]:,:] = offsprings_mutated
    
    #show the best result:
    print("Best Fitness:", np.max(np.sum(population*cost_function_weights, axis=1)))
    
#show the best set and best result:
fitness_values = ga.fitness_calculator(cost_function_weights, population)
best_fitness_index = np.where(fitness_values == np.max(fitness_values))
print("Best Solution:", population[best_fitness_index,:])
print("Best Solution Fitness:", fitness_values[best_fitness_index])