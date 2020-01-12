# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:32:03 2020

@author: nir
"""

import numpy as np

def fitness_calculator(cost_function, parameters):
    # calulates the result of a linear problem (for now...)
    fitness_values = np.sum(parameters*cost_function, axis=1)
    return fitness_values

def select_parents(population, fitness_values, number_of_parents):
    #creating an empty array to hold the parameters of the selected parents to mate:
    parents = np.empty((number_of_parents, population.shape[1]))
    #selecting the parents by decreasing fitness values:
    for parent_index in range(number_of_parents):
        max_fitness_index = np.where(fitness_values == np.max(fitness_values))
        max_fitness_index = max_fitness_index[0][0]
        parents[parent_index, :] = population[max_fitness_index,:]
        #erasing the best current fitness to move to the next best fitness:
        fitness_values[max_fitness_index] = -999999999
    return parents

def crossover_mating(parents, number_of_offsprings):
    # completeing the population size with new sets
    # the sets are combinations of the parents sets
    # each parent contributes half a set (for now...)
    
    #creatubg an empty array to hold the parameters of the new offsprings:
    offsprings = np.empty(number_of_offsprings)
    #choosing the point of crossover 
    crossover_at = np.uint8(number_of_offsprings[1]/2)
    
    for index in range(number_of_offsprings[0]):
        #selecting the parents by order of fitness:
        first_parent = index%parents.shape[0]
        second_parent = (index+1)%parents.shape[0]
        #creating the offspring by dividing the parents at the crossover point:
        offsprings[index, 0:crossover_at] = parents[first_parent, 0:crossover_at]
        offsprings[index, crossover_at:] = parents[second_parent, crossover_at:]
    return offsprings

def mutate(offsprings):
    #mutating a random parameter in each offspring:
    for index in range(offsprings.shape[0]):
        mutation_size = np.random.uniform(-1.0,1.0,1)
        random_gene = np.random.randint(0,offsprings.shape[1],1)
        offsprings[index,random_gene] = offsprings[index,random_gene] + mutation_size
    return offsprings


