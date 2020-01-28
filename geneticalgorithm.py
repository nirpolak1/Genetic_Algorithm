# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 20:32:03 2020

@author: nir
"""

import numpy as np
import cv2


###Problem 2
def cost_function_image(parameters):
    image = cv2.imread('64652.jpg',0).flatten()

    result = np.sum(abs(parameters - image))

    return result

def fitness_calculator(population):
    # calulates the result of a linear problem (for now...)
    fitness_values = np.zeros(population.shape[0])
    for i in range (population.shape[0]):
        fitness_values[i] = 1/(cost_function_image(population[i,:]))
    return fitness_values

###Problem 1
def cost_function_1(x,y):
    result = ((y*np.sin(x+y))+(x*np.cos(y)))/(20+np.power(x,2))
    return result

def fitness_calculator_1(population):
    # calulates the result of a linear problem (for now...)
    fitness_values = np.zeros(population.shape[0])
    for i in range (population.shape[0]):
        x=population[i,0]
        y=population[i,1]
        fitness_values[i] = cost_function_1(x,y)
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


def sus_selection(population, fitness_values,number_of_parents):
    #creating an empty array to hold the indexes of selected parents among the population
    parents_index = np.empty(number_of_parents)
    #making sure all fitness values for roulette sections are positive
    fitness_values = fitness_values - np.min(fitness_values)
    sum_of_fitness = np.sum(fitness_values)
    pointer_step = 1/number_of_parents
    pointer = np.random.uniform(0.0,1.0)%pointer_step
    roulette_i = 0.0
    selection_index = 0
    #selecting the individuals where a roulette pointer lies and recording their index:
    for i in range(0,fitness_values.shape[0]):
        roulette_f = roulette_i + fitness_values[i]/sum_of_fitness
        if (roulette_i <= pointer < roulette_f):
            parents_index[selection_index] = np.uint8(i)
            selection_index += 1
            pointer += pointer_step
        roulette_i = roulette_f
    
    return parents_index

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

def sus_crossover(parents_index, population, number_of_offsprings):
    offsprings = np.empty(number_of_offsprings)
    for index in range(number_of_offsprings[0]):
        #selecting the parents by order of fitness:
        first_parent = np.uint8(parents_index[index])
        second_parent = np.uint8(parents_index[(index+1)%parents_index.shape[0]])
        #randomly choosing which parameters would be taken from the first or second parent:
        crossover_list = np.random.randint(0,2,number_of_offsprings[1])
        #creating the offspring:
        offsprings[index,:] = population[first_parent,:]*crossover_list + population[second_parent,:]*(1-crossover_list)
    return offsprings

def mutate(offsprings,parameters_range):
    #mutating a random parameter in each offspring:
    for index in range(offsprings.shape[0]):
        mutation_range = np.random.uniform(-1.0,1.0,1)
        mutation_magnitude = 1.0
        random_gene = np.random.randint(0,offsprings.shape[1],1)
        mutation_size = mutation_range*mutation_magnitude*np.min([offsprings[index,random_gene]-parameters_range[0,random_gene],parameters_range[1,random_gene]-offsprings[index,random_gene]])
        offsprings[index,random_gene] = offsprings[index,random_gene] + mutation_size
    return offsprings

def mutation_rating(fitness_values,base_rate):
    top_range = np.max(fitness_values) - np.mean(fitness_values)
    bottom_range = np.mean(fitness_values) - np.min(fitness_values)
    mutation_rate = base_rate*np.amax([np.amin([top_range/bottom_range,10.0]),0.1])
    return mutation_rate
    
def sus_mutate(offsprings,parameters_range,rate):
    #mutating a random parameter in each offspring:
    for index in range(offsprings.shape[0]):
        mutation_vector = np.random.uniform(0.0,1.0,offsprings.shape[1])
        mutation_vector = np.where(mutation_vector <= rate, 1 ,0)
        offsprings[index,:] = mutation_vector*np.random.uniform(parameters_range[0,0],parameters_range[1,0],offsprings.shape[1]) + (1-mutation_vector)*offsprings[index,:]
    return offsprings

def sus_insertion(offsprings, population, number_of_parents, fitness_values):
    parents = np.empty((number_of_parents, population.shape[1]))
    #selecting the parents by decreasing fitness values:
    for parent_index in range(number_of_parents):
        max_fitness_index = np.where(fitness_values == np.max(fitness_values))
        max_fitness_index = max_fitness_index[0][0]
        parents[parent_index, :] = population[max_fitness_index,:]
        #erasing the best current fitness to move to the next best fitness:
        fitness_values[max_fitness_index] = -999999999
    population[0:number_of_parents,:] = parents
    population[number_of_parents:,:] = offsprings
    return population