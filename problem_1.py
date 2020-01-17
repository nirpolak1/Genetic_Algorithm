# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 22:18:33 2020

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
import geneticalgorithm as ga

#defining a cost function:
number_of_parameters = 2

#defining the size of the population:
number_of_individuals = 5
number_of_parents = int(number_of_individuals/2)
population_size = (number_of_individuals, number_of_parameters)

#defining the number of generations:
generations = 20

#defining range:
lower_bound = -5.0
upper_bound = 5.0
parameters_range = np.zeros((2,number_of_parameters))
parameters_range[:][0] = lower_bound
parameters_range[:][1] = upper_bound

#plotting the function space:
x = np.outer(np.linspace(lower_bound,upper_bound,50),np.ones(50))
y = x.copy().T
z = ga.cost_function(x,y)

a_fig = plt.figure(figsize=(15,10))
search_space = a_fig.add_subplot(1,1,1)
search_space = plt.axes(projection='3d')
search_space.plot_surface(x,y,z,cmap='afmhot',edgecolor='none', alpha =0.5)
search_space.set_title('Cost Function Search Space')

b_fig = plt.figure(figsize=(15,10))
progress_max = np.zeros(generations)
progress_mean = np.zeros(generations)
progress_min = np.zeros(generations)
progress_generations = np.linspace(0,generations-1,generations)
progress_fig = b_fig.add_subplot(1,1,1)

#plt.show()


#creating the first generation population:
population = np.random.uniform(lower_bound,upper_bound,population_size)
#showing first generation in red
search_space.scatter(population[:,0],population[:,1],ga.fitness_calculator(population),marker='o',alpha = 1, c='r')


for generation in range(generations):
    print("Generation #", generation)
    #calculating the fitness value of each individual:
    fitness_values = ga.fitness_calculator(population)
    progress_max[generation] = np.max(fitness_values)
    progress_mean[generation] = np.mean(fitness_values)
    progress_min[generation] = np.min(fitness_values)
    #showing each generation in blue, less faded with each generation
    search_space.scatter(population[:,0],population[:,1],fitness_values,marker='o',alpha = generation/generations, c='b')
    #selecting the parents:
    parents = ga.select_parents(population, fitness_values, number_of_parents)
    #creating offsprings:
    offsprings = ga.crossover_mating(parents,((number_of_individuals-number_of_parents),(number_of_parameters)))
    #mutating offsprings:
    offsprings_mutated = ga.mutate(offsprings,parameters_range)
    #assembling the new population:
    population[0:parents.shape[0],:] = parents
    population[parents.shape[0]:,:] = offsprings_mutated
    
    #show the best result:
    print("Best Fitness:", np.max(ga.fitness_calculator(population)))

progress_fig.fill_between(progress_generations,progress_max,progress_min,edgecolor='black',facecolor='grey')
progress_fig.plot(progress_generations, progress_mean,c='black')
progress_fig.plot(progress_generations, progress_max,c='r',linewidth=2)
a_fig.show()
#show the best set and best result:
fitness_values = ga.fitness_calculator(population)
best_fitness_index = np.where(fitness_values == np.max(fitness_values))
print("Best Solution:", population[best_fitness_index,:])
print("Best Solution Fitness:", fitness_values[best_fitness_index])