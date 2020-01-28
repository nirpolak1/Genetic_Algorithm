# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 22:18:33 2020

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
import geneticalgorithm as ga

#defining a cost function:
number_of_parameters = 12604

#defining mutation base rate:
base_rate=0.005

#defining the size of the population:
number_of_individuals = 100
number_of_parents = int(number_of_individuals/2)
population_size = (number_of_individuals, number_of_parameters)

#defining the number of generations:
generations = 1500
#defining range:
lower_bound = 0
upper_bound = 255
parameters_range = np.zeros((2,number_of_parameters))
parameters_range[:][0] = lower_bound
parameters_range[:][1] = upper_bound


b_fig = plt.figure(figsize=(15,10))
progress_max = np.zeros(generations)
progress_mean = np.zeros(generations)
progress_min = np.zeros(generations)
progress_generations = np.linspace(0,generations-1,generations)
progress_fig = b_fig.add_subplot(1,1,1)


#creating the first generation population:
population = np.random.uniform(lower_bound,upper_bound,population_size)


for generation in range(generations):
    print("Generation #", generation)
    #calculating the fitness value of each individual:
    fitness_values = ga.fitness_calculator(population)
    progress_max[generation] = np.max(fitness_values)
    progress_mean[generation] = np.mean(fitness_values)
    progress_min[generation] = np.min(fitness_values)
    #selecting the parents:
    parents_index = ga.sus_selection(population, fitness_values,number_of_parents)
    #creating offsprings:
    offsprings = ga.sus_crossover(parents_index, population, ((number_of_individuals-number_of_parents),(number_of_parameters)))
    #mutating offsprings:
    rate = ga.mutation_rating(fitness_values,base_rate)
    offsprings_mutated = ga.sus_mutate(offsprings,parameters_range,rate)
    population = ga.sus_insertion(offsprings, population, number_of_parents, fitness_values)
    
    #show the best result:
    print("Best Fitness:", np.max(ga.fitness_calculator(population)))

progress_fig.fill_between(progress_generations,progress_max,progress_min,edgecolor='black',facecolor='grey')
progress_fig.plot(progress_generations, progress_mean,c='black')
progress_fig.plot(progress_generations, progress_max,c='r',linewidth=2)
plt.show()

#show the best set and best result:
fitness_values = ga.fitness_calculator(population)
best_fitness_index = np.where(fitness_values == np.max(fitness_values))
print("Best Solution:", population[best_fitness_index,:])
print("Best Solution Fitness:", fitness_values[best_fitness_index])

solution_image = np.reshape(np.uint8(population[best_fitness_index,:]),(137,92))
plt.imshow(solution_image, cmap = 'gray')
plt.show()
