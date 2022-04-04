#!/usr/bin/env python
# coding: utf-8

# ### Genetic Algorithm on  equation x^3 - 2x^2 + 5x -6

# In[21]:


import random
random.seed(100)

def selection(population, fit_chromosomes):
    fitnessscore=[]
    for x in population:
        individual_fitness= x*x*x - 2*x**2 + 5*x -6
        fitnessscore.append(individual_fitness)

    print('Fitness score:', fitnessscore)
    total_fitness=sum(fitnessscore)
    print('Totalfitness:', total_fitness)
    score_card=list(zip(fitnessscore,population))
    print('Score card:', score_card)
    score_card.sort(reverse=True)
    print('Score card sorted:', score_card)

    for individual in score_card:
        if individual[0]>11000:
            if individual[1] not in fit_chromosomes:
                fit_chromosomes.append(individual[1])
    #print(fit_chromosomes)
    print("we got the maximum value at {}".format(fit_chromosomes))
    
    score_card=score_card[:4]
    score, population=zip(*score_card)
    return list(population)

fit_chromosomes=[]
population=[random.randint(0,31) for i in range(4) ]
print("Initial Populataion : ",population)
population = selection(population, fit_chromosomes)
print("current Populataion : ",population)


# In[ ]:




