# Tested on all Benchmark functions, the current fitness function itself is a benchmark function.

import random
import numpy as np
import math

# Candidate Function/Objective Function
def FitnessFn(x1, x2):
    # return (x1**2-x2**2)*np.sin(x1+x2)/(x1**2+x2**2)
    # return x1**2 - x1*x2 + x2**2 + 2*x1 + 4*x2 + 3
    return x1**2 + x2**2
    
#Updation of the solution based on the concept of Levy Flight Apprach.
def LevyFlight(X_old, alpha, step, best):
    X_new = X_old + alpha*step*(X_old - best)
    return X_new

#Determining the step for the Cuckoo which determines how far or closer the next solution would be determined in the search flight.
def Step(beta):
    sigma_u = ((math.gamma(1 + beta) * math.sin(math.pi * beta/2.0)) / (math.gamma((1 + beta)/2.0) * beta * 2**((beta-1)/2)))**(1.0/beta)
    u = np.random.normal(0, sigma_u, size = 2)
    v = np.random.normal(0, 1, size = 2)
    return u/np.power(np.fabs(v), (1.0/beta))


#Depending on the probability of the abandonment (Pa), the newer solutions are computed.
def Abandon_with_Pa(index, nest, P_a, new_matrix, old_matrix): # matrix = nests
    random_soln = np.random.rand(2, )
    d1 = random.randint(0, 4)
    d2 = random.randint(0, 4)
    if (random_soln[0] < P_a):
        rand = random.random()
        new_matrix[index][0] = nest[0] + rand*(old_matrix[d1][0] - old_matrix[d2][0])

    if (random_soln[1] < P_a):
        rand = random.random()
        new_matrix[index][1] = nest[1] + rand*(old_matrix[d1][1] - old_matrix[d2][1])

    new_matrix[index][-1] = FitnessFn(nest[0], nest[1])

    old_fitness_value = old_matrix[index][-1]
    new_fitness_value = new_matrix[index][-1]

    if(new_fitness_value > old_fitness_value):
        new_matrix[index] = old_matrix[index]
    return new_matrix

