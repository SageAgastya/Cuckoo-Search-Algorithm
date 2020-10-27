#18921
from Functions import FitnessFn, LevyFlight, Step, Abandon_with_Pa
import numpy as np
import random, copy, json

class CuckooSearch:

    def __init__(self):
        loader = json.load(open("config.json", "r"))
        self.P_a = loader["P_a"]
        self.beta = loader["beta"]
        self.alpha = loader["alpha"]
        self.lb = loader["lb"]
        self.ub = loader["ub"]
        self.dim = loader["dim"]
        self.nests = loader["nests"]
        self.max_generations = loader["max_generations"]
        self.matrix = self.lb + np.random.rand(self.nests, self.dim) * (self.ub - self.lb)
        print(self.matrix)

    def InitializeCandidateSoln(self):
        matrix = self.matrix.T
        x1, x2 = matrix[0], matrix[1]
        fitness_value = FitnessFn(x1, x2)
        matrix = np.concatenate([matrix, [fitness_value]], axis=0)
        self.matrix = np.concatenate([self.matrix, np.array([fitness_value]).T], axis=-1)
        return self.matrix, matrix          # matrix in shape(nest,dim), matrix in shape(dim, nest)

    def Search(self):
        best_soln = None
        mat_nest_dim, mat_dim_nest = self.InitializeCandidateSoln()
        nest_indices = np.array(range(self.nests))
        for i in range(self.max_generations):
            best_index = np.argmin(mat_dim_nest[-1])
            best_soln = mat_nest_dim[best_index]
            best_soln, best_fitness_val = best_soln[:-1], best_soln[-1]

            # Now, choose a random nest one by one (giving opportunity to all nests) and update using Levy-Flight
            new_cuckoo_pos = np.array([np.clip(LevyFlight(nest[:-1], self.alpha, Step(self.beta), best_soln), a_min=self.lb, a_max=self.ub) for nest in mat_nest_dim])
            new_fitness_val = [FitnessFn(new_pos[0], new_pos[1]) for new_pos in new_cuckoo_pos]

            np.random.shuffle(nest_indices)

            # updating the pre-existing solution with new solution if new solution is better than pre-existing solution
            for index, new_pos, new_fitness in zip(nest_indices, new_cuckoo_pos, new_fitness_val):
                old_fitness = mat_nest_dim[index][-1]
                if(new_fitness < old_fitness):       # greedy approach
                    mat_nest_dim[index] = np.concatenate([new_pos, [new_fitness]], axis=0)

            # abandon the nests with P_a probability and build newer ones
            # initilaize random-fraction, and keep the original solution if fraction > P_a, else perform X_new = X + rand*(X[d1] - X[d2])
            # here X_d1 means d1-th row of matrix the value of d1 and d2 is random integer.

            new_matrix = copy.deepcopy(mat_nest_dim)
            for index, nest in enumerate(mat_nest_dim):
                new_matrix = Abandon_with_Pa(index=index, nest=nest, P_a=self.P_a, new_matrix=new_matrix, old_matrix=mat_nest_dim)
            mat_nest_dim = new_matrix
        return mat_nest_dim, best_soln

a = CuckooSearch()
print(a.Search())