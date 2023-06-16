import utility_functions
import numpy as np
from ABC_algorithm_SVM import ArtificialBeeColony


def rosenbrock(vector, a=1, b=100):
    """f(x, y) = (a-x)^2 + b(y-x^2)^2"""

    vector = np.array(vector)

    return (a - vector[0])**2 + b * (vector[1] - vector[0]**2)**2

# test case
def test_case1():

    # create model
    ndim = int(5)
    model = ArtificialBeeColony(l_bound=[0]*ndim,
                                u_bound=[10]*ndim,
                                function=rosenbrock,
                                hive_size=10,
                                max_iterations=50,
                                seed=144)

    # run model
    cost = model.run()

    utility_functions.ConvergencePlot(cost)

    # print best solution
    print("Fitness value ABC algorithm: {0}".format(model.best_place))
    print("Solution vector ABC algorithm: {0}".format(model.solution))

# if __name__ == "__main__":
#     test_case1()

def rastrigin(vector):
    """                     n
            f(x) = 10*n + Sigma { x_i^2 - 10*cos(2*PI*x_i) }
                           i=1
                           with xi within [-5.12:5.12]

    """

    vector = np.array(vector)

    return 10 * vector.size + sum(vector * vector - 10 * np.cos(2 * np.pi * vector))

def radial_basis_function(vector):
    # radial basis function
    return abs(vector)

def test_case2():

    # create model
    ndim = int(5)
    model = ArtificialBeeColony(l_bound=[-5.12]*ndim,
                                u_bound=[5.12]*ndim,
                                function=rastrigin,
                                hive_size=50,
                                max_iterations=100,
                                seed=144)

    # run model
    cost = model.run()

    utility_functions.ConvergencePlot(cost)

    # print best solution
    print("Fitness value ABC algorithm: {0}".format(model.best_place))
    print("Solution vector ABC algorithm: {0}".format(model.solution))

# if __name__ == "__main__":
#     test_case2()




