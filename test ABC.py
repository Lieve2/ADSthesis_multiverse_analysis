import utility_functions
from ABC_algorithm_SVM import ArtificialBeeColony
from utility_functions import rosenbrock, rastrigin


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
