import random
import sys
import copy
import numpy as np
import time
import pipeline_SVM

# ---- Important information ---- #

__doc__ = """

This code is based on the Hive algorithm described by Romain Wuilbercq (2017)
but is adjusted for the purposes of the thesis project. 

More specifically, adjustments have been made to link this algorithm to the 
parameter search algorithms for finding the optimal set of parameters for various models
like SVM. 

Moreover, the algorithm described below diverges slightly from the classical ABC algorithm. 
That is, it has been optimized by updating three dimensions per iteration, instead of one. 
This is based on the findings of Alizadegan et al. (2013). 
A link can be found here: https://doi.org/10.1016/j.amc.2013.09.012


The inspirational Hive code can be found here: https://github.com/rwuilbercq/Hive

"""


# single bee
class ArtificialBee(object):
    """
            Create artificial bee object.

    """

    def __init__(self, l_bound, u_bound, function):

        """
                Random instantiation of artificial bee object.

                :param l_bound:     lower bound
                :param u_bound:     upper bound
                :param function:    optimization function

        """

        # random solution vector
        self._random(l_bound, u_bound)

        t = time.process_time()

        # compute fitness
        if (function != None and isinstance(function, pipeline_SVM.OptRFParameters)):
            self.fitness_val = function.objective_function_value(self.sol_vector)[0][0]
        elif (function != None and not isinstance(function, pipeline_SVM.OptRFParameters)):
            self.fitness_val = function(self.sol_vector)
        else:
            self.fitness_val = sys.float_info.max

        self._fitness()

        elapsed_t = time.process_time() - t
        print(f"Elapsed time computing fitness: {elapsed_t} sec")

        # trial limit counter initialization
        self.trial_counter = 0

    def _random(self, l_bound, u_bound):
        """
                Random initialization of solution vector.

                :param l_bound:     lower bound
                :param u_bound:     upper bound

        """

        self.sol_vector = np.array(l_bound) + random.random() * (np.array(u_bound) - np.array(l_bound))

    def _fitness(self):
        """
                Evaluate fitness of solution vector.

        """

        if (self.fitness_val >= 0):
            self.fitness = 1 / (1 + self.fitness_val)
        else:
            self.fitness = 1 + abs(self.fitness_val)


class ArtificialBeeColony(object):
    """
            Create ABC algorithm.
    """

    def run(self):
        """
                Run the ABC algorithm.

                :return:    the cost

        """

        cost = {};
        cost['best'] = [];
        cost['mean'] = []

        t = time.process_time()

        for iteration in range(self.max_iterations):

            t = time.process_time()
            # employees
            for j in range(self.nr_emp):
                self.send_emp(j)

            elapsed_t = time.process_time() - t
            print(f"Elapsed time employees: {elapsed_t} sec")

            t = time.process_time()
            # onlookers
            self.send_onl()

            elapsed_t = time.process_time() - t
            print(f"Elapsed time onlookers: {elapsed_t} sec")

            t = time.process_time()
            # scouts
            self.send_sct()

            elapsed_t = time.process_time() - t
            print(f"Elapsed time scouts: {elapsed_t} sec")

            t = time.process_time()
            # find best path
            self.find_best_bee()

            elapsed_t = time.process_time() - t
            print(f"Elapsed time best bee: {elapsed_t} sec")

            # store info of convergence
            cost['best'].append(self.best_place)
            cost['mean'].append(sum([b.fitness_val for b in self.colony]) / self.nr_emp)

            # print info if verbose=True
            if self.verbose:
                self._verbose(iteration, cost)

        elapsed_t = time.process_time() - t
        print(f"Elapsed time ABC iteration for-loop: {elapsed_t} sec")

        return cost

    def __init__(self,
                 l_bound,
                 u_bound,
                 function,
                 hive_size=30,
                 max_iterations=100,
                 max_trials=None,
                 custom_function=None,
                 seed=None,
                 verbose=False,
                 additional_pars=None,
                 ):

        """
                Instantiate bee hive object.

        """

        # check input
        assert (len(u_bound) == len(l_bound)), "upper and lower bound must be a list of the same length."

        # generate seed for RNG
        if (seed == None):
            self.seed = random.randint(0, 1000)
        else:
            self.seed = seed
        random.seed(self.seed)

        # compute nr of employees
        self.nr_emp = int((hive_size + hive_size % 2))  # classic 1:1 ratio

        # assign algo properties
        self.dim = len(l_bound)
        self.max_iterations = max_iterations
        if (max_trials == None):
            self.max_trials = 0.6 * self.nr_emp * self.dim
        else:
            self.max_trials = max_trials

        self.custom_function = custom_function
        self.additional_pars = additional_pars

        # assign optimization problem props
        self.evaluate = function
        self.l_bound = l_bound
        self.u_bound = u_bound

        # initialize current best as solution vector
        self.best_place = sys.float_info.max
        self.solution = None

        # create hive
        self.colony = [ArtificialBee(l_bound, u_bound, function) for i in range(self.nr_emp)]

        # initialize best solution vector
        self.find_best_bee()

        # compute proba
        self.comp_prob()

        # verbose
        self.verbose = verbose

    def find_best_bee(self):
        """
                Find current best.

        """

        vals = [b.fitness_val for b in self.colony]

        i = vals.index(min(vals))

        if (vals[i] < self.best_place):
            self.best_place = vals[i]
            self.solution = self.colony[i].sol_vector

    def comp_prob(self):
        """
                Compute relative chance a given solution is chosen by another
                onlooker bee in 4th phase (employees back in hive).

                :return:     probability intervals
        """

        # retrieve fitness in-hive bees
        vals = [b.fitness for b in self.colony]
        max_vals = max(vals)

        # compute proba similar to classic ABC by Karaboga
        if (self.custom_function == None):
            self.probs = 0.9 * np.array(vals) / max_vals + 0.1

        else:
            if (self.extra_params != None):
                self.probs = self.custom_function(list(vals), **self.additional_pars)
            else:
                self.probs = self.custom_function(vals)

        # return prob intervals
        return [sum(self.probs[:i + 1]) for i in range(self.nr_emp)]

    def send_emp(self, i):
        """
                2nd phase: new candidate solutions produced and
                solution is updated if applicable.
        """

        # deepcopy current bee solution vector
        clonebee = copy.deepcopy(self.colony[i])

        # draw dimension
        d = random.randint(0, self.dim - 1)  # classic

        if (self.dim > 2):
            d2 = random.randint(0, self.dim - 1)  # add second dim (optim ABC) Alizadegan et al. (2013)
            d3 = random.randint(0, self.dim - 1)  # add third dim (optim ABC) Alizadegan et al. (2013)

        # select other bee
        b_ix = i;

        while (b_ix == i):
            b_ix = random.randint(0, self.nr_emp - 1)

        # produce mutant based on current bee and its friend
        clonebee.sol_vector[d] = self._mutate(d, i, b_ix)  # classic

        if (self.dim > 2):
            clonebee.sol_vector[d2] = self._mutate(d2, i, b_ix)  # optim second dim Alizadegan et al. (2013)
            clonebee.sol_vector[d3] = self._mutate(d3, i, b_ix)  # optim third dim Alizadegan et al. (2013)

        # check boundaries
        clonebee.sol_vector = self._check(clonebee.sol_vector, dim=d)

        # compute fitness
        clonebee.val = self.evaluate(clonebee.sol_vector)
        clonebee._fitness()

        # crowding (deterministic)
        if (clonebee.fitness > self.colony[i].fitness):
            self.colony[i] = copy.deepcopy(clonebee)
            self.colony[i].trial_counter = 0
        else:
            self.colony[i].trial_counter += 1

    def send_onl(self):
        """
                Locally improve solution path.
        """

        # send onlookers
        n_onl = 0;
        beta = 0
        while (n_onl < self.nr_emp):
            # draw random nr [0-1]
            phi = random.random()

            # increment roulette wheel par beta
            beta += phi * max(self.probs)
            beta %= max(self.probs)

            # select new onlooker based on phase 4
            i = self.recruit(beta)

            # send new onlooker
            self.send_emp(i)

            # increment nr onlookers
            n_onl += 1

    def recruit(self, beta):
        """
                Recruit onlooker bees using roulette wheel selection.

                :param beta:    value to compare probability with
                :return:        new potential onlooker
        """

        # (re)compute proba intervals after each onlooker
        probs = self.comp_prob()

        # select new potential onlooker
        for i in range(self.nr_emp):
            if (beta < probs[i]):
                return i

    def send_sct(self):
        """
                Abandon bees exceeding trials limit.
        """

        # retrieve nr trials for all bees
        trials = [self.colony[i].trial_counter for i in range(self.nr_emp)]

        # identify bee with max nr trials
        i = trials.index(max(trials))

        # check if max nr trials exceeds pre-set max
        if (trials[i] > self.max_trials):
            # create new scout bee (at random)
            self.colony[i] = ArtificialBee(self.l_bound, self.u_bound, self.evaluate)

            # send scout to exploit solution vector
            self.send_emp(i)

    def _mutate(self, dim, index_current, index_other):
        """
                Mutate given solution vector.

                :param dim:             dimension
                :param index_current:   index value of current bee
                :param index_other:     index value of other bee
                :return:                 mutated solution vector
        """

        return self.colony[index_current].sol_vector[dim] + \
               (random.random() - 0.5) * 2 * \
               (self.colony[index_current].sol_vector[dim] -
                self.colony[index_other].sol_vector[dim])

    def _check(self, vec, dim=None):
        """
                Check if solution vector is contained within pre-set bounds.

                :param vec:     solution vector
                :param dim:     dimension
                :return:         solution vector
        """

        if (dim == None):
            range_ = range(self.dim)
        else:
            range_ = [dim]

        for r in range_:

            # check lb
            if (vec[r] < self.l_bound[r]):
                vec[r] = self.l_bound[r]

            # check ub
            elif (vec[r] > self.u_bound[r]):
                vec[r] = self.u_bound[r]

        return vec

    def _verbose(self, iteration, cost):
        """
                Display info.

                :param iteration:   iteration number
                :param cost:        cost in current iteration
        """

        message = "Iter nr. = {} | Best evaluation value = {} | Mean evaluation value = {} "
        print(message.format(int(iteration), cost['best'][iteration], cost['mean'][iteration]))














