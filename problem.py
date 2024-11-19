from copy import deepcopy
from pymoo.problems.multi import ElementwiseProblem
from pymoo.core.sampling import Sampling
# from pymoo.core.crossover import Crossover
from tournament import binary_tournament
from crossover_override import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.operators.selection.tournament import TournamentSelection
from simple_cut import Sheet, Cut
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.visualization.scatter import Scatter

from pymoo.optimize import minimize
from data import load_data, display_sheet
import numpy as np

class Problem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=2, n_constr=0, type=Sheet)

    def _evaluate(self, sheet ,out, *args, **kwargs):
        # print(sheet[0])
        f1 = sheet[0].average_weighted_worst_percentile()
        f2 = sheet[0].average_rectangle_size()

        out["F"] = np.column_stack([f1, -f2])


def get_random_sheet()->Sheet:
    sh = Sheet(
        sheet=load_data("data.csv")[0]
    )

    for i in range(2,10,2):
        if np.random.random() < 0.5:
            sh.add_cut(Cut(i, -1))

    part = int(0.2*sh.sheet.shape[1])

    for i in range(part,sh.sheet.shape[1],part):
        if np.random.random() < 0.5:
            sh.add_cut(Cut(-1, i))       
    
    return sh



class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        a = a.x[0]
        b = b.x[0]
        for i in range(len(a.cuts)):
            if len(a.cuts) != len(b.cuts):
                return False
            if a.cuts[i].x != b.cuts[i].x or a.cuts[i].y != b.cuts[i].y:
                return False
        return True

class RandomSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, 1), None, dtype=Sheet)

        for i in range(n_samples):
            X[i, 0] = get_random_sheet()

        return X

class MyCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        # The input of has the following shape (n_parents, n_matings, n_var)
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]
            off_a = Sheet(a.get_sheet())
            off_b = Sheet(a.get_sheet())
            off_a.cuts = a.cuts[:len(a.cuts)//2] + b.cuts[len(b.cuts)//2:]
            off_b.cuts = a.cuts[len(a.cuts)//2:] + b.cuts[:len(b.cuts)//2]


            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = off_a, off_b
        # print("crossover")
        # print(Y[:10])
        return deepcopy(Y)

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):
        # print("before mutation")
        # print(X)
        # for each individual
        for i in range(len(X)):
            # print(X[i, 0])
            r = np.random.random()

            # with a probabilty of 40% - change the order of characters
            if r < 0.7:
                
                cuts_len = len(X[i, 0].cuts)
                if cuts_len > 2:
                    idx = np.random.randint(0, cuts_len-1)
                    if X[i, 0].cuts[idx].x + 1 < X[i, 0].sheet.shape[0]:
                        X[i, 0].cuts[idx].x = X[i, 0].cuts[idx].x + 1

            # also with a probabilty of 40% - change a character randomly
            elif r < 0.8:
                cuts_len = len(X[i, 0].cuts)
                if cuts_len > 2:
                    idx = np.random.randint(0, cuts_len-1)
                    if X[i, 0].cuts[idx].y + 1 < X[i, 0].sheet.shape[1]:
                        X[i, 0].cuts[idx].y = X[i, 0].cuts[idx].y + 1
        # print("mutation")
        # print(X[:10])
        return deepcopy(X)
    

POPULATION_SIZE = 500
GENERATIONS = 100

if __name__ == "__main__":
    sheet = load_data("data.csv")[0]
    s = Sheet(sheet)
    problem = Problem()
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        # eliminate_duplicates=MyDuplicateElimination(),
        selection=TournamentSelection(func_comp=binary_tournament),
        eliminate_duplicates=NoDuplicateElimination(),
        sampling=RandomSampling(),
        crossover=MyCrossover(),
        mutation=MyMutation(),
    )
    res = minimize(
        problem, algorithm, ("n_gen", GENERATIONS), seed=0xC0FFEE, verbose=True
    )

    # print(res.X.shape)
    # x1, x2 = res.X[0][0], res.X[1][0]
    # print(x1, x2)
    # print(res.X)
    # print(res.F)
    Scatter().add(res.F).show()
    print(res.X[0][0])
    sss = res.X[0][0]
    sss.add_cuts_to_sheet()
    display_sheet(sss.sheet)
    print(sss.cuts)