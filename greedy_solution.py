from problem import CoilSlitting
from data import load_data
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch

if __name__ == "__main__":
    single_sheet = load_data("./data.csv")[0]

    problem = CoilSlitting(
        max_rectangle_size=-1,
        min_rectangle_size=-1,
        sheet_width=100,
        sheet_height=100,
        sensors_sheet=single_sheet,
    )
    algorithm = PatternSearch()
    res = minimize(
        problem, algorithm, verbose=False, seed=1)
    print(res.X)