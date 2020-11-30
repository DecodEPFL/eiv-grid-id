import cvxpy

DEFAULT_SOLVER = 'default'


def _solve_problem_with_solver(problem: cvxpy.Problem, solver, verbose: bool, warm_start: bool = False):
    if solver == DEFAULT_SOLVER:
        problem.solve(verbose=verbose, warm_start=warm_start, qcp=True)
    else:
        problem.solve(solver=solver, verbose=verbose, warm_start=warm_start, qcp=True)
