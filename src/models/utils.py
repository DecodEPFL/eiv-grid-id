import cvxpy

DEFAULT_SOLVER = 'default'


def _solve_problem_with_solver(problem: cvxpy.Problem, solver, verbose: bool):
    if solver == DEFAULT_SOLVER:
        problem.solve(verbose=verbose)
    else:
        problem.solve(solver=solver, verbose=verbose)
