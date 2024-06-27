"""
This module contains the implementation of the myopic knapsack policy for project selection using a small weighted cost to select the cheapest optimal solution.

The Myopic Knapsack policy uses the PuLP package with CBC solver to pick the optimal set of projects. It considers the current state of the simulation - including the current year, budgets, and available projects - to make the selection. The policy aims to maximize the value of selected projects while minimizing the cost.

The module provides a function `pulp_solver` that takes the current state and the total number of years as input and returns a list of 1's and 0's representing which project has been picked for this run from the available projects list.

Example Usage:
--------------
state = {
    'current_year': 0,
    'budgets': [100, 23, 100],
    'available_projects':[ ['Project A', 10, 5, 8, 12], ['Project B', 15, 6, 10, 14], ['Project C', 20, 8, 15, 18]]
    }

total_years = 3
    
project_selection = pulp_solver(state, total_years)

print(project_selection)  # [1, 0, 1]

"""
import pulp

def pulp_solver(state, total_years):
    """Myopic Knapsack policy for project selection using weighted costs
    
    The solution uses a myopic knapsack solver to pick the optimal set of projects using PuLP with the CBC solver.
    This myopic knapsack policy uses a linear weighted sum to aggregate the maximizing value and minimizing total cost objectives. An order of magnitude weight is used for the total cost objective such that its influence on the objective function is always secondary to maximizing the total value.

    Parameters
    ----------
    state : dict
        A dictionary containing the current state of the simulation, including the current year, budgets, and available projects.
        The 'current_year' key represents the year that the simulation is currently at.
        The 'budgets' key is a list of current budgets to check if a project can be afforded.
        The 'available_projects' key is a list of projects that are available to pick from.
    
    total_years : int
        The total number of years for the planning horizon.

    Returns
    -------
    list
        A list of 1's and 0's representing which project has been picked for this run from the available projects list.
    """
    current_year = state['current_year']
    budgets = state['budgets']
    available_projects = state['available_projects']

    problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)
    project_vars = [pulp.LpVariable(f"Project_{p[0]}", cat="Binary") for p in available_projects]

    weight_for_cost = 1e-4  # Small weight to minimize cost without undermining value maximization
    
    problem += pulp.lpSum([p[1] * var - weight_for_cost * pulp.lpSum(p[2:2+len(budgets)-current_year]) * var for p, var in zip(available_projects, project_vars)])

    for year in range(len(budgets) - current_year):
        if year < len(available_projects[0]) - 2:  # Ensure we don't exceed available project data
            problem += pulp.lpSum([p[2 + year] * var for p, var in zip(available_projects, project_vars)]) <= budgets[current_year + year], f"Budget_Year_{current_year + year + 1}"

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    project_selection = [int(var.value()) for var in project_vars]

    return project_selection
