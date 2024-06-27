"""
This module contains a function for solving the project selection problem using a myopic knapsack solution policy with a percentage of the annual budget being withheld in years beyond the current year.

The alpha-myopic policy is a method for selecting projects based on their value and cost, while considering the annual budget constraints. The policy allocates certain budget percent as a parameter alpha which will be held back for future selections.

Functions:
- alpha_solver: Solves the project selection problem using the alpha-myopic policy.

"""

import pulp

def alpha_solver(state, total_years, alpha=None):
    """Policy for the project selection problem using the myopic knapsack solution with alpha budget withheld.

    Args:
        state (dict): The state of the environment, including the current year, budgets, and available projects.
        total_years (int): The total number of years in the planning horizon.
        alpha (float, optional): The parameter alpha for adjusting the budget allocation. Ratio of each annual budget to be withheld except in the current year, ranges from 0 to 1. For example: a value of 1 would mean 100% of each annual budget beyond the current year being withheld, a value of 0.4 would mean 40% of each annual budget beyond the current year being withheld. Defaults to None.

    Returns:
        list: A list of binary values indicating the selected projects.

    """
    current_year = state['current_year']
    budgets = state['budgets'] 
    available_projects = state['available_projects']
    high_value_project_value = 100

    problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)

    # Create binary variables for each project
    project_vars = [pulp.LpVariable(f"Project_{p[0]}", cat="Binary") for p in available_projects]

    # Weight for the project cost in the objective function
    weight_for_cost = 1e-4

    # Check if there is a high-value project available
    # high_value_project_present = any(project[1] == high_value_project_value for project in available_projects)

    # Adjust budgets based on the PSR policy
    # If high-value project is present or alpha is 1, no budget is reserved; otherwise, budget is reduced
    adjusted_budgets = [budgets[current_year]] + [(1 - alpha) * b for b in budgets[current_year + 1:]]

    # Objective function: Maximize project values while considering costs
    problem += pulp.lpSum([
        p[1] * var - weight_for_cost * pulp.lpSum(p[2:2 + len(budgets) - current_year]) * var
        for p, var in zip(available_projects, project_vars)
    ])

    # Budget constraints for each year
    for year in range(len(budgets) - current_year):
        if year < len(available_projects[0]) - 2:
            problem += pulp.lpSum([
                p[2 + year] * var for p, var in zip(available_projects, project_vars)
            ]) <= adjusted_budgets[year], f"Budget_Year_{current_year + year + 1}"

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # 1 for selected, 0 for not selected
    project_selection = [int(var.value()) for var in project_vars]

    # If alpha is 1, filter out the non-high-value projects
    # if alpha == 1:
    #     project_selection = [1 if project_vars[idx].value() == 1 and available_projects[idx][1] == high_value_project_value else 0 for idx in range(len(available_projects))]

    return project_selection

# def alpha_solver(state, total_years, alpha):
#     current_year = state['current_year']
#     budgets = state['budgets']
#     available_projects = state['available_projects']
    
#     # Create a binary variable for each project to determine if it's selected
#     problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)
#     project_vars = [pulp.LpVariable(f"Project_{p[0]}", cat="Binary") for p in available_projects]
    
#     # Objective function: Maximize value minus a small weighted cost
#     weight_for_cost = 1e-4  # Small weight to minimize cost impact
#     problem += pulp.lpSum([p[1] * var - weight_for_cost * pulp.lpSum(p[2:2+len(budgets)-current_year]) * var for p, var in zip(available_projects, project_vars)])
    
#     # Determine if high value projects are available
#     high_value_vars = [var for p, var in zip(available_projects, project_vars) if p[1] == 100]
    
#     # Budget constraints for each year
#     for year in range(len(budgets) - current_year):
#         if year < len(available_projects[0]) - 2:  # Ensure we don't exceed project data bounds
#             # If high value projects exist, attempt to use full budget for them
#             if high_value_vars:
#                 problem += pulp.lpSum([p[2 + year] * var for p, var in zip(available_projects, project_vars)]) <= budgets[current_year + year], f"Full_Budget_Year_{current_year + year + 1}"
#             else:
#                 # If no high value projects, apply alpha to budget
#                 adjusted_budget = [budgets[current_year]] + [(1 - alpha) * b for b in budgets[current_year + 1:]]
#                 problem += pulp.lpSum([p[2 + year] * var for p, var in zip(available_projects, project_vars)]) <= adjusted_budget[year], f"Adjusted_Budget_Year_{current_year + year + 1}"
    
#     # Solve the problem using the CBC solver
#     problem.solve(pulp.PULP_CBC_CMD(msg=False))
#     project_selection = [int(var.value()) for var in project_vars]

#     return project_selection


# def alpha_solver(state, total_years, alpha):
#     current_year = state['current_year']
#     budgets = state['budgets']
#     available_projects = state['available_projects']
#     high_value_project_value = 100  # Value that indicates a high-value project

#     problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)
#     project_vars = [pulp.LpVariable(f"Project_{p[0]}", cat="Binary") for p in available_projects]

#     weight_for_cost = 1e-4  # Small weight to minimize cost without undermining value maximization

#     # Check for high-value project availability
#     high_value_project_present = any(project[1] == high_value_project_value for project in available_projects)

#     # Adjust budgets based on PSR policy
#     adjusted_budgets = [b if high_value_project_present else (1 - alpha) * b for b in budgets]

#     # Objective Function
#     problem += pulp.lpSum([
#         p[1] * var - weight_for_cost * pulp.lpSum(p[2:2 + total_years - current_year]) * var
#         for p, var in zip(available_projects, project_vars)
#     ])

#     # Budget Constraints
#     for year in range(total_years - current_year):
#         if year < len(available_projects[0]) - 2:  # Ensure we don't exceed available project data
#             problem += pulp.lpSum([
#                 p[2 + year] * var for p, var in zip(available_projects, project_vars)
#             ]) <= adjusted_budgets[current_year + year], f"Budget_Year_{current_year + year + 1}"

#     # Solve the problem
#     problem.solve(pulp.PULP_CBC_CMD(msg=False))
#     project_selection = [int(var.value()) for var in project_vars]

#     return project_selection

