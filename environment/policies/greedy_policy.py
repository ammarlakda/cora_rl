"""
This module contains a function for solving the project selection problem using a Greedy policy with a fractional heuristic.

The greedy policy is a method for selecting projects based on their value and total cost, while considering annual budget constraints. The policy ranks all projects by using a value/total cost reverse list then picks projects from highest to lowest according to their value/cost ratio, until an annual budget constraint is encountered.

Functions:
- greedy: Solves the project selection problem using the greedy policy.

"""
def greedy(state, total_years):
    """Greedy policy for project selection based on value//total cost
    
    Picks projects using a greedy heuristic policy by ranked list by value/sum of yearly expenditures. 
    Highest ratio is picked first.

    Parameters
    ----------
    available_projects : list
        List of projects that are available to pick from
    budgets : list
        List of current budgets to see if a project can be afforded
    current_year : int
        The year that the simulation is at currently. Increments each step
    total_years : int
        The total amount of years for the planning horizon

    Returns
    -------
    list
        List of 1's for picked and 0's for not picked, representing which project has been picked for this step from the available projects list.
    """
    current_year = state['current_year']
    budgets = state['budgets']
    available_projects = state['available_projects']
    selected_projects = [0] * len(available_projects)
    
    available_projects_with_index = sorted(
        enumerate(available_projects),
        key=lambda x: x[1][1] / sum(x[1][2:]),  # Adjust the sum to consider all costs
        reverse=True
    )

    temp_budgets = budgets.copy()

    for index, project in available_projects_with_index:
        project_id, project_value, *project_costs = project

        # Check if the project can be afforded in the current and subsequent years
        can_afford = True
        for year_offset, cost in enumerate(project_costs):
            if year_offset >= total_years - current_year:
                break  # Do not consider costs beyond the total simulation years

            if temp_budgets[current_year + year_offset] - cost < 0:
                can_afford = False
                break

        if can_afford:
            selected_projects[index] = 1  # Select this project
            # Deduct the costs from the temporary budgets
            for year_offset, cost in enumerate(project_costs):
                if year_offset >= total_years - current_year:
                    break
                temp_budgets[current_year + year_offset] -= cost

    return selected_projects
