"""
This module contains a function for solving the project selection problem using a Greedy policy using a ranked list based on value of project.

It is a method for selecting projects based on their value, while considering annual budget constraints. The policy ranks all projects by using a value reverse list to pick projects to select projects with the highest value without taking cost into account.

Functions:
- greedy_value: Solves the project selection problem using the greedy policy.

"""
def greedy_value(state, total_years):
    """Greedy policy for project selection based on value only.
    
    Picks projects using a greedy policy by ranked list by value alone. 
    Highest value projects are picked first.

    Parameters
    ----------
    state : dict
        The state dict containing information about current year, budgets, and available projects.
    total_years : int
        The total amount of years for the planning horizon.

    Returns
    -------
    list
        List of 1's and 0's representing which projects have been picked for this run from the available projects list.
    """
    current_year = state['current_year']
    budgets = state['budgets']
    available_projects = state['available_projects']
    selected_projects = [0] * len(available_projects)
    
    # Sort available projects by their value in descending order
    available_projects_with_index = sorted(
        enumerate(available_projects),
        key=lambda x: x[1][1],  # Represents the project value
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
