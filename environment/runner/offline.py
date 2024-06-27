import itertools
import pulp
from offline_env import ProjectSelectionEnv
from tqdm import tqdm

def offline_solver(state):
    """
    Solves the project selection problem using a linear programming approach to maximize project value while adhering 
    to annual budget constraints across multiple years. The objective is to select projects that maximize the total value,
    adjusting the selection based on a weight that minimizes the project costs.

    Each project can be selected as a binary decision and applies constraints for each year's budget, and handles project costs dynamically based on project duration and start year.

    To be used with offline_env.py as the non-sequential environment.

    Parameters
    ----------
    state (dict)
        A dictionary representing the current state of the environment which includes:
        - 'current_year': the ongoing year of project selection
        - 'budgets': list of available budgets for each year
        - 'available_projects': list of projects available for selection where each project is represented
          as a list with the structure [project_id, project_value, cost_year_1, ..., cost_year_n]

    Returns
    -------
    tuple
        Returns a tuple containing:
        - list of binary integers representing project selection,
        - total value of selected projects,
        - list of remaining budgets after project selection.
        
    """
    current_year = state['current_year']
    budgets = state['budgets']
    available_projects = state['available_projects']

    # Define the optimization problem
    problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)

    # Create decision variables for each project
    project_vars = [pulp.LpVariable(f"Project_{p[0]}_{i}", cat="Binary") for i, p in enumerate(available_projects)]

    weight_for_cost = 1e-4  # Small weight to minimize cost without undermining value maximization

    # Objective function: maximize value while minimizing costs with a small weight
    problem += pulp.lpSum([
        p[1] * var - weight_for_cost * pulp.lpSum(p[2:2+len(budgets)-current_year]) * var 
        for p, var in zip(available_projects, project_vars)
    ])

    # Constraints: ensure total costs do not exceed the budget for each year
    for year in range(len(budgets) - current_year):
        if year < len(available_projects[0]) - 2:  # Ensure we don't exceed available project data
            problem += (
                pulp.lpSum([p[2 + year] * var for p, var in zip(available_projects, project_vars)]) <= budgets[current_year + year],
                f"Budget_Year_{current_year + year + 1}"
            )

    # Constraint: ensure each project ID is picked at most once
    project_id_to_vars = {}
    for p, var in zip(available_projects, project_vars):
        if p[0] not in project_id_to_vars:
            project_id_to_vars[p[0]] = []
        project_id_to_vars[p[0]].append(var)

    for project_id, vars in project_id_to_vars.items():
        problem += pulp.lpSum(vars) <= 1, f"One_Version_Of_Project_{project_id}"

    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the results
    project_selection = [int(var.value()) for var in project_vars]

    # Calculate and print remaining budgets and total value
    remaining_budgets = budgets.copy()
    total_value = 0
    for p, var in zip(available_projects, project_vars):
        if var.value() == 1:  # Project is selected
            for year in range(len(budgets) - current_year):
                if year < len(p) - 2:  # Ensure we don't exceed available project data
                    remaining_budgets[current_year + year] -= p[2 + year]
            total_value += p[1]  # Add project value once

    return project_selection, total_value, remaining_budgets

def main():
    cumulative_offline = []
    N = 25

    with open("offline_results.txt", "w") as file:
        file.write("Offline Optimization Results\n")
        file.write("============================\n\n")

    for budget in tqdm(range(50, 160, 10), desc="Budgets", position=0, leave=True):
        project_list = []

        init_state = {
            'years': 25,
            'annual_budget': budget,  # in millions
            'num_slots': 20,
            'slot_probability': 0.5,
            'project_values': [1, 10, 100],  # Low, Medium, High values
            'project_probabilities': [0.50, 0.35, 0.15],  # Probabilities for project values
            'project_duration_choices': [5, 6, 7],
            'project_duration_probabilities': [0.25, 0.5, 0.25],
            'minimum_cost': 1200000,
            'maximum_cost': 85015000,
            'average_cost': 25706000
        }

        # Initialize the environment
        env = ProjectSelectionEnv(init_state=init_state, seed=42)
        env.reset()

        for _ in range(N):
            state, reward, done = env.step()
            project_list.append([list(project) for project in state['available_projects']])
            if done:
                break

        env.close()

        final_list = list(itertools.chain.from_iterable(project_list))

        state1 = {
            'current_year': 0,
            'budgets': [budget] * 32,
            'available_projects': final_list
        }

        selection, total_value, budget_left = offline_solver(state1)
        cumulative_offline.append(total_value)
        
        with open("offline_results.txt", "a") as file:
            file.write(f"Budget: {budget}\n")
            file.write(f"Budget Left: {budget_left}\n")
            file.write(f"Total Value: {total_value}\n")
            file.write(f"Selected Projects: {selection}\n\n")

        print(f"Total value for budget {budget}: {total_value}")

    print("Cumulative offline values:", cumulative_offline)

if __name__ == "__main__":
    main()