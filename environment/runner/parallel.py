import itertools
import pulp
from offline_env import ProjectSelectionEnv
from tqdm import tqdm
import concurrent.futures
import numpy as np

def offline_solver(state):
    """Solves the project selection problem offline using a knapsack-like optimization approach in which the environment has knowledge of all future variants of every project. Sets an unrealistic benchmark for other policies to achieve.

    Parameters
    ----------
    state : dict
        The current state of the simulation, including the current year, budgets, and available projects.

    Returns
    -------
    total_value (int)
        The cumulative value of the selected projects
    remaining_budgets (list)
        The remaining budget for each year after selecting the projects
    project_selection (list)
        A list of 1's and 0's representing which projects have been picked from the simulation.
    """
    current_year = state['current_year']
    budgets = state['budgets']
    available_projects = state['available_projects']

    problem = pulp.LpProblem("Optimize_Project_Selection", pulp.LpMaximize)
    project_vars = [pulp.LpVariable(f"Project_{p[0]}_{i}", cat="Binary") for i, p in enumerate(available_projects)]

    weight_for_cost = 1e-4

    problem += pulp.lpSum([
        p[1] * var - weight_for_cost * pulp.lpSum(p[2:2+len(budgets)-current_year]) * var 
        for p, var in zip(available_projects, project_vars)
    ])

    for year in range(len(budgets) - current_year):
        if year < len(available_projects[0]) - 2:
            problem += (
                pulp.lpSum([p[2 + year] * var for p, var in zip(available_projects, project_vars)]) <= budgets[current_year + year],
                f"Budget_Year_{current_year + year + 1}"
            )

    project_id_to_vars = {}
    for p, var in zip(available_projects, project_vars):
        if p[0] not in project_id_to_vars:
            project_id_to_vars[p[0]] = []
        project_id_to_vars[p[0]].append(var)

    for project_id, vars in project_id_to_vars.items():
        problem += pulp.lpSum(vars) <= 1, f"One_Version_Of_Project_{project_id}"

    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    project_selection = [int(var.value()) for var in project_vars]

    remaining_budgets = budgets.copy()
    total_value = 0
    for p, var in zip(available_projects, project_vars):
        if var.value() == 1:
            for year in range(len(budgets) - current_year):
                if year < len(p) - 2:
                    remaining_budgets[current_year + year] -= p[2 + year]
            total_value += p[1]

    return total_value, remaining_budgets, project_selection

def run_simulation(budget, seed):
    try:
        project_list = []

        init_state = {
            'years': 25,
            'annual_budget': budget,
            'num_slots': 20,
            'slot_probability': 0.5,
            'project_values': [1, 10, 100],
            'project_probabilities': [0.50, 0.35, 0.15],
            'project_duration_choices': [5, 6, 7],
            'project_duration_probabilities': [0.25, 0.5, 0.25],
            'minimum_cost': 1200000,
            'maximum_cost': 85015000,
            'average_cost': 25706000
        }

        env = ProjectSelectionEnv(init_state=init_state, seed=seed)
        env.reset()

        N = 25
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

        total_value, budget_left, selections = offline_solver(state1)
        return budget, total_value, budget_left, selections
    except Exception as e:
        print(f"Error in budget {budget} with seed {seed}: {e}")
        return budget, None, None, None

def main():
    budget_values = list(range(50, 160, 10))
    seeds = list(range(20))
    cumulative_offline = []

    with open("offline_results.txt", "w") as file:
        file.write("Offline Optimization Results\n")
        file.write("============================\n\n")

    all_results = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_simulation, budget, seed)
            for budget in budget_values
            for seed in seeds
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(budget_values)*len(seeds), desc="Simulations", position=0, leave=True):
            result = future.result()
            all_results.append(result)
    
    for budget in budget_values:
        budget_results = [result[1] for result in all_results if result[0] == budget and result[1] is not None]
        if budget_results:
            average_value = np.mean(budget_results)
            cumulative_offline.append(average_value)
            with open("offline_results.txt", "a") as file:
                file.write(f"Budget: {budget}\n")
                file.write(f"Average Total Value: {average_value}\n\n")
            print(f"Average total value for budget {budget}: {average_value}")

    print("Cumulative offline values:", cumulative_offline)

if __name__ == "__main__":
    main()
