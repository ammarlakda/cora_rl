""" This script is used to run the environment with a specific policy. Gives a command line output of the environment state after each step."""
from fisher_env.project_env import ProjectSelectionEnv
from policies.myopic_knapsack_policy import pulp_solver
from policies.greedy_policy import greedy
from policies.greedy_value_policy import greedy_value
from policies.alpha_myopic import alpha_solver

N = 25

init_state = {
    'years' : 25,
    'annual_budget' : 80,  # in millions
    'num_slots' : 20,
    'slot_probability' : 0.5,
    'project_values' : [1, 10, 100],  # Low, Medium, High values
    'project_probabilities' : [0.50, 0.35, 0.15], # Probabilities for project values
    'project_duration_choices' : [5, 6, 7],
    'project_duration_probabilities' : [0.25, 0.5, 0.25],
    'minimum_cost' : 1200000,
    'maximum_cost' : 85015000,
    'average_cost' : 25706000
}
print()
env = ProjectSelectionEnv(init_state = init_state, seed = 42)
env.reset()
print("Initial available projects before Year 1:")
env.render()


for _ in range(N):

    action = alpha_solver(state=env.state, total_years=init_state['years'], alpha = 0.4)

    bud = env.state['budgets']
    print(f"Year: {env.state['current_year']}, \n")
    print()
    for project in env.state['available_projects']:
        print(project)
    print("Before picking:\n")
    print(bud[env.state['current_year']:])
    print(env.state['picked_projects_ids'])
    state, reward, done = env.step(action) 
    print()
    print("After picking:\n")
    print(bud[env.state['current_year'] - 1:])
    print(env.state['picked_projects_ids'],"\n")
    env.render(reward = reward)
    print("---------------------------------")

    if done:
        break
env.close()

#[200, 0.6830134553650705, 6.21, 12.42, 24.85, 12.42, 6.21, 0, 0]