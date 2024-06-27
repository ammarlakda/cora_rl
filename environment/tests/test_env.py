import pytest
from fisher_env.project_env import ProjectSelectionEnv

# Fixture to set up the environment
@pytest.fixture
def project_env():
    """Fixture to set up the environment with default initialization settings."""
    init_state = {
        'years' : 25,
        'annual_budget' : 55,  # in millions
        'num_slots' : 5,
        'slot_probability' : 0.9,
        'project_values' : [1, 10, 100],  # Low, Medium, High values
        'project_probabilities' : [0.50, 0.35, 0.15], # Probabilities for project values
        'project_duration_choices' : [5, 6, 7],
        'project_duration_probabilities' : [0.25, 0.5, 0.25],
        'minimum_cost' : 1200000,
        'maximum_cost' : 85015000,
        'average_cost' : 25706000
    }
    return ProjectSelectionEnv(init_state=init_state, seed=42)

def test_environment_initialization(project_env):
    """Checks that the environment is initialized correctly with the current_year set to 0 and the number of budget years equal to the specified years in the initialization."""
    assert project_env.state['current_year'] == 0
    assert len(project_env.state['budgets']) == project_env.init_dict['years']

def test_reset_functionality(project_env):
    """Verifies that the environment's reset function correctly resets the current_year to 0 and generates an initial set of available projects."""
    project_env.reset()
    assert project_env.state['current_year'] == 0
    assert len(project_env.state['available_projects']) > 0  # Assuming some projects are generated

def test_step_functionality(project_env):
    """Tests the environment's step function by selecting all available projects, ensuring the current year increments by 1 and the budget decreases."""
    initial_budget = project_env.state['budgets'][0]
    project_env.reset()
    print(project_env.state['available_projects'])
    action = [1] * len(project_env.state['available_projects'])  # Select all projects
    state, reward, done = project_env.step(action)
    print(state)
    assert state['current_year'] == 1
    assert project_env.state['budgets'][0] < initial_budget  # Budget should decrease

def test_project_generation(project_env):
    """Confirms that projects are generated and available at the start of the simulation after resetting the environment."""
    project_env.reset()
    assert len(project_env.state['available_projects']) > 0

def test_environment_constraints(project_env):
    """Checks that rejecting all projects results in no rewards and the budgets remain within a 10% tolerance of the initial setup."""
    project_env.reset()
    action = [0] * len(project_env.state['available_projects'])  # Reject all projects
    state, reward, done = project_env.step(action)
    assert reward == 0  # No reward since no projects are selected

    # Allow a small tolerance for budget fluctuations
    tolerance = 0.1  # 10% tolerance
    for budget in state['budgets']:
        assert abs(budget - project_env.init_dict['annual_budget']) <= tolerance * project_env.init_dict['annual_budget']


def test_end_of_simulation(project_env):
    """Verifies that the simulation ends after running for the predefined number of years specified in the initialization."""
    project_env.reset()
    for _ in range(project_env.init_dict['years']):
        action = [0] * len(project_env.state['available_projects'])  # No projects selected
        _, _, done = project_env.step(action)
    assert done  # The simulation should end after the specified number of years

def test_reward_accumulation(project_env):
    """Assesses whether the cumulative reward matches the total reward obtained over two steps when all available projects are selected."""
    project_env.reset()
    total_reward = 0
    action = [1] * len(project_env.state['available_projects'])  # Select all available projects
    for _ in range(2):  # Test over two steps
        _, reward, _ = project_env.step(action)
        total_reward += reward
    assert project_env.state['cumulative_reward'] == total_reward  # Check if rewards are accumulated correctly

def test_project_carry_over(project_env):
    """Ensures that projects not selected in one step are carried over to the next step, maintaining or increasing the total count of available projects."""
    project_env.reset()
    initial_project_count = len(project_env.state['available_projects'])
    action = [0] * initial_project_count  # Reject all projects
    project_env.step(action)
    assert len(project_env.state['available_projects']) >= initial_project_count  # Ensure projects are carried over

def test_budget_overrun(project_env):
    """Tests the system's response to attempting to select a project that exceeds the available budget, expecting a negative budget as a result."""
    project_env.reset()
    project_env.state['available_projects'] = [[0, 300, 2000]]  # Unaffordable project
    action = [1]  # Try to select the unaffordable project
    _, _, _ = project_env.step(action)
    assert project_env.state['budgets'][0] < 0  # Budget should be negative indicating overrun

def test_random_seed_reproducibility(project_env):
    """Confirms that initializing two environments with the same seed results in identical initial states."""
    env1 = ProjectSelectionEnv(project_env.init_dict, seed=42)
    env2 = ProjectSelectionEnv(project_env.init_dict, seed=42)
    env1.reset()
    env2.reset()
    assert env1.state == env2.state  # States should be identical after the same number of steps

def test_budget_deduction(project_env):
    """Validates that the budget decreases appropriately after selecting all available projects."""
    project_env.reset()
    initial_budget = project_env.state['budgets'][0]
    action = [1] * len(project_env.state['available_projects'])  # Select all projects
    _, _, _ = project_env.step(action)
    assert project_env.state['budgets'][0] < initial_budget  # Budget should decrease after selecting projects

def test_project_value_decay(project_env):
    """Ensures that the values of projects not selected decay by 10% from their initial values."""
    project_env.reset()
    initial_values = [project[1] for project in project_env.state['available_projects']]
    action = [0] * len(project_env.state['available_projects'])  # Reject all projects
    project_env.step(action)
    decayed_values = [project[1] for project in project_env.state['available_projects']]
    for initial, decayed in zip(initial_values, decayed_values):
        assert decayed == initial / 1.1  # Check if the project values are decayed correctly

def test_project_cost_distribution(project_env):
    """Checks that the total cost of projects lies within the specified cost range in the initialization settings."""
    project_env.reset()
    for project in project_env.state['available_projects']:
        project_id, project_value, *project_costs = project
        total_cost = sum(project_costs)
        if project_costs:
            assert total_cost > 0  # Ensure project has a positive total cost
            assert project_env.init_dict['minimum_cost'] / 1000000 <= total_cost <= project_env.init_dict['maximum_cost'] / 1000000

def test_future_budget_percent_change(project_env):
    """Verifies that budget changes from year to year stay within a ±10% range after rejecting all projects."""
    project_env.reset()
    initial_budgets = project_env.state['budgets'][project_env.state['current_year']:]
    action = [0] * len(project_env.state['available_projects'])  # Reject all projects
    project_env.step(action)
    new_budgets = project_env.state['budgets'][project_env.state['current_year']:]
    for initial, new in zip(initial_budgets, new_budgets):
        percent_change = new / initial
        assert 0.90 <= percent_change <= 1.10  # Ensure budget changes are within ±10%