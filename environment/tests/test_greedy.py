import pytest
from fisher_env.project_env import ProjectSelectionEnv
from policies.greedy_policy import greedy

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


def test_greedy_selection_varied_values_costs(project_env):
    """Validates that the greedy policy correctly selects projects, returning a list of binary values (0 or 1), where each element represents whether a project is selected (1) or not (0). This test ensures that the output format is correct and respects the binary nature of project selection."""
    project_env.reset()
    total_years = project_env.init_dict['years']
    selected = greedy(project_env.state, total_years)
    assert isinstance(selected, list) and all(x in [0, 1] for x in selected), "Selection should be a list of binary values."

def test_budget_limitations(project_env):
    """Tests the greedy policy to ensure that the total cost of selected projects does not exceed the initial budget. This test calculates the aggregated cost of all selected projects and checks if it remains within the financial constraints set by the environment's budget."""
    project_env.reset()
    total_years = project_env.init_dict['years']
    initial_budget = project_env.state['budgets'][0]
    selected = greedy(project_env.state, total_years)
    # Ensure total cost of selected projects does not exceed the initial budget
    total_cost = sum(project_env.state['available_projects'][i][2] * selected[i] for i in range(len(selected)))
    assert total_cost <= initial_budget * 1000000, "Selected projects should not exceed the initial annual budget."

def test_multi_year_budget_impact(project_env):
    """Evaluates the long-term financial impact of the greedy policy across multiple years, ensuring that no annual budget turns negative after project selections. This test confirms that the policy responsibly manages the budget over the simulation period."""
    project_env.reset()
    total_years = project_env.init_dict['years']
    greedy(project_env.state, total_years)
    # Check future budgets to ensure costs are appropriately allocated
    assert all(budget >= 0 for budget in project_env.state['budgets']), "Budgets should not be negative in any year."

def test_extreme_cost_projects(project_env):
    """Assesses how the greedy policy handles projects with extremely high costs, checking if such projects are rejected to avoid budget overrun. This test introduces a project with a cost at the upper limit of the environment's cost range to see if the policy correctly decides not to select it due to its prohibitive expense."""
    project_env.reset()
    total_years = project_env.init_dict['years']
    # Modify a project to have a maximum possible cost
    project_env.state['available_projects'].append([999, 100, 85015000])
    selected = greedy(project_env.state, total_years)
    assert selected[-1] == 0, "Extremely high cost projects should not be selected."

def test_greedy_policy_selects_most_valuable_projects(project_env):
    """
    Test that the greedy policy selects the projects with the highest value-to-cost ratio.
    """

    project_env.state = {
        'current_year': 0,
        'budgets': [25, 25, 25, 25, 25],
        'available_projects': [
            [0, 30, 20, 20, 0, 0, 0],
            [1, 20, 10, 0, 0, 0, 0],
            [2, 50, 25, 25, 0, 0, 0]
        ]
    }
    total_years = 5
    selected_projects = greedy(project_env.state, total_years)
    
    assert selected_projects == [0, 1, 0], "Greedy policy should select the projects with the highest value-to-cost ratio."

def test_greedy_policy_respects_budget_constraints(project_env):
    """
    Test that the greedy policy does not select projects that exceed the budget.
    """
    project_env.state = {
        'current_year': 0,
        'budgets': [50, 50, 50, 50, 50],  # Limited budget scenario
        'available_projects': [
            [0, 100, 60, 0, 0, 0, 0],  # Exceeds budget in first year
            [1, 30, 30, 20, 20, 0, 0],  # Affordable over multiple years
            [2, 20, 20, 0, 0, 0, 0]     # Affordable
        ]
    }
    total_years = 5
    selected_projects = greedy(project_env.state, total_years)
    
    assert selected_projects == [0, 1, 1], "Greedy policy should respect budget constraints and not select projects that cannot be afforded."


def test_greedy_policy_with_identical_value_to_cost_ratios(project_env):
    """
    Test that the greedy policy can handle projects with identical value-to-cost ratios,
    prioritizing those that fit within the budget constraints.
    """
    project_env.state = {
        'current_year': 0,
        'budgets': [100, 100],
        'available_projects': [
            [0, 40, 50, 0],
            [1, 40, 50, 0],
            [2, 20, 10, 0]
        ]
    }
    total_years = 2
    selected_projects = greedy(project_env.state, total_years)
    
    # The greedy policy might select both projects 0 and 1,
    # or just one of them plus project 2, to maximize the value within constraints.
    # Checks for one possible correct outcome.
    assert selected_projects == [1, 0, 1], "Greedy policy should handle projects with identical value-to-cost ratios."

def test_greedy_policy_full_budget_utilization(project_env):
    """
    Test that the greedy policy fully utilizes the budget by selecting a mix of projects
    that maximize the value without leaving significant unused budget.
    """
    project_env.state = {
        'current_year': 0,
        'budgets': [100], 
        'available_projects': [
            [0, 60, 60],
            [1, 30, 30], 
            [2, 10, 10]  
        ]
    }
    total_years = 1
    selected_projects = greedy(project_env.state, total_years)
    
    # Expected to select both medium and low-cost projects to maximize the budget utilization
    assert selected_projects == [1, 1, 1], "Greedy policy should strive for full budget utilization."