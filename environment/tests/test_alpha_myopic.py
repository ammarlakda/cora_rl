import pytest
from fisher_env.project_env import ProjectSelectionEnv
from policies.alpha_myopic import alpha_solver

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

def test_alpha_solver(project_env):
    """Tests the alpha_solver function by setting up a state with predefined projects and budgets, and asserts that the solver chooses the last project while holding back a fraction of the budget as determined by the alpha parameter."""
    # Set up initial state
    state = {
        'current_year': 0,
        'budgets': [500, 10, 10, 10],
        'available_projects': [
            [1, 10, 100, 5, 6, 7],
            [2, 20, 200, 5, 6, 7],
            [3, 30, 300, 5, 6, 7]
        ]
    }
    
    # Call the alpha_solver function
    project_selection = alpha_solver(state, total_years=5, alpha=0.2)
    
    # Assert the expected project selection
    assert project_selection == [0, 0, 1]
    
def test_alpha_solver_no_budget_withheld(project_env):
    """Verifies that when alpha is set to 0 (no budget is withheld), the alpha_solver selects all available projects, regardless of their cost, as long as the budget permits."""
    # Set up initial state
    state = {
        'current_year': 0,
        'budgets': [6, 15, 18 , 21, 9],
        'available_projects': [
            [1, 10, 1, 5, 6, 7],
            [2, 20, 2, 5, 6, 7],
            [3, 30, 3, 5, 6, 7]
        ]
    }
    
    # Call the alpha_solver function
    project_selection = alpha_solver(state, total_years=5, alpha=0.0)
    
    # Assert the expected project selection
    assert project_selection == [1, 1, 1]

def test_alpha_solver_budget_withheld(project_env):
    """Checks the functionality of the alpha_solver when a high alpha value is set, leading to significant budget withholding, ensuring that only projects that fit within the severely restricted budget are selected."""
    # Set up initial state
    state = {
        'current_year': 0,
        'budgets': [10, 15, 25 , 20],
        'available_projects': [
            [1, 10, 2, 3, 5, 4],
            [2, 20, 3, 7, 6, 6],
            [3, 30, 1, 4, 8, 5]
        ]
    }
    
    # Call the alpha_solver function
    project_selection = alpha_solver(state, total_years=5, alpha=0.8)
    
    # Assert the expected project selection
    # Should be just enough budget for the 1st project
    assert project_selection == [1, 0, 0]

def test_alpha_solver_no_projects_available(project_env):
    """Assesses the alpha_solver function in a scenario where no affordable projects are available within the given budget constraints, expecting no projects to be selected."""
    # Set up initial state
    project_env.state = {
        'current_year': 0,
        'budgets': [10, 15, 25 , 20],
        'available_projects': [[3, 30, 100, 400, 800, 500]]
    }
    
    # Call the alpha_solver function
    project_selection = alpha_solver(project_env.state, total_years=5, alpha=0.8)
    
    # Assert the expected project selection
    assert project_selection == [0]

def test_alpha_solver_no_budgets_available(project_env):
    """Tests the alpha_solver when there is no budget available at all, ensuring that no projects are selected due to financial constraints."""
    # Set up initial state
    project_env.state = {
        'current_year': 0,
        'budgets': [0, 0, 0, 0],
        'available_projects': [
            [1, 10, 2, 3, 5, 4],
            [2, 20, 3, 7, 6, 6],
            [3, 30, 1, 4, 8, 5]
        ]
    }
    
    # Call the alpha_solver function
    project_selection = alpha_solver(project_env.state, total_years=5, alpha=0.8)
    
    # Assert the expected project selection
    assert project_selection == [0, 0, 0]

def test_alpha_solver_holds_back_budget(project_env):
    """Evaluates the alpha_solver in a scenario where it should hold back a specific percentage of the budget (specified by alpha), and checks if the function selects projects accordingly to meet this constraint."""
    state = {
        'current_year': 0,
        'budgets': [10, 10, 10, 10],
        'available_projects': [
            [0, 10, 5, 5, 0],
            [1, 10, 5, 5, 0],
            [2, 10, 5, 5, 0]
        ]
    }
    total_years = 4
    alpha = 0.5
    selected_projects = alpha_solver(state, total_years, alpha)
    
    assert selected_projects == [0, 0, 1], "Alpha solver should hold back alpha percent of the budget."

def test_alpha_solver_cheapest_combination(project_env):
    """Tests whether the alpha_solver selects the cheapest viable combination of projects when multiple options are available that fit within the budget constraints."""
    state = {
        'current_year': 0,
        'budgets': [5, 5, 5],
        'available_projects': [
            [0, 10, 2, 3, 4],
            [1, 10, 1, 2, 3],
            [2, 10, 3, 4, 5]
        ]
    }
    total_years = 4
    alpha = 0
    selected_projects = alpha_solver(state, total_years, alpha)

    assert selected_projects == [0, 1, 0], "Alpha solver should select the cheapest combination of projects."