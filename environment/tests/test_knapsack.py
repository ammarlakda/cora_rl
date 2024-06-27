import pytest
from fisher_env.project_env import ProjectSelectionEnv
from policies.myopic_knapsack_policy import pulp_solver


# Fixture to set up the environment
@pytest.fixture
def env_setup():
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

def test_knapsack_policy_optimizes_value_within_budget(env_setup):
    """Verifies that the pulp_solver, implementing a knapsack-like policy, selects projects that maximize value without exceeding the budget. This test checks that the selected projects represent the optimal combination for value maximization within financial constraints."""
    state = {
        'current_year': 0,
        'budgets': [100, 100, 100],
        'available_projects': [
            [0, 60, 50, 0, 0],  # High value, fits first year budget
            [1, 40, 30, 20, 0], # Spans two years, good value
            [2, 30, 20, 20, 20] # Spans all years
        ]
    }
    total_years = 3
    selected_projects = pulp_solver(state, total_years)
    
    assert selected_projects == [1, 1, 1], "Knapsack policy should select projects that optimize value within budget."

def test_knapsack_policy_when_budget_is_tight(env_setup):
    """Assesses the policy effectiveness in tightly constrained budget scenarios, ensuring that it selects the best possible combination of projects to maximize value while adhering strictly to the budget limits."""
    state = {
        'current_year': 0,
        'budgets': [50, 50, 50],
        'available_projects': [
            [0, 50, 40, 0, 0],  # High value, nearly exhausts budget in year 1
            [1, 30, 20, 10, 0], # Good value, spread out cost
            [2, 20, 10, 0, 0]   # Affordable
        ]
    }
    total_years = 3
    selected_projects = pulp_solver(state, total_years)
    
    assert selected_projects == [1, 0, 1], "Knapsack policy should make optimal selections when budget is tight."

def test_optimal_selection_with_exact_budget_fit(env_setup):
    """Tests the policy ability to make selections that exactly match the available budget, confirming that it optimally uses the entire budget without any excess or deficit."""

    state = {
        'current_year': 0,
        'budgets': [50, 25, 0],
        'available_projects': [
        [0, 60, 50, 25, 0],  # High value but fits precisely over two years
        [1, 40, 75, 0, 0],   # Fits year one budget perfectly
        [2, 30, 30, 0, 0]    # Does not fit
    ]
    }
    total_years = 3
    selected_projects = pulp_solver(state, total_years)

    assert selected_projects == [1, 0, 0], "Should select projects that fit the budget perfectly."

def test_all_projects_below_budget(env_setup):
    """Evaluates whether the pulp_solver selects all projects when each individual project's cost is within the available budget, thus maximizing the number of projects selected under favorable financial conditions."""

    state = {
        'current_year': 0,
        'budgets': [100, 100, 100],
        'available_projects': [
        [0, 30, 20, 10, 0],
        [1, 25, 15, 10, 0],
        [2, 40, 30, 20, 0]
    ]
    }
    total_years = 3
    selected_projects = pulp_solver(state, total_years)
    assert selected_projects == [1, 1, 1], "Should select all projects when each fits within the budget."

def test_future_budget_constraints(env_setup):
    """Checks the policy's foresight by evaluating how it manages selections considering future budget constraints, ensuring that costs allocated to future years do not exceed upcoming budgets."""

    state = {
        'current_year': 0,
        'budgets': [100, 50, 50],
        'available_projects': [
        [0, 50, 10, 50, 50],  # Spreads cost to future years
        [1, 60, 100, 0, 0],   # Fits current year but no room for future costs
        [2, 40, 40, 20, 0]    # Balances across years
    ]
    }
    total_years = 3
    selected_projects = pulp_solver(state, total_years)
    assert selected_projects == [0, 1, 0], "Should make optimal selections considering future year budget constraints."

def test_high_value_priority_under_budget_limit(env_setup):
    """Tests the policy's decision-making under scenarios where project costs are close to the budget limit, verifying that it prioritizes higher-value projects that still fit within the financial constraints."""

    state = {
        'current_year': 0,
        'budgets': [200, 200],
        'available_projects': [
        [0, 120, 199, 0],  # Almost exhausts the budget
        [1, 110, 198, 0],  # Slightly less value and cost
        [2, 100, 50, 150]  # Spreads costs but lower value
    ]
    }
    total_years = 2
    selected_projects = pulp_solver(state, total_years)
    assert selected_projects == [1, 0, 0], "Should select the highest value projects that are within budget constraints."


@pytest.fixture
def negative_budget_state():
    """Provides a state setup with negative budgets."""
    return {
        'current_year': 0,
        'budgets': [80, -9, 50],  # Negative budgets for the simulation
        'available_projects': [
            [0, 60, 50, 20, 20],  # A sample project
            [1, 40, 30, 10, 10],
            [2, 30, 10, 10, 10]   # Another sample project
        ]
    }

def test_negative_budgets(negative_budget_state):
    """Verifies that the pulp_solver does not select any projects when the budget is negative. This test ensures the policy robustness in handling non-ideal financial conditions."""
    total_years = 1
    selected_projects = pulp_solver(negative_budget_state, total_years)
    # Check that no projects are selected
    assert all(x == 0 for x in selected_projects), "No projects should be selected with negative budgets."
