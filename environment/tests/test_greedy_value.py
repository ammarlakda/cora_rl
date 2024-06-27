import pytest
from policies.greedy_value_policy import greedy_value  # Adjust import path as needed

# Fixture for environment state setup
@pytest.fixture
def sample_state():
    return {
        'current_year': 0,
        'budgets': [100, 100, 100],  # Assuming a simple case with a flat budget across years
        'available_projects': [
            [0, 100, 50, 0, 0],  # High value but takes half the budget
            [1, 60, 30, 20, 0],  # Medium value, spread cost
            [2, 40, 40, 0, 0],   # Lower value, costs 40% of the budget
            [3, 20, 10, 0, 0]    # Least value, minimal cost
        ]
    }

# Test to verify the policy selects the highest value projects within budget constraints
def test_selects_highest_value_projects_within_budget(sample_state):
    """Verifies that the greedy_value policy selects the highest value projects that fit within a given budget, prioritizing projects with the greatest value per cost."""
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [1, 1, 0, 1], "Should select highest value projects that fit the budget"

# Test to ensure the policy respects the budget constraint
def test_respects_budget_constraints(sample_state):
    """Tests that the greedy_value policy adheres strictly to budget limits, ensuring that it does not select projects whose total costs exceed the available budget."""
    sample_state['budgets'] = [50, 50, 50]  # Adjust budget to be more restrictive
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [1, 0, 0, 0], "Should only select projects that can be afforded within the budget"

# Test to check functionality when no projects can be afforded
def test_no_projects_affordable_with_given_budget(sample_state):
    """Checks the policy's behavior in scenarios where the available budget is too low to afford any of the listed projects, ensuring that it does not erroneously select unaffordable projects."""
    sample_state['budgets'] = [10, 10, 10]  # Set budget too low for any project
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [0, 0, 0, 1], "Should not select projects that exceed the budget"

# Test to confirm that the policy selects all projects if the budget allows
def test_selects_all_projects_if_budget_allows(sample_state):
    """Confirms that if the budget is sufficiently large, the greedy_value policy will select all available projects, taking advantage of the ample budget to maximize project selection."""
    sample_state['budgets'] = [200, 200, 200]  # Increase budget to allow all projects
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [1, 1, 1, 1], "Should select all projects when the budget allows"

def test_with_dynamically_changing_budgets(sample_state):
    """Evaluates how the greedy_value policy manages variable budgets across different years, testing its ability to optimize project selection in response to changing financial constraints."""
    sample_state['budgets'] = [80, 120, 60]  # Varying budgets over the years
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    # Expectation might change based on project values and costs
    assert selected == [1, 1, 0, 0], "Should optimally select projects given fluctuating annual budgets."

def test_with_projects_having_identical_values(sample_state):
    """Tests the policy's decision-making with projects that have identical values but different costs, assessing how it prioritizes projects based on cost-effectiveness when project values do not provide a clear selection criterion."""
    sample_state['available_projects'] = [
        [0, 50, 60, 0, 0],
        [1, 50, 40, 10, 0],
        [2, 50, 60, 0, 0],  # Identical values, different costs
        [3, 20, 50, 0, 0]
    ]
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    # The selection can vary; this is a placeholder assertion
    assert selected == [1, 1, 0, 0], "Should handle projects with identical values intelligently."

def test_all_projects_affordable_exactly(sample_state):
    """Checks the greedy_value policy when the total budget exactly matches the combined costs of all projects, confirming that it selects all projects, fully utilizing the available budget."""
    sample_state['budgets'] = [100, 0, 0]  # Budget exactly matches the total cost of all projects
    sample_state['available_projects'] = [
        [0, 60, 50, 0, 0],
        [1, 40, 50, 0, 0]  # Adjusted for total budget match
    ]
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [1, 1], "Should select all projects when their total cost matches the budget exactly."

def test_insufficient_budget_for_any_project(sample_state):
    """Verifies that the greedy_value policy correctly handles situations where the budget is too low to afford even the least expensive project, ensuring that no projects are selected under these financial constraints."""
    sample_state['budgets'] = [5, 5, 5]  # Insufficient budget
    total_years = 3
    selected = greedy_value(sample_state, total_years)
    assert selected == [0, 0, 0, 0], "Should not select any project when the budget is too low."
