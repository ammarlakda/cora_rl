import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ProjectSelectionEnv(gym.Env):
    def __init__(self, init_state, seed=None):
        """Initialize the environment with a given initial state and an optional random seed.

        Parameters
        ----------
            init_state (dict): Initial settings for the environment including years, budget, project values, etc.
            seed (int, optional): Random seed for generating reproducible random effects within the environment.
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.max_val = max(init_state['project_duration_choices'])
        self.total_years = init_state['years']

        self.init_dict = {
            'years': init_state['years'] + self.max_val,
            'annual_budget': init_state['annual_budget'],
            'num_slots': init_state['num_slots'],
            'slot_probability': init_state['slot_probability'],
            'project_values': init_state['project_values'],
            'project_probabilities': init_state['project_probabilities'],
            'project_duration_choices': init_state['project_duration_choices'],
            'project_duration_probabilities': init_state['project_duration_probabilities'],
            'minimum_cost': init_state['minimum_cost'],
            'maximum_cost': init_state['maximum_cost'],
            'average_cost': init_state['average_cost']
        }

        self.state = {
            "current_year": 0,
            "project_id_counter": 0,
            "cumulative_reward": 0,
            "budgets": [self.init_dict['annual_budget']] * self.init_dict['years'],
            "available_projects": [],
            "picked_projects_ids": []
        }

        self.action_space = spaces.MultiBinary(self.state['available_projects'])  # Changed from number of projects high to available projects

        self.observation_space = spaces.Dict({
            "budgets": spaces.Box(low=0, high=np.inf, shape=(self.init_dict['years'],), dtype=np.float32),
            "projects": spaces.Box(low=0, high=np.inf, shape=(len(self.state['available_projects']), 2 + self.init_dict['years']), dtype=np.float32)  # Changed from number of projects high to available projects
        })

    def reset(self):
        """
        Resets the environment to its initial state. This includes setting the current year to zero,
        regenerating the initial set of projects, and resetting the project id counter and budgets.

        Returns
        -------
        dict
            The initial state of the environment with refreshed values.
        """
        self.state['current_year'] = 0
        self.state['project_id_counter'] = 0
        self.state['budgets'] = [self.init_dict['annual_budget']] * self.init_dict['years']
        self.state['available_projects'] = []
        self.state['picked_projects_ids'] = []
        self._generate_new_projects()
        self._update_observation()
        return self.state

    def _generate_new_projects(self):
        """Generates new projects for the current year based on the initialized slot probabilities and project parameters.
        Each project's cost distribution is determined by its duration and adjusted for the ongoing year."""
        slots = self.init_dict['num_slots']
        slot_prob = self.init_dict['slot_probability']

        cost_distributions = {
            5: [0.1, 0.2, 0.4, 0.2, 0.1],
            6: [0.05, 0.15, 0.3, 0.3, 0.15, 0.05],
            7: [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        }

        for _ in range(slots):
            if self.rng.random() < slot_prob:
                project_id = self.state['project_id_counter']
                self.state['project_id_counter'] += 1

                project_value = self.rng.choice(self.init_dict['project_values'], p=self.init_dict['project_probabilities'])
                project_duration = self.rng.choice(self.init_dict['project_duration_choices'], p=self.init_dict['project_duration_probabilities'])
                triangle = self.rng.triangular(self.init_dict['minimum_cost'], self.init_dict['average_cost'], self.init_dict['maximum_cost'])
                total_cost = round(triangle / 1000000, 2)

                if project_duration in cost_distributions:
                    distribution = cost_distributions[project_duration]
                    project_costs = [round(total_cost * proportion, 2) for proportion in distribution]
                else:
                    project_costs = [round(total_cost / project_duration, 2)] * project_duration

                project_costs.extend([0] * (self.max_val - project_duration))
                padded_costs = [0] * self.state['current_year'] + project_costs
                padded_costs.extend([0] * (self.init_dict['years'] - len(padded_costs)))

                project = [project_id, project_value] + padded_costs
                self.state['available_projects'].append(project)

    def _update_observation(self):
        """Updates the observation space to reflect the current internal state of the environment.
        This method is typically called after changes to the state such as after generating new projects or advancing the year."""
        self.state = {
            "current_year": self.state['current_year'],
            "project_id_counter": self.state['project_id_counter'],
            "cumulative_reward": self.state['cumulative_reward'],
            "budgets": self.state['budgets'],
            "available_projects": self.state['available_projects'],
            "picked_projects_ids": self.state['picked_projects_ids']
        }

    def step(self):
        """Transitions through the environment by selecting projects, updating the state, and calculating rewards.
        """
        reward = 0

        # Decay project values by a factor of 1 / 1.1 each year
        for project in self.state['available_projects']:
            project[1] /= 1.1

            # Shift costs by 1 year
            project[2:] = [0] + project[2:-1]

        # Move to the next year without making any selections or deductions
        self.state['current_year'] += 1
        done = self.state['current_year'] >= self.total_years

        # Generate new projects only if it's not the last year
        if self.state['current_year'] < self.total_years:
            self._generate_new_projects()

        # # Apply random budget changes for all future years
        # budget_changes = np.arange(-0.10, 0.11, 0.01)
        # for year in range(self.state['current_year'], self.init_dict['years']):
        #     self.state['budgets'][year] = round(
        #         self.state['budgets'][year] * (1 + self.rng.choice(budget_changes)), 2
        #     )

        self._update_observation()
        return self.state, reward, done

    def render(self, mode='console', reward=0):
        """
        Prints the current state of the environment to the console. Useful for debugging and visualizing the
        environment's progression through the simulation.

        Parameters
        ----------
            mode (str): The medium through which to render the environment's state. Currently, only 'console' is supported.
            reward (float): The latest reward obtained, which can be displayed alongside the state.
        """
        if mode == 'console':
            print(f"Current Year: {self.state['current_year']}")
            print(f"Budgets: {self.state['budgets']}")
            print(f"Available Projects: {self.state['available_projects']}")
            print(f"Cumulative Reward: {self.state['cumulative_reward']}")
            print()