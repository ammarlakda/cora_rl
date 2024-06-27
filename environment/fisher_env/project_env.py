"""Project Selection Environment for Reinforcement Learning.

This module implements a custom environment for simulating 
project selection decisions within a specified planning horizon.
The environment is built on top of the OpenAI Gymnasium framework, 
allowing it to be integrated with various reinforcement learning algorithms. 
It simulates the challenge of selecting from a set of available projects each year, 
given a fixed annual budget, 
with the goal of maximizing cumulative rewards over the entire planning horizon.
The environment supports decisions based on project values, costs, and durations, 
incorporating randomness in project generation and budget allocations to mimic real-world uncertainty. 
It is designed for experiments in resource allocation, 
strategic planning, and decision-making under uncertainty.

How to Use:
    To use the environment, instantiate the ProjectSelectionEnv class 
    with an initial state configuration and optional seed for reproducibility. 
    Agents can interact with the environment using the `step` method 
    by passing actions that return project selections. 
    The environment transitions to the new state, updates the reward, and checks the flag indicating if the simulation has ended. 

Dependencies:
    gymnasium: OpenAI Gymnasium framework for developing and comparing reinforcement learning algorithms.
    numpy: The fundamental package for scientific computing with Python.

Author:
    [Ammar Lakdawala] 
    [Mark Rempel] [Carolyn Chen]
    [github.com/ammarlakda]

License:
    [The MIT License (MIT)
    Copyright (c) 2023, Crown Copyright]
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class ProjectSelectionEnv(gym.Env):
    """A custom Gymnasium environment for simulating project selection
    in a resource-constrained scenario.

    This environment challenges agents to select from a set of available projects each year, 
    given a fixed annual budget, aiming to maximize cumulative rewards
    over a planning horizon. 
    It incorporates randomness in project generation and budget allocations to mimic real-world uncertainty,
    for experiments in resource allocation, strategic planning,
    and decision-making under uncertainty.

    Parameters
    ----------
    init_state : dict
        A dictionary containing the initial configuration for the environment. 
        Keys include:

        - 'years': Planning horizon in years.
        - 'annual_budget': Annual budget for each year.
        - num_slots: Number of available project spots to be filled.
        - slot_probability: Probability of each slot being filled.
        - 'project_values': Values of projects
        - 'project_probabilities': Project value probabilities.
        - 'project_duration_choices': Project durations
        - 'project_duration_probabilities': Project durations probabilities.
        - 'minimum_cost' : The minium total cost for a project for sampling from a triangular distribution,
        - 'maximum_cost' : The maximum total cost for a project for sampling from a triangular distribution,
        - 'average_cost' : The average total cost for a project for sampling from a triangular distribution
    seed : int, optional
        Seed for random number generation to ensure reproducibility.

    Attributes
    ----------
    action_space : gymnasium.spaces
        The space of possible actions (project selections) an agent can take being binary decisions for each project being selected or not.
    observation_space : gymnasium.spaces
        The space of observable states, including budgets and project costs, value, and ID.
    state : dict
        The current state of the environment, containing information about
        the current year, available projects, and financial status.
    """
    def __init__(self, init_state, seed=None):
        """Initialize the environment with the specified initial state and an optional random seed.

        Parameters
        ----------
        init_state : dict
            A dictionary containing initial configuration for the environment including:
            - years: The number of years for the planning horizon.
            - annual_budget: Annual budget for each year.
            - num_slots: Number of available project spots to be filled.
            - slot_probability: Probability of each slot being filled.
            - project_values: List of possible project values.
            - project_probabilities: Probabilities associated with each project value.
            - project_duration_choices: List of possible project durations.
            - project_duration_probabilities: Probabilities associated with each project duration.
            - minimum_cost : The minium total cost for a project for triangular distribution.
            - maximum_cost : The maximum total cost for a project for triangular distribution.
            - average_cost : The average total cost for a project for triangular distribution.
        seed : int, optional
            An optional random seed for reproducibility.
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.max_val = max(init_state['project_duration_choices'])


        self.init_dict = {
            'years' : init_state['years'] + self.max_val,
            'annual_budget': init_state['annual_budget'],
            'num_slots': init_state['num_slots'],
            'slot_probability': init_state['slot_probability'],
            'project_values' : init_state['project_values'],
            'project_probabilities' : init_state['project_probabilities'],
            'project_duration_choices' : init_state['project_duration_choices'],
            'project_duration_probabilities' : init_state['project_duration_probabilities'],
            'minimum_cost' : init_state['minimum_cost'],
            'maximum_cost' : init_state['maximum_cost'],
            'average_cost' : init_state['average_cost']
        }


        self.state = {
            "current_year": 0,
            "project_id_counter": 0,
            "cumulative_reward": 0,
            "budgets": [self.init_dict['annual_budget']] * (self.init_dict['years']),
            "available_projects": [],
            "picked_projects_ids": []
        }


        # Action space: binary decision for each project (select or not)
        self.action_space = spaces.MultiBinary(self.state['available_projects']) # Changed from Number of projects high to available projects
        # Observation space: Budgets for each year, projects with ID, value, and costs
        self.observation_space = spaces.Dict({
            "budgets": 
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(self.init_dict['years'],),
                       dtype=np.float32),
            "projects": 
            spaces.Box(low=0,
                       high=np.inf,
                       shape=(len(self.state['available_projects']), 2 + self.max_val), # Changed from Number of projects high to available projects
                       dtype=np.float32)
        })

    def reset(self):
        """Reset the environment to its initial state and generate a new set of projects for initial selections.

        Parameters
        ----------
        None

        Returns
        -------
        state (dict)
            The state of the environment from the initial state. This includes the current year, budgets, available and picked projects, and cumulative reward. In other words, this is the environment in year 0.
        
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
        """Generate a new set of projects based on the environment's initial configuration.

        This method updates the environment's state by adding newly generated projects
        to the list of available projects. Each project is generated with attributes
        according to predefined probabilities for value and duration and random costs.
        """
        slots = self.init_dict['num_slots']
        slot_prob = self.init_dict['slot_probability']

        # Define the cost distribution based on project duration
        cost_distributions = {
            5: [0.1, 0.2, 0.4, 0.2, 0.1],
            6: [0.05, 0.15, 0.3, 0.3, 0.15, 0.05],
            7: [0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05]
        }

        for _ in range(slots):
            if self.rng.random() < slot_prob:
                project_id = self.state['project_id_counter']
                self.state['project_id_counter'] += 1

                project_value = self.rng.choice(self.init_dict['project_values'],
                                                p=self.init_dict['project_probabilities'])
                project_duration = self.rng.choice(self.init_dict['project_duration_choices'],
                                                p=self.init_dict['project_duration_probabilities'])
                triangle = self.rng.triangular(self.init_dict['minimum_cost'], self.init_dict['average_cost'], self.init_dict['maximum_cost'])
                total_cost = round(triangle / 1000000, 2)

                # Generate project costs based on the specified distribution
                if project_duration in cost_distributions:
                    distribution = cost_distributions[project_duration]
                    project_costs = [round(total_cost * proportion, 2) for proportion in distribution]
                else:
                    # Default to equal split if duration not specified
                    project_costs = [round(total_cost / project_duration, 2)] * project_duration

                project_costs.extend([0] * (self.max_val - project_duration))  # Fill the rest with 0

                project = [project_id, project_value] + project_costs
                self.state['available_projects'].append(project)


    def _update_observation(self):
        """Update the observable state of the environment with all relevant information.

        This method is used internally to ensure the environment's state is correctly
        updated after actions are taken or when the environment is reset.
        """
        self.state = {
        "current_year": self.state['current_year'],
        "project_id_counter": self.state['project_id_counter'],
        "cumulative_reward": self.state['cumulative_reward'],
        "budgets": self.state['budgets'],
        "available_projects": self.state['available_projects'],
        "picked_projects_ids": self.state['picked_projects_ids']
        }


    def step(self, action):
        """Execute one step in the environment using the provided action.

        Parameters
        ----------
        action : list of int
            A list indicating the selection (1) or rejection (0) of each available project.

        Returns
        -------
        state (dict)
            The updated state of the environment after executing the action.
        reward (int)
            The reward obtained after executing the action.
        done (bool)
            A flag indicating if the simulation has reached its conclusion (True) or not (False).
        """
        ## DOES NOT DISTRIBUTE LEFTOVER COST
        reward = 0
        projects_to_carry_over = []

        for index, decision in enumerate(action):
            project = self.state['available_projects'][index]
            project_id, project_value, *project_costs = project

            if decision == 1:  # If the project is selected
                # Deduct project costs
                for year_offset, cost in enumerate(project_costs):
                    if self.state['current_year'] + year_offset < self.init_dict['years']:
                        self.state['budgets'][self.state['current_year'] + year_offset] -= cost

                reward += project_value
                self.state['picked_projects_ids'].append(project_id)
            else:
                # Carry over the project to the next year if not selected
                projects_to_carry_over.append(project)
        self.state['cumulative_reward']+=reward
        # Update available projects for the next year
        self.state['available_projects'] = projects_to_carry_over

        # Decay the all project values by a factor of 1 / 1.1 each year
        for i in self.state['available_projects']:
            i[1] /= 1.1

        self.state['current_year'] += 1
        done = self.state['current_year'] >= self.init_dict['years']

        # Generate new projects only if it's not the last year
        if self.state['current_year'] < self.init_dict['years'] - 1:
            self._generate_new_projects()

        # Apply random budget changes for all future years
        budget_changes = np.arange(-0.10, 0.11, 0.01)
        for year in range(self.state['current_year'], self.init_dict['years']):
            self.state['budgets'][year] = round(
                self.state['budgets'][year] * (1 + self.rng.choice(budget_changes)), 2
            )

        self._update_observation()

        return self.state, reward, done

    def render(self, mode='console', reward=0):
        """Render the current state of the environment to the console.

        Parameters
        ----------
        mode : str, optional
            The mode of rendering. Currently, only 'console' is supported, which prints
            the state information to the console. Default is 'console'.
        reward : int, optional
            The reward obtained in the last step. Default is 0.
        """
        if mode == 'console':
            # print(f"Year: {self.state['current_year']}, \n")
            #print(f"Picked Projects IDs: \n{self.state['picked_projects_ids']}\n")

            # print("Available Projects:")
            # for project in self.state['available_projects']:
            #     print(project)
            print(f"Reward: \n{reward}")
            print(f"Cumulative Reward: {self.state['cumulative_reward']}")
            print()