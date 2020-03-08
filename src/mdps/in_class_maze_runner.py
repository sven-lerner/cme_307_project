"""
Maze Runner Problem from Lecture
"""


class BasicMazeRunner:

    def __init__(self):
        self.maze_runner_actions = {
            0: {'s', 'j'},
            1: {'s', 'j'},
            2: {'s', 'j'},
            3: {'s', 'j'},
            4: {'s', },
            5: {'stay'}
        }

        self.maze_runner_transitions = {
            0: {'s': {1: 1}, 'j': {2: 0.5, 3: 0.25, 4: 0.125, 5: 0.125}},
            1: {'s': {2: 1}, 'j': {3: 0.5, 4: 0.25, 5: 0.25}},
            2: {'s': {3: 1}, 'j': {4: 0.5, 5: 0.5}},
            #     3: {'s': {4: 1}, 'j': {4: 0.5, 5:0.5}},
            3: {'s': {4: 1}, 'j': {5: 1}},
            4: {'s': {5: 1}},
            5: {'stay': {5: 1}}
        }

        self.maze_runner_rewards = {
            0: {'s': 0, 'j': 0},
            1: {'s': 0, 'j': 0},
            2: {'s': 0, 'j': 0},
            3: {'s': 0, 'j': 0},
            4: {'s': 1},
            5: {'stay': 0}
        }

    def get_mdp(self):
        return self.maze_runner_actions, self.maze_runner_transitions, self.maze_runner_rewards
