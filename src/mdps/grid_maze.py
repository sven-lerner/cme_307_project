import numpy as np


class RandomGridMaze:

    def __init__(self, x_dim, y_dim, num_terminal, num_actions=8, seed=None, deterministic=True, dense_rewards=False):
        self.state_actions, self.state_transitions, self.state_rewards, self.terminal_states = \
            self._get_maze(x_dim, y_dim, num_terminal, num_actions=num_actions, seed=seed, deterministic=deterministic,
                           dense_rewards=dense_rewards)

    def get_mdp(self):
        return self.state_actions, self.state_transitions, self.state_rewards, self.terminal_states

    def _get_maze(self, x_dim, y_dim, num_terminal, num_actions=8, seed=None, deterministic=True,
                  dense_rewards=False):
        if seed is not None:
            np.random.seed(seed)
        states = [(x, y) for x in range(x_dim) for y in range(y_dim)]
        terminal_states_idx = np.random.choice(list(range(len(states))), size=num_terminal)
        terminal_states = [states[idx] for idx in terminal_states_idx]

        if num_actions ==1:
            actions = {'l'}
        elif num_actions ==2:
            actions = {'l', 'r'}
        elif num_actions ==4:
            actions = {'l', 'r', 'u', 'd'}
        elif num_actions ==8:
            actions = {'l', 'r', 'u', 'd', 'lu', 'ld', 'ru', 'rd'}

        state_transitions = {}
        state_actions = {}
        state_rewards = {}
        for state in states:
            state_transitions[state] = {}
            state_rewards[state] = {}
            if state in terminal_states:
                state_actions[state] = {'stay'}
            else:
                state_actions[state] = actions
            if deterministic:
                for action in state_actions[state]:
                    next_coord = None
                    if action == 'stay':
                        next_coord = state
                    if action == 'l':
                        next_coord = (state[0], max(0, state[1] - 1))
                    if action == 'r':
                        next_coord = (state[0], min(x_dim - 1, state[1] + 1))
                    if action == 'u':
                        next_coord = (min(y_dim - 1, state[0] + 1), state[1])
                    if action == 'd':
                        next_coord = (max(0, state[0] - 1), state[1])
                    if action == 'lu':
                        next_coord = (min(y_dim - 1, state[0] + 1), max(0, state[1] - 1))
                    if action == 'ru':
                        next_coord = (min(y_dim - 1, state[0] + 1), min(x_dim - 1, state[1] + 1))
                    if action == 'ld':
                        next_coord = (max(0, state[0] - 1), max(0, state[1] - 1))
                    if action == 'rd':
                        next_coord = (max(0, state[0] - 1),  min(x_dim - 1, state[1] + 1))
                    state_transitions[state][action] = {next_coord: 1}
                    if next_coord != state and next_coord in terminal_states:
                        state_rewards[state][action] = np.random.choice([-10, 0, 10])
                    else:
                        if not dense_rewards:
                            state_rewards[state][action] = 0
                        else:
                            state_rewards[state][action] = np.random.rand()
            else:
                raise NotImplemented('have not implemented non-deterministic transitions')
        return state_actions, state_transitions, state_rewards, terminal_states
