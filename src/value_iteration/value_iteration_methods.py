from typing import Set,  Mapping, Tuple
import numpy as np

from src.util.typevars import MDPActions, MDPTransitions, MDPRewards, A, S
from src.value_iteration.core import extract_value_of_action, check_value_fuction_equivalence, get_influence_tree

def record_convergence(record, max_diffs, gt_vf, new_vf):
    if record:
        max_diffs.append(max(abs(gt_vf[state] - new_vf[state]) for state in gt_vf))
        return max_diffs
    else:
        return max_diffs


def value_iteration(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                    discount: float, vi_method: str = 'normal', k=None,
                    log_converence=False, gt_vf=None) -> Tuple[Mapping[S, float], int, list]:
    max_diffs = []
    next_value_function = {s: 0 for s in actions.keys()}
    base_value_function = None

    num_iter = 0
    if vi_method == 'normal':
        while base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function = iterate_on_value_function(actions, transitions, rewards, base_value_function,
                                                            discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
    elif vi_method == 'random-k':
        while base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function):
            num_iter += 1
            base_value_function = next_value_function

            next_value_function = random_k_iterate_on_value_function(actions, transitions, rewards, base_value_function,
                                                                     discount, k)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            if log_converence:
                base_value_function = gt_vf
    elif vi_method == 'influence-tree':
        influence_tree = get_influence_tree(transitions)
        next_states_to_update = set(actions.keys())
        while len(next_states_to_update) > 0:
            num_iter += 1
            base_value_function = next_value_function
            next_value_function, updated_states = iterate_on_value_function_specific_states(actions, transitions,
                                                                                            rewards,
                                                                                            base_value_function,
                                                                                            discount,
                                                                                            next_states_to_update)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            next_states_to_update = set()
            for state in updated_states:
                next_states_to_update.update(influence_tree[state])
    elif vi_method == 'cyclic-vi':
        while base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function = cycle_iterate_on_value_function(actions, transitions, rewards,
                                                                  base_value_function,
                                                                  discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
    elif vi_method == 'cyclic-vi-rp':
        while base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function = cycle_iterate_on_value_function_rp(actions, transitions, rewards,
                                                                     base_value_function,
                                                                     discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
    else:
        raise NotImplemented(f'have not implemented {vi_method} value iteration yet')
    return base_value_function, num_iter, max_diffs

def iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                              base_vf: Mapping[S, float], discount: float) -> Mapping[S, float]:
    new_vf = {}
    for s in actions.keys():
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf


def cycle_iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                    base_vf: Mapping[S, float], discount: float) -> Mapping[S, float]:
    new_vf = base_vf.copy()
    for s in actions.keys():
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, new_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf


def cycle_iterate_on_value_function_rp(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                       base_vf: Mapping[S, float], discount: float) -> Mapping[S, float]:
    new_vf = base_vf.copy()
    states = list(actions.keys())
    np.random.shuffle(states)
    for s in states:
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, new_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf


def random_k_iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                       base_vf: Mapping[S, float], discount: float, k: int) -> Mapping[S, float]:
    new_vf = {}
    states = list(actions.keys())
    states_to_update_idx = np.random.choice(range(len(states)), size=k)
    states_to_update = [states[idx] for idx in states_to_update_idx]
    for s in states_to_update:
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    for s in set(actions.keys()) - set(states_to_update):
        new_vf[s] = base_vf[s]
    return new_vf


def iterate_on_value_function_specific_states(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                              base_vf: Mapping[S, float], discount: float,
                                              states_to_update: Set[S]) -> Mapping[S, float]:
    new_vf = {}
    updated_states = set()
    states = list(actions.keys())
    for s in states_to_update:
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
        if abs(new_vf[s] - base_vf[s]) > 1e-8:
            updated_states.add(s)
    for s in set(actions.keys()) - set(states_to_update):
        new_vf[s] = base_vf[s]
    return new_vf, updated_states
