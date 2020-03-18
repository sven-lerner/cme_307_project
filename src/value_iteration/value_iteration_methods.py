from typing import Set, Mapping, Tuple
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
                    discount: float, vi_method: str = 'normal', k=None, sample_action_size=None,
                    log_converence=False, gt_vf=None,
                    max_iter=1e4, tolerance=1e-4) -> Tuple[Mapping[S, float], int, list, list]:
    max_diffs = []
    updates_per_iter = []
    next_value_function = {s: 0 for s in actions.keys()}
    base_value_function = None
    num_iter = 0

    if vi_method == 'normal':
        while num_iter < max_iter and (base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function,
                                                    tolerance)):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function, updates = iterate_on_value_function(actions, transitions, rewards, base_value_function,
                                                                     discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            updates_per_iter.append(updates)
    elif vi_method == 'random-k':
        while num_iter < max_iter and (base_value_function is None or log_converence and max_diffs[-1] > tolerance
                                       or (not log_converence and
                                           not check_value_fuction_equivalence(base_value_function,
                                                                               next_value_function,
                                                                               tolerance))):
            num_iter += 1
            base_value_function = next_value_function

            next_value_function, updates = random_k_iterate_on_value_function(actions, transitions, rewards,
                                                                              base_value_function,
                                                                              discount, k)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            updates_per_iter.append(updates)

    elif vi_method == 'influence-tree':
        influence_tree = get_influence_tree(transitions)
        next_states_to_update = set(actions.keys())
        best_actions_by_state = {}
        if k is not None:
            next_states_to_update = list(next_states_to_update)
            next_sample_size = min(len(next_states_to_update), k)
            states_to_update_idx = np.random.choice(range(len(next_states_to_update)), size=next_sample_size,
                                                    replace=False)
            next_states_to_update = [next_states_to_update[idx] for idx in states_to_update_idx]
        while num_iter < max_iter and len(next_states_to_update) > 0 and (len(max_diffs) == 0 or max_diffs[-1] > tolerance):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function, updated_states = iterate_on_value_function_specific_states(actions, transitions,
                                                                                            rewards,
                                                                                            base_value_function,
                                                                                            discount,
                                                                                            next_states_to_update,
                                                                                            sample_action_size=sample_action_size,
                                                                                            best_actions_by_state=best_actions_by_state)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            updates_per_iter.append(len(next_states_to_update))
            next_states_to_update = set()
            if log_converence:
                base_value_function = gt_vf
            for state in updated_states:
                next_states_to_update.update(influence_tree[state])
            if k is not None:
                if len(next_states_to_update) == 0 and log_converence and max_diffs[-1] > tolerance:
                    next_states_to_update = list(actions.keys())
                else:
                    next_states_to_update = list(next_states_to_update)
                    if len(next_states_to_update) < k:
                        states = list(actions.keys())
                        extra_states_inds = np.random.choice(range(len(states)),
                                                                size=k-len(next_states_to_update),
                                                             replace=False)
                        next_states_to_update += [states[ind] for ind in extra_states_inds]

                next_sample_size = min(len(next_states_to_update), k)
                states_to_update_idx = np.random.choice(range(len(next_states_to_update)), size=next_sample_size,
                                                        replace=False)
                next_states_to_update = [next_states_to_update[idx] for idx in states_to_update_idx]

    elif vi_method == 'cyclic-vi':
        while num_iter < max_iter and (base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function,
                                                    tolerance)):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function, updates = cycle_iterate_on_value_function(actions, transitions, rewards,
                                                                           base_value_function,
                                                                           discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            updates_per_iter.append(updates)
    elif vi_method == 'cyclic-vi-rp':
        while num_iter < max_iter and (base_value_function is None or \
                not check_value_fuction_equivalence(base_value_function, next_value_function,
                                                    tolerance)):
            num_iter += 1
            base_value_function = next_value_function
            next_value_function, updates = cycle_iterate_on_value_function_rp(actions, transitions, rewards,
                                                                              base_value_function,
                                                                              discount)
            max_diffs = record_convergence(log_converence, max_diffs, gt_vf, next_value_function)
            updates_per_iter.append(updates)
    else:
        raise NotImplemented(f'have not implemented {vi_method} value iteration yet')
    return base_value_function, num_iter, max_diffs, updates_per_iter


def iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                              base_vf: Mapping[S, float], discount: float) -> Tuple[Mapping[S, float], float]:
    new_vf = {}
    for s in actions.keys():
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf, len(actions.keys())


def cycle_iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                    base_vf: Mapping[S, float], discount: float) -> Tuple[Mapping[S, float], float]:
    new_vf = base_vf.copy()
    for s in actions.keys():
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, new_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf, len(actions.keys())


def cycle_iterate_on_value_function_rp(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                       base_vf: Mapping[S, float], discount: float) -> Tuple[Mapping[S, float], float]:
    new_vf = base_vf.copy()
    states = list(actions.keys())
    np.random.shuffle(states)
    for s in states:
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, new_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf, len(actions.keys())


def random_k_iterate_on_value_function(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                       base_vf: Mapping[S, float], discount: float, k: int) -> Tuple[
    Mapping[S, float], float]:
    new_vf = {}
    states = list(actions.keys())
    states_to_update_idx = np.random.choice(range(len(states)), size=k, replace=False)
    states_to_update = [states[idx] for idx in states_to_update_idx]
    for s in states_to_update:
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in actions[s]]
        best_action_reward = min([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    for s in set(actions.keys()) - set(states_to_update):
        new_vf[s] = base_vf[s]
    return new_vf, k


def iterate_on_value_function_specific_states(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                                              base_vf: Mapping[S, float], discount: float,
                                              states_to_update: Set[S], sample_action_size=None,
                                              best_actions_by_state: dict = {}) -> Mapping[S, float]:
    new_vf = {}
    updated_states = set()
    if sample_action_size != None:
        num_s = len(states_to_update)
        random_pool = np.random.choice(np.arange(8), size=num_s * sample_action_size).reshape(num_s, sample_action_size)
    for i, s in enumerate(states_to_update):
        sampled_actions = actions[s]
        if sample_action_size != None and len(actions[s]) > 1:
            # action_idx = np.random.choice(range(len(actions[s])), size=sample_action_size, replace=False)
            action_idx = random_pool[i]
            sampled_actions = [list(actions[s])[idx] for idx in action_idx]
            if s in best_actions_by_state:
                sampled_actions += [best_actions_by_state[s]]
        action_values = [(action, extract_value_of_action(actions, transitions, rewards,
                                                          action, s, base_vf, discount)) for action in sampled_actions]
        best_action_reward, best_action = min([(x[1], x[0]) for x in action_values])
        best_actions_by_state[s] = best_action
        new_vf[s] = best_action_reward
        if abs(new_vf[s] - base_vf[s]) > 1e-5:
            updated_states.add(s)
    for s in set(actions.keys()) - set(states_to_update):
        new_vf[s] = base_vf[s]
    return new_vf, updated_states
