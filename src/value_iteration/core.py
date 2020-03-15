from typing import Tuple, Mapping, Set
from collections import defaultdict

from src.util.typevars import MDPActions, MDPTransitions, MDPRewards, A, S


def extract_value_of_action(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                            action: A, state: S, value_function, discount: float):
    return rewards[state][action] + discount * sum([p * value_function[s_prime]
                                                    for s_prime, p in
                                                    transitions[state][action].items()])


def check_value_fuction_equivalence(v1, v2, epsilon=1e-4) -> bool:
    assert v1.keys() == v2.keys(), "comparing policies with different state spaces"
    for state in v1:
        if not abs(v1[state] - v2[state]) <= epsilon:
            return False
    return True


def check_policy_equivalence(p1, p2) -> bool:
    assert p1.keys() == p2.keys(), "comparing policies with different state spaces"
    for state in p1:
        if p1[state] != p2[state]:
            return False
    return True


def get_greedy_policy(actions: MDPActions, transitions: MDPTransitions, rewards: MDPRewards,
                      value_function: Mapping[S, float], terminal_states: Set[S],
                      discount: float) -> Mapping[S, A]:
    policy = {}
    non_terminal_states = set(actions.keys()) - terminal_states
    for s in non_terminal_states:
        actions_rewards = {}
        for action in actions[s]:
            actions_rewards[action] = extract_value_of_action(actions, transitions, rewards,
                                                              action, s, value_function, discount)
        policy[s] = {(min(actions_rewards, key=actions_rewards.get), 1)}
    for s in terminal_states:
        policy[s] = {(list(actions[s])[0], 1)}
    return policy


def get_influence_tree(transitions) -> Mapping[S, Set[S]]:
    """
    returns a mapping from state to all states that depend on that state in bellman equantions
    """
    influence_tree = defaultdict(set)
    for state in transitions:
        for action in transitions[state]:
            for next_state in transitions[state][action]:
                influence_tree[next_state].add(state)
    return influence_tree
