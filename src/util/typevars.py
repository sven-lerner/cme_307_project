from typing import Mapping, TypeVar, Set

S = TypeVar('S')
A = TypeVar('A')
MDPTransitions = Mapping[S, Mapping[A, Mapping[S, float]]]
MDPActions = Mapping[S, Set[A]]
MDPRewards = Mapping[S, Mapping[A, float]]

