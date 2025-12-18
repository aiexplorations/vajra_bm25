"""
Coalgebras: The categorical structure for dynamics and unfolding

A coalgebra for a functor F is:
- A carrier object X (the state space)
- A structure map α: X → F(X) (how states unfold)

Coalgebras capture:
- Streams (infinite sequences)
- Transition systems
- Automata
- Search tree generation
- Any "productive" or "generative" process

The dual of algebras (which fold/consume), coalgebras unfold/produce.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Callable
from dataclasses import dataclass
from vajra_bm25.categorical.functor import Functor, ListFunctor, Tree, TreeFunctor


X = TypeVar('X')  # Carrier/state type
FX = TypeVar('FX')  # F(X)


class Coalgebra(ABC, Generic[X, FX]):
    """
    A coalgebra (X, α) where α: X → F(X)

    X is the state space
    α is the transition/unfolding function
    """

    @abstractmethod
    def structure_map(self, state: X) -> FX:
        """
        α: X → F(X)

        Given a state, produce its unfolding in the functor F.
        This is the heart of dynamics.
        """
        pass

    def unfold(self, initial_state: X, depth: int = 1) -> FX:
        """
        Unfold the coalgebra from an initial state.
        This generates the dynamics.
        """
        if depth <= 0:
            return initial_state  # type: ignore
        return self.structure_map(initial_state)

    def trajectory(self, initial_state: X, steps: int) -> List[X]:
        """
        Generate a trajectory of states by repeatedly applying the structure map.
        Note: This only works for deterministic coalgebras (would need modification for branching)
        """
        trajectory = [initial_state]
        current = initial_state

        for _ in range(steps):
            unfolded = self.structure_map(current)
            # For deterministic case, extract next state
            # This is a simplification; full categorical treatment would use corecursion
            current = unfolded  # type: ignore
            trajectory.append(current)

        return trajectory


@dataclass
class SearchCoalgebra(Coalgebra[X, List[X]]):
    """
    A coalgebra for search: X → List[X]

    Each state unfolds into a list of successor states.
    This is the categorical essence of search.

    The functor is List, capturing nondeterministic branching.
    """
    successor_function: Callable[[X], List[X]]

    def structure_map(self, state: X) -> List[X]:
        """
        α: State → List[State]

        Given current state, produce all possible next states.
        This is search space exploration as coalgebraic unfolding.
        """
        return self.successor_function(state)


@dataclass
class TreeSearchCoalgebra(Coalgebra[X, Tree[X]]):
    """
    A coalgebra for tree search: X → Tree[X]

    Each state unfolds into a tree of successor states.
    Captures hierarchical/recursive search structure.
    """
    successor_function: Callable[[X], List[X]]

    def structure_map(self, state: X) -> Tree[X]:
        """
        α: State → Tree[State]

        Unfold state into a tree with current state as root
        and successors as children.
        """
        successors = self.successor_function(state)
        # Create single-level tree (children are leaves for now)
        children = [Tree(value=succ, children=[]) for succ in successors]
        return Tree(value=state, children=children)

    def full_unfold(self, initial_state: X, max_depth: int) -> Tree[X]:
        """
        Fully unfold the search tree to a given depth.
        This is corecursion: the categorical generative process.
        """
        if max_depth <= 0:
            return Tree(value=initial_state, children=[])

        successors = self.successor_function(initial_state)
        children = [
            self.full_unfold(succ, max_depth - 1)
            for succ in successors
        ]

        return Tree(value=initial_state, children=children)


@dataclass
class ConditionalCoalgebra(Coalgebra[X, List[X]]):
    """
    A coalgebra with termination conditions.

    X → 1 + List[X]

    Either terminates (1) or continues with successors (List[X]).
    This is closer to real search with goal states.
    """
    successor_function: Callable[[X], List[X]]
    is_terminal: Callable[[X], bool]

    def structure_map(self, state: X) -> List[X]:
        """
        If terminal, return empty list (no successors).
        Otherwise, return successors.
        """
        if self.is_terminal(state):
            return []
        return self.successor_function(state)
