"""
Functors: Structure-preserving maps between categories

A functor F: C → D consists of:
- Object mapping: F(A) for each object A in C
- Morphism mapping: F(f: A → B) for each morphism f in C

Must preserve:
- Identity: F(id_A) = id_F(A)
- Composition: F(g ∘ f) = F(g) ∘ F(f)
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional
from dataclasses import dataclass
from vajra_bm25.categorical.category import Morphism


A = TypeVar('A')
B = TypeVar('B')
FA = TypeVar('FA')  # F(A)
FB = TypeVar('FB')  # F(B)


class Functor(ABC, Generic[A, FA]):
    """
    A functor maps objects and morphisms from one category to another,
    preserving categorical structure.
    """

    @abstractmethod
    def fmap_object(self, a: A) -> FA:
        """Map an object A to F(A)"""
        pass

    @abstractmethod
    def fmap_morphism(self, f: Morphism[A, B]) -> Morphism[FA, FB]:
        """Map a morphism f: A → B to F(f): F(A) → F(B)"""
        pass


class ListFunctor(Functor[A, List[A]]):
    """
    The List functor: maps objects to lists of that object

    This captures nondeterministic or branching behavior:
    - One state can unfold into multiple possible next states
    - List represents all possible futures
    """

    def fmap_object(self, a: A) -> List[A]:
        """Wrap a single object in a list"""
        return [a]

    def fmap_morphism(self, f: Morphism[A, B]) -> Morphism[List[A], List[B]]:
        """Lift a morphism to work on lists"""
        from vajra_bm25.categorical.category import FunctionMorphism
        return FunctionMorphism(lambda xs: [f.apply(x) for x in xs])


class MaybeFunctor(Functor[A, Optional[A]]):
    """
    The Maybe functor: captures computation that might fail

    F(A) = A ∪ {None}
    Useful for search paths that might terminate or fail
    """

    def fmap_object(self, a: A) -> Optional[A]:
        """Wrap object in Maybe (Some)"""
        return a

    def fmap_morphism(self, f: Morphism[A, B]) -> Morphism[Optional[A], Optional[B]]:
        """Lift morphism to work on Maybe values"""
        from vajra_bm25.categorical.category import FunctionMorphism

        def maybe_apply(ma: Optional[A]) -> Optional[B]:
            if ma is None:
                return None
            return f.apply(ma)

        return FunctionMorphism(maybe_apply)


@dataclass
class TreeFunctor(Functor[A, 'Tree[A]']):
    """
    Tree functor: captures hierarchical branching

    Perfect for search trees where each state unfolds into a tree of possibilities
    """

    def fmap_object(self, a: A) -> 'Tree[A]':
        """Wrap object as a leaf"""
        return Tree(a, [])

    def fmap_morphism(self, f: Morphism[A, B]) -> Morphism['Tree[A]', 'Tree[B]']:
        """Lift morphism to work on trees"""
        from vajra_bm25.categorical.category import FunctionMorphism

        def tree_map(ta: 'Tree[A]') -> 'Tree[B]':
            return Tree(
                value=f.apply(ta.value),
                children=[tree_map(child) for child in ta.children]
            )

        return FunctionMorphism(tree_map)


@dataclass
class Tree(Generic[A]):
    """A tree data structure"""
    value: A
    children: List['Tree[A]']
