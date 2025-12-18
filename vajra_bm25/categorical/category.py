"""
Core Category Theory Primitives

A category consists of:
- Objects
- Morphisms (arrows between objects)
- Composition of morphisms
- Identity morphisms
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable, List, Tuple
from dataclasses import dataclass


# Type variables for objects
A = TypeVar('A')
B = TypeVar('B')
C = TypeVar('C')


class Morphism(ABC, Generic[A, B]):
    """
    A morphism (arrow) from object A to object B in a category.

    Morphisms are the fundamental structure-preserving maps.
    They must support composition and have identities.
    """

    @abstractmethod
    def apply(self, a: A) -> B:
        """Apply this morphism to an object"""
        pass

    def compose(self, other: 'Morphism[B, C]') -> 'Morphism[A, C]':
        """
        Categorical composition: if f: A → B and g: B → C,
        then g ∘ f: A → C
        """
        return ComposedMorphism(self, other)

    def __rshift__(self, other: 'Morphism[B, C]') -> 'Morphism[A, C]':
        """Operator for composition: f >> g means g ∘ f"""
        return self.compose(other)


@dataclass
class ComposedMorphism(Morphism[A, C]):
    """The composition of two morphisms"""
    first: Morphism[A, B]
    second: Morphism[B, C]

    def apply(self, a: A) -> C:
        return self.second.apply(self.first.apply(a))


@dataclass
class IdentityMorphism(Morphism[A, A]):
    """
    Identity morphism: id_A: A → A
    For any morphism f: A → B:
    - f ∘ id_A = f
    - id_B ∘ f = f
    """

    def apply(self, a: A) -> A:
        return a


@dataclass
class FunctionMorphism(Morphism[A, B]):
    """A morphism represented as a function"""
    func: Callable[[A], B]

    def apply(self, a: A) -> B:
        return self.func(a)
