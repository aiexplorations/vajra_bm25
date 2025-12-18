"""
Categorical Framework: Core Category Theory Abstractions

This module provides pure category theory primitives for building
categorical systems like coalgebraic search algorithms and morphism-based
transformations.

Core Concepts:
- Categories: Objects and morphisms with composition
- Functors: Structure-preserving maps between categories
- Coalgebras: Dynamics and unfolding (α: X → F(X))

Usage:
    from vajra_bm25.categorical import Morphism, Functor, Coalgebra
    from vajra_bm25.categorical import FunctionMorphism, ListFunctor
"""

# Category primitives
from vajra_bm25.categorical.category import (
    Morphism,
    ComposedMorphism,
    IdentityMorphism,
    FunctionMorphism,
)

# Functors
from vajra_bm25.categorical.functor import (
    Functor,
    ListFunctor,
    MaybeFunctor,
    TreeFunctor,
    Tree,
)

# Coalgebras
from vajra_bm25.categorical.coalgebra import (
    Coalgebra,
    SearchCoalgebra,
    TreeSearchCoalgebra,
    ConditionalCoalgebra,
)

__all__ = [
    # Category
    'Morphism',
    'ComposedMorphism',
    'IdentityMorphism',
    'FunctionMorphism',
    # Functors
    'Functor',
    'ListFunctor',
    'MaybeFunctor',
    'TreeFunctor',
    'Tree',
    # Coalgebras
    'Coalgebra',
    'SearchCoalgebra',
    'TreeSearchCoalgebra',
    'ConditionalCoalgebra',
]
