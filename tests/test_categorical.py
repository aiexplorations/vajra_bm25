"""
Tests for categorical abstractions: Functors, Coalgebras, and Morphisms.

Tests cover:
- Functor laws (identity and composition preservation)
- Coalgebra structure maps
- Morphism composition
- Search coalgebra behavior
"""

import pytest

from vajra_bm25.categorical import (
    Morphism,
    FunctionMorphism,
    IdentityMorphism,
    Functor,
    ListFunctor,
    Coalgebra,
    SearchCoalgebra,
)
from vajra_bm25.categorical.functor import MaybeFunctor, TreeFunctor, Tree
from vajra_bm25.categorical.coalgebra import TreeSearchCoalgebra, ConditionalCoalgebra


# ============================================================================
# Morphism Tests
# ============================================================================

class TestMorphism:
    """Tests for morphism composition."""

    def test_function_morphism_apply(self):
        """Test basic function morphism application."""
        double = FunctionMorphism(lambda x: x * 2)

        assert double.apply(5) == 10
        assert double.apply(0) == 0
        assert double.apply(-3) == -6

    def test_morphism_composition(self):
        """Test morphism composition (f >> g)."""
        add_one = FunctionMorphism(lambda x: x + 1)
        double = FunctionMorphism(lambda x: x * 2)

        # (x + 1) * 2
        composed = add_one >> double

        assert composed.apply(5) == 12  # (5 + 1) * 2 = 12
        assert composed.apply(0) == 2   # (0 + 1) * 2 = 2

    def test_morphism_composition_associativity(self):
        """Test that morphism composition is associative: (f >> g) >> h = f >> (g >> h)."""
        f = FunctionMorphism(lambda x: x + 1)
        g = FunctionMorphism(lambda x: x * 2)
        h = FunctionMorphism(lambda x: x - 3)

        left = (f >> g) >> h
        right = f >> (g >> h)

        for x in [0, 5, -10, 100]:
            assert left.apply(x) == right.apply(x)

    def test_identity_morphism(self):
        """Test identity morphism."""
        identity = IdentityMorphism()

        assert identity.apply(5) == 5
        assert identity.apply("hello") == "hello"
        assert identity.apply([1, 2, 3]) == [1, 2, 3]

    def test_identity_left_law(self):
        """Test left identity law: id >> f = f."""
        identity = IdentityMorphism()
        f = FunctionMorphism(lambda x: x * 2)

        composed = identity >> f

        for x in [0, 5, -10, 100]:
            assert composed.apply(x) == f.apply(x)

    def test_identity_right_law(self):
        """Test right identity law: f >> id = f."""
        identity = IdentityMorphism()
        f = FunctionMorphism(lambda x: x * 2)

        composed = f >> identity

        for x in [0, 5, -10, 100]:
            assert composed.apply(x) == f.apply(x)

    def test_morphism_with_complex_types(self):
        """Test morphisms with complex types."""
        to_list = FunctionMorphism(lambda x: [x])
        length = FunctionMorphism(lambda xs: len(xs))

        composed = to_list >> length

        assert composed.apply(5) == 1
        assert composed.apply("hello") == 1


# ============================================================================
# Functor Tests
# ============================================================================

class TestListFunctor:
    """Tests for ListFunctor."""

    def test_fmap_object_wraps_in_list(self):
        """Test that fmap_object wraps value in list."""
        functor = ListFunctor()

        assert functor.fmap_object(5) == [5]
        assert functor.fmap_object("hello") == ["hello"]
        assert functor.fmap_object(None) == [None]

    def test_fmap_morphism_lifts_function(self):
        """Test that fmap_morphism lifts a morphism to work on lists."""
        functor = ListFunctor()
        double = FunctionMorphism(lambda x: x * 2)

        lifted = functor.fmap_morphism(double)

        assert lifted.apply([1, 2, 3]) == [2, 4, 6]
        assert lifted.apply([]) == []

    def test_functor_identity_law(self):
        """Test functor preserves identity: F(id) = id."""
        functor = ListFunctor()
        identity = IdentityMorphism()

        lifted_id = functor.fmap_morphism(identity)

        test_list = [1, 2, 3, 4, 5]
        assert lifted_id.apply(test_list) == test_list

    def test_functor_composition_law(self):
        """Test functor preserves composition: F(g . f) = F(g) . F(f)."""
        functor = ListFunctor()
        f = FunctionMorphism(lambda x: x + 1)
        g = FunctionMorphism(lambda x: x * 2)

        # F(g . f)
        composed = f >> g
        lifted_composed = functor.fmap_morphism(composed)

        # F(g) . F(f)
        lifted_f = functor.fmap_morphism(f)
        lifted_g = functor.fmap_morphism(g)
        sequential = lifted_f >> lifted_g

        test_list = [1, 2, 3]
        assert lifted_composed.apply(test_list) == sequential.apply(test_list)


class TestMaybeFunctor:
    """Tests for MaybeFunctor."""

    def test_fmap_object_returns_value(self):
        """Test that fmap_object wraps value (Some)."""
        functor = MaybeFunctor()

        # For MaybeFunctor, fmap_object just returns the value
        assert functor.fmap_object(5) == 5

    def test_fmap_morphism_applies_to_some(self):
        """Test fmap_morphism applies function to Some value."""
        functor = MaybeFunctor()
        double = FunctionMorphism(lambda x: x * 2)

        lifted = functor.fmap_morphism(double)

        assert lifted.apply(5) == 10

    def test_fmap_morphism_propagates_none(self):
        """Test fmap_morphism propagates None."""
        functor = MaybeFunctor()
        double = FunctionMorphism(lambda x: x * 2)

        lifted = functor.fmap_morphism(double)

        assert lifted.apply(None) is None


class TestTreeFunctor:
    """Tests for TreeFunctor."""

    def test_fmap_object_creates_leaf(self):
        """Test that fmap_object creates a leaf tree."""
        functor = TreeFunctor()

        tree = functor.fmap_object(5)

        assert isinstance(tree, Tree)
        assert tree.value == 5
        assert tree.children == []

    def test_fmap_morphism_maps_tree(self):
        """Test fmap_morphism maps over tree structure."""
        functor = TreeFunctor()
        double = FunctionMorphism(lambda x: x * 2)

        # Create a simple tree: root=1, children=[2, 3]
        tree = Tree(value=1, children=[
            Tree(value=2, children=[]),
            Tree(value=3, children=[])
        ])

        lifted = functor.fmap_morphism(double)
        result = lifted.apply(tree)

        assert result.value == 2  # 1 * 2
        assert len(result.children) == 2
        assert result.children[0].value == 4  # 2 * 2
        assert result.children[1].value == 6  # 3 * 2


# ============================================================================
# Coalgebra Tests
# ============================================================================

class TestSearchCoalgebra:
    """Tests for SearchCoalgebra."""

    def test_structure_map_returns_successors(self):
        """Test that structure_map returns successor states."""
        def successors(n):
            return [n + 1, n + 2] if n < 5 else []

        coalgebra = SearchCoalgebra(successor_function=successors)

        assert coalgebra.structure_map(0) == [1, 2]
        assert coalgebra.structure_map(4) == [5, 6]
        assert coalgebra.structure_map(5) == []

    def test_unfold_single_step(self):
        """Test unfold with depth=1."""
        def successors(n):
            return [n + 1, n + 2]

        coalgebra = SearchCoalgebra(successor_function=successors)

        result = coalgebra.unfold(0, depth=1)
        assert result == [1, 2]

    def test_search_coalgebra_with_complex_state(self):
        """Test SearchCoalgebra with complex state type."""
        def successors(state):
            x, y = state
            return [(x + 1, y), (x, y + 1)]

        coalgebra = SearchCoalgebra(successor_function=successors)

        result = coalgebra.structure_map((0, 0))
        assert result == [(1, 0), (0, 1)]


class TestTreeSearchCoalgebra:
    """Tests for TreeSearchCoalgebra."""

    def test_structure_map_creates_tree(self):
        """Test that structure_map creates a tree with successors as children."""
        def successors(n):
            return [n * 2, n * 2 + 1] if n < 4 else []

        coalgebra = TreeSearchCoalgebra(successor_function=successors)

        tree = coalgebra.structure_map(1)

        assert isinstance(tree, Tree)
        assert tree.value == 1
        assert len(tree.children) == 2
        assert tree.children[0].value == 2
        assert tree.children[1].value == 3

    def test_full_unfold_depth_0(self):
        """Test full_unfold with max_depth=0."""
        def successors(n):
            return [n + 1]

        coalgebra = TreeSearchCoalgebra(successor_function=successors)

        tree = coalgebra.full_unfold(0, max_depth=0)

        assert tree.value == 0
        assert tree.children == []

    def test_full_unfold_depth_2(self):
        """Test full_unfold with max_depth=2."""
        def successors(n):
            return [n + 1] if n < 10 else []

        coalgebra = TreeSearchCoalgebra(successor_function=successors)

        tree = coalgebra.full_unfold(0, max_depth=2)

        # Root: 0
        # Level 1: [1]
        # Level 2: [2] (child of 1)
        assert tree.value == 0
        assert len(tree.children) == 1
        assert tree.children[0].value == 1
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].value == 2

    def test_full_unfold_binary_tree(self):
        """Test full_unfold generating binary tree."""
        def binary_successors(n):
            return [2 * n, 2 * n + 1] if n < 4 else []

        coalgebra = TreeSearchCoalgebra(successor_function=binary_successors)

        tree = coalgebra.full_unfold(1, max_depth=2)

        # Root: 1
        # Level 1: [2, 3]
        # Level 2: [4, 5] under 2, [6, 7] under 3
        assert tree.value == 1
        assert len(tree.children) == 2
        assert tree.children[0].value == 2
        assert tree.children[1].value == 3


class TestConditionalCoalgebra:
    """Tests for ConditionalCoalgebra."""

    def test_terminal_state_returns_empty(self):
        """Test that terminal states return empty list."""
        def successors(n):
            return [n + 1]

        def is_terminal(n):
            return n >= 5

        coalgebra = ConditionalCoalgebra(
            successor_function=successors,
            is_terminal=is_terminal
        )

        assert coalgebra.structure_map(5) == []
        assert coalgebra.structure_map(10) == []

    def test_non_terminal_state_returns_successors(self):
        """Test that non-terminal states return successors."""
        def successors(n):
            return [n + 1]

        def is_terminal(n):
            return n >= 5

        coalgebra = ConditionalCoalgebra(
            successor_function=successors,
            is_terminal=is_terminal
        )

        assert coalgebra.structure_map(0) == [1]
        assert coalgebra.structure_map(4) == [5]

    def test_goal_based_search(self):
        """Test conditional coalgebra for goal-based search."""
        goal = 10

        def successors(n):
            return [n + 1, n + 2]

        def is_terminal(n):
            return n == goal

        coalgebra = ConditionalCoalgebra(
            successor_function=successors,
            is_terminal=is_terminal
        )

        # At goal, no successors
        assert coalgebra.structure_map(10) == []

        # Before goal, has successors
        assert coalgebra.structure_map(8) == [9, 10]


class TestCoalgebraTrajectory:
    """Tests for coalgebra trajectory generation."""

    def test_trajectory_generation(self):
        """Test generating trajectory of states."""
        # Simple deterministic successor (increment)
        coalgebra = SearchCoalgebra(successor_function=lambda n: n + 1)

        trajectory = coalgebra.trajectory(0, steps=3)

        # Note: trajectory method is simplified for deterministic case
        assert trajectory[0] == 0


class TestCategoricalIntegration:
    """Integration tests combining categorical concepts."""

    def test_morphism_with_functor(self):
        """Test using morphisms with functors."""
        # Score morphism: Document -> Float
        score = FunctionMorphism(lambda doc: len(doc) * 0.5)

        # Lift to lists
        list_functor = ListFunctor()
        batch_score = list_functor.fmap_morphism(score)

        docs = ["hello", "world", "category"]
        scores = batch_score.apply(docs)

        assert scores == [2.5, 2.5, 4.0]

    def test_coalgebra_unfold_for_search(self):
        """Test using coalgebra to model search unfolding."""
        # Simple search state: current position
        def search_successors(pos):
            # Can move left or right
            if 0 <= pos <= 10:
                return [pos - 1, pos + 1]
            return []

        coalgebra = SearchCoalgebra(successor_function=search_successors)

        # From position 5, can go to 4 or 6
        successors = coalgebra.structure_map(5)
        assert 4 in successors
        assert 6 in successors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
