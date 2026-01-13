"""
HNSW Graph Data Structure

The Hierarchical Navigable Small World graph is a multi-layer structure:
- Layer 0: Contains all nodes with dense local connections
- Higher layers: Sparse subsets with long-range connections
- Entry point at the highest layer for fast navigation

Properties:
- Logarithmic expected search complexity
- High recall with tunable parameters
- Efficient incremental construction
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
import numpy as np
import math


@dataclass
class HNSWGraph:
    """
    Hierarchical Navigable Small World graph.

    Multi-layer structure where:
    - Layer 0: All nodes, dense local connections (M0 neighbors)
    - Layer L > 0: Sparse subset, long-range connections (M neighbors)
    - Entry point at highest layer

    Parameters:
        dimension: Vector dimensionality
        M: Max connections per node at layers > 0
        M0: Max connections at layer 0 (typically 2*M)
        ef_construction: Beam width during index construction
        ml: Level multiplier for random level assignment
    """

    dimension: int
    M: int = 16
    M0: int = 32
    ef_construction: int = 200
    ml: float = field(default=None)

    # Node data
    vectors: Optional[np.ndarray] = field(default=None, repr=False)
    ids: List[str] = field(default_factory=list)
    id_to_idx: Dict[str, int] = field(default_factory=dict, repr=False)

    # Graph structure: layers[level][node_idx] = List[neighbor_idx]
    layers: List[Dict[int, List[int]]] = field(default_factory=list, repr=False)

    # Entry point
    entry_point: int = -1
    max_level: int = -1

    def __post_init__(self):
        if self.ml is None:
            self.ml = 1.0 / math.log(self.M)
        if self.M0 is None:
            self.M0 = 2 * self.M

    def get_random_level(self) -> int:
        """
        Sample level with exponential decay.

        P(level = l) proportional to exp(-l / ml)

        Most nodes are at level 0, with exponentially fewer at higher levels.
        This creates the hierarchical structure.
        """
        return int(-math.log(np.random.random()) * self.ml)

    def get_neighbors(self, node_idx: int, level: int) -> List[int]:
        """Get neighbors of node at given level"""
        if level >= len(self.layers):
            return []
        return self.layers[level].get(node_idx, [])

    def set_neighbors(self, node_idx: int, level: int, neighbors: List[int]) -> None:
        """Set neighbors for a node at a level"""
        while len(self.layers) <= level:
            self.layers.append({})
        self.layers[level][node_idx] = neighbors

    def add_edge(self, node_a: int, node_b: int, level: int) -> None:
        """
        Add bidirectional edge between nodes at level.

        Does not check for duplicates or enforce M limit (caller's responsibility).
        """
        while len(self.layers) <= level:
            self.layers.append({})

        if node_a not in self.layers[level]:
            self.layers[level][node_a] = []
        if node_b not in self.layers[level]:
            self.layers[level][node_b] = []

        # Add edges
        if node_b not in self.layers[level][node_a]:
            self.layers[level][node_a].append(node_b)
        if node_a not in self.layers[level][node_b]:
            self.layers[level][node_b].append(node_a)

    def get_max_neighbors(self, level: int) -> int:
        """Get max neighbors allowed at this level"""
        return self.M0 if level == 0 else self.M

    def node_level(self, node_idx: int) -> int:
        """Get the maximum level at which this node exists"""
        for level in range(len(self.layers) - 1, -1, -1):
            if node_idx in self.layers[level]:
                return level
        return -1

    def num_nodes(self) -> int:
        """Total number of nodes in the graph"""
        return len(self.ids)

    def num_edges(self, level: Optional[int] = None) -> int:
        """
        Count edges at a level or total.

        Each edge is counted once (divides bidirectional count by 2).
        """
        if level is not None:
            if level >= len(self.layers):
                return 0
            total = sum(len(neighbors) for neighbors in self.layers[level].values())
            return total // 2

        return sum(self.num_edges(l) for l in range(len(self.layers)))

    def stats(self) -> Dict:
        """Get graph statistics"""
        level_stats = []
        for level, layer in enumerate(self.layers):
            if not layer:
                continue
            degrees = [len(neighbors) for neighbors in layer.values()]
            level_stats.append(
                {
                    "level": level,
                    "nodes": len(layer),
                    "edges": sum(degrees) // 2,
                    "avg_degree": np.mean(degrees) if degrees else 0,
                    "max_degree": max(degrees) if degrees else 0,
                }
            )

        return {
            "dimension": self.dimension,
            "total_nodes": len(self.ids),
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "M": self.M,
            "M0": self.M0,
            "ef_construction": self.ef_construction,
            "levels": level_stats,
        }

    def __repr__(self) -> str:
        return (
            f"HNSWGraph(dimension={self.dimension}, nodes={len(self.ids)}, "
            f"levels={self.max_level + 1}, M={self.M})"
        )
