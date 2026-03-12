"""
Network Analyzer — builds entity co-occurrence graphs using NetworkX.
Generates data structures for frontend network visualization.
"""

from collections import Counter
from itertools import combinations
from typing import List, Dict, Optional
from .models import EntityResult, NetworkData, NetworkNode, NetworkEdge


class NetworkAnalyzer:
    """Builds co-occurrence networks from NER results."""

    def build_network(
        self,
        documents_entities: List[List[EntityResult]],
        min_frequency: int = 2,
        top_n_nodes: int = 50,
    ) -> NetworkData:
        """
        Build a co-occurrence network from entity results across multiple documents.

        Args:
            documents_entities: List of entity lists (one per document)
            min_frequency: Minimum entity frequency to include as a node
            top_n_nodes: Maximum number of nodes to include
        """
        # Count entity frequencies
        entity_counter = Counter()
        entity_types = {}

        for doc_entities in documents_entities:
            for ent in doc_entities:
                key = ent.word.strip()
                if key:
                    entity_counter[key] += 1
                    entity_types[key] = ent.entity_group

        # Filter by minimum frequency and take top N
        top_entities = {
            word for word, count in entity_counter.most_common(top_n_nodes)
            if count >= min_frequency
        }

        if not top_entities:
            return NetworkData()

        # Count co-occurrences (entities appearing in the same document)
        edge_counter = Counter()
        for doc_entities in documents_entities:
            doc_words = list({
                ent.word.strip() for ent in doc_entities
                if ent.word.strip() in top_entities
            })
            for a, b in combinations(sorted(doc_words), 2):
                edge_counter[(a, b)] += 1

        # Build nodes
        nodes = []
        for word in top_entities:
            nodes.append(NetworkNode(
                id=word,
                label=word,
                entity_type=entity_types.get(word, "MISC"),
                frequency=entity_counter[word],
            ))

        # Build edges
        edges = []
        for (source, target), weight in edge_counter.items():
            if weight >= 1:
                edges.append(NetworkEdge(
                    source=source,
                    target=target,
                    weight=weight,
                ))

        return NetworkData(nodes=nodes, edges=edges)

    def get_entity_stats(
        self, documents_entities: List[List[EntityResult]], top_n: int = 20
    ) -> Dict[str, List[Dict]]:
        """
        Get top entities by type (PER, ORG, LOC).

        Returns: {"PER": [{"word": ..., "count": ...}], "ORG": [...], ...}
        """
        by_type: Dict[str, Counter] = {}

        for doc_entities in documents_entities:
            for ent in doc_entities:
                etype = ent.entity_group
                if etype not in by_type:
                    by_type[etype] = Counter()
                by_type[etype][ent.word.strip()] += 1

        result = {}
        for etype, counter in by_type.items():
            result[etype] = [
                {"word": word, "count": count}
                for word, count in counter.most_common(top_n)
            ]

        return result
