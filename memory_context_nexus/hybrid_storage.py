"""Hybrid Storage layer integrating ChromaDB for vector search and NetworkX for knowledge graph relationships."""

import uuid
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import networkx as nx


class HybridStorage:
    """Manages simultaneous storage in vector DB and knowledge graph."""

    def __init__(self, chroma_path: str = "./chroma_db", graph_path: Optional[str] = None):
        """Initialize ChromaDB client and NetworkX graph."""
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=chroma_path
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            name="compressed_memories",
            metadata={"hnsw:space": "cosine"}
        )
        self.graph = nx.MultiDiGraph()
        self.graph_path = graph_path
        if graph_path:
            try:
                self.graph = nx.read_graphml(graph_path)
            except FileNotFoundError:
                pass

    def add_compressed_content(self, content: str, metadata: Dict[str, Any],
                               relations: List[Tuple[str, str, Optional[Dict]]] = None) -> str:
        """
        Add compressed content to both vector store and knowledge graph.

        Args:
            content: Compressed text content to store
            metadata: Additional metadata (timestamp, source, etc.)
            relations: List of (target_node_id, relation_type, edge_attrs)

        Returns:
            Unique ID for the stored node
        """
        node_id = str(uuid.uuid4())

        # Add to ChromaDB
        embedding = self._generate_embedding(content)
        self.collection.add(
            ids=[node_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )

        # Add to NetworkX graph
        self.graph.add_node(node_id, content=content, **metadata)

        # Add relations if provided
        if relations:
            for target, rel_type, edge_attrs in relations:
                if self.graph.has_node(target):
                    attrs = edge_attrs or {}
                    self.graph.add_edge(node_id, target, type=rel_type, **attrs)

        return node_id

    def hybrid_search(self, query: str, top_k: int = 5,
                      graph_depth: int = 2, vector_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and graph traversal.

        Args:
            query: Search query
            top_k: Number of results to return
            graph_depth: Depth of graph traversal from vector matches
            vector_weight: Weight for vector score (graph weight = 1 - vector_weight)

        Returns:
            List of result dictionaries with 'id', 'content', 'score', 'metadata'
        """
        # Vector search
        query_embedding = self._generate_embedding(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2
        )

        if not vector_results['ids'] or not vector_results['ids'][0]:
            return []

        # Build candidate set from vector matches and their graph neighbors
        candidate_nodes = set(vector_results['ids'][0])
        for node_id in list(candidate_nodes):
            neighbors = set()
            for depth in range(1, graph_depth + 1):
                for source in list(candidate_nodes if depth == 1 else neighbors):
                    neighbors.update(self.graph.successors(source))
                    neighbors.update(self.graph.predecessors(source))
            candidate_nodes.update(neighbors)

        # Score each candidate
        scores = []
        for node_id in candidate_nodes:
            # Vector similarity score
            if node_id in vector_results['ids'][0]:
                idx = vector_results['ids'][0].index(node_id)
                vector_score = 1.0 - vector_results['distances'][0][idx]  # Convert distance to similarity
            else:
                vector_score = 0.0

            # Graph centrality score (simple closeness to vector matches)
            graph_score = 0.0
            if node_id not in vector_results['ids'][0]:
                for match_id in vector_results['ids'][0]:
                    if nx.has_path(self.graph, node_id, match_id):
                        path_len = nx.shortest_path_length(self.graph, node_id, match_id)
                        graph_score += 1.0 / (path_len + 1)
            graph_score = min(1.0, graph_score / len(vector_results['ids'][0]))

            # Combined score
            combined = (vector_weight * vector_score) + ((1 - vector_weight) * graph_score)

            # Get node data
            node_data = self.graph.nodes[node_id]
            scores.append({
                'id': node_id,
                'content': node_data.get('content', ''),
                'score': combined,
                'metadata': {k: v for k, v in node_data.items() if k != 'content'}
            })

        # Sort and return top_k
        scores.sort(key=lambda x: x['score'], reverse=True)
        return scores[:top_k]

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate a simple deterministic embedding for demonstration.
        Replace with actual embedding model (e.g., sentence-transformers).
        """
        # Simplified: hash-based dummy embedding (for demo only)
        np.random.seed(hash(text) % 2**32)
        emb = np.random.randn(384).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        return emb.tolist()

    def get_node_relations(self, node_id: str) -> Dict[str, List[Dict]]:
        """Retrieve all incoming and outgoing relations for a node."""
        if not self.graph.has_node(node_id):
            return {'incoming': [], 'outgoing': []}

        outgoing = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            outgoing.append({'target': target, 'type': data.get('type'), 'data': data})

        incoming = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            incoming.append({'source': source, 'type': data.get('type'), 'data': data})

        return {'incoming': incoming, 'outgoing': outgoing}

    def persist(self):
        """Persist graph to disk if path provided."""
        if self.graph_path:
            nx.write_graphml(self.graph, self.graph_path)

    def delete_node(self, node_id: str):
        """Delete node from both stores."""
        self.collection.delete(ids=[node_id])
        self.graph.remove_node(node_id)
