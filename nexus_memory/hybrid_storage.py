import chromadb
import networkx as nx
import hashlib
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class ChainArchive:
    """Tamper-evident linear storage using blockchain-like hashing."""
    
    def __init__(self, storage_path: str = "chain_archive.json"):
        self.storage_path = storage_path
        self.chain = []
        self.load()
    
    def load(self):
        try:
            with open(self.storage_path, 'r') as f:
                self.chain = json.load(f)
        except FileNotFoundError:
            self.chain = []
    
    def save(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.chain, f, indent=2)
    
    def add_block(self, data: Dict[str, Any]) -> str:
        prev_hash = self.chain[-1]['hash'] if self.chain else '0' * 64
        block = {
            'index': len(self.chain),
            'timestamp': datetime.utcnow().isoformat(),
            'data': data,
            'prev_hash': prev_hash,
            'hash': self._compute_hash(prev_hash, data)
        }
        self.chain.append(block)
        self.save()
        return block['hash']
    
    def _compute_hash(self, prev_hash: str, data: Dict) -> str:
        content = prev_hash + json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_integrity(self) -> Tuple[bool, Optional[int]]:
        for i, block in enumerate(self.chain):
            if i == 0:
                expected_prev = '0' * 64
            else:
                expected_prev = self.chain[i-1]['hash']
            if block['prev_hash'] != expected_prev:
                return False, i
            if block['hash'] != self._compute_hash(block['prev_hash'], block['data']):
                return False, i
        return True, None
    
    def get_chain(self) -> List[Dict]:
        return self.chain


class KnowledgeGraph:
    """NetworkX-based graph for entity relationships."""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
    
    def add_entity(self, entity_id: str, attributes: Dict[str, Any] = None):
        self.graph.add_node(entity_id, **(attributes or {}))
    
    def add_relation(self, from_entity: str, to_entity: str, relation_type: str, metadata: Dict = None):
        self.graph.add_edge(from_entity, to_entity, type=relation_type, metadata=metadata or {})
    
    def get_entity(self, entity_id: str) -> Dict:
        return dict(self.graph.nodes[entity_id]) if entity_id in self.graph else {}
    
    def get_neighbors(self, entity_id: str, relation_type: str = None) -> List[Tuple[str, str, Dict]]:
        neighbors = []
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if relation_type is None or data.get('type') == relation_type:
                neighbors.append((entity_id, target, data))
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            if relation_type is None or data.get('type') == relation_type:
                neighbors.append((source, entity_id, data))
        return neighbors
    
    def get_all_entities(self) -> List[str]:
        return list(self.graph.nodes)
    
    def get_graph_data(self) -> Dict:
        nodes = [{'id': n, **self.graph.nodes[n]} for n in self.graph.nodes]
        edges = [{'from': u, 'to': v, 'type': d.get('type'), 'metadata': d.get('metadata')} 
                 for u, v, d in self.graph.edges(data=True)]
        return {'nodes': nodes, 'edges': edges}
    
    def export_to_json(self, path: str):
        data = nx.node_link_data(self.graph)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


class HybridStore:
    """Integrates ChromaDB, NetworkX, and ChainArchive."""
    
    def __init__(self, chroma_path: str = "./chroma_db", chain_path: str = "chain_archive.json"):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection("memories")
        self.knowledge_graph = KnowledgeGraph()
        self.chain_archive = ChainArchive(chain_path)
    
    def add_memory(self, text: str, metadata: Dict = None) -> str:
        metadata = metadata or {}
        memory_id = hashlib.sha256(f"{text}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        
        # Store in ChromaDB
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[memory_id]
        )
        
        # Extract entities and add to knowledge graph
        entities = self._extract_entities(text)
        for ent in entities:
            self.knowledge_graph.add_entity(ent['id'], ent.get('attributes'))
        
        # Add relations between entities
        relations = self._extract_relations(text, entities)
        for rel in relations:
            self.knowledge_graph.add_relation(rel['from'], rel['to'], rel['type'], rel.get('metadata'))
        
        # Store in tamper-evident chain
        block_data = {
            'memory_id': memory_id,
            'text': text,
            'metadata': metadata,
            'entities': entities,
            'relations': relations,
            'timestamp': datetime.utcnow().isoformat()
        }
        chain_hash = self.chain_archive.add_block(block_data)
        
        return memory_id
    
    def search_memories(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        memories = []
        for i, doc in enumerate(results['documents'][0]):
            memories.append({
                'id': results['ids'][0][i],
                'text': doc,
                'metadata': results['metadatas'][0][i]
            })
        return memories
    
    def get_entity_relations(self, entity_id: str) -> List[Tuple[str, str, Dict]]:
        return self.knowledge_graph.get_neighbors(entity_id)
    
    def get_all_entities(self) -> List[str]:
        return self.knowledge_graph.get_all_entities()
    
    def verify_chain_integrity(self) -> Tuple[bool, Optional[int]]:
        return self.chain_archive.verify_integrity()
    
    def get_chain_history(self) -> List[Dict]:
        return self.chain_archive.get_chain()
    
    def export_graph(self, path: str):
        self.knowledge_graph.export_to_json(path)
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Simple rule-based entity extraction. Override for NLP integration."""
        entities = []
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entity_id = word.strip('.,!?')
                entities.append({'id': entity_id, 'attributes': {'type': 'UNKNOWN'}})
        return list({e['id']: e for e in entities}.values())
    
    def _extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Simple co-occurrence relation extraction."""
        relations = []
        entity_ids = [e['id'] for e in entities]
        if len(entity_ids) >= 2:
            for i in range(len(entity_ids)-1):
                relations.append({
                    'from': entity_ids[i],
                    'to': entity_ids[i+1],
                    'type': 'CO_OCCURS',
                    'metadata': {'context': text[:100]}
                })
        return relations