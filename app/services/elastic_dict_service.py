"""
Service for managing the Elastic Dictionary
"""
import os
from typing import List, Dict, Any, Optional, Tuple
import networkx as nx
import matplotlib.colors as mcolors

from elastic_dict import ElasticDictionary, Node
from app.core.config import settings
from app.models.elastic_dict_models import (
    NodeModel, GraphDataModel, SearchResult, DictionaryStateResponse
)


class ElasticDictionaryService:
    """Service for managing the Elastic Dictionary"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance of the dictionary exists"""
        if cls._instance is None:
            cls._instance = super(ElasticDictionaryService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the dictionary"""
        # Check if a saved dictionary exists
        if os.path.exists(settings.DICTIONARY_SAVE_PATH):
            try:
                self.dictionary = ElasticDictionary.load(settings.DICTIONARY_SAVE_PATH)
                print(f"Loaded dictionary from {settings.DICTIONARY_SAVE_PATH}")
            except Exception as e:
                print(f"Error loading dictionary: {e}")
                self.dictionary = ElasticDictionary(model_name=settings.DICTIONARY_MODEL)
        else:
            self.dictionary = ElasticDictionary(model_name=settings.DICTIONARY_MODEL)
            print(f"Created new dictionary with model {settings.DICTIONARY_MODEL}")
    
    def add_item(self, item: str) -> NodeModel:
        """Add a single item to the dictionary"""
        node = self.dictionary.add(item)
        self._save_dictionary()
        return self._node_to_model(node)
    
    def add_batch(self, items: List[str]) -> List[NodeModel]:
        """Add multiple items to the dictionary"""
        nodes = self.dictionary.add_batch(items)
        self._save_dictionary()
        return [self._node_to_model(node) for node in nodes]
    
    def add_paragraph(self, paragraph: str) -> List[NodeModel]:
        """Add a paragraph to the dictionary"""
        nodes = self.dictionary.add_paragraph(paragraph)
        self._save_dictionary()
        return [self._node_to_model(node) for node in nodes]
    
    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search the dictionary for items related to the query"""
        results = self.dictionary.find(query)
        
        # Convert to SearchResult models
        search_results = []
        for node, similarity in results[:limit]:
            search_results.append(
                SearchResult(
                    key=node.key,
                    value=node.value,
                    similarity=similarity
                )
            )
        
        return search_results
    
    def get_dictionary_state(self) -> DictionaryStateResponse:
        """Get the current state of the dictionary"""
        node_count = len(self.dictionary.all_nodes)
        graph_data = self._generate_graph_data()
        
        return DictionaryStateResponse(
            node_count=node_count,
            graph_data=graph_data
        )
    
    def _generate_graph_data(self) -> GraphDataModel:
        """Generate graph data for visualization"""
        G = nx.DiGraph()
        node_depths = {}
        
        # Process nodes recursively
        def process_node(node, depth=0):
            if node.key not in node_depths:
                node_depths[node.key] = depth
                
                # Add node to graph
                G.add_node(
                    node.key,
                    name=node.key,
                    val=5 + (len(node.children) * 2),
                    is_category=node.is_category_node
                )
                
                # Process children
                for child in node.children:
                    # Add edge
                    G.add_edge(node.key, child.key)
                    process_node(child, depth + 1)
        
        # Start processing from root
        process_node(self.dictionary.root)
        
        # Generate color map based on depth
        max_depth = max(node_depths.values()) if node_depths else 0
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Create nodes and links for the graph data
        nodes = []
        for node_id in G.nodes():
            depth = node_depths.get(node_id, 0)
            color_idx = min(depth, len(colors) - 1)
            
            nodes.append({
                "id": node_id,
                "name": node_id,
                "val": G.nodes[node_id]["val"],
                "color": colors[color_idx],
                "is_category": G.nodes[node_id]["is_category"]
            })
        
        links = []
        for source, target in G.edges():
            links.append({
                "source": source,
                "target": target,
                "value": 1
            })
        
        return GraphDataModel(nodes=nodes, links=links)
    
    def _node_to_model(self, node: Node) -> NodeModel:
        """Convert a Node object to a NodeModel"""
        return NodeModel(
            key=node.key,
            value=node.value,
            children=[child.key for child in node.children],
            is_category_node=node.is_category_node
        )
    
    def _save_dictionary(self):
        """Save the dictionary to disk"""
        try:
            self.dictionary.save(settings.DICTIONARY_SAVE_PATH)
            print(f"Saved dictionary to {settings.DICTIONARY_SAVE_PATH}")
        except Exception as e:
            print(f"Error saving dictionary: {e}")


# Create a singleton instance
elastic_dict_service = ElasticDictionaryService() 