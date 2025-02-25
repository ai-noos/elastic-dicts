import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import sklearn
from packaging import version
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
import pickle
import os
from typing import List, Dict, Any, Optional, Union, Tuple
import json

class Node:
    """A node in the elastic dictionary tree."""
    
    def __init__(self, key: str, value: str = None, embedding=None):
        self.key = key
        self.value = value if value is not None else key
        self.embedding = embedding
        self.children = []
        self.parent = None
        self.similarity_threshold = 0.6  # Default threshold for similarity
        self.is_category_node = False  # Whether this node is a category (not a leaf)
    
    def add_child(self, child):
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def remove_child(self, child):
        """Remove a child node."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def __repr__(self):
        return f"Node({self.key})"


class ElasticDictionary:
    """
    An adaptive dictionary that organizes data into a hierarchical tree structure
    based on semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the ElasticDictionary.
        
        Args:
            model_name: The sentence-transformer model to use for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.root = Node("root")
        self.all_nodes = {self.root.key: self.root}
        self.restructure_threshold = 20  # Number of items before considering restructuring
        self.min_similarity_threshold = 0.5  # Minimum similarity to consider related
        self.max_similarity_threshold = 0.8  # Similarity threshold to consider same category
        self.enable_auto_restructure = True  # Whether to automatically restructure
        self.min_cluster_size = 4  # Minimum size for a cluster to form a category
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a text."""
        return self.model.encode(text, show_progress_bar=False)
    
    def _find_best_placement(self, embedding: np.ndarray, node: Node = None) -> Tuple[Node, float]:
        """
        Recursively find the best node placement for a new entry based on embedding similarity.
        
        Returns:
            Tuple of (best_node, similarity_score)
        """
        if node is None:
            node = self.root
        
        # If no children, return current node
        if not node.children:
            if node == self.root:
                return node, 0.0
            else:
                similarity = cosine_similarity([embedding], [node.embedding])[0][0]
                return node, similarity
        
        # Check similarity with current node (if not root)
        current_similarity = 0.0
        if node != self.root and node.embedding is not None:
            current_similarity = cosine_similarity([embedding], [node.embedding])[0][0]
        
        # Check similarity with children
        best_child = None
        best_similarity = 0.0
        
        for child in node.children:
            if child.embedding is not None:
                child_similarity = cosine_similarity([embedding], [child.embedding])[0][0]
                
                if child_similarity > best_similarity:
                    best_similarity = child_similarity
                    best_child = child
        
        # If best child has higher similarity than threshold, recurse
        if best_similarity > self.min_similarity_threshold and best_child is not None:
            child_best_node, child_best_similarity = self._find_best_placement(embedding, best_child)
            
            # If child's placement is better, return it
            if child_best_similarity > current_similarity:
                return child_best_node, child_best_similarity
        
        # Otherwise, return current node
        return node, current_similarity
    
    def add(self, item: str) -> Node:
        """
        Add a single item to the elastic dictionary.
        
        Args:
            item: The string item to add
            
        Returns:
            The created Node
        """
        # Generate embedding
        embedding = self._get_embedding(item)
        
        # Create new node
        new_node = Node(key=item, value=item, embedding=embedding)
        self.all_nodes[new_node.key] = new_node
        
        # Find best placement
        if len(self.all_nodes) == 1:  # Only root exists
            self.root.add_child(new_node)
            return new_node
        
        parent_node, similarity = self._find_best_placement(embedding)
        
        # If similarity is very high, this might be a duplicate or very similar item
        if similarity > self.max_similarity_threshold:
            # For now, still add it but as child of the similar node
            parent_node.add_child(new_node)
        else:
            parent_node.add_child(new_node)
        
        # Consider restructuring if we've added many items
        if self.enable_auto_restructure and len(self.all_nodes) % self.restructure_threshold == 0:
            self._consider_restructuring()
        
        return new_node
    
    def add_batch(self, items: List[str]) -> List[Node]:
        """Add multiple items at once."""
        return [self.add(item) for item in tqdm(items, desc="Adding items")]
    
    def add_paragraph(self, paragraph: str) -> List[Node]:
        """
        Process a paragraph by breaking it into sentences and adding each.
        
        Simple implementation that splits on periods, question marks, and exclamation points.
        A more sophisticated NLP approach could be used for better sentence segmentation.
        """
        # Simple sentence splitting - in a real implementation, use a proper NLP library
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return self.add_batch(sentences)
    
    def _create_category_node(self, items: List[Node], category_name: str = None) -> Node:
        """
        Create a category node to group similar items.
        
        Args:
            items: List of nodes to group
            category_name: Optional name for the category. If None, will use representative item.
            
        Returns:
            The created category node
        """
        # If no category name provided, use first item as representative
        if category_name is None and items:
            category_name = f"Category: {items[0].key[:30]}..."
        elif category_name is None:
            category_name = "Category"
        
        # Create embedding for category by averaging the embeddings of items
        if items:
            embeddings = [item.embedding for item in items if item.embedding is not None]
            if embeddings:
                category_embedding = np.mean(embeddings, axis=0)
            else:
                category_embedding = None
        else:
            category_embedding = None
        
        # Create the category node
        category_node = Node(key=category_name, embedding=category_embedding)
        category_node.is_category_node = True
        self.all_nodes[category_node.key] = category_node
        
        return category_node
    
    def _simple_clustering(self, embeddings, n_clusters):
        """
        A simple implementation of hierarchical clustering that doesn't rely on 
        scikit-learn's specific parameter configurations.
        
        Args:
            embeddings: Array of embeddings to cluster
            n_clusters: Number of clusters to create
            
        Returns:
            Array of cluster labels
        """
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        # Convert to distance matrix (1 - similarity)
        distance_matrix = 1 - similarity_matrix
        
        # Number of items
        n_items = len(embeddings)
        
        # Start with each item in its own cluster
        labels = np.arange(n_items)
        
        # Track active clusters
        active_clusters = set(labels)
        
        # Hierarchical clustering by merging closest clusters until we have n_clusters
        while len(active_clusters) > n_clusters:
            min_dist = float('inf')
            merge_a, merge_b = -1, -1
            
            # Find the closest pair of clusters
            for i in range(n_items):
                if labels[i] not in active_clusters:
                    continue
                    
                for j in range(i + 1, n_items):
                    if labels[j] not in active_clusters or labels[i] == labels[j]:
                        continue
                        
                    if distance_matrix[i, j] < min_dist:
                        min_dist = distance_matrix[i, j]
                        merge_a, merge_b = labels[i], labels[j]
            
            # Merge the closest clusters
            if merge_a != -1 and merge_b != -1:
                # Replace all occurrences of merge_b with merge_a
                labels[labels == merge_b] = merge_a
                active_clusters.remove(merge_b)
        
        # Relabel clusters to be consecutive integers starting from 0
        relabel_map = {old_label: i for i, old_label in enumerate(active_clusters)}
        for i in range(n_items):
            labels[i] = relabel_map[labels[i]]
            
        return labels
    
    def _consider_restructuring(self):
        """
        Consider restructuring the tree to better organize nodes.
        
        Uses hierarchical clustering to identify groups of related nodes
        and reorganizes the tree accordingly.
        """
        print("Considering tree restructuring...")
        
        # 1. Collect all leaf nodes (non-category nodes)
        leaf_nodes = [node for _, node in self.all_nodes.items() 
                     if node != self.root and not node.is_category_node]
        
        if len(leaf_nodes) < self.min_cluster_size * 2:
            print("Not enough leaf nodes for clustering. Skipping restructuring.")
            return
        
        # 2. Extract embeddings
        node_embeddings = []
        valid_nodes = []
        
        for node in leaf_nodes:
            if node.embedding is not None:
                node_embeddings.append(node.embedding)
                valid_nodes.append(node)
        
        if len(valid_nodes) < self.min_cluster_size * 2:
            print("Not enough valid nodes for clustering. Skipping restructuring.")
            return
        
        # 3. Perform hierarchical clustering
        node_embeddings = np.array(node_embeddings)
        
        # Determine optimal number of clusters (simplified approach)
        max_clusters = min(20, len(valid_nodes) // self.min_cluster_size)
        if max_clusters < 2:
            max_clusters = 2
            
        # Use our simple clustering method instead of scikit-learn's
        try:
            # First try using scikit-learn for better performance
            cluster_labels = AgglomerativeClustering(
                n_clusters=max_clusters, 
                linkage='average'
            ).fit_predict(node_embeddings)
        except Exception as e:
            print(f"Scikit-learn clustering failed: {e}")
            print("Falling back to simple clustering implementation")
            # Fall back to our simple implementation
            cluster_labels = self._simple_clustering(node_embeddings, max_clusters)
        
        # 4. Reorganize the tree based on clusters
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_nodes[i])
        
        print(f"Identified {len(clusters)} clusters")
        
        # 5. Create category nodes for each cluster with enough nodes
        for label, cluster_nodes in clusters.items():
            if len(cluster_nodes) >= self.min_cluster_size:
                # Find a representative name for this cluster
                # For simplicity, we'll just use the first node's key
                category_name = f"Category: {cluster_nodes[0].key[:20]}..."
                
                # Create a new category node
                category_node = self._create_category_node(
                    cluster_nodes, category_name=category_name
                )
                
                # Add the category node to the root
                self.root.add_child(category_node)
                
                # Move the cluster nodes to the category node
                for node in cluster_nodes:
                    # Remove from current parent
                    if node.parent:
                        node.parent.remove_child(node)
                    
                    # Add to new category
                    category_node.add_child(node)
                
                print(f"Created category '{category_name}' with {len(cluster_nodes)} nodes")
    
    def visualize(self, max_depth: int = None, layout: str = 'spring', 
                  node_size_metric: str = None, title: str = "Elastic Dictionary Structure",
                  figsize: tuple = (12, 8), save_path: str = None):
        """
        Visualize the current tree structure with enhanced features.
        
        Args:
            max_depth: Maximum depth to visualize, None for no limit
            layout: Layout algorithm to use ('spring', 'hierarchical', 'circular', 'kamada_kawai')
            node_size_metric: Metric to determine node size ('children_count', 'depth', 'none')
            title: Title of the visualization
            figsize: Figure size as tuple (width, height)
            save_path: Path to save the visualization, None to display only
        """
        G = nx.DiGraph()
        node_attrs = {}
        
        def add_nodes_to_graph(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return
                
            # Add node to graph
            if node.parent:
                G.add_edge(node.parent.key, node.key)
            else:
                G.add_node(node.key)
            
            # Store node attributes for visualization
            if node == self.root:
                node_attrs[node.key] = {"color": "green", "type": "root", "depth": depth}
            elif node.is_category_node:
                node_attrs[node.key] = {"color": "orange", "type": "category", "depth": depth}
            else:
                node_attrs[node.key] = {"color": "skyblue", "type": "item", "depth": depth}
            
            node_attrs[node.key]["children_count"] = len(node.children)
            node_attrs[node.key]["text"] = node.text if hasattr(node, 'text') else node.key
                
            for child in node.children:
                add_nodes_to_graph(child, depth+1)
        
        add_nodes_to_graph(self.root)
        
        # Calculate node sizes based on selected metric
        node_sizes = []
        node_colors = []
        for node in G.nodes():
            if node_size_metric == 'children_count':
                # Size based on number of children (min 500, max 3000)
                size = max(500, min(3000, node_attrs[node]["children_count"] * 300 + 500))
            elif node_size_metric == 'depth':
                # Size based on inverse of depth (deeper nodes are smaller)
                depth = node_attrs[node]["depth"]
                size = max(500, 3000 - (depth * 300))
            else:
                # Default fixed size
                size = 1500
            
            node_sizes.append(size)
            node_colors.append(node_attrs[node]["color"])
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Apply the selected layout
        if layout == 'hierarchical':
            try:
                # First try with pygraphviz if available
                import pygraphviz
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except (ImportError, ModuleNotFoundError):
                print("Warning: pygraphviz not available. Using pydot layout instead.")
                try:
                    import pydot
                    pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
                except (ImportError, ModuleNotFoundError, Exception):
                    print("Warning: pydot also not available. Using tree layout instead.")
                    pos = nx.spring_layout(G, seed=42)
                    # Apply some vertical spacing based on depth
                    for node in G.nodes():
                        if node in node_attrs:
                            depth = node_attrs[node]["depth"]
                            pos[node] = (pos[node][0], -depth * 0.2)  # Adjust y-position based on depth
            except Exception as e:
                print(f"Warning: Hierarchical layout failed: {e}. Falling back to spring layout.")
                pos = nx.spring_layout(G, seed=42)
            is_3d = False
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:  # default to spring
            pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw(G, pos, with_labels=True, node_color=node_colors, 
                node_size=node_sizes, arrows=True, edge_color='gray',
                font_size=10, font_weight='bold')
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Root'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Category'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Item')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(title)
        
        # Save or show the figure
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_interactive(self, max_depth: int = None, layout: str = 'force', 
                             node_size_metric: str = None, show_labels: bool = True,
                             focus_node: str = None, filter_types: list = None,
                             height: int = 800, width: int = 1000):
        """
        Create an enhanced interactive visualization of the tree using Plotly.
        
        Args:
            max_depth: Maximum depth to visualize, None for no limit
            layout: Layout algorithm ('force', '3d', 'hierarchy')
            node_size_metric: Metric for node sizing ('children_count', 'depth', None)
            show_labels: Whether to show node labels directly on the graph
            focus_node: Key of the node to focus on (center the visualization)
            filter_types: List of node types to show ('root', 'category', 'item')
            height: Height of the visualization in pixels
            width: Width of the visualization in pixels
        """
        G = nx.DiGraph()
        node_attrs = {}
        
        def add_nodes_to_graph(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return
            
            # Assign attributes
            if node == self.root:
                node_attrs[node.key] = {
                    "color": "green", 
                    "type": "root", 
                    "depth": depth,
                    "children_count": len(node.children),
                    "text": node.text if hasattr(node, 'text') else node.key,
                    "hover_text": f"Root: {node.key}<br>Children: {len(node.children)}"
                }
            elif node.is_category_node:
                node_attrs[node.key] = {
                    "color": "orange", 
                    "type": "category", 
                    "depth": depth,
                    "children_count": len(node.children),
                    "text": node.text if hasattr(node, 'text') else node.key,
                    "hover_text": f"Category: {node.key}<br>Depth: {depth}<br>Children: {len(node.children)}"
                }
            else:
                node_attrs[node.key] = {
                    "color": "skyblue", 
                    "type": "item", 
                    "depth": depth,
                    "children_count": len(node.children),
                    "text": node.text if hasattr(node, 'text') else node.key,
                    "hover_text": f"Item: {node.key}<br>Depth: {depth}"
                }
                
            if node.parent:
                G.add_edge(node.parent.key, node.key)
            else:
                G.add_node(node.key)
                
            for child in node.children:
                add_nodes_to_graph(child, depth+1)
        
        add_nodes_to_graph(self.root)
        
        # Apply filters if specified
        if filter_types:
            nodes_to_keep = [node for node in G.nodes() if node_attrs[node]["type"] in filter_types]
            G = G.subgraph(nodes_to_keep)
        
        # Focus on a specific node if requested
        if focus_node and focus_node in G:
            # Get the subgraph containing the focus node and its neighbors
            neighbors = list(G.predecessors(focus_node)) + list(G.successors(focus_node))
            nodes_to_keep = [focus_node] + neighbors
            G = G.subgraph(nodes_to_keep)
        
        # Choose layout algorithm
        if layout == 'hierarchy':
            # For hierarchical layout, we'll use nx's graphviz wrapper first, then convert to Plotly
            try:
                # First try with pygraphviz if available
                import pygraphviz
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except (ImportError, ModuleNotFoundError):
                print("Warning: pygraphviz not available. Using pydot layout instead.")
                try:
                    import pydot
                    pos = nx.drawing.nx_pydot.pydot_layout(G, prog='dot')
                except (ImportError, ModuleNotFoundError, Exception):
                    print("Warning: pydot also not available. Using tree layout instead.")
                    pos = nx.spring_layout(G, seed=42)
                    # Apply some vertical spacing based on depth
                    for node in G.nodes():
                        if node in node_attrs:
                            depth = node_attrs[node]["depth"]
                            pos[node] = (pos[node][0], -depth * 0.2)  # Adjust y-position based on depth
            except Exception as e:
                print(f"Warning: Hierarchical layout failed: {e}. Falling back to spring layout.")
                pos = nx.spring_layout(G, seed=42)
            is_3d = False
        elif layout == '3d':
            pos = nx.spring_layout(G, dim=3, seed=42)
            is_3d = True
        else:  # Default to force-directed layout
            pos = nx.spring_layout(G, seed=42)
            is_3d = False
            
        # Process nodes and edges
        if is_3d:
            # 3D visualization
            edge_x, edge_y, edge_z = [], [], []
            
            for edge in G.edges():
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_z.extend([z0, z1, None])
                
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                line=dict(width=1.5, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            # Create node traces - separate by type for legend
            node_groups = {'root': [], 'category': [], 'item': []}
            
            for node in G.nodes():
                if node not in node_attrs:
                    continue
                    
                attrs = node_attrs[node]
                node_type = attrs["type"]
                
                # Calculate node size
                if node_size_metric == 'children_count':
                    size = max(5, min(20, attrs["children_count"] * 2 + 5))
                elif node_size_metric == 'depth':
                    size = max(5, 20 - (attrs["depth"] * 2))
                else:
                    size = 10
                
                node_trace = go.Scatter3d(
                    x=[pos[node][0]], 
                    y=[pos[node][1]], 
                    z=[pos[node][2]],
                    mode='markers+text' if show_labels else 'markers',
                    hovertext=[attrs["hover_text"]],
                    text=[node] if show_labels else None,
                    textposition="top center",
                    name=node_type.capitalize(),
                    marker=dict(
                        size=size,
                        color=attrs["color"],
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    showlegend=True if node == next(iter(G.nodes())) or len(node_groups[node_type]) == 0 else False
                )
                node_groups[node_type].append(node_trace)
            
            # Flatten node groups
            node_traces = [trace for group in node_groups.values() for trace in group]
            
            # Create figure
            fig = go.Figure(data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title="Interactive Elastic Dictionary Visualization (3D)",
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    height=height,
                    width=width,
                    margin=dict(b=0, l=0, r=0, t=40),
                    scene=dict(
                        xaxis=dict(showbackground=False, showticklabels=False, title=''),
                        yaxis=dict(showbackground=False, showticklabels=False, title=''),
                        zaxis=dict(showbackground=False, showticklabels=False, title='')
                    ),
                    updatemenus=[
                        dict(
                            buttons=[
                                dict(
                                    args=[{'visible': [True] * len([edge_trace] + node_traces)}],
                                    label="Show All",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name == "Root" for trace in node_traces]}],
                                    label="Root Only",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name in ["Root", "Category"] for trace in node_traces]}],
                                    label="Categories Only",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name == "Item" for trace in node_traces]}],
                                    label="Items Only",
                                    method="update"
                                ),
                            ],
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        ),
                    ]
                )
            )
            
        else:
            # 2D visualization
            edge_trace = go.Scatter(
                x=[], y=[],
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines')
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)
            
            # Create node traces grouped by type
            node_traces = []
            for node_type, color in [('root', 'green'), ('category', 'orange'), ('item', 'skyblue')]:
                # Filter nodes by type
                nodes_of_type = [node for node in G.nodes() if node in node_attrs and node_attrs[node]["type"] == node_type]
                
                if not nodes_of_type:
                    continue
                
                # Calculate positions and sizes
                x_vals = []
                y_vals = []
                hover_texts = []
                text_vals = []
                sizes = []
                
                for node in nodes_of_type:
                    attrs = node_attrs[node]
                    x, y = pos[node]
                    x_vals.append(x)
                    y_vals.append(y)
                    hover_texts.append(attrs["hover_text"])
                    text_vals.append(node if show_labels else "")
                    
                    # Calculate node size
                    if node_size_metric == 'children_count':
                        size = max(10, min(40, attrs["children_count"] * 4 + 10))
                    elif node_size_metric == 'depth':
                        size = max(10, 40 - (attrs["depth"] * 4))
                    else:
                        size = 20
                    
                    sizes.append(size)
                
                node_trace = go.Scatter(
                    x=x_vals, 
                    y=y_vals,
                    mode='markers+text' if show_labels else 'markers',
                    hovertext=hover_texts,
                    text=text_vals,
                    textposition="top center",
                    name=node_type.capitalize(),
                    marker=dict(
                        size=sizes,
                        color=color,
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                )
                node_traces.append(node_trace)
            
            # Create figure
            title_text = "Interactive Elastic Dictionary Visualization"
            if layout == 'hierarchy':
                title_text += " (Hierarchical)"
            
            fig = go.Figure(data=[edge_trace] + node_traces,
                layout=go.Layout(
                    title=title_text,
                    showlegend=True,
                    height=height,
                    width=width,
                    margin=dict(b=0, l=0, r=0, t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ),
                    hovermode='closest',
                    clickmode='event+select',
                    # Add buttons for filtering
                    updatemenus=[
                        dict(
                            buttons=[
                                dict(
                                    args=[{'visible': [True] * len([edge_trace] + node_traces)}],
                                    label="Show All",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name == "Root" for trace in node_traces]}],
                                    label="Root Only",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name in ["Root", "Category"] for trace in node_traces]}],
                                    label="Categories Only",
                                    method="update"
                                ),
                                dict(
                                    args=[{'visible': [True] + [trace.name == "Item" for trace in node_traces]}],
                                    label="Items Only",
                                    method="update"
                                ),
                            ],
                            direction="down",
                            pad={"r": 10, "t": 10},
                            showactive=True,
                            x=0.1,
                            xanchor="left",
                            y=1.1,
                            yanchor="top"
                        ),
                    ]
                )
            )
            
            # Add annotations for node labels if requested but not shown directly
            if not show_labels:
                annotations = []
                for node in G.nodes():
                    if node in node_attrs:
                        annotations.append(
                            dict(
                                x=pos[node][0],
                                y=pos[node][1],
                                text=node,
                                showarrow=False,
                                font=dict(size=10)
                            )
                        )
                fig.update_layout(annotations=annotations)
        
        # Add download button
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=[
                        dict(
                            args=["toImageButtonOptions", {"format": "png", "filename": "elastic_dict_viz"}],
                            label="Download PNG",
                            method="relayout"
                        ),
                        dict(
                            args=["toImageButtonOptions", {"format": "svg", "filename": "elastic_dict_viz"}],
                            label="Download SVG",
                            method="relayout"
                        ),
                        dict(
                            args=["toImageButtonOptions", {"format": "pdf", "filename": "elastic_dict_viz"}],
                            label="Download PDF",
                            method="relayout"
                        ),
                    ],
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.9,
                    xanchor="right",
                    y=1.1,
                    yanchor="top"
                ),
            ] + fig.layout.updatemenus
        )
        
        # Add help annotation
        help_text = """
        Controls:
        - Click and drag to rotate (3D) or pan (2D)
        - Scroll to zoom
        - Double-click to reset view
        - Click legend items to toggle visibility
        - Use the dropdown to filter by node type
        """
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            text=help_text,
            showarrow=False,
            font=dict(size=10),
            align="left"
        )
        
        fig.show()
        
        # Return the figure object to allow additional customization if needed
        return fig
    
    def export_graph(self, format='graphml', filename='elastic_dict_graph'):
        """
        Export the graph structure to file in various formats.
        
        Args:
            format: Export format ('graphml', 'gexf', 'json', 'dot')
            filename: Base filename to use (without extension)
        
        Returns:
            Path to saved file
        """
        G = nx.DiGraph()
        
        def add_nodes_to_graph(node):
            # Add node attributes
            G.add_node(
                node.key, 
                type=('root' if node == self.root else 'category' if node.is_category_node else 'item'),
                text=node.text if hasattr(node, 'text') else node.key,
                children_count=len(node.children)
            )
            
            # Add edges to children
            for child in node.children:
                G.add_edge(node.key, child.key)
                add_nodes_to_graph(child)
        
        add_nodes_to_graph(self.root)
        
        # Export to the specified format
        full_filename = f"{filename}.{format}"
        if format == 'graphml':
            nx.write_graphml(G, full_filename)
        elif format == 'gexf':
            nx.write_gexf(G, full_filename)
        elif format == 'json':
            data = nx.node_link_data(G)
            with open(full_filename, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'dot':
            nx.drawing.nx_pydot.write_dot(G, full_filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return full_filename
    
    def find(self, query: str) -> List[Tuple[Node, float]]:
        """
        Find nodes in the tree that are semantically similar to the query.
        
        Args:
            query: The search query string
            
        Returns:
            List of tuples (node, similarity_score) ordered by similarity
        """
        query_embedding = self._get_embedding(query)
        results = []
        
        for key, node in self.all_nodes.items():
            if node == self.root or node.embedding is None:
                continue
                
            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
            results.append((node, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_node(self, key: str) -> Optional[Node]:
        """Get a node by its key."""
        return self.all_nodes.get(key)
    
    def save(self, filepath: str):
        """
        Save the elastic dictionary to a file.
        
        Args:
            filepath: Path to save the dictionary to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # First, we need to remove parent references to avoid circular references
        # We'll restore them when loading
        node_parents = {}
        
        for key, node in self.all_nodes.items():
            if node.parent:
                node_parents[key] = node.parent.key
                node.parent = None
        
        # Save the dictionary
        data = {
            'all_nodes': self.all_nodes,
            'node_parents': node_parents,
            'model_name': self.model.get_sentence_embedding_dimension(),
            'thresholds': {
                'restructure': self.restructure_threshold,
                'min_similarity': self.min_similarity_threshold,
                'max_similarity': self.max_similarity_threshold,
                'min_cluster_size': self.min_cluster_size,
            },
            'enable_auto_restructure': self.enable_auto_restructure
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Restore parent references
        for key, parent_key in node_parents.items():
            self.all_nodes[key].parent = self.all_nodes[parent_key]
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load an elastic dictionary from a file.
        
        Args:
            filepath: Path to load the dictionary from
            
        Returns:
            ElasticDictionary: The loaded dictionary
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new dictionary
        ed = cls()
        
        # Replace the nodes
        ed.all_nodes = data['all_nodes']
        
        # Restore parent references
        for key, parent_key in data['node_parents'].items():
            ed.all_nodes[key].parent = ed.all_nodes[parent_key]
        
        # Get the root
        for key, node in ed.all_nodes.items():
            if node.parent is None:
                ed.root = node
                break
        
        # Restore thresholds
        thresholds = data.get('thresholds', {})
        ed.restructure_threshold = thresholds.get('restructure', 10)
        ed.min_similarity_threshold = thresholds.get('min_similarity', 0.5)
        ed.max_similarity_threshold = thresholds.get('max_similarity', 0.8)
        ed.min_cluster_size = thresholds.get('min_cluster_size', 4)
        
        # Restore auto restructure setting
        ed.enable_auto_restructure = data.get('enable_auto_restructure', True)
        
        return ed
    
    def force_restructure(self):
        """Force a restructuring of the tree, regardless of threshold."""
        self._consider_restructuring()
    
    def __len__(self):
        """Return the number of nodes in the dictionary."""
        return len(self.all_nodes)
    
    def __contains__(self, key):
        """Check if a key exists in the dictionary."""
        return key in self.all_nodes


# Advanced Features To Be Implemented:
# 1. Automatic topic extraction to name internal nodes
# 2. Support for different embedding models with auto-downloading
# 3. Handling of non-text data types via custom embeddings 