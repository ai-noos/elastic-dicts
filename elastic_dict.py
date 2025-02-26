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
    
    def visualize(self, max_depth: int = None, figsize=(12, 8), node_size_factor=1.0, label_offset=0.1,
                 title="Elastic Dictionary Structure", layout="hierarchical"):
        """
        Visualize the current tree structure with improved styling and layout.
        
        Args:
            max_depth: Maximum depth to visualize, None for no limit
            figsize: Size of the figure as a tuple (width, height)
            node_size_factor: Factor to scale all node sizes
            label_offset: Offset for node labels to prevent overlap
            title: Title of the visualization
            layout: Layout algorithm to use ('hierarchical', 'radial', 'spring', 'kamada_kawai')
        """
        G = nx.DiGraph()
        node_colors = {}
        node_sizes = {}
        node_depths = {}
        node_counts = {}  # Count of descendants for each node
        
        # First pass to build the graph and calculate properties
        def process_node(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return 0  # No descendants at this level
            
            # Assign depths
            node_depths[node.key] = depth
            
            # Count descendants recursively
            descendant_count = 1  # Count self
            for child in node.children:
                if max_depth is None or depth < max_depth:
                    descendant_count += process_node(child, depth + 1)
            
            node_counts[node.key] = descendant_count
            
            # Assign colors: categories are orange, regular nodes are blue, root is green
            if node == self.root:
                node_colors[node.key] = '#2ca02c'  # Green
            elif node.is_category_node:
                node_colors[node.key] = '#ff7f0e'  # Orange
            else:
                node_colors[node.key] = '#1f77b4'  # Blue
            
            # Add to graph
            if node.parent:
                G.add_edge(node.parent.key, node.key)
            else:
                G.add_node(node.key)
                
            return descendant_count
        
        process_node(self.root)
        
        # Calculate node sizes based on descendants and depth
        max_count = max(node_counts.values()) if node_counts else 1
        min_size = 300 * node_size_factor
        max_size = 1200 * node_size_factor
        
        for node_key in G.nodes():
            # Nodes higher in the tree and with more descendants are larger
            depth_factor = 1.0 - 0.1 * (node_depths.get(node_key, 0))
            depth_factor = max(0.5, depth_factor)  # Don't go below 0.5
            
            count_factor = node_counts.get(node_key, 1) / max_count
            node_sizes[node_key] = min_size + (max_size - min_size) * count_factor * depth_factor
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Create custom layouts without using graphviz
        if layout == "hierarchical":
            # Create a hierarchical layout using a custom implementation
            pos = self._create_hierarchical_layout(G, node_depths)
        elif layout == "radial":
            # Create a radial layout centered on root
            pos = self._create_radial_layout(G, node_depths)
        elif layout == "kamada_kawai":
            # Use networkx's kamada_kawai layout
            pos = nx.kamada_kawai_layout(G)
        else:
            # Default to spring layout with depth as initial positions
            pos = nx.spring_layout(
                G, 
                pos={n: (0, -d) for n, d in node_depths.items()},
                fixed=[self.root.key],
                k=1.5/np.sqrt(len(G.nodes())),
                iterations=50
            )
        
        # Draw the graph
        nx.draw_networkx_edges(G, pos, alpha=0.4, edge_color='gray', width=1.0, arrows=True, 
                             arrowstyle='-|>', arrowsize=15)
        
        # Draw nodes
        node_list = list(G.nodes())
        node_color_list = [node_colors.get(node, '#1f77b4') for node in node_list]
        node_size_list = [node_sizes.get(node, min_size) for node in node_list]
        
        nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_color_list, 
                              node_size=node_size_list, alpha=0.8, linewidths=1.0, 
                              edgecolors='black')
        
        # Draw labels with adjusted position to avoid overlap
        label_pos = {n: (p[0], p[1] + label_offset) for n, p in pos.items()}
        nx.draw_networkx_labels(G, label_pos, font_size=10, font_family='sans-serif', 
                               font_weight='bold')
        
        # Add legend for node types
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Root'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Category'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Item')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add title and adjust layout
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        return plt.gcf()  # Return the figure for potential saving

    def _create_hierarchical_layout(self, G, node_depths):
        """
        Create a hierarchical layout without using graphviz.
        Places nodes in layers based on their depth.
        
        Args:
            G: NetworkX graph
            node_depths: Dictionary of node depths
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        max_depth = max(node_depths.values()) if node_depths else 0
        
        # Group nodes by depth
        nodes_by_depth = {}
        for node, depth in node_depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
        
        # Position nodes in layers
        for depth, nodes in nodes_by_depth.items():
            num_nodes = len(nodes)
            width = max(1.0, num_nodes * 0.5)  # Width increases with number of nodes
            
            # Place nodes at this depth in a horizontal line
            for i, node in enumerate(nodes):
                x = (i / max(1, num_nodes - 1)) * width - (width / 2)
                y = -depth  # Negative to place root at top
                pos[node] = (x, y)
        
        return pos
    
    def _create_radial_layout(self, G, node_depths):
        """
        Create a radial layout without using graphviz.
        Places nodes in concentric circles with root at center.
        
        Args:
            G: NetworkX graph
            node_depths: Dictionary of node depths
            
        Returns:
            Dictionary of node positions
        """
        pos = {}
        max_depth = max(node_depths.values()) if node_depths else 0
        
        # Group nodes by depth
        nodes_by_depth = {}
        for node, depth in node_depths.items():
            if depth not in nodes_by_depth:
                nodes_by_depth[depth] = []
            nodes_by_depth[depth].append(node)
        
        # Position nodes in concentric circles
        for depth, nodes in nodes_by_depth.items():
            num_nodes = len(nodes)
            radius = depth * 1.0  # Radius increases with depth
            
            # Place nodes in a circle at this depth
            for i, node in enumerate(nodes):
                angle = (i / num_nodes) * 2 * np.pi
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pos[node] = (x, y)
        
        return pos
    
    def visualize_interactive(self, max_depth: int = None, layout_type="3d_hierarchy", height=800, width=1000):
        """
        Create an enhanced interactive visualization of the tree using Plotly.
        
        Args:
            max_depth: Maximum depth to visualize, None for no limit
            layout_type: Type of layout to use ('3d_hierarchy', '3d_radial', '3d_spring')
            height: Height of the plot in pixels
            width: Width of the plot in pixels
            
        Returns:
            Plotly figure
        """
        G = nx.DiGraph()
        node_colors = {}  # Store colors for nodes
        node_sizes = {}   # Store sizes for nodes
        node_info = {}    # Store additional node info for hover
        node_depths = {}  # Store depth of each node
        
        # First pass to build the graph and compute properties
        def process_node(node, depth=0):
            if max_depth is not None and depth > max_depth:
                return 0  # No descendants
                
            node_depths[node.key] = depth
            
            # Store node info for hover
            info = {
                'type': 'Root' if node == self.root else ('Category' if node.is_category_node else 'Item'),
                'depth': depth,
                'children': len(node.children),
                'value': str(node.value)[:50] + ('...' if len(str(node.value)) > 50 else '')
            }
            node_info[node.key] = info
            
            # Assign colors based on node type
            if node == self.root:
                node_colors[node.key] = '#2ca02c'  # Green
            elif node.is_category_node:
                node_colors[node.key] = '#ff7f0e'  # Orange
            else:
                node_colors[node.key] = '#1f77b4'  # Blue
                
            # Add to graph
            if node.parent:
                G.add_edge(node.parent.key, node.key)
            else:
                G.add_node(node.key)
            
            # Calculate descendants for node sizing
            descendant_count = 1  # Count self
            for child in node.children:
                if max_depth is None or depth < max_depth:
                    descendant_count += process_node(child, depth + 1)
            
            # Calculate node size based on descendants and depth
            max_depth_factor = 1.0 - 0.1 * depth
            size = 10 + (5 * descendant_count * max_depth_factor)
            node_sizes[node.key] = min(30, size)  # Cap size for very large nodes
            
            return descendant_count
            
        process_node(self.root)
        
        # Create position layout based on selected type
        if layout_type == "3d_hierarchy":
            # Create a hierarchical 3D layout
            pos = {}
            max_depth = max(node_depths.values()) if node_depths else 0
            
            # Group nodes by depth
            nodes_by_depth = {}
            for node, depth in node_depths.items():
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(node)
            
            # Position nodes in layers
            for depth, nodes in nodes_by_depth.items():
                num_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    # Calculate position in a circular pattern at each depth
                    if depth == 0:  # Root at the top
                        pos[node] = (0, 0, max_depth)
                    else:
                        radius = depth * 1.5
                        theta = (i / num_nodes) * 2 * np.pi
                        x = radius * np.cos(theta)
                        y = radius * np.sin(theta)
                        z = max_depth - depth
                        pos[node] = (x, y, z)
        
        elif layout_type == "3d_radial":
            # Create a radial 3D layout
            pos = {}
            max_depth = max(node_depths.values()) if node_depths else 0
            
            # Group nodes by depth
            nodes_by_depth = {}
            for node, depth in node_depths.items():
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(node)
            
            # Position nodes in a radial pattern
            for depth, nodes in nodes_by_depth.items():
                num_nodes = len(nodes)
                for i, node in enumerate(nodes):
                    if depth == 0:  # Root at the center
                        pos[node] = (0, 0, 0)
                    else:
                        # Spherical coordinates
                        radius = depth * 2
                        theta = (i / num_nodes) * 2 * np.pi
                        phi = (depth / max_depth) * np.pi / 2  # Angle from z-axis
                        
                        x = radius * np.sin(phi) * np.cos(theta)
                        y = radius * np.sin(phi) * np.sin(theta)
                        z = radius * np.cos(phi)
                        pos[node] = (x, y, z)
        
        else:  # Default 3D spring layout
            pos = nx.spring_layout(G, dim=3, seed=42)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        edge_z = []
        
        for edge in G.edges():
            x0, y0, z0 = pos[edge[0]]
            x1, y1, z1 = pos[edge[1]]
            
            # Create smooth curved lines for the edges
            points = 10  # Number of points to create the curve
            for i in range(points):
                t = i / (points - 1)
                # Simple linear interpolation between points
                x = x0 * (1 - t) + x1 * t
                y = y0 * (1 - t) + y1 * t
                z = z0 * (1 - t) + z1 * t
                
                edge_x.append(x)
                edge_y.append(y)
                edge_z.append(z)
            
            # Add None to create separation between edges
            edge_x.append(None)
            edge_y.append(None)
            edge_z.append(None)
            
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            line=dict(width=2, color='rgba(150,150,150,0.5)'),
            hoverinfo='none',
            mode='lines')
        
        # Create node traces - separate by color
        node_groups = {}
        for node in G.nodes():
            color = node_colors.get(node, '#1f77b4')
            if color not in node_groups:
                node_groups[color] = {
                    'x': [], 'y': [], 'z': [], 
                    'text': [], 'size': [], 
                    'hover_info': [], 'color_name': []
                }
            
            x, y, z = pos[node]
            node_groups[color]['x'].append(x)
            node_groups[color]['y'].append(y)
            node_groups[color]['z'].append(z)
            node_groups[color]['text'].append(node)
            node_groups[color]['size'].append(node_sizes.get(node, 10))
            
            # Create hover text with node info
            info = node_info.get(node, {})
            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Type: {info.get('type', 'Unknown')}<br>"
            hover_text += f"Depth: {info.get('depth', 'Unknown')}<br>"
            hover_text += f"Children: {info.get('children', 0)}<br>"
            if 'value' in info:
                hover_text += f"Value: {info['value']}<br>"
            
            node_groups[color]['hover_info'].append(hover_text)
            
            # Assign color names for legend
            if color == '#2ca02c':
                node_groups[color]['color_name'].append('Root')
            elif color == '#ff7f0e':
                node_groups[color]['color_name'].append('Category')
            else:
                node_groups[color]['color_name'].append('Item')
        
        node_traces = []
        for color, data in node_groups.items():
            # Get the type name for this color group
            type_name = data['color_name'][0] if data['color_name'] else 'Unknown'
            
            node_trace = go.Scatter3d(
                x=data['x'], y=data['y'], z=data['z'],
                mode='markers',
                hovertext=data['hover_info'],
                hoverinfo='text',
                name=type_name,
                marker=dict(
                    size=data['size'],
                    color=color,
                    sizemode='diameter',
                    line=dict(width=1, color='DarkSlateGrey'),
                    opacity=0.8
                ),
                text=data['text'],
                textposition="top center"
            )
            node_traces.append(node_trace)
        
        # Add legend items for types
        legend_trace_root = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            name='Root',
            marker=dict(size=15, color='#2ca02c'),
            showlegend=True
        )
        
        legend_trace_category = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            name='Category',
            marker=dict(size=15, color='#ff7f0e'),
            showlegend=True
        )
        
        legend_trace_item = go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            name='Item',
            marker=dict(size=15, color='#1f77b4'),
            showlegend=True
        )
        
        # Create figure with improved layout
        fig = go.Figure(
            data=[edge_trace] + node_traces,
            layout=go.Layout(
                title={
                    'text': "Interactive Elastic Dictionary Visualization",
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255, 255, 255, 0.5)"
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                scene=dict(
                    xaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                    yaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                    zaxis=dict(showticklabels=False, title='', showgrid=False, zeroline=False),
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.0)
                    ),
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=0.8)
                ),
                height=height,
                width=width,
                hovermode='closest',
                updatemenus=[
                    dict(
                        buttons=[
                            dict(
                                args=[{'scene.camera.eye': dict(x=1.5, y=1.5, z=1.0)}],
                                label="Default View",
                                method="relayout"
                            ),
                            dict(
                                args=[{'scene.camera.eye': dict(x=0, y=0, z=2.5)}],
                                label="Top View",
                                method="relayout"
                            ),
                            dict(
                                args=[{'scene.camera.eye': dict(x=2.5, y=0, z=0)}],
                                label="Side View",
                                method="relayout"
                            )
                        ],
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=0,
                        yanchor="bottom"
                    )
                ]
            )
        )
        
        return fig
    
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