# Elastic Dictionary

An adaptive, hierarchical data structure that dynamically organizes string data and text into a semantic tree structure.

## Features

- **Dynamic Organization**: Elements find their natural place in a hierarchical structure
- **Semantic Understanding**: Uses embeddings to understand meaning and relationships between entries
- **Adaptive Structure**: Tree evolves and reorganizes as new data is added
- **Support for Different Data Types**: Handles simple strings, lists, and paragraphs
- **Visualization**: Advanced tools to visualize and explore the tree structure

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from elastic_dict import ElasticDictionary

# Create a new elastic dictionary
ed = ElasticDictionary()

# Add single elements
ed.add("apple")
ed.add("banana")
ed.add("orange")

# Add a list of elements
ed.add_batch(["computer", "keyboard", "mouse"])

# Add a paragraph
ed.add_paragraph("Machine learning is a subset of artificial intelligence that focuses on developing systems that learn from data.")

# Basic visualization
ed.visualize()

# Enhanced visualization options
ed.visualize(
    layout='hierarchical',           # Options: 'spring', 'hierarchical', 'circular', 'kamada_kawai'
    node_size_metric='children_count',  # Options: 'children_count', 'depth', None
    title="My Elastic Dictionary",
    figsize=(12, 10),
    save_path="my_visualization.png"
)

# Interactive 3D visualization
ed.visualize_interactive(
    layout='3d',                    # Options: 'force', '3d', 'hierarchy'
    node_size_metric='depth',
    show_labels=True,
    focus_node="apple",            # Focus on a particular node
    filter_types=['root', 'category', 'item']  # Filter by node types
)
```

## How It Works

The Elastic Dictionary uses a combination of:
1. Sentence embeddings to understand meaning
2. Dynamic tree structures to organize related concepts
3. Adaptive algorithms to reorganize as needed

This allows for intuitive organization of data without predefined categories.

## Visualization Features

The Elastic Dictionary provides powerful visualization capabilities:

### Static Visualization
- Multiple layout algorithms (force-directed, hierarchical, circular)
- Node sizing based on metrics (number of children, depth)
- Color-coded nodes (categories, items, root)
- Legend and customizable appearance
- Export to various image formats

### Interactive Visualization
- 2D and 3D visualization modes
- Dynamic filtering of node types
- Focus on specific nodes or branches
- Detailed hover information
- Download options (PNG, SVG, PDF)
- Interactive controls (zoom, pan, rotate)

### Graph Export
Export the graph structure for analysis in external tools:
```python
ed.export_graph(format='graphml', filename='my_graph')  # Formats: 'graphml', 'gexf', 'json', 'dot'
``` 