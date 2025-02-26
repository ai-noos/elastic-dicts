# Elastic Dictionary

An adaptive, hierarchical data structure that dynamically organizes string data and text into a semantic tree structure.

## Features

- **Dynamic Organization**: Elements find their natural place in a hierarchical structure
- **Semantic Understanding**: Uses embeddings to understand meaning and relationships between entries
- **Adaptive Structure**: Tree evolves and reorganizes as new data is added
- **Support for Different Data Types**: Handles simple strings, lists, and paragraphs
- **Visualization**: Tools to visualize the tree structure

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

# Visualize the tree
ed.visualize()
```

## How It Works

The Elastic Dictionary uses a combination of:
1. Sentence embeddings to understand meaning
2. Dynamic tree structures to organize related concepts
3. Adaptive algorithms to reorganize as needed

This allows for intuitive organization of data without predefined categories. 