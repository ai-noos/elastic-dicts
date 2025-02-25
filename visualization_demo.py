#!/usr/bin/env python3
"""
Visualization Demo for Elastic Dictionary

This script demonstrates the enhanced visualization capabilities of the Elastic Dictionary.
It creates a sample dictionary with various items, then shows different visualization techniques.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from elastic_dict import ElasticDictionary
import sys

# Create output directory for figures if it doesn't exist
os.makedirs('figs', exist_ok=True)

def main():
    """Main demonstration function"""
    print("Creating sample Elastic Dictionary...")
    dict_obj = ElasticDictionary()
    
    # Add some sample items from different domains
    print("Adding items...")
    # Fruits
    dict_obj.add("apple", "A red or green fruit")
    dict_obj.add("banana", "A yellow curved fruit")
    dict_obj.add("orange", "A round orange citrus fruit")
    dict_obj.add("strawberry", "A small red fruit with seeds on the outside")
    dict_obj.add("pear", "A sweet fruit with a bulbous end")
    dict_obj.add("grape", "A small round fruit growing in clusters")
    
    # Animals
    dict_obj.add("dog", "A domesticated carnivorous mammal")
    dict_obj.add("cat", "A small domesticated carnivorous mammal")
    dict_obj.add("elephant", "A very large herbivorous mammal with a trunk")
    dict_obj.add("lion", "A large wild cat with a mane")
    dict_obj.add("tiger", "A large wild cat with striped fur")
    dict_obj.add("zebra", "A wild horse with black and white stripes")
    
    # Technology
    dict_obj.add("computer", "An electronic device for processing data")
    dict_obj.add("smartphone", "A mobile phone with advanced features")
    dict_obj.add("laptop", "A portable computer")
    dict_obj.add("tablet", "A flat portable computer with a touchscreen")
    dict_obj.add("headphones", "Devices worn over the ears to listen to audio")
    dict_obj.add("keyboard", "An input device with keys for typing")
    
    # Weather
    dict_obj.add("rain", "Water falling from the clouds")
    dict_obj.add("snow", "Frozen water vapor falling from the sky")
    dict_obj.add("wind", "Movement of air")
    dict_obj.add("storm", "Violent disturbance of the atmosphere")
    dict_obj.add("sunshine", "Direct light from the sun")
    dict_obj.add("cloud", "A visible mass of water droplets in the atmosphere")
    
    print(f"Added {len(dict_obj.root.children)} categories with a total of 24 items")
    
    # Restructure to form categories
    dict_obj._consider_restructuring(force=True)
    
    print("\nDemonstrating different visualization methods...")
    
    # 1. Basic Spring Layout Visualization with node sizing based on children count
    print("\n1. Basic Spring Layout with node sizing by children count")
    dict_obj.visualize(
        layout='spring',
        node_size_metric='children_count',
        title="Spring Layout with Node Sizing by Children Count",
        figsize=(10, 8),
        save_path="figs/viz_spring.pdf"
    )
    
    # 2. Try Hierarchical Layout, fallback to another if not available
    print("\n2. Hierarchical Layout (or alternative if not available)")
    try:
        import pygraphviz
        has_pygraphviz = True
    except (ImportError, ModuleNotFoundError):
        has_pygraphviz = False
        print("Note: pygraphviz not available, will use alternative layout")
    
    dict_obj.visualize(
        layout='hierarchical' if has_pygraphviz else 'circular',
        node_size_metric='depth',
        title=f"{'Hierarchical' if has_pygraphviz else 'Circular'} Layout with Node Sizing by Depth",
        figsize=(12, 10),
        save_path="figs/viz_hierarchical.pdf"
    )
    
    # 3. Create a 3D interactive visualization
    print("\n3. 3D Interactive Visualization")
    try:
        fig = dict_obj.visualize_interactive(
            layout='3d',
            node_size_metric='children_count',
            show_labels=True,
            height=800,
            width=1000
        )
        
        # Check if kaleido is available for static image export
        try:
            import kaleido
            # Save static version for the paper
            fig.write_image("figs/viz_3d.pdf")
            print("Saved 3D visualization to figs/viz_3d.pdf")
        except (ImportError, ModuleNotFoundError):
            print("Note: kaleido not available, skipping static export of 3D visualization")
    except Exception as e:
        print(f"Error with 3D visualization: {e}")
    
    # 4. Create a filtered visualization showing only categories
    print("\n4. Filtered Visualization (Categories Only)")
    try:
        fig = dict_obj.visualize_interactive(
            layout='force',  # Use force layout which should work everywhere 
            filter_types=['root', 'category'],
            show_labels=True,
            height=600,
            width=800
        )
        
        # Try to save if kaleido is available
        try:
            import kaleido
            fig.write_image("figs/viz_filtered.pdf")
            print("Saved filtered visualization to figs/viz_filtered.pdf")
        except (ImportError, ModuleNotFoundError):
            print("Note: kaleido not available, skipping static export of filtered visualization")
    except Exception as e:
        print(f"Error with filtered visualization: {e}")
    
    # 5. Export the graph structure for external analysis
    print("\n5. Exporting Graph Structure")
    try:
        output_file = dict_obj.export_graph(format='json', filename='elastic_dict_graph')
        print(f"Graph exported to {output_file}")
    except Exception as e:
        print(f"Error exporting graph: {e}")
        # Fallback to a simple JSON export
        import json
        data = {"nodes": []}
        for node_key, node in dict_obj.all_nodes.items():
            data["nodes"].append({
                "key": node_key,
                "type": "root" if node == dict_obj.root else 
                       "category" if node.is_category_node else "item",
                "children": [child.key for child in node.children]
            })
        with open("elastic_dict_graph.json", "w") as f:
            json.dump(data, f, indent=2)
        print("Fallback: Graph exported to elastic_dict_graph.json")

    print("\nDemonstration complete! Visualization files saved to the 'figs' directory.")

if __name__ == "__main__":
    main() 