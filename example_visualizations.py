#!/usr/bin/env python3
"""
Example script demonstrating the improved visualization capabilities of the Elastic Dictionary.
"""

import numpy as np
import matplotlib.pyplot as plt
from elastic_dict import ElasticDictionary
import plotly.io as pio

# Set Plotly to render in browser for better interaction
pio.renderers.default = "browser"

def main():
    # Create a sample dictionary with various types of content
    print("Creating sample Elastic Dictionary...")
    ed = ElasticDictionary()
    
    # Add some fruits
    print("Adding fruits...")
    fruits = ["apple", "banana", "orange", "strawberry", "blueberry", "raspberry", 
              "grape", "kiwi", "mango", "pineapple", "watermelon", "peach"]
    for fruit in fruits:
        ed.add(fruit)
    
    # Add some animals
    print("Adding animals...")
    animals = ["cat", "dog", "elephant", "giraffe", "lion", "tiger", "zebra", 
               "monkey", "gorilla", "bear", "wolf", "fox", "deer", "moose"]
    for animal in animals:
        ed.add(animal)
    
    # Add some technology terms
    print("Adding technology items...")
    tech = ["computer", "smartphone", "tablet", "laptop", "server", "database", 
            "cloud", "algorithm", "network", "internet", "software", "hardware"]
    for item in tech:
        ed.add(item)
    
    # Add some weather terms
    print("Adding weather items...")
    weather = ["rain", "snow", "sunshine", "cloud", "storm", "thunder", "lightning",
              "hurricane", "tornado", "fog", "hail", "wind", "breeze"]
    for item in weather:
        ed.add(item)
    
    # Add some paragraphs
    print("Adding paragraphs...")
    paragraphs = [
        "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
        "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains.",
        "Data science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from data.",
        "Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.",
        "Cloud computing is the on-demand availability of computer system resources, especially data storage and computing power, without direct active management by the user."
    ]
    for i, para in enumerate(paragraphs):
        ed.add_paragraph(para)
    
    # Trigger restructuring
    print("Restructuring dictionary...")
    ed.force_restructure()
    
    # Create different 2D visualizations
    print("Creating 2D visualizations...")
    
    # Hierarchical layout
    fig1 = ed.visualize(layout="hierarchical", title="Hierarchical Layout")
    fig1.savefig("viz_hierarchical.png", dpi=300, bbox_inches="tight")
    
    # Radial layout
    fig2 = ed.visualize(layout="radial", title="Radial Layout")
    fig2.savefig("viz_radial.png", dpi=300, bbox_inches="tight")
    
    # Spring layout
    fig3 = ed.visualize(layout="spring", title="Spring Layout")
    fig3.savefig("viz_spring.png", dpi=300, bbox_inches="tight")
    
    # Kamada-Kawai layout
    fig4 = ed.visualize(layout="kamada_kawai", title="Kamada-Kawai Layout")
    fig4.savefig("viz_kamada_kawai.png", dpi=300, bbox_inches="tight")
    
    # Create 3D visualizations
    print("Creating 3D visualizations...")
    
    # Hierarchical 3D
    fig3d_1 = ed.visualize_interactive(layout_type="3d_hierarchy")
    fig3d_1.write_html("viz_3d_hierarchical.html")
    
    # Radial 3D
    fig3d_2 = ed.visualize_interactive(layout_type="3d_radial")
    fig3d_2.write_html("viz_3d_radial.html")
    
    # Spring 3D
    fig3d_3 = ed.visualize_interactive(layout_type="3d_spring")
    fig3d_3.write_html("viz_3d_spring.html")
    
    print("Visualizations created and saved to files.")
    print("2D visualizations: viz_*.png")
    print("3D visualizations: viz_3d_*.html")
    
    # Show the 2D visualizations
    plt.show()
    
if __name__ == "__main__":
    main() 