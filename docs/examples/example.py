"""
Example usage of the ElasticDictionary.

This script demonstrates how to use the ElasticDictionary to organize various types of data
and visualize the resulting structure.
"""

from elastic_dict import ElasticDictionary
import time

def main():
    print("Initializing ElasticDictionary...")
    ed = ElasticDictionary()
    
    # Example 1: Adding related fruit items
    print("\n--- Example 1: Adding Fruits ---")
    fruits = ["apple", "banana", "orange", "pear", "strawberry", "blueberry", "raspberry", 
              "watermelon", "kiwi", "mango", "pineapple", "grape"]
    ed.add_batch(fruits)
    print(f"Added {len(fruits)} fruits to the dictionary")
    
    # Visualize the current structure
    print("\nVisualizing the fruit structure...")
    ed.visualize()
    
    # Example 2: Adding technology-related terms
    print("\n--- Example 2: Adding Technology Items ---")
    tech_items = ["computer", "keyboard", "mouse", "monitor", "laptop", "smartphone", 
                  "tablet", "processor", "memory", "hard drive", "USB drive", "cloud storage"]
    ed.add_batch(tech_items)
    print(f"Added {len(tech_items)} technology items to the dictionary")
    
    # Visualize again to see the new structure
    print("\nVisualizing the updated structure...")
    ed.visualize()
    
    # Example 3: Adding paragraphs
    print("\n--- Example 3: Adding Paragraphs ---")
    paragraphs = [
        "Machine learning is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns, and make decisions with minimal human intervention.",
        "Fruits are the mature ovaries of plants with their seeds. They provide essential nutrients like vitamins, minerals, and fiber that are important for human health.",
        "The history of computing hardware covers the developments from early simple devices to aid calculation to modern computers. Before the 20th century, most calculations were done by humans."
    ]
    
    for i, para in enumerate(paragraphs):
        print(f"\nProcessing paragraph {i+1}...")
        nodes = ed.add_paragraph(para)
        print(f"Added {len(nodes)} sentences from the paragraph")
    
    # Example 4: Search for related items
    print("\n--- Example 4: Searching the Dictionary ---")
    search_queries = ["fruit", "technology", "artificial intelligence", "nutrition", "history"]
    
    for query in search_queries:
        print(f"\nSearching for items related to '{query}':")
        results = ed.find(query)
        
        # Display top 5 results
        for i, (node, similarity) in enumerate(results[:5]):
            print(f"  {i+1}. {node.key} (similarity: {similarity:.2f})")
    
    # Example 5: Interactive visualization
    print("\n--- Example 5: Interactive Visualization ---")
    print("Generating interactive 3D visualization...")
    ed.visualize_interactive()
    
    print("\nElasticDictionary demo completed.")

if __name__ == "__main__":
    main() 