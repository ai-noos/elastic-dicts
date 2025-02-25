"""
Example demonstrating the auto-restructuring feature of ElasticDictionary.

This script shows how the dictionary automatically reorganizes its structure
to group semantically similar items together as the tree grows.
"""

from elastic_dict import ElasticDictionary
import time
import os

def main():
    print("Initializing ElasticDictionary with auto-restructuring enabled...")
    
    # Create a new dictionary or load existing one
    save_path = "restructured_dict.pkl"
    
    if os.path.exists(save_path):
        ed = ElasticDictionary.load(save_path)
        print(f"Loaded existing dictionary with {len(ed)} nodes")
    else:
        ed = ElasticDictionary()
        
        # Configure restructuring settings
        ed.restructure_threshold = 15  # Restructure after every 15 items
        ed.min_cluster_size = 3        # Minimum cluster size to form a category
        
        print("Dictionary configured for frequent restructuring")
    
    # First, add items from different domains to see clustering in action
    
    # Technology terms
    tech_items = [
        "computer", "laptop", "smartphone", "tablet", "processor", 
        "memory", "RAM", "CPU", "GPU", "hardware", "software", 
        "programming", "algorithm", "data structure", "compiler"
    ]
    
    # Animal terms
    animal_items = [
        "dog", "cat", "elephant", "tiger", "lion", "zebra", 
        "giraffe", "monkey", "gorilla", "bear", "wolf", 
        "fox", "rabbit", "squirrel", "deer"
    ]
    
    # Food terms
    food_items = [
        "pizza", "pasta", "burger", "sandwich", "salad", 
        "steak", "chicken", "fish", "vegetable", "fruit", 
        "apple", "banana", "orange", "strawberry", "chocolate"
    ]
    
    # Weather terms
    weather_items = [
        "rain", "snow", "storm", "hurricane", "tornado", 
        "sunny", "cloudy", "windy", "fog", "humidity", 
        "temperature", "climate", "weather forecast", "meteorology", "precipitation"
    ]
    
    # Add items in mixed order to see how they get organized
    all_items = []
    for i in range(len(tech_items)):
        if i < len(tech_items): all_items.append(tech_items[i])
        if i < len(animal_items): all_items.append(animal_items[i])
        if i < len(food_items): all_items.append(food_items[i])
        if i < len(weather_items): all_items.append(weather_items[i])
    
    # Add items in batches to see restructuring happen
    batch_size = 10
    total_batches = len(all_items) // batch_size + (1 if len(all_items) % batch_size > 0 else 0)
    
    for batch in range(total_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(all_items))
        
        batch_items = all_items[start_idx:end_idx]
        print(f"\nAdding batch {batch+1}/{total_batches} ({len(batch_items)} items)...")
        
        for item in batch_items:
            print(f"  Adding: {item}")
            ed.add(item)
        
        print("\nCurrent structure after this batch:")
        ed.visualize()
    
    # Manually force restructuring
    print("\nManually forcing final restructuring to optimize the tree...")
    ed.force_restructure()
    
    # Save the restructured dictionary
    print(f"\nSaving restructured dictionary to {save_path}...")
    ed.save(save_path)
    
    # Show the final structure with interactive 3D visualization
    print("\nFinal tree structure (interactive):")
    ed.visualize_interactive()
    
    # Demonstrate search capabilities on the restructured tree
    search_queries = ["computer technology", "wild animals", "food and drinks", "weather conditions"]
    
    print("\n--- Searching the Restructured Dictionary ---")
    for query in search_queries:
        print(f"\nResults for '{query}':")
        results = ed.find(query)[:5]  # Top 5 results
        
        for i, (node, similarity) in enumerate(results):
            # Show if it's a category node or not
            node_type = "Category" if node.is_category_node else "Item"
            print(f"  {i+1}. [{similarity:.2f}] {node_type}: {node.key}")
    
    print("\nRestructuring example completed successfully.")


if __name__ == "__main__":
    main() 