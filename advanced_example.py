"""
Advanced example of the ElasticDictionary.

This script demonstrates more advanced features:
1. Processing and organizing paragraphs of text
2. Saving and loading the dictionary structure
3. Search and retrieve semantically related information
"""

from elastic_dict import ElasticDictionary
import os
import time
from tqdm import tqdm

# Sample paragraphs about various topics
SAMPLE_PARAGRAPHS = [
    # Technology
    "Artificial intelligence is transforming industries through machine learning algorithms. These systems analyze data patterns to make predictions and automate decision-making processes. Deep learning, a subset of machine learning, uses neural networks with many layers to process complex information.",
    
    "Cloud computing provides on-demand access to computing resources over the internet. Services include storage, processing power, and software applications. Organizations benefit from scalability, cost-efficiency, and reduced need for physical infrastructure.",
    
    "Cybersecurity protects systems, networks, and programs from digital attacks. These attacks often aim to access, change, or destroy sensitive information, extort money, or interrupt business processes. Implementing effective security measures is especially challenging as attackers become more innovative.",
    
    # Biology
    "Cellular biology focuses on the physiological properties, metabolic processes, signaling pathways, life cycle, and interactions of cells with their environment. The cell is the fundamental unit of structure and function in living organisms.",
    
    "Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch of biology and gives insight into why organisms look and behave the way they do. Genetic information is passed from parent to child through DNA sequences.",
    
    "Ecology examines the relationships between organisms and their environment. This includes interactions with the physical environment and with other organisms. Ecological systems can be studied at different levels, from individual organisms to entire ecosystems.",
    
    # History
    "The Renaissance was a period of European cultural, artistic, political, and scientific rebirth following the Middle Ages. Generally described as taking place from the 14th century to the 17th century, the Renaissance promoted the rediscovery of classical philosophy, literature, and art.",
    
    "The Industrial Revolution marked a major turning point in history. Almost every aspect of daily life was influenced in some way. Most notably, average income and population began to exhibit unprecedented sustained growth. Some economists say that the major impact of the Industrial Revolution was that the standard of living for the general population began to increase consistently for the first time in history.",
    
    "World War II was a global war that lasted from 1939 to 1945. It involved the vast majority of the world's countries forming two opposing military alliances: the Allies and the Axis. It was the most widespread war in history, directly involving more than 100 million people from more than 30 countries.",
    
    # Cooking
    "Italian cuisine is characterized by its simplicity, with many dishes having only two to four main ingredients. Italian cooks rely chiefly on the quality of the ingredients rather than on elaborate preparation. Ingredients and dishes vary by region. Many dishes that were once regional have proliferated with variations throughout the country.",
    
    "Japanese cuisine is based on combining staple foods, typically rice or noodles, with a soup and okazu — dishes made from fish, vegetable, tofu and the like — to add flavor to the staple food. Foods of similar flavors are generally combined, and to achieve a balanced meal, foods with opposing tastes are paired.",
    
    "Mexican cuisine is primarily a fusion of indigenous Mesoamerican cooking with European, especially Spanish, influences. Native ingredients include tomatoes, squashes, avocados, cocoa, and vanilla, while European contributions include pork, chicken, beef, cheese, herbs and spices."
]


def main():
    # Check if we have a saved dictionary
    save_path = "elastic_dict_save.pkl"
    if os.path.exists(save_path):
        print(f"Loading existing dictionary from {save_path}...")
        ed = ElasticDictionary.load(save_path)
        print(f"Loaded dictionary with {len(ed)} nodes")
    else:
        print("Creating new ElasticDictionary...")
        ed = ElasticDictionary()
        
        # Process all paragraphs
        print("Processing sample paragraphs...")
        for i, para in enumerate(tqdm(SAMPLE_PARAGRAPHS, desc="Processing paragraphs")):
            ed.add_paragraph(para)
        
        print(f"Created dictionary with {len(ed)} nodes")
        
        # Save the dictionary
        print(f"Saving dictionary to {save_path}...")
        ed.save(save_path)
    
    # Demonstrate searching capabilities
    topics = [
        "artificial intelligence", 
        "biology", 
        "history", 
        "food", 
        "computers",
        "world war",
        "genetic research",
        "ecosystems",
        "renaissance art"
    ]
    
    print("\n--- Searching for topics ---")
    for topic in topics:
        print(f"\nResults for '{topic}':")
        results = ed.find(topic)[:3]  # Get top 3 results
        
        for i, (node, similarity) in enumerate(results):
            print(f"  {i+1}. [{similarity:.2f}] {node.key}")
    
    # Visualize the tree structure
    print("\n--- Visualizing Tree Structure ---")
    ed.visualize()
    
    # Add a new domain of knowledge
    print("\n--- Adding New Domain: Physics ---")
    physics_paragraphs = [
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles. It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.",
        
        "Thermodynamics is the branch of physics that deals with heat, work, and temperature, and their relation to energy, radiation, and physical properties of matter. The behavior of these quantities is governed by the four laws of thermodynamics which convey a quantitative description using measurable macroscopic physical quantities."
    ]
    
    for para in physics_paragraphs:
        print(f"\nAdding paragraph: {para[:50]}...")
        nodes = ed.add_paragraph(para)
        print(f"Added {len(nodes)} nodes")
    
    # Save the updated dictionary
    print(f"\nSaving updated dictionary to {save_path}...")
    ed.save(save_path)
    
    # Search for physics-related topics
    print("\n--- Searching for Physics Topics ---")
    physics_topics = ["quantum", "thermodynamics", "energy", "atoms"]
    
    for topic in physics_topics:
        print(f"\nResults for '{topic}':")
        results = ed.find(topic)[:3]
        
        for i, (node, similarity) in enumerate(results):
            print(f"  {i+1}. [{similarity:.2f}] {node.key}")
    
    # Interactive visualization
    print("\n--- Generating Interactive Visualization ---")
    ed.visualize_interactive()
    
    print("\nElasticDictionary advanced demo completed.")


if __name__ == "__main__":
    main() 