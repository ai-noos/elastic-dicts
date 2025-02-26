import { useState, useEffect } from 'react';
import Graph from './components/Graph';
import InputForm from './components/InputForm';
import SearchForm from './components/SearchForm';
import { dictionaryApi } from './services/api';

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [activeTab, setActiveTab] = useState('add'); // 'add' or 'search'
  const [selectedNode, setSelectedNode] = useState(null);
  const [error, setError] = useState(null);

  // Fetch initial dictionary state
  useEffect(() => {
    fetchDictionaryState();
  }, []);

  const fetchDictionaryState = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await dictionaryApi.getDictionaryState();
      setGraphData(data.graph_data);
    } catch (error) {
      console.error('Error fetching dictionary state:', error);
      setError('Failed to load dictionary data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddItem = async (item) => {
    try {
      setIsLoading(true);
      setError(null);
      
      if (Array.isArray(item)) {
        // Handle batch of items
        await dictionaryApi.addBatch(item);
      } else {
        // Handle single item
        await dictionaryApi.addItem(item);
      }
      
      // Refresh the graph data
      await fetchDictionaryState();
    } catch (error) {
      console.error('Error adding item:', error);
      setError('Failed to add item. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleAddParagraph = async (paragraph) => {
    try {
      setIsLoading(true);
      setError(null);
      await dictionaryApi.addParagraph(paragraph);
      
      // Refresh the graph data
      await fetchDictionaryState();
    } catch (error) {
      console.error('Error adding paragraph:', error);
      setError('Failed to add paragraph. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async (query, limit) => {
    try {
      setIsLoading(true);
      setError(null);
      const results = await dictionaryApi.search(query, limit);
      setSearchResults(results);
    } catch (error) {
      console.error('Error searching:', error);
      setError('Failed to search. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeClick = (node) => {
    setSelectedNode(node);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-indigo-600 text-white p-6">
        <div className="container mx-auto">
          <h1 className="text-3xl font-bold">Elastic Dictionary</h1>
          <p className="mt-2">An adaptive semantic data structure that organizes information based on meaning</p>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            <p>{error}</p>
          </div>
        )}

        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Interactive Visualization</h2>
          <Graph 
            graphData={graphData} 
            onNodeClick={handleNodeClick} 
          />
          
          {selectedNode && (
            <div className="mt-4 p-4 bg-white rounded-md shadow">
              <h3 className="text-lg font-semibold">{selectedNode.name}</h3>
              <p className="text-gray-600">Type: {selectedNode.is_category ? 'Category' : 'Item'}</p>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-md overflow-hidden">
          <div className="flex border-b">
            <button
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === 'add' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600'
              }`}
              onClick={() => setActiveTab('add')}
            >
              Add Items
            </button>
            <button
              className={`flex-1 py-3 px-4 text-center font-medium ${
                activeTab === 'search' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600'
              }`}
              onClick={() => setActiveTab('search')}
            >
              Search
            </button>
          </div>

          <div className="p-6">
            {activeTab === 'add' ? (
              <InputForm 
                onAddItem={handleAddItem} 
                onAddParagraph={handleAddParagraph} 
                isLoading={isLoading} 
              />
            ) : (
              <SearchForm 
                onSearch={handleSearch} 
                isLoading={isLoading} 
                searchResults={searchResults} 
              />
            )}
          </div>
        </div>
      </main>

      <footer className="bg-gray-800 text-white py-8 px-4">
        <div className="container mx-auto">
          <p className="text-center">Elastic Dictionary - A semantic data structure for organizing information</p>
        </div>
      </footer>
    </div>
  );
}

export default App; 