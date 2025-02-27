import { useState, useEffect } from 'react';
import Graph from './components/Graph';
import InputForm from './components/InputForm';
import SearchForm from './components/SearchForm';
import Settings from './components/Settings';
import Documentation from './components/Documentation';
import { dictionaryApi } from './services/api';
import { SpeedInsights } from "@vercel/speed-insights/react";
import {
  PlusCircle,
  Search,
  Settings as SettingsIcon,
  BookOpen,
  Github,
  Brain,
  Loader2,
  AlertCircle
} from 'lucide-react';

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [isLoading, setIsLoading] = useState(false);
  const [searchResults, setSearchResults] = useState(null);
  const [activeTab, setActiveTab] = useState('add'); // 'add', 'search', 'settings', or 'docs'
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

  const handleDictionaryReset = async () => {
    // Refresh the graph data after reset
    await fetchDictionaryState();
    // Clear any selected node
    setSelectedNode(null);
    // Clear search results
    setSearchResults(null);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-8">
        <div className="container mx-auto flex items-center space-x-4">
          <Brain className="w-10 h-10" />
          <div>
            <h1 className="text-4xl font-bold">Elastic Dictionary</h1>
            <p className="mt-2 text-indigo-100">An adaptive semantic data structure that organizes information based on meaning</p>
          </div>
        </div>
      </header>

      <main className="container mx-auto py-8 px-4">
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 flex items-center space-x-2">
            <AlertCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        )}

        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4 flex items-center space-x-2">
            <div className="p-2 bg-indigo-100 rounded-lg">
              <Brain className="w-6 h-6 text-indigo-600" />
            </div>
            <span>Interactive Visualization</span>
          </h2>
          
          {isLoading && (
            <div className="absolute inset-0 bg-white/50 flex items-center justify-center z-10">
              <Loader2 className="w-8 h-8 text-indigo-600 animate-spin" />
            </div>
          )}
          
          <Graph 
            graphData={graphData} 
            onNodeClick={handleNodeClick} 
          />
          
          {selectedNode && (
            <div className="mt-4 p-4 bg-white rounded-md shadow-lg border border-indigo-100">
              <h3 className="text-lg font-semibold text-indigo-700">{selectedNode.name}</h3>
              <p className="text-gray-600">Type: {selectedNode.is_category ? 'Category' : 'Item'}</p>
            </div>
          )}
        </div>

        <div className="bg-white rounded-lg shadow-lg overflow-hidden border border-gray-200">
          <div className="flex border-b">
            <button
              className={`flex-1 py-4 px-6 text-center font-medium flex items-center justify-center space-x-2 ${
                activeTab === 'add' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('add')}
            >
              <PlusCircle className="w-5 h-5" />
              <span>Add Items</span>
            </button>
            <button
              className={`flex-1 py-4 px-6 text-center font-medium flex items-center justify-center space-x-2 ${
                activeTab === 'search' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('search')}
            >
              <Search className="w-5 h-5" />
              <span>Search</span>
            </button>
            <button
              className={`flex-1 py-4 px-6 text-center font-medium flex items-center justify-center space-x-2 ${
                activeTab === 'settings' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('settings')}
            >
              <SettingsIcon className="w-5 h-5" />
              <span>Settings</span>
            </button>
            <button
              className={`flex-1 py-4 px-6 text-center font-medium flex items-center justify-center space-x-2 ${
                activeTab === 'docs' ? 'bg-indigo-50 text-indigo-700 border-b-2 border-indigo-500' : 'text-gray-600 hover:bg-gray-50'
              }`}
              onClick={() => setActiveTab('docs')}
            >
              <BookOpen className="w-5 h-5" />
              <span>Documentation</span>
            </button>
          </div>

          <div className="p-6">
            {activeTab === 'add' ? (
              <InputForm 
                onAddItem={handleAddItem} 
                onAddParagraph={handleAddParagraph} 
                isLoading={isLoading} 
              />
            ) : activeTab === 'search' ? (
              <SearchForm 
                onSearch={handleSearch} 
                isLoading={isLoading} 
                searchResults={searchResults} 
              />
            ) : activeTab === 'docs' ? (
              <Documentation />
            ) : (
              <Settings
                onReset={handleDictionaryReset}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            )}
          </div>
        </div>
      </main>

      <footer className="bg-gradient-to-r from-gray-800 to-gray-900 text-white py-8 px-4">
        <div className="container mx-auto flex justify-between items-center">
          <p className="flex items-center space-x-2">
            <Brain className="w-5 h-5" />
            <span>Elastic Dictionary - A semantic data structure for organizing information</span>
          </p>
          <div className="flex items-center space-x-4">
            <a 
              href="https://github.com/ai-noos/elastic-dicts" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-white hover:text-indigo-300 transition-colors flex items-center space-x-2 bg-gray-700 px-4 py-2 rounded-lg hover:bg-gray-600"
            >
              <Github className="w-5 h-5" />
              <span>GitHub</span>
            </a>
            <span className="text-gray-400">© {new Date().getFullYear()}</span>
          </div>
          <SpeedInsights />
        </div>
      </footer>
    </div>
  );
}

export default App; 