import React, { useState } from 'react';
import { dictionaryApi } from '../services/api';

const Settings = ({ onReset, isLoading, setIsLoading }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [resetStatus, setResetStatus] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [deleteStatus, setDeleteStatus] = useState(null);
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);

  const handleResetClick = () => {
    setShowConfirmation(true);
  };

  const handleConfirmReset = async () => {
    try {
      setIsLoading(true);
      setResetStatus({ type: 'loading', message: 'Resetting dictionary...' });
      await dictionaryApi.resetDictionary();
      setResetStatus({ type: 'success', message: 'Dictionary reset successfully!' });
      setShowConfirmation(false);
      
      // Call the parent component's reset handler
      if (onReset) {
        onReset();
      }
    } catch (error) {
      console.error('Error resetting dictionary:', error);
      setResetStatus({ 
        type: 'error', 
        message: 'Failed to reset dictionary. Please try again.' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelReset = () => {
    setShowConfirmation(false);
  };

  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    
    if (!searchQuery.trim()) {
      return;
    }
    
    try {
      setIsLoading(true);
      setDeleteStatus(null);
      const results = await dictionaryApi.search(searchQuery);
      setSearchResults(results.results || []);
      setSelectedNode(null);
    } catch (error) {
      console.error('Error searching for nodes:', error);
      setDeleteStatus({
        type: 'error',
        message: 'Failed to search for nodes. Please try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleNodeSelect = (node) => {
    setSelectedNode(node);
    setShowDeleteConfirmation(false);
    setDeleteStatus(null);
  };

  const handleDeleteClick = () => {
    if (selectedNode) {
      setShowDeleteConfirmation(true);
    }
  };

  const handleConfirmDelete = async () => {
    try {
      setIsLoading(true);
      setDeleteStatus({ type: 'loading', message: 'Deleting node...' });
      
      await dictionaryApi.deleteNode(selectedNode.key);
      
      setDeleteStatus({ 
        type: 'success', 
        message: `Node "${selectedNode.key}" deleted successfully!` 
      });
      
      setShowDeleteConfirmation(false);
      setSelectedNode(null);
      setSearchResults([]);
      
      // Call the parent component's reset handler to refresh the graph
      if (onReset) {
        onReset();
      }
    } catch (error) {
      console.error('Error deleting node:', error);
      setDeleteStatus({ 
        type: 'error', 
        message: 'Failed to delete node. Please try again.' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteConfirmation(false);
  };

  return (
    <div className="form-container">
      <h2 className="text-2xl font-bold mb-4">Settings</h2>
      
      {/* Node Deletion Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-6">
        <h3 className="text-xl font-semibold mb-4">Delete Node</h3>
        
        <div className="mb-6">
          <p className="text-gray-600 mb-4">
            Search for a node to delete. Deleting a node will reorganize its children in the tree.
          </p>
          
          <form onSubmit={handleSearchSubmit} className="mb-4">
            <div className="flex">
              <input
                type="text"
                className="flex-grow input-field mb-0 rounded-r-none"
                placeholder="Search for a node..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={isLoading}
              />
              <button
                type="submit"
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-l-none rounded-r-md"
                disabled={isLoading || !searchQuery.trim()}
              >
                Search
              </button>
            </div>
          </form>
          
          {searchResults.length > 0 && (
            <div className="mb-4">
              <h4 className="text-lg font-medium mb-2">Search Results</h4>
              <div className="bg-white border border-gray-300 rounded-md max-h-60 overflow-y-auto shadow-sm">
                <ul className="divide-y divide-gray-300">
                  {searchResults.map((result) => (
                    <li 
                      key={result.key}
                      className={`p-3 cursor-pointer hover:bg-indigo-50 ${
                        selectedNode && selectedNode.key === result.key ? 'bg-indigo-100' : ''
                      }`}
                      onClick={() => handleNodeSelect(result)}
                    >
                      <div className="flex justify-between">
                        <span className="font-medium text-gray-900">{result.key}</span>
                        <span className="text-indigo-700 font-medium">
                          Similarity: {(result.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                      {result.value && result.value !== result.key && (
                        <p className="text-sm text-gray-800 mt-1">{result.value}</p>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {selectedNode && !showDeleteConfirmation && (
            <div className="mb-4 p-4 bg-indigo-100 border border-indigo-200 rounded-md">
              <h4 className="text-lg font-medium mb-2">Selected Node</h4>
              <p><span className="font-medium">Key:</span> {selectedNode.key}</p>
              {selectedNode.value && selectedNode.value !== selectedNode.key && (
                <p><span className="font-medium">Value:</span> {selectedNode.value}</p>
              )}
              <div className="mt-3">
                <button
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md"
                  onClick={handleDeleteClick}
                  disabled={isLoading}
                >
                  Delete This Node
                </button>
              </div>
            </div>
          )}
          
          {showDeleteConfirmation && (
            <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
              <p className="text-red-700 font-medium mb-3">
                Are you sure you want to delete the node "{selectedNode.key}"? This will reorganize its children and cannot be undone.
              </p>
              <div className="flex space-x-3">
                <button
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md"
                  onClick={handleConfirmDelete}
                  disabled={isLoading}
                >
                  Yes, Delete Node
                </button>
                <button
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-md"
                  onClick={handleCancelDelete}
                  disabled={isLoading}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
          
          {deleteStatus && (
            <div className={`mt-4 p-3 rounded-md ${
              deleteStatus.type === 'success' 
                ? 'bg-green-50 text-green-700 border border-green-200' 
                : deleteStatus.type === 'error'
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-blue-50 text-blue-700 border border-blue-200'
            }`}>
              <p>{deleteStatus.message}</p>
            </div>
          )}
        </div>
      </div>
      
      {/* Reset Dictionary Section */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-xl font-semibold mb-4">Dictionary Management</h3>
        
        <div className="mb-6">
          <p className="text-gray-600 mb-4">
            Reset the dictionary to remove all items and start fresh. This action cannot be undone.
          </p>
          
          {!showConfirmation ? (
            <button
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md"
              onClick={handleResetClick}
              disabled={isLoading}
            >
              Reset Dictionary
            </button>
          ) : (
            <div className="bg-red-50 border border-red-200 rounded-md p-4">
              <p className="text-red-700 font-medium mb-3">
                Are you sure you want to reset the dictionary? This will remove all items and cannot be undone.
              </p>
              <div className="flex space-x-3">
                <button
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md"
                  onClick={handleConfirmReset}
                  disabled={isLoading}
                >
                  Yes, Reset Dictionary
                </button>
                <button
                  className="bg-gray-200 hover:bg-gray-300 text-gray-800 px-4 py-2 rounded-md"
                  onClick={handleCancelReset}
                  disabled={isLoading}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
        
        {resetStatus && (
          <div className={`mt-4 p-3 rounded-md ${
            resetStatus.type === 'success' 
              ? 'bg-green-50 text-green-700 border border-green-200' 
              : resetStatus.type === 'error'
                ? 'bg-red-50 text-red-700 border border-red-200'
                : 'bg-blue-50 text-blue-700 border border-blue-200'
          }`}>
            <p>{resetStatus.message}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings; 