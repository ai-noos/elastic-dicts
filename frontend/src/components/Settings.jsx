import React, { useState } from 'react';
import { dictionaryApi } from '../services/api';
import {
  Settings2,
  Search,
  Trash2,
  AlertTriangle,
  RefreshCw,
  X,
  Check,
  ArrowRight,
  Loader2
} from 'lucide-react';

const Settings = ({ onReset, isLoading, setIsLoading }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [resetStatus, setResetStatus] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [deleteStatus, setDeleteStatus] = useState(null);
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);
  const [rebuildStatus, setRebuildStatus] = useState(null);
  const [showRebuildConfirmation, setShowRebuildConfirmation] = useState(false);

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

  const handleRebuildClick = () => {
    setShowRebuildConfirmation(true);
  };

  const handleConfirmRebuild = async () => {
    try {
      setIsLoading(true);
      setRebuildStatus({ type: 'loading', message: 'Rebuilding tree structure...' });
      
      await dictionaryApi.rebuildTree();
      
      setRebuildStatus({ 
        type: 'success', 
        message: 'Tree structure rebuilt successfully!' 
      });
      
      setShowRebuildConfirmation(false);
      
      // Call the parent component's reset handler to refresh the graph
      if (onReset) {
        onReset();
      }
    } catch (error) {
      console.error('Error rebuilding tree:', error);
      setRebuildStatus({ 
        type: 'error', 
        message: 'Failed to rebuild tree. Please try again.' 
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelRebuild = () => {
    setShowRebuildConfirmation(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-indigo-100 rounded-lg">
          <Settings2 className="w-5 h-5 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Settings</h2>
      </div>
      
      {/* Node Deletion Section */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 space-y-6">
        <div className="flex items-center space-x-3">
          <Trash2 className="w-5 h-5 text-gray-400" />
          <h3 className="text-xl font-semibold text-gray-800">Delete Node</h3>
        </div>
        
        <div className="space-y-4">
          <p className="text-gray-600">
            Search for a node to delete. Deleting a node will reorganize its children in the tree.
          </p>
          
          <form onSubmit={handleSearchSubmit}>
            <div className="relative">
              <input
                type="text"
                className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
                placeholder="Search for a node..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={isLoading}
              />
              <Search className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
              <button
                type="submit"
                className={`absolute right-2 top-1/2 transform -translate-y-1/2 px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  isLoading || !searchQuery.trim()
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-indigo-600 text-white hover:bg-indigo-700'
                }`}
                disabled={isLoading || !searchQuery.trim()}
              >
                Search
              </button>
            </div>
          </form>
          
          {searchResults.length > 0 && (
            <div className="space-y-3">
              <h4 className="text-lg font-medium text-gray-800">Search Results</h4>
              <div className="bg-white border border-gray-200 rounded-xl max-h-60 overflow-y-auto divide-y divide-gray-100">
                {searchResults.map((result) => (
                  <div 
                    key={result.key}
                    className={`p-4 cursor-pointer transition-colors hover:bg-gray-50 ${
                      selectedNode && selectedNode.key === result.key ? 'bg-indigo-50' : ''
                    }`}
                    onClick={() => handleNodeSelect(result)}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <ArrowRight className="w-4 h-4 text-indigo-600" />
                        <span className="font-medium text-gray-900">{result.key}</span>
                      </div>
                      <div className="flex items-center space-x-1 px-3 py-1 bg-indigo-50 rounded-full">
                        <span className="text-sm font-medium text-indigo-700">
                          {(result.similarity * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {result.value && result.value !== result.key && (
                      <p className="text-sm text-gray-600 mt-2 ml-7">{result.value}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {selectedNode && !showDeleteConfirmation && (
            <div className="p-4 bg-indigo-50 border border-indigo-100 rounded-xl space-y-3">
              <h4 className="text-lg font-medium text-gray-800">Selected Node</h4>
              <div className="space-y-2">
                <p className="flex items-center space-x-2">
                  <span className="font-medium text-gray-700">Key:</span>
                  <span className="text-gray-600">{selectedNode.key}</span>
                </p>
                {selectedNode.value && selectedNode.value !== selectedNode.key && (
                  <p className="flex items-center space-x-2">
                    <span className="font-medium text-gray-700">Value:</span>
                    <span className="text-gray-600">{selectedNode.value}</span>
                  </p>
                )}
              </div>
              <button
                className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors disabled:bg-red-300 disabled:cursor-not-allowed"
                onClick={handleDeleteClick}
                disabled={isLoading}
              >
                <Trash2 className="w-5 h-5" />
                <span>Delete This Node</span>
              </button>
            </div>
          )}
          
          {showDeleteConfirmation && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-xl space-y-4">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-1" />
                <p className="text-red-700">
                  Are you sure you want to delete the node "{selectedNode.key}"? This will reorganize its children and cannot be undone.
                </p>
              </div>
              <div className="flex space-x-3">
                <button
                  className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex-1 justify-center"
                  onClick={handleConfirmDelete}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      <span>Deleting...</span>
                    </>
                  ) : (
                    <>
                      <Check className="w-5 h-5" />
                      <span>Yes, Delete Node</span>
                    </>
                  )}
                </button>
                <button
                  className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex-1 justify-center"
                  onClick={handleCancelDelete}
                >
                  <X className="w-5 h-5" />
                  <span>Cancel</span>
                </button>
              </div>
            </div>
          )}
          
          {deleteStatus && (
            <div className={`p-4 rounded-xl flex items-center space-x-3 ${
              deleteStatus.type === 'error' ? 'bg-red-50 text-red-700' :
              deleteStatus.type === 'success' ? 'bg-green-50 text-green-700' :
              'bg-blue-50 text-blue-700'
            }`}>
              {deleteStatus.type === 'error' ? (
                <AlertTriangle className="w-5 h-5 flex-shrink-0" />
              ) : deleteStatus.type === 'success' ? (
                <Check className="w-5 h-5 flex-shrink-0" />
              ) : (
                <Loader2 className="w-5 h-5 animate-spin flex-shrink-0" />
              )}
              <p>{deleteStatus.message}</p>
            </div>
          )}
        </div>
      </div>

      {/* Tree Management Section */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-200 space-y-6">
        <div className="flex items-center space-x-3">
          <RefreshCw className="w-5 h-5 text-gray-400" />
          <h3 className="text-xl font-semibold text-gray-800">Tree Management</h3>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          {/* Reset Dictionary */}
          <div className="p-4 border border-gray-200 rounded-xl space-y-3">
            <h4 className="font-medium text-gray-800">Reset Dictionary</h4>
            <p className="text-sm text-gray-600">
              Clear all items from the dictionary and start fresh.
            </p>
            {!showConfirmation ? (
              <button
                className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors w-full justify-center"
                onClick={handleResetClick}
                disabled={isLoading}
              >
                <RefreshCw className="w-5 h-5" />
                <span>Reset Dictionary</span>
              </button>
            ) : (
              <div className="space-y-3">
                <div className="flex items-start space-x-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 flex-shrink-0 mt-1" />
                  <p className="text-red-700 text-sm">
                    Are you sure? This will delete all items and cannot be undone.
                  </p>
                </div>
                <div className="flex space-x-3">
                  <button
                    className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors flex-1 justify-center"
                    onClick={handleConfirmReset}
                    disabled={isLoading}
                  >
                    <Check className="w-5 h-5" />
                    <span>Yes, Reset</span>
                  </button>
                  <button
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex-1 justify-center"
                    onClick={handleCancelReset}
                  >
                    <X className="w-5 h-5" />
                    <span>Cancel</span>
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Rebuild Tree */}
          <div className="p-4 border border-gray-200 rounded-xl space-y-3">
            <h4 className="font-medium text-gray-800">Rebuild Tree</h4>
            <p className="text-sm text-gray-600">
              Reorganize the tree structure for optimal arrangement.
            </p>
            {!showRebuildConfirmation ? (
              <button
                className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors w-full justify-center"
                onClick={handleRebuildClick}
                disabled={isLoading}
              >
                <RefreshCw className="w-5 h-5" />
                <span>Rebuild Tree</span>
              </button>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-gray-700">
                  This will reorganize all items in the tree for optimal structure.
                </p>
                <div className="flex space-x-3">
                  <button
                    className="flex items-center space-x-2 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex-1 justify-center"
                    onClick={handleConfirmRebuild}
                    disabled={isLoading}
                  >
                    <Check className="w-5 h-5" />
                    <span>Yes, Rebuild</span>
                  </button>
                  <button
                    className="flex items-center space-x-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex-1 justify-center"
                    onClick={handleCancelRebuild}
                  >
                    <X className="w-5 h-5" />
                    <span>Cancel</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {(resetStatus || rebuildStatus) && (
          <div className={`p-4 rounded-xl flex items-center space-x-3 ${
            (resetStatus?.type === 'error' || rebuildStatus?.type === 'error') ? 'bg-red-50 text-red-700' :
            (resetStatus?.type === 'success' || rebuildStatus?.type === 'success') ? 'bg-green-50 text-green-700' :
            'bg-blue-50 text-blue-700'
          }`}>
            {(resetStatus?.type === 'error' || rebuildStatus?.type === 'error') ? (
              <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            ) : (resetStatus?.type === 'success' || rebuildStatus?.type === 'success') ? (
              <Check className="w-5 h-5 flex-shrink-0" />
            ) : (
              <Loader2 className="w-5 h-5 animate-spin flex-shrink-0" />
            )}
            <p>{resetStatus?.message || rebuildStatus?.message}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings; 