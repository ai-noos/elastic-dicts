import React, { useState } from 'react';
import {
  Search,
  Loader2,
  Hash,
  ArrowRight,
  AlertCircle
} from 'lucide-react';

const SearchForm = ({ onSearch, isLoading, searchResults }) => {
  const [query, setQuery] = useState('');
  const [limit, setLimit] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (query.trim()) {
      onSearch(query.trim(), limit);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-indigo-100 rounded-lg">
          <Search className="w-5 h-5 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Search Elastic Dictionary</h2>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="query" className="block text-sm font-medium text-gray-700 mb-2">
            Search query:
          </label>
          <div className="relative">
            <input
              type="text"
              id="query"
              className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., fruit"
              disabled={isLoading}
            />
            <Search className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
          </div>
        </div>
        
        <div>
          <label htmlFor="limit" className="block text-sm font-medium text-gray-700 mb-2">
            Maximum results:
          </label>
          <div className="relative">
            <input
              type="number"
              id="limit"
              className="w-full pl-10 pr-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
              min="1"
              max="50"
              disabled={isLoading}
            />
            <Hash className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
          </div>
        </div>
        
        <button
          type="submit"
          className={`w-full flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
            isLoading || !query.trim()
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Searching...</span>
            </>
          ) : (
            <>
              <Search className="w-5 h-5" />
              <span>Search</span>
            </>
          )}
        </button>
      </form>
      
      {searchResults && searchResults.results && searchResults.results.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-xl font-semibold text-gray-800">Search Results</h3>
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm divide-y divide-gray-100">
            {searchResults.results.map((result, index) => (
              <div key={index} className="p-4 transition-colors hover:bg-gray-50">
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
      
      {searchResults && searchResults.results && searchResults.results.length === 0 && (
        <div className="flex items-center justify-center space-x-3 p-8 bg-gray-50 border border-gray-200 rounded-xl text-gray-500">
          <AlertCircle className="w-5 h-5" />
          <p>No results found for "{query}"</p>
        </div>
      )}
    </div>
  );
};

export default SearchForm; 