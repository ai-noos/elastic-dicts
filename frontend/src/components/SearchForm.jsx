import React, { useState } from 'react';

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
    <div className="form-container">
      <h2 className="text-2xl font-bold mb-4">Search Elastic Dictionary</h2>
      
      <form onSubmit={handleSubmit} className="mb-6">
        <div className="mb-4">
          <label htmlFor="query" className="block text-sm font-medium mb-1">
            Search query:
          </label>
          <input
            type="text"
            id="query"
            className="input-field"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., fruit"
            disabled={isLoading}
          />
        </div>
        
        <div className="mb-4">
          <label htmlFor="limit" className="block text-sm font-medium mb-1">
            Maximum results:
          </label>
          <input
            type="number"
            id="limit"
            className="input-field"
            value={limit}
            onChange={(e) => setLimit(parseInt(e.target.value) || 10)}
            min="1"
            max="50"
            disabled={isLoading}
          />
        </div>
        
        <button
          type="submit"
          className="button"
          disabled={isLoading || !query.trim()}
        >
          {isLoading ? 'Searching...' : 'Search'}
        </button>
      </form>
      
      {searchResults && searchResults.results && searchResults.results.length > 0 && (
        <div>
          <h3 className="text-xl font-semibold mb-2">Search Results</h3>
          <div className="bg-white p-4 rounded-md border border-gray-300 shadow-sm">
            <ul className="divide-y divide-gray-300">
              {searchResults.results.map((result, index) => (
                <li key={index} className="py-3">
                  <div className="flex justify-between">
                    <span className="font-medium text-gray-900">{result.key}</span>
                    <span className="text-indigo-700 font-medium">Similarity: {(result.similarity * 100).toFixed(1)}%</span>
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
      
      {searchResults && searchResults.results && searchResults.results.length === 0 && (
        <div className="text-center p-4 bg-white border border-gray-300 rounded-md">
          <p className="text-gray-800">No results found for "{query}"</p>
        </div>
      )}
    </div>
  );
};

export default SearchForm; 