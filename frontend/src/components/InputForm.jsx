import React, { useState } from 'react';

const InputForm = ({ onAddItem, onAddParagraph, isLoading }) => {
  const [inputType, setInputType] = useState('item'); // 'item' or 'paragraph'
  const [inputValue, setInputValue] = useState('');
  const [batchItems, setBatchItems] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (inputType === 'item') {
      if (inputValue.trim()) {
        onAddItem(inputValue.trim());
        setInputValue('');
      }
    } else if (inputType === 'batch') {
      const items = batchItems
        .split('\n')
        .map(item => item.trim())
        .filter(item => item.length > 0);
      
      if (items.length > 0) {
        onAddItem(items);
        setBatchItems('');
      }
    } else if (inputType === 'paragraph') {
      if (inputValue.trim()) {
        onAddParagraph(inputValue.trim());
        setInputValue('');
      }
    }
  };

  return (
    <div className="form-container">
      <h2 className="text-2xl font-bold mb-4">Add to Elastic Dictionary</h2>
      
      <div className="mb-4">
        <div className="flex space-x-4">
          <button
            type="button"
            className={`px-4 py-2 rounded-md ${inputType === 'item' ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setInputType('item')}
          >
            Single Item
          </button>
          <button
            type="button"
            className={`px-4 py-2 rounded-md ${inputType === 'batch' ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setInputType('batch')}
          >
            Multiple Items
          </button>
          <button
            type="button"
            className={`px-4 py-2 rounded-md ${inputType === 'paragraph' ? 'bg-indigo-600 text-white' : 'bg-gray-200 text-gray-800'}`}
            onClick={() => setInputType('paragraph')}
          >
            Paragraph
          </button>
        </div>
      </div>
      
      <form onSubmit={handleSubmit}>
        {inputType === 'item' && (
          <div className="mb-4">
            <label htmlFor="item" className="block text-sm font-medium mb-1">
              Enter a single item:
            </label>
            <input
              type="text"
              id="item"
              className="input-field"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="e.g., apple"
              disabled={isLoading}
            />
          </div>
        )}
        
        {inputType === 'batch' && (
          <div className="mb-4">
            <label htmlFor="batch" className="block text-sm font-medium mb-1">
              Enter multiple items (one per line):
            </label>
            <textarea
              id="batch"
              className="input-field"
              value={batchItems}
              onChange={(e) => setBatchItems(e.target.value)}
              placeholder="e.g.,&#10;apple&#10;banana&#10;orange"
              rows={5}
              disabled={isLoading}
            />
          </div>
        )}
        
        {inputType === 'paragraph' && (
          <div className="mb-4">
            <label htmlFor="paragraph" className="block text-sm font-medium mb-1">
              Enter a paragraph:
            </label>
            <textarea
              id="paragraph"
              className="input-field"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="e.g., Fruits are nutritious and delicious. Apples and bananas are popular choices."
              rows={5}
              disabled={isLoading}
            />
          </div>
        )}
        
        <button
          type="submit"
          className="button"
          disabled={isLoading || (inputType === 'item' && !inputValue.trim()) || (inputType === 'batch' && !batchItems.trim()) || (inputType === 'paragraph' && !inputValue.trim())}
        >
          {isLoading ? 'Adding...' : 'Add to Dictionary'}
        </button>
      </form>
    </div>
  );
};

export default InputForm; 