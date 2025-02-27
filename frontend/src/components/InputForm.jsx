import React, { useState } from 'react';
import {
  PlusCircle,
  Type,
  ListPlus,
  FileText,
  Loader2
} from 'lucide-react';

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
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-indigo-100 rounded-lg">
          <PlusCircle className="w-5 h-5 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-800">Add to Elastic Dictionary</h2>
      </div>
      
      <div className="grid grid-cols-3 gap-4">
        <button
          type="button"
          className={`p-4 rounded-xl border-2 transition-all ${
            inputType === 'item' 
              ? 'border-indigo-600 bg-indigo-50 text-indigo-700' 
              : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'
          }`}
          onClick={() => setInputType('item')}
        >
          <div className="flex flex-col items-center space-y-2">
            <Type className={`w-6 h-6 ${inputType === 'item' ? 'text-indigo-600' : 'text-gray-400'}`} />
            <span className="font-medium">Single Item</span>
          </div>
        </button>
        <button
          type="button"
          className={`p-4 rounded-xl border-2 transition-all ${
            inputType === 'batch' 
              ? 'border-indigo-600 bg-indigo-50 text-indigo-700' 
              : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'
          }`}
          onClick={() => setInputType('batch')}
        >
          <div className="flex flex-col items-center space-y-2">
            <ListPlus className={`w-6 h-6 ${inputType === 'batch' ? 'text-indigo-600' : 'text-gray-400'}`} />
            <span className="font-medium">Multiple Items</span>
          </div>
        </button>
        <button
          type="button"
          className={`p-4 rounded-xl border-2 transition-all ${
            inputType === 'paragraph' 
              ? 'border-indigo-600 bg-indigo-50 text-indigo-700' 
              : 'border-gray-200 hover:border-indigo-300 hover:bg-gray-50'
          }`}
          onClick={() => setInputType('paragraph')}
        >
          <div className="flex flex-col items-center space-y-2">
            <FileText className={`w-6 h-6 ${inputType === 'paragraph' ? 'text-indigo-600' : 'text-gray-400'}`} />
            <span className="font-medium">Paragraph</span>
          </div>
        </button>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {inputType === 'item' && (
          <div>
            <label htmlFor="item" className="block text-sm font-medium text-gray-700 mb-2">
              Enter a single item:
            </label>
            <input
              type="text"
              id="item"
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="e.g., apple"
              disabled={isLoading}
            />
          </div>
        )}
        
        {inputType === 'batch' && (
          <div>
            <label htmlFor="batch" className="block text-sm font-medium text-gray-700 mb-2">
              Enter multiple items (one per line):
            </label>
            <textarea
              id="batch"
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
              value={batchItems}
              onChange={(e) => setBatchItems(e.target.value)}
              placeholder="e.g.,&#10;apple&#10;banana&#10;orange"
              rows={5}
              disabled={isLoading}
            />
          </div>
        )}
        
        {inputType === 'paragraph' && (
          <div>
            <label htmlFor="paragraph" className="block text-sm font-medium text-gray-700 mb-2">
              Enter a paragraph:
            </label>
            <textarea
              id="paragraph"
              className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-colors disabled:bg-gray-100 disabled:cursor-not-allowed"
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
          className={`w-full flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
            isLoading || (inputType === 'item' && !inputValue.trim()) || 
            (inputType === 'batch' && !batchItems.trim()) || 
            (inputType === 'paragraph' && !inputValue.trim())
              ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
              : 'bg-indigo-600 text-white hover:bg-indigo-700'
          }`}
          disabled={isLoading || (inputType === 'item' && !inputValue.trim()) || (inputType === 'batch' && !batchItems.trim()) || (inputType === 'paragraph' && !inputValue.trim())}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Adding...</span>
            </>
          ) : (
            <>
              <PlusCircle className="w-5 h-5" />
              <span>Add to Dictionary</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default InputForm; 