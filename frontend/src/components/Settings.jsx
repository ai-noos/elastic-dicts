import React, { useState } from 'react';
import { dictionaryApi } from '../services/api';

const Settings = ({ onReset, isLoading }) => {
  const [showConfirmation, setShowConfirmation] = useState(false);
  const [resetStatus, setResetStatus] = useState(null);

  const handleResetClick = () => {
    setShowConfirmation(true);
  };

  const handleConfirmReset = async () => {
    try {
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
    }
  };

  const handleCancelReset = () => {
    setShowConfirmation(false);
  };

  return (
    <div className="form-container">
      <h2 className="text-2xl font-bold mb-4">Settings</h2>
      
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