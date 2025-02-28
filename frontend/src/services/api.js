import axios from 'axios';

// Get the environment-specific API URL
const API_URL = import.meta.env.VITE_API_URL;

// Log the current environment and API URL
console.log('Environment:', import.meta.env.MODE);
console.log('API URL:', API_URL);

if (!API_URL) {
  console.error('API URL not configured for current environment!');
}

// Function to get or create a user ID
const getUserId = () => {
  let userId = localStorage.getItem('userId');
  if (!userId) {
    // Generate a random user ID if none exists
    userId = 'user_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('userId', userId);
  }
  return userId;
};

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add user ID to every request
api.interceptors.request.use((config) => {
  config.headers['X-User-ID'] = getUserId();
  return config;
});

export const dictionaryApi = {
  // Get the current state of the dictionary
  getDictionaryState: async () => {
    try {
      const response = await api.get('/dictionary/state');
      return response.data;
    } catch (error) {
      console.error('Error getting dictionary state:', error);
      throw error;
    }
  },

  // Add a single item to the dictionary
  addItem: async (item) => {
    try {
      const response = await api.post('/dictionary/add', { item });
      return response.data;
    } catch (error) {
      console.error('Error adding item:', error);
      throw error;
    }
  },

  // Add multiple items to the dictionary
  addBatch: async (items) => {
    try {
      const response = await api.post('/dictionary/add-batch', { items: items });
      return response.data;
    } catch (error) {
      console.error('Error adding batch:', error);
      throw error;
    }
  },

  // Add a paragraph to the dictionary
  addParagraph: async (paragraph) => {
    try {
      const response = await api.post('/dictionary/add-paragraph', { paragraph });
      return response.data;
    } catch (error) {
      console.error('Error adding paragraph:', error);
      throw error;
    }
  },

  // Search the dictionary
  search: async (query, limit = 10) => {
    try {
      const response = await api.post('/dictionary/search', { query, limit });
      return response.data;
    } catch (error) {
      console.error('Error searching:', error);
      throw error;
    }
  },
  
  // Delete a node from the dictionary
  deleteNode: async (nodeKey) => {
    try {
      const response = await api.post('/dictionary/delete-node', { node_key: nodeKey });
      return response.data;
    } catch (error) {
      console.error('Error deleting node:', error);
      throw error;
    }
  },
  
  // Reset the dictionary (remove all items)
  resetDictionary: async () => {
    try {
      const response = await api.post('/dictionary/reset');
      return response.data;
    } catch (error) {
      console.error('Error resetting dictionary:', error);
      throw error;
    }
  },
  
  // Rebuild the dictionary tree structure
  rebuildTree: async () => {
    try {
      const response = await api.post('/dictionary/rebuild');
      return response.data;
    } catch (error) {
      console.error('Error rebuilding dictionary tree:', error);
      throw error;
    }
  }
};

export default api; 