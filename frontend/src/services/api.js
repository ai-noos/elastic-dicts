import axios from 'axios';

const API_URL = '/api/v1';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
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
      const response = await api.post('/dictionary/add-batch', { items });
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
};

export default api; 