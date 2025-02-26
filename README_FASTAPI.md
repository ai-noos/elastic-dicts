# Elastic Dictionary Web Application

This project provides a web interface for the Elastic Dictionary, a semantic data structure that organizes information based on meaning. The application consists of a FastAPI backend and a React frontend.

## Project Structure

```
elastic-dict/
├── app/                    # FastAPI backend
│   ├── api/                # API routes
│   │   └── endpoints/      # API endpoints
│   ├── core/               # Core application components
│   ├── models/             # Data models
│   └── services/           # Business logic
├── data/                   # Data storage
├── frontend/               # React frontend
│   ├── public/             # Static assets
│   └── src/                # Source code
│       ├── components/     # React components
│       └── services/       # API services
├── elastic_dict.py         # Core elastic dictionary implementation
└── run_api.py              # Script to run the FastAPI application
```

## Features

- **Interactive 3D Visualization**: Dynamic graph with visible node labels and customizable appearance
- **Data Management**: Add single items, multiple items, or paragraphs to the dictionary
- **Semantic Search**: Find items in the dictionary based on meaning, not just exact matches
- **Real-time Updates**: Live visualization updates as items are added
- **Dictionary Reset**: Clear the dictionary and start fresh through the Settings tab
- **Responsive UI**: User-friendly interface with tabbed navigation

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn

### Backend Setup

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Run the FastAPI backend:

```bash
python run_api.py
```

The API will be available at http://localhost:8000.

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install the required npm packages:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

The frontend will be available at http://localhost:5173.

## API Documentation

Once the backend is running, you can access the API documentation at http://localhost:8000/docs.

## How It Works

The Elastic Dictionary is a data structure that organizes text entries based on semantic similarity, creating a hierarchical structure. It uses sentence transformers for embeddings and provides visualization capabilities.

### Key Components:

1. **Backend (FastAPI)**:
   - RESTful API for dictionary operations
   - Singleton service pattern for dictionary management
   - Persistent storage of dictionary state

2. **Frontend (React)**:
   - 3D force-directed graph visualization using Three.js
   - Intuitive forms for adding content
   - Semantic search interface
   - Settings panel for dictionary management

3. **Data Flow**:
   - User inputs text through the frontend
   - Data is sent to the backend API
   - Backend processes the text and updates the dictionary
   - Frontend fetches the updated state and renders the visualization
   - Graph updates in real-time to show the new structure

## Visualization Features

- **Node Representation**: 
  - Categories shown as spheres
  - Items shown as cubes
  - Size indicates importance in the hierarchy
  - Color indicates depth in the tree

- **Labels**: 
  - Always visible node labels
  - Automatic truncation for long text
  - Special handling for paragraphs (first word + "...")

- **Interaction**:
  - Click nodes to see details
  - Rotate, zoom, and pan the 3D view
  - Hover for additional information

## License

This project is licensed under the MIT License - see the LICENSE file for details. 