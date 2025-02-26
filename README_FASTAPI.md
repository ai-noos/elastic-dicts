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

- Interactive 3D visualization of the elastic dictionary
- Add single items, multiple items, or paragraphs to the dictionary
- Search for items in the dictionary
- Real-time updates to the visualization as items are added

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

When you add items to the dictionary through the web interface, they are sent to the backend API, which processes them and updates the dictionary. The frontend then fetches the updated state and renders the visualization.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 