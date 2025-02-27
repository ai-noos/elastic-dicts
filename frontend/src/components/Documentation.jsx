import React from 'react';
import {
  Brain,
  Sparkles,
  PlayCircle,
  Code2,
  Lightbulb,
  Search,
  Eye,
  Settings2,
  Cpu
} from 'lucide-react';

const Documentation = () => {
  return (
    <div className="space-y-8">
      <section className="bg-gradient-to-r from-indigo-50 to-purple-50 p-6 rounded-xl border border-indigo-100">
        <div className="flex items-start space-x-4">
          <div className="p-3 bg-white rounded-lg shadow-md">
            <Brain className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3 text-indigo-700">What is Elastic Dictionary?</h3>
            <p className="text-gray-600 leading-relaxed">
              Elastic Dictionary is an adaptive, hierarchical data structure that dynamically organizes text data based on semantic meaning. 
              Unlike traditional dictionaries, it creates a self-organizing tree structure that evolves as new information is added.
            </p>
          </div>
        </div>
      </section>

      <section>
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <Sparkles className="w-5 h-5 text-indigo-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-800">Key Features</h3>
        </div>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3 mb-2">
              <Brain className="w-5 h-5 text-indigo-600" />
              <h4 className="font-medium text-gray-800">Dynamic Organization</h4>
            </div>
            <p className="text-gray-600">Automatically organizes entries based on their semantic meaning</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3 mb-2">
              <Lightbulb className="w-5 h-5 text-indigo-600" />
              <h4 className="font-medium text-gray-800">Semantic Understanding</h4>
            </div>
            <p className="text-gray-600">Uses advanced embeddings to understand relationships between entries</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3 mb-2">
              <Cpu className="w-5 h-5 text-indigo-600" />
              <h4 className="font-medium text-gray-800">Adaptive Structure</h4>
            </div>
            <p className="text-gray-600">The tree structure evolves and reorganizes as new data is added</p>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
            <div className="flex items-center space-x-3 mb-2">
              <Eye className="w-5 h-5 text-indigo-600" />
              <h4 className="font-medium text-gray-800">Interactive Visualization</h4>
            </div>
            <p className="text-gray-600">Explore semantic relationships through 3D visualization</p>
          </div>
        </div>
      </section>

      <section>
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-indigo-100 rounded-lg">
            <PlayCircle className="w-5 h-5 text-indigo-600" />
          </div>
          <h3 className="text-xl font-semibold text-gray-800">How to Use</h3>
        </div>
        <div className="space-y-4">
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-indigo-50 rounded-lg">
                <PlayCircle className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h4 className="font-medium text-gray-800 mb-2">1. Adding Content</h4>
                <p className="text-gray-600">Use the "Add Items" tab to input single items, multiple items, or paragraphs. The dictionary will automatically organize them based on meaning.</p>
              </div>
            </div>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-indigo-50 rounded-lg">
                <Search className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h4 className="font-medium text-gray-800 mb-2">2. Searching</h4>
                <p className="text-gray-600">Use the "Search" tab to find semantically related items. The search goes beyond exact matches to find conceptually similar entries.</p>
              </div>
            </div>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-indigo-50 rounded-lg">
                <Eye className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h4 className="font-medium text-gray-800 mb-2">3. Visualization</h4>
                <p className="text-gray-600">Explore the 3D visualization to understand how items are related. Click nodes to see details, and use your mouse to rotate and zoom.</p>
              </div>
            </div>
          </div>
          <div className="p-4 bg-white rounded-lg border border-gray-200">
            <div className="flex items-start space-x-3">
              <div className="p-2 bg-indigo-50 rounded-lg">
                <Settings2 className="w-5 h-5 text-indigo-600" />
              </div>
              <div>
                <h4 className="font-medium text-gray-800 mb-2">4. Management</h4>
                <p className="text-gray-600">Use the "Settings" tab to manage the dictionary, including deleting specific nodes or resetting the entire structure.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gradient-to-r from-gray-50 to-gray-100 p-6 rounded-xl border border-gray-200">
        <div className="flex items-start space-x-4">
          <div className="p-3 bg-white rounded-lg shadow-md">
            <Code2 className="w-6 h-6 text-indigo-600" />
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-3 text-gray-800">Technical Details</h3>
            <p className="text-gray-600 leading-relaxed">
              Built using a combination of modern technologies including sentence transformers for embeddings, 
              hierarchical clustering for organization, and Three.js for visualization. The backend uses FastAPI 
              and the frontend is built with React.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Documentation; 