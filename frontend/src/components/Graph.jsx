import React, { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

const Graph = ({ graphData, onNodeClick }) => {
  const fgRef = useRef();

  useEffect(() => {
    if (fgRef.current && graphData.nodes.length > 0) {
      // Add some initial animation
      fgRef.current.d3Force('charge').strength(-120);
      
      // Aim at the center of the graph
      const { nodes } = graphData;
      if (nodes.length) {
        fgRef.current.cameraPosition(
          { x: 0, y: 0, z: 200 },
          { x: 0, y: 0, z: 0 },
          2000
        );
      }
    }
  }, [graphData]);

  return (
    <div className="graph-container">
      {graphData.nodes.length > 0 ? (
        <ForceGraph3D
          ref={fgRef}
          graphData={graphData}
          nodeLabel="name"
          nodeColor={node => node.color}
          nodeVal={node => node.val}
          nodeThreeObject={node => {
            // Use a sphere for category nodes and a box for regular nodes
            const geometry = node.is_category 
              ? new THREE.SphereGeometry(node.val) 
              : new THREE.BoxGeometry(node.val, node.val, node.val);
            
            const material = new THREE.MeshLambertMaterial({
              color: node.color,
              transparent: true,
              opacity: 0.8
            });
            
            return new THREE.Mesh(geometry, material);
          }}
          linkWidth={1}
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleSpeed={0.01}
          onNodeClick={onNodeClick}
          backgroundColor="#ffffff"
        />
      ) : (
        <div className="flex items-center justify-center h-full">
          <p className="text-lg text-gray-500">No data available. Add some items to see the graph.</p>
        </div>
      )}
    </div>
  );
};

export default Graph; 