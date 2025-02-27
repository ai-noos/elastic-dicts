import React, { useRef, useEffect } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

const Graph = ({ graphData, onNodeClick }) => {
  const fgRef = useRef();

  // Function to truncate node labels
  const getNodeLabel = (node) => {
    const text = node.name || '';
    
    // If text contains spaces (likely a paragraph), truncate after first space
    if (text.includes(' ')) {
      const firstSpaceIndex = text.indexOf(' ');
      return text.substring(0, firstSpaceIndex) + '...';
    }
    
    return text;
  };

  // Function to create a text sprite
  const createTextSprite = (text, size = 1) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = 256;
    canvas.height = 64;
    
    // Draw background with rounded corners
    context.fillStyle = 'rgba(255, 255, 255, 0.8)';
    context.strokeStyle = '#666666';
    context.lineWidth = 2;
    roundRect(context, 0, 0, canvas.width, canvas.height, 10, true, true);
    
    // Draw text
    const fontSize = Math.max(12, Math.min(24, 24 * size));
    context.font = `${fontSize}px Arial, sans-serif`;
    context.fillStyle = '#000000';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    
    // Handle text that's too long
    const maxWidth = canvas.width - 20;
    let displayText = text;
    let textWidth = context.measureText(displayText).width;
    
    if (textWidth > maxWidth) {
      // Truncate text if it's too long
      let truncated = false;
      while (textWidth > maxWidth && displayText.length > 3) {
        displayText = displayText.slice(0, -1);
        textWidth = context.measureText(displayText + '...').width;
        truncated = true;
      }
      
      if (truncated) {
        displayText += '...';
      }
    }
    
    context.fillText(displayText, canvas.width / 2, canvas.height / 2);
    
    // Create texture from canvas
    const texture = new THREE.Texture(canvas);
    texture.needsUpdate = true;
    
    // Create sprite material
    const spriteMaterial = new THREE.SpriteMaterial({ 
      map: texture,
      transparent: true 
    });
    
    // Create sprite
    const sprite = new THREE.Sprite(spriteMaterial);
    
    // Scale sprite based on text length and node size
    const aspectRatio = canvas.width / canvas.height;
    sprite.scale.set(8 * aspectRatio * size, 8 * size, 1);
    
    return sprite;
  };
  
  // Helper function to draw rounded rectangles
  const roundRect = (ctx, x, y, width, height, radius, fill, stroke) => {
    if (typeof radius === 'number') {
      radius = {tl: radius, tr: radius, br: radius, bl: radius};
    } else {
      radius = {...{tl: 0, tr: 0, br: 0, bl: 0}, ...radius};
    }
    
    ctx.beginPath();
    ctx.moveTo(x + radius.tl, y);
    ctx.lineTo(x + width - radius.tr, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius.tr);
    ctx.lineTo(x + width, y + height - radius.br);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius.br, y + height);
    ctx.lineTo(x + radius.bl, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius.bl);
    ctx.lineTo(x, y + radius.tl);
    ctx.quadraticCurveTo(x, y, x + radius.tl, y);
    ctx.closePath();
    
    if (fill) {
      ctx.fill();
    }
    
    if (stroke) {
      ctx.stroke();
    }
  };

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
          nodeThreeObject={(node) => {
            // Create a group to hold both the node geometry and the label
            const group = new THREE.Group();
            
            // Use a sphere for category nodes and a box for regular nodes
            const geometry = node.is_category 
              ? new THREE.SphereGeometry(node.val) 
              : new THREE.BoxGeometry(node.val, node.val, node.val);
            
            const material = new THREE.MeshLambertMaterial({
              color: node.color,
              transparent: true,
              opacity: 0.8
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            group.add(mesh);
            
            // Add text label
            const truncatedLabel = getNodeLabel(node);
            if (truncatedLabel) {
              // Scale label size based on node value
              const labelSize = 0.5 + (node.val / 10);
              const textSprite = createTextSprite(truncatedLabel, labelSize);
              textSprite.position.y = node.val + 6; // Position above the node
              group.add(textSprite);
            }
            
            return group;
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