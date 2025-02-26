"""
Pydantic models for the Elastic Dictionary API
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple


class NodeModel(BaseModel):
    """Model representing a node in the elastic dictionary"""
    key: str
    value: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    is_category_node: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "key": "apple",
                "value": "apple",
                "children": [],
                "is_category_node": False
            }
        }


class GraphDataModel(BaseModel):
    """Model representing graph data for visualization"""
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]
    
    class Config:
        schema_extra = {
            "example": {
                "nodes": [
                    {"id": "root", "name": "root", "val": 10, "color": "#1f77b4", "is_category": True},
                    {"id": "apple", "name": "apple", "val": 5, "color": "#ff7f0e", "is_category": False}
                ],
                "links": [
                    {"source": "root", "target": "apple", "value": 1}
                ]
            }
        }


class AddItemRequest(BaseModel):
    """Request model for adding a single item"""
    item: str
    
    class Config:
        schema_extra = {
            "example": {
                "item": "apple"
            }
        }


class AddBatchRequest(BaseModel):
    """Request model for adding multiple items"""
    items: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "items": ["apple", "banana", "orange"]
            }
        }


class AddParagraphRequest(BaseModel):
    """Request model for adding a paragraph"""
    paragraph: str
    
    class Config:
        schema_extra = {
            "example": {
                "paragraph": "Fruits are nutritious and delicious. Apples and bananas are popular choices."
            }
        }


class SearchRequest(BaseModel):
    """Request model for searching the dictionary"""
    query: str
    limit: Optional[int] = 10
    
    class Config:
        schema_extra = {
            "example": {
                "query": "fruit",
                "limit": 5
            }
        }


class SearchResult(BaseModel):
    """Model representing a search result"""
    key: str
    value: Optional[str] = None
    similarity: float
    
    class Config:
        schema_extra = {
            "example": {
                "key": "apple",
                "value": "apple",
                "similarity": 0.85
            }
        }


class SearchResponse(BaseModel):
    """Response model for search results"""
    results: List[SearchResult]
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {"key": "apple", "value": "apple", "similarity": 0.85},
                    {"key": "banana", "value": "banana", "similarity": 0.75}
                ]
            }
        }


class DictionaryStateResponse(BaseModel):
    """Response model for the dictionary state"""
    node_count: int
    graph_data: GraphDataModel
    
    class Config:
        schema_extra = {
            "example": {
                "node_count": 10,
                "graph_data": {
                    "nodes": [
                        {"id": "root", "name": "root", "val": 10, "color": "#1f77b4", "is_category": True},
                        {"id": "apple", "name": "apple", "val": 5, "color": "#ff7f0e", "is_category": False}
                    ],
                    "links": [
                        {"source": "root", "target": "apple", "value": 1}
                    ]
                }
            }
        } 