"""
API endpoints for the Elastic Dictionary
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.services.elastic_dict_service import elastic_dict_service
from app.models.elastic_dict_models import (
    NodeModel, AddItemRequest, AddBatchRequest, AddParagraphRequest,
    SearchRequest, SearchResponse, DictionaryStateResponse, DeleteNodeRequest
)


router = APIRouter()


@router.post("/add", response_model=NodeModel, summary="Add a single item")
async def add_item(request: AddItemRequest):
    """
    Add a single item to the elastic dictionary.
    
    - **item**: The text item to add
    """
    try:
        return elastic_dict_service.add_item(request.item)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding item: {str(e)}")


@router.post("/add-batch", response_model=List[NodeModel], summary="Add multiple items")
async def add_batch(request: AddBatchRequest):
    """
    Add multiple items to the elastic dictionary.
    
    - **items**: List of text items to add
    """
    try:
        return elastic_dict_service.add_batch(request.items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding batch: {str(e)}")


@router.post("/add-paragraph", response_model=List[NodeModel], summary="Add a paragraph")
async def add_paragraph(request: AddParagraphRequest):
    """
    Add a paragraph to the elastic dictionary. The paragraph will be split into sentences.
    
    - **paragraph**: The paragraph text to add
    """
    try:
        return elastic_dict_service.add_paragraph(request.paragraph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding paragraph: {str(e)}")


@router.post("/search", response_model=SearchResponse, summary="Search the dictionary")
async def search(request: SearchRequest):
    """
    Search the elastic dictionary for items related to the query.
    
    - **query**: The search query
    - **limit**: Maximum number of results to return (default: 10)
    """
    try:
        results = elastic_dict_service.search(request.query, request.limit)
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")


@router.get("/state", response_model=DictionaryStateResponse, summary="Get dictionary state")
async def get_state():
    """
    Get the current state of the elastic dictionary, including node count and graph data for visualization.
    """
    try:
        return elastic_dict_service.get_dictionary_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting state: {str(e)}")


@router.post("/delete-node", response_model=dict, summary="Delete a node from the dictionary")
async def delete_node(request: DeleteNodeRequest):
    """
    Delete a node from the elastic dictionary and reorganize the tree.
    
    - **node_key**: The key of the node to delete
    """
    try:
        success = elastic_dict_service.delete_node(request.node_key)
        if success:
            return {"status": "success", "message": f"Node '{request.node_key}' deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Node '{request.node_key}' not found or cannot be deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting node: {str(e)}")


@router.post("/reset", response_model=dict, summary="Reset the dictionary")
async def reset_dictionary():
    """
    Reset the elastic dictionary, removing all items and starting fresh.
    """
    try:
        elastic_dict_service.reset_dictionary()
        return {"status": "success", "message": "Dictionary reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting dictionary: {str(e)}")


@router.post("/rebuild", response_model=dict, summary="Rebuild the dictionary tree")
async def rebuild_tree():
    """
    Rebuild the elastic dictionary tree structure without deleting any nodes.
    This forces a complete restructuring of the tree based on semantic similarity.
    """
    try:
        success = elastic_dict_service.rebuild_tree()
        if success:
            return {"status": "success", "message": "Dictionary tree rebuilt successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to rebuild dictionary tree")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding dictionary tree: {str(e)}") 