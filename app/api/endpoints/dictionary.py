"""
API endpoints for the Elastic Dictionary
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import List

from app.services.elastic_dict_service import elastic_dict_service
from app.models.elastic_dict_models import (
    NodeModel, AddItemRequest, AddBatchRequest, AddParagraphRequest,
    SearchRequest, SearchResponse, DictionaryStateResponse
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