import shutil
import os
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from app.schemas.document import DocumentUploadResponse, DocumentResponse
from app.api.dependencies import get_ingestion_pipeline
from ingestion_pipeline import IngestionPipeline
from app.core.config import settings
from pathlib import Path

router = APIRouter()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline)
):
    """
    Upload a document and trigger ingestion.
    """
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "text/markdown"]:
         # Note: Strictly validation might need to check extensions too as MIME types vary
         pass

    # Save file
    upload_dir = Path(settings.raw_documents_path_absolute)
    if not upload_dir.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
        
    file_path = upload_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {len(e)}")

    # Trigger ingestion immediately (synchronous for simplicity/feedback)
    try:
        # Run pipeline only for this file if possible? 
        # The current IngestionPipeline scans the directory. 
        # Ideally we'd modify it to process one file, but per instructions we keep it.
        # We will run the full pipeline.
        
        import time
        start = time.time()
        
        # We can pass skip_stages if we want partial run, but here we want full.
        result = pipeline.run()
        
        duration = time.time() - start
        
        if not result.success:
             raise HTTPException(status_code=500, detail="Ingestion failed")
             
        return DocumentUploadResponse(
            filename=file.filename,
            message="File uploaded and processed successfully",
            chunks_processed=result.summary.get('total_chunks_created', 0),
            duration=duration
        )

    except Exception as e:
        # If ingestion fails, maybe delete the file? keeping it for now.
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.get("/", response_model=List[DocumentResponse])
async def list_documents():
    """
    List all documents in the raw and processed directories.
    """
    docs = []
    
    # 1. Check Raw Directory (Uploaded but not processed yet)
    upload_dir = Path(settings.raw_documents_path_absolute)
    if upload_dir.exists():
        for file_path in upload_dir.glob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                 stat = file_path.stat()
                 docs.append(DocumentResponse(
                     filename=file_path.name,
                     file_type=file_path.suffix,
                     size=stat.st_size,
                     upload_date=None, 
                     status="uploaded"
                 ))
    
    # 2. Check Processed Directory (Successfully ingested)
    processed_dir = Path(settings.processed_documents_path_absolute)
    if processed_dir.exists():
        # Recursively find files in processed directory
        for file_path in processed_dir.rglob("*"):
             if file_path.is_file() and not file_path.name.startswith('.') and not file_path.name.endswith('.json'):
                 # Skip JSON metadata files, show only original docs
                 stat = file_path.stat()
                 docs.append(DocumentResponse(
                     filename=file_path.name,
                     file_type=file_path.suffix,
                     size=stat.st_size,
                     upload_date=None,
                     status="processed"
                 ))

    return docs

@router.delete("/{filename}")
async def delete_document(filename: str):
    """
    Delete a document completely from the system.
    1. Removes file from filesystem (raw or processed)
    2. Removes embeddings from Qdrant vector database
    3. Removes intermediate JSON processing results
    """
    messages = []
    
    # Initialize Vector DB Helper
    try:
        from app.services.vector_db.dbstoring import QdrantClient
        from app.services.vector_db.operations import VectorDBOperations
        
        qdrant_client = QdrantClient()
        if qdrant_client.collection_exists():
            ops = VectorDBOperations(qdrant_client)
            # Delete by file_name in metadata
            ops.delete_by_filter({"file_name": filename})
            messages.append(f"Deleted embeddings for {filename}")
        else:
             messages.append("Vector DB collection not found, skipping embedding deletion")
    except Exception as e:
        messages.append(f"Failed to delete embeddings: {str(e)}")

    # Delete JSON result file (if exists)
    try:
        json_path = Path(settings.processed_documents_path_absolute) / "results" / (Path(filename).stem + ".json")
        if json_path.exists():
            os.remove(json_path)
            messages.append(f"Deleted metadata file {json_path.name}")
    except Exception as e:
        messages.append(f"Failed to delete metadata file: {str(e)}")

    # Delete Actual File
    # 1. Try to find in Raw
    upload_dir = Path(settings.raw_documents_path_absolute)
    file_path_raw = upload_dir / filename
    
    deleted_file = False
    
    if file_path_raw.exists():
        try:
            os.remove(file_path_raw)
            messages.append(f"Deleted source file from raw")
            deleted_file = True
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to delete raw file: {str(e)}")
             
    # 2. Try to find in Processed (Recursive search)
    if not deleted_file:
        processed_dir = Path(settings.processed_documents_path_absolute)
        if processed_dir.exists():
            for path in processed_dir.rglob(filename):
                if path.is_file():
                    try:
                        os.remove(path)
                        messages.append(f"Deleted source file from processed")
                        deleted_file = True
                        break 
                    except Exception as e:
                        raise HTTPException(status_code=500, detail=f"Failed to delete processed file: {str(e)}")
    
    if not deleted_file:
         # If we scraped embeddings but couldn't find the file, still return success with warning context
         if "Deleted embeddings" in messages[0]:
             return {"message": "Document file not found, but embeddings were deleted", "details": messages}
         
         raise HTTPException(status_code=404, detail=f"Document {filename} not found in raw or processed directories")
        
    return {
        "message": f"Successfully deleted {filename}",
        "details": messages
    }
