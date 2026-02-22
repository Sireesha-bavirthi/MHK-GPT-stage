"""
Simple script to store embeddings in Qdrant.
Run this from the backend directory: python3 store_to_qdrant.py
"""

import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.vector_db.dbstoring import QdrantClient
from app.services.vector_db.operations import VectorDBOperations
from app.core.config import settings


def main():
    print("\n" + "="*60)
    print("STORING EMBEDDINGS IN QDRANT")
    print("="*60)
    
    # Path to embeddings folder (new structure: individual JSON files)
    embeddings_dir = settings.project_root / "data" / "documents" / "embeddings" / "results"
    
    if not embeddings_dir.exists():
        print(f"\n Embeddings folder not found: {embeddings_dir}")
        print("Run embed_pipeline.py first to generate embeddings")
        return 1
    
    # Find all JSON files in the embeddings directory
    json_files = list(embeddings_dir.glob("*.json"))
    
    if not json_files:
        print(f"\n No embedding files found in: {embeddings_dir}")
        print("Run embed_pipeline.py first to generate embeddings")
        return 1
    
    print(f"\n Found {len(json_files)} embedding files:")
    for file in json_files:
        print(f"   - {file.name}")
    
    # Load all embeddings from individual JSON files
    all_documents = []
    for json_file in json_files:
        print(f"\n Loading: {json_file.name}")
        with open(json_file, 'r') as f:
            documents = json.load(f)
            all_documents.extend(documents)
    
    print(f"\n Loaded {len(all_documents)} total documents from {len(json_files)} files")
    
    # Prepare data
    texts = []
    embeddings = []
    metadatas = []
    
    for doc in all_documents:
        if doc.get("page_content") and doc.get("embedding"):
            texts.append(doc["page_content"])
            embeddings.append(doc["embedding"])
            metadatas.append(doc.get("metadata", {}))
    
    print(f" Prepared {len(texts)} embeddings from JSON files")

    
    # Connect to Qdrant
    print(f"\n Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}...")
    client = QdrantClient()
    
    # Create collection if it doesn't exist
    print(f" Setting up collection: {settings.QDRANT_COLLECTION_NAME}")
    client.create_collection(recreate=False)  # Keep existing data
    
    #  existing file names from Qdrant to check what's already stored
    print(f"\n Checking for existing embeddings..")
    ops = VectorDBOperations(client)
    
    try:
        existing_docs = ops.get_all_documents(limit=1000)
        existing_files = {}
        
        # Group existing documents by file name and count chunks
        for doc in existing_docs:
            file_name = doc['metadata'].get('file_name', '')
            if file_name:
                if file_name not in existing_files:
                    existing_files[file_name] = 0
                existing_files[file_name] += 1
        
        print(f" Found {len(existing_files)} files already in database:")
        for file_name, count in existing_files.items():
            print(f"   - {file_name}: {count} chunks")
    except:
        existing_files = {}
        print(f" No existing embeddings found (new database)")
    
    # Filter out embeddings that are already in the database
    new_texts = []
    new_embeddings = []
    new_metadatas = []
    skipped_count = 0
    
    for text, embedding, metadata in zip(texts, embeddings, metadatas):
        file_name = metadata.get('file_name', '')
        
        # Check if this file already exists in database
        if file_name in existing_files:
            skipped_count += 1
            continue
        
        # This is a new file, add it
        new_texts.append(text)
        new_embeddings.append(embedding)
        new_metadatas.append(metadata)
    
    if not new_texts:
        print(f"\n All embeddings already exist in database")
        print(f" Skipped: {skipped_count} embeddings")
        print(f" Nothing to add.")
        
        # Show current stats
        info = client.get_collection_info()
        print(f"\n Current database stats:")
        print(f" Collection: {info['name']}")
        print(f" Total points: {info['points_count']}")
        print(f" Vector size: {info['vector_size']}")
        return 0
    
    print(f"\n Found {len(new_texts)} NEW embeddings to add")
    print(f" Skipped {skipped_count} existing embeddings")
    
    # Store only NEW embeddings
    print(f"\n Storing {len(new_texts)} new embeddings...")
    point_ids = ops.upsert(
        texts=new_texts,
        embeddings=new_embeddings,
        metadatas=new_metadatas
    )
    
    # Get final stats
    info = client.get_collection_info()
    
    print("\n" + "="*60)
    print(" SUCCESS!")
    print("="*60)
    print(f"New embeddings added: {len(point_ids)}")
    print(f"Existing embeddings skipped: {skipped_count}")
    print(f"Total points in DB: {info['points_count']}")
    print(f"Collection: {info['name']}")
    print(f"Vector size: {info['vector_size']}")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
