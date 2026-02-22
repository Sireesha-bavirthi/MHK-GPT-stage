# MHK-GPT Backend

## Overview

MHK-GPT is a production-ready Retrieval-Augmented Generation (RAG) chatbot system designed to provide intelligent, context-aware responses based on company documentation. The system combines advanced document processing, vector search, and large language models to deliver accurate answers to user queries.

## System Architecture

The backend consists of two primary pipeline orchestrators that handle the complete RAG workflow:

### 1. Ingestion Pipeline
Processes raw documents and prepares them for semantic search by:
- Extracting and cleaning text from various document formats
- Splitting documents into semantically meaningful chunks
- Generating vector embeddings using OpenAI's embedding models
- Storing embeddings in Qdrant vector database

### 2. RAG Pipeline
Handles real-time query processing and response generation through:
- Query reformulation using conversation context
- Semantic document retrieval from vector database
- AI-powered answer generation using GPT models
- Conversation memory management

## Technology Stack

- **Language**: Python 3.9+
- **LLM Provider**: OpenAI (GPT-4, text-embedding-3-small)
- **Vector Database**: Qdrant
- **Framework**: FastAPI (for API endpoints)
- **Document Processing**: PyPDF2, python-docx, docx2txt
- **Machine Learning**: LangChain, Sentence Transformers

## Current System Status

The system is fully configured and operational:

- Virtual environment: Created and configured
- Dependencies: All packages installed
- Configuration: Environment variables set in `.env`
- Documents: 4 files processed (143 chunks)
- Vector Database: 143 embeddings stored in Qdrant
- Testing: Both pipelines validated and working

**Performance Metrics:**
- Ingestion Pipeline: 4.88 seconds for 4 documents
- RAG Pipeline: 2.91 seconds average query time
- Vector Database: 143 embeddings with 1536 dimensions

---

## Quick Start Guide

### Prerequisites

Before running the pipelines, ensure the following are installed:

1. **Python 3.9 or higher**
2. **Docker** (for Qdrant vector database)
3. **OpenAI API Key** (already configured in `.env`)

### Step 1: Environment Setup

#### 1.1 Navigate to Backend Directory

```bash
cd MHK-GPT/backend
```

#### 1.2 Activate Virtual Environment

The virtual environment is already created. Activate it before running any commands:

```bash
source .venv/bin/activate
```

**Note:** All subsequent commands assume the virtual environment is activated.

#### 1.3 Verify Dependencies

Dependencies are already installed. To verify or reinstall:

```bash
pip install -r requirements.txt
```

### Step 2: Start Qdrant Vector Database

Qdrant must be running before using either pipeline.

```bash
# Start Qdrant in detached mode
docker run -d -p 6333:6333 qdrant/qdrant

# Verify Qdrant is running
curl http://localhost:6333/collections
```

Expected response: JSON listing of collections

### Step 3: Run Ingestion Pipeline

The ingestion pipeline processes documents and stores them in the vector database.

#### Basic Usage

```bash
python3 ingestion_pipeline.py
```

#### Expected Output

```
================================================================================
DATA INGESTION PIPELINE - START
================================================================================

STAGE 1: DOCUMENT PROCESSING
  - Processing: CLOUD Solutions.docx
  - Processing: Data-Engineering.docx
  - Processing: AI-ML.docx
  - Processing: Company-Overview.docx

STAGE 2: EMBEDDING GENERATION
  - Generated embeddings for 143 chunks

STAGE 3: VECTOR DATABASE STORAGE
  - Stored 143 embeddings in Qdrant

PIPELINE COMPLETED SUCCESSFULLY
Total Duration: 4.88s
```

#### Advanced Options

```bash
# Skip specific stages (if already completed)
python3 ingestion_pipeline.py --skip-stages 1

# Use specific chunking strategy
python3 ingestion_pipeline.py --use-semantic
```

### Step 4: Run RAG Pipeline

The RAG pipeline handles user queries and generates responses.

#### Option A: Interactive Chat Mode (Recommended)

```bash
python3 rag_pipeline.py
```

This starts an interactive session where you can type questions:

```
Query > What cloud services does MHK Tech offer?
[AI generates response based on retrieved documents]

Query > Tell me more about data engineering
[AI continues conversation with context]

Query > quit
[Exits the session]
```

**Special Commands:**
- `stats` - Display performance statistics
- `clear` - Clear conversation memory
- `quit` or `exit` - Exit the application

#### Option B: Single Query Mode

For one-off queries or automation:

```bash
python3 rag_pipeline.py --query "What services does MHK Tech offer?"
```

#### Option C: Programmatic API

For integration into applications:

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline(
    retrieval_strategy="mmr",
    top_k=5,
    score_threshold=0.5,
    enable_reformulation=True
)

# Execute query
result = pipeline.query("What cloud services do you offer?")

# Access results
print(result.response)
print(f"Duration: {result.total_duration}s")
print(f"Sources: {len(result.retrieval_results)}")
```

---

## How the Application Works

### Document Ingestion Flow

1. **Document Loading**
   - System scans `data/documents/raw/` directory
   - Supports formats: PDF, DOCX, TXT, MD
   - Extracts text using format-specific loaders

2. **Text Processing**
   - Cleans and normalizes extracted text
   - Removes special characters and formatting artifacts
   - Preserves semantic structure

3. **Semantic Chunking**
   - Splits documents into meaningful segments
   - Uses sentence embeddings to detect topic boundaries
   - Maintains context within chunks
   - Default chunk size: 100-2000 characters

4. **Embedding Generation**
   - Converts text chunks to vector embeddings
   - Uses OpenAI's text-embedding-3-small model
   - Generates 1536-dimensional vectors
   - Batch processes for efficiency

5. **Vector Storage**
   - Stores embeddings in Qdrant vector database
   - Includes metadata (filename, chunk index, timestamps)
   - Implements deduplication to avoid redundant storage
   - Creates optimized indexes for fast retrieval

### Query Processing Flow

1. **Query Reception**
   - Accepts user query as natural language text
   - Maintains conversation context if in interactive mode

2. **Query Reformulation**
   - Analyzes conversation history (if available)
   - Enhances query with contextual information
   - Expands ambiguous references
   - Improves retrieval accuracy

3. **Semantic Retrieval**
   - Converts query to vector embedding
   - Performs similarity search in Qdrant
   - Uses MMR (Maximal Marginal Relevance) for diversity
   - Retrieves top-k most relevant chunks (default: 5)

4. **Context Preparation**
   - Aggregates retrieved document chunks
   - Formats context for LLM consumption
   - Includes source metadata for citations

5. **Answer Generation**
   - Sends context + query to GPT-4 model
   - Generates natural language response
   - Includes source attribution
   - Maintains conversation coherence

6. **Response Delivery**
   - Returns formatted response to user
   - Updates conversation memory
   - Logs performance metrics

---

## Configuration

### Environment Variables

All configuration is managed through the `.env` file located at:
```
/Users/kishore/Documents/MHK-GPT/backend/.env
```

#### Core Settings

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...           # Your OpenAI API key
OPENAI_MODEL=gpt-4o                   # GPT model for generation
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=company_docs
QDRANT_VECTOR_SIZE=1536

# Document Processing
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEMANTIC_SIMILARITY_THRESHOLD=0.75
SEMANTIC_MIN_CHUNK_SIZE=100
SEMANTIC_MAX_CHUNK_SIZE=2000

# Retrieval Settings
RETRIEVAL_TOP_K=5
RETRIEVAL_FETCH_K=20
RETRIEVAL_LAMBDA_MULT=0.7

# Conversation Settings
MAX_CONVERSATION_HISTORY=10
COMPANY_NAME=MHKTech
```

### Modifying Configuration

To update configuration:

```bash
# Open .env file in text editor
nano .env

# Or use your preferred editor
code .env
```

After modifying `.env`, restart the pipeline for changes to take effect.

---

## Directory Structure

```
backend/
├── .env                        # Environment configuration
├── .venv/                      # Virtual environment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── COMPLETE_GUIDE.md          # Comprehensive documentation
│
├── ingestion_pipeline.py      # Document ingestion orchestrator
├── rag_pipeline.py            # RAG query orchestrator
│
├── app/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   └── logging.py         # Logging setup
│   │
│   ├── services/
│   │   ├── document_processor/
│   │   │   ├── loader.py      # Document loading
│   │   │   ├── cleaner.py     # Text cleaning
│   │   │   ├── chunker.py     # Semantic chunking
│   │   │   └── pipeline.py    # Processing pipeline
│   │   │
│   │   ├── rag/
│   │   │   ├── embeddings.py  # Embedding generation
│   │   │   ├── retriever.py   # Document retrieval
│   │   │   ├── generator.py   # Answer generation
│   │   │   └── query_reformulation.py
│   │   │
│   │   ├── vector_db/
│   │   │   ├── dbstoring.py   # Qdrant client
│   │   │   └── operations.py  # DB operations
│   │   │
│   │   └── llm/
│   │       └── openai_client.py
│   │
│   └── models/                # Data models
│
└── data/
    └── documents/
        ├── raw/               # Input documents (add files here)
        ├── processed/         # Processed chunks
        └── embeddings/        # Generated embeddings
```

---

## Usage Scenarios

### Scenario 1: Adding New Documents

When you need to add new company documentation:

```bash
# 1. Add documents to raw directory
cp /path/to/new-document.pdf data/documents/raw/

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run ingestion pipeline
python3 ingestion_pipeline.py

# 4. Verify ingestion
# Check output for successful processing confirmation
```

### Scenario 2: Querying the System

For general queries:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start interactive chat
python3 rag_pipeline.py

# Or single query
python3 rag_pipeline.py --query "What are MHK's AI capabilities?"
```

### Scenario 3: API Integration

For embedding in applications:

```python
from rag_pipeline import RAGPipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = RAGPipeline()

@app.route('/api/chat', methods=['POST'])
def chat():
    query = request.json.get('query')
    result = pipeline.query(query)
    
    return jsonify({
        'response': result.response,
        'duration': result.total_duration,
        'sources': len(result.retrieval_results)
    })

if __name__ == '__main__':
    app.run(port=5000)
```

### Scenario 4: Batch Processing

For processing multiple queries:

```python
from rag_pipeline import RAGPipeline
import json

pipeline = RAGPipeline()

queries = [
    "What cloud services does MHK offer?",
    "Describe MHK's data engineering capabilities",
    "What AI/ML solutions are available?"
]

results = []
for query in queries:
    result = pipeline.query(query)
    results.append({
        'query': query,
        'answer': result.response,
        'duration': result.total_duration
    })

# Save results
with open('batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: ModuleNotFoundError

**Symptom:** `ModuleNotFoundError: No module named 'langchain'`

**Cause:** Virtual environment not activated

**Solution:**
```bash
cd /Users/kishore/Documents/MHK-GPT/backend
source .venv/bin/activate
pip install -r requirements.txt
```

#### Issue 2: Qdrant Connection Refused

**Symptom:** `Connection refused` when accessing Qdrant

**Cause:** Qdrant is not running

**Solution:**
```bash
# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Wait for initialization
sleep 5

# Verify
curl http://localhost:6333/collections
```

#### Issue 3: OpenAI API Errors

**Symptom:** `AuthenticationError` or `RateLimitError`

**Cause:** Invalid API key or rate limit exceeded

**Solution:**
```bash
# Verify API key in .env file
cat .env | grep OPENAI_API_KEY

# If needed, update the key
nano .env
```

For rate limits, implement retry logic or reduce request frequency.

#### Issue 4: No Documents Found

**Symptom:** Ingestion pipeline reports "No files to process"

**Cause:** No documents in raw directory

**Solution:**
```bash
# Check raw directory
ls -la data/documents/raw/

# Add documents
cp /path/to/documents/* data/documents/raw/
```

#### Issue 5: Poor Retrieval Quality

**Symptom:** Retrieved documents not relevant to query

**Cause:** Suboptimal retrieval parameters

**Solution:** Adjust parameters in `.env`:
```bash
RETRIEVAL_TOP_K=10              # Retrieve more documents
RETRIEVAL_LAMBDA_MULT=0.5       # Increase diversity
SEMANTIC_SIMILARITY_THRESHOLD=0.70  # Lower threshold
```

---

## Performance Optimization

### Ingestion Pipeline

1. **Batch Processing**: Process multiple documents in parallel
2. **Incremental Updates**: Only process new/modified documents
3. **Chunk Size Tuning**: Adjust based on document type
4. **Index Optimization**: Configure Qdrant for your use case

### RAG Pipeline

1. **Caching**: Enable embedding cache for repeated queries
2. **Top-K Tuning**: Balance between accuracy and speed
3. **Streaming Responses**: Use streaming for long responses
4. **Conversation Pruning**: Limit conversation history length

---

## Testing

### Validation Tests

The system includes validation scripts to verify functionality:

```bash
# Run all validation tests
python3 validate_pipelines.py

# Expected output: All tests pass
```

### Manual Testing

Test ingestion:
```bash
python3 ingestion_pipeline.py
# Verify: Documents processed successfully
```

Test RAG:
```bash
python3 rag_pipeline.py --query "Test query"
# Verify: Response generated successfully
```

---

## API Reference

### Ingestion Pipeline

```python
from ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline(
    use_semantic_chunking=True,
    skip_stages=[]
)

result = pipeline.run()
print(result.summary)
```

### RAG Pipeline

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    retrieval_strategy="mmr",  # or "basic"
    top_k=5,
    score_threshold=0.5,
    enable_reformulation=True
)

# Single query
result = pipeline.query("Your question here")

# Access results
print(result.response)
print(result.total_duration)
print(result.retrieval_results)
print(result.metadata)

# Conversation management
pipeline.clear_memory()
history = pipeline.get_conversation_history()
stats = pipeline.get_performance_stats()
```

---

## Monitoring and Logging

### Log Files

Logs are stored in:
```
data/logs/backend/app.log
```

### Log Levels

Configure in `.env`:
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Monitoring Metrics

Key metrics tracked:
- Query processing time
- Retrieval accuracy scores
- Token usage
- Error rates
- Document processing throughput

---

## Production Deployment

### Checklist

- [ ] Update `SECRET_KEY` in `.env`
- [ ] Configure proper CORS origins
- [ ] Set up rate limiting
- [ ] Enable metrics collection
- [ ] Configure log rotation
- [ ] Set up monitoring alerts
- [ ] Implement backup strategy for Qdrant
- [ ] Use environment-specific `.env` files

### Security Considerations

1. **API Key Management**: Use secure vault for OpenAI API key
2. **Access Control**: Implement authentication for API endpoints
3. **Rate Limiting**: Prevent abuse with rate limits
4. **Input Validation**: Sanitize all user inputs
5. **HTTPS**: Use TLS for all communications

---

## Maintenance

### Regular Tasks

**Daily:**
- Monitor error logs
- Check system performance metrics

**Weekly:**
- Review and update documents
- Run validation tests
- Check Qdrant disk usage

**Monthly:**
- Update dependencies
- Review and optimize performance
- Backup vector database

### Updating Documents

```bash
# 1. Add new/updated documents to raw/
cp updated-doc.pdf data/documents/raw/

# 2. Run ingestion
source .venv/bin/activate
python3 ingestion_pipeline.py

# 3. Verify updates
python3 rag_pipeline.py --query "Query about new content"
```

---

## Support and Documentation

### Additional Resources

- **Complete Guide**: See `COMPLETE_GUIDE.md` for detailed documentation
- **API Documentation**: See individual module docstrings
- **Configuration Reference**: See `.env.example`

### Contact

For issues or questions, contact the MHK Tech development team.

---

## License

Copyright (c) 2026 MHK Tech Inc. All rights reserved.

---

## Version History

**Version 1.0.0** (2026-02-06)
- Initial release
- Ingestion pipeline with semantic chunking
- RAG pipeline with conversation memory
- Qdrant vector database integration
- OpenAI GPT-4 integration
- Comprehensive documentation
