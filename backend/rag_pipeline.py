"""
__Title__ : MHK chatbot
__Author_name__: Bhavani Kishore
__Verison__: 1.0
__Created_Date__: 06 february 2026
__Updated_date__: 06 february 2026
__Employee_id: 800341
__Description__ : RAG Pipeline Orchestrator.

This pipeline orchestrates the complete RAG (Retrieval-Augmented Generation) process:
1. Query Reformulation - Enhances queries using conversation history
2. Document Retrieval - Retrieves relevant documents from vector DB
3. Answer Generation - Generates AI responses using LLM

"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag.retriever import get_retriever_service, RetrieverService, RetrievalResult
from app.services.rag.generator import get_generator_service, GeneratorService, GenerationResult, ConversationMemory
from app.services.rag.query_reformulation import get_query_reformulator, QueryReformulator
from app.core.config import settings
from app.core.logging import setup_logging


# =============================================================================
# Data Classes for Pipeline Results
# =============================================================================
@dataclass
class RAGStageResult:
    """Result from a single RAG stage."""
    stage_name: str
    success: bool
    duration: float  # seconds
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "success": self.success,
            "duration": self.duration,
            "data": self.data,
            "error": self.error
        }


@dataclass
class RAGPipelineResult:
    """Result from the complete RAG pipeline execution."""
    success: bool
    query: str
    reformulated_query: Optional[str]
    response: str
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    total_duration: float = 0.0  # seconds
    stages: List[RAGStageResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "query": self.query,
            "reformulated_query": self.reformulated_query,
            "response": self.response,
            "retrieval_results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "text": r.text if hasattr(r, 'text') else r.preview,
                    "metadata": r.metadata
                } for r in self.retrieval_results
            ],
            "total_duration": self.total_duration,
            "stages": [stage.to_dict() for stage in self.stages],
            "metadata": self.metadata
        }


# =============================================================================
# RAG Pipeline Orchestrator Class
# =============================================================================
class RAGPipeline:
    """
    Orchestrates the complete RAG pipeline.
    
    This class coordinates the entire RAG workflow:
    1. Query Reformulation (with conversation history)
    2. Document Retrieval (from vector DB)
    3. Answer Generation (using LLM)
    
    Features:
    - Comprehensive error handling
    - Performance monitoring
    - Logging at each stage
    - Conversation memory management
    - Flexible configuration
    """
    
    def __init__(
        self,
        retriever: Optional[RetrieverService] = None,
        generator: Optional[GeneratorService] = None,
        reformulator: Optional[QueryReformulator] = None,
        memory: Optional[ConversationMemory] = None,
        retrieval_strategy: str = "mmr",
        top_k: int = None,
        score_threshold: float = None,
        enable_reformulation: bool = True,
        log_level: str = "INFO"
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            retriever: Custom retriever service (or use default)
            generator: Custom generator service (or use default)
            reformulator: Custom query reformulator (or use default)
            memory: Conversation memory (or create new)
            retrieval_strategy: Strategy for retrieval ("mmr" or "basic")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            enable_reformulation: Whether to enable query reformulation
            log_level: Logging level
        """
        # Setup logging
        setup_logging()
        self.logger = self._setup_file_logger()
        self.logger.setLevel(log_level)
        
        # Initialize services
        self.retriever = retriever or get_retriever_service()
        self.generator = generator or get_generator_service()
        self.reformulator = reformulator or get_query_reformulator()
        self.memory = memory or ConversationMemory(max_messages=settings.MAX_CONVERSATION_HISTORY)
        
        # Configuration
        self.retrieval_strategy = retrieval_strategy or settings.RETRIEVAL_STRATEGY  # Use config default
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.enable_reformulation = enable_reformulation
        
        # Performance tracking
        self.total_queries_processed = 0
        self.total_processing_time = 0.0
        self.stages_completed: List[RAGStageResult] = []
        
        self.logger.info("RAG Pipeline initialized")
        self.logger.info(f"  - Retrieval Strategy: {self.retrieval_strategy}")
        self.logger.info(f"  - Top K: {self.top_k or 'default'}")
        self.logger.info(f"  - Score Threshold: {self.score_threshold or 'default'}")
        self.logger.info(f"  - Query Reformulation: {self.enable_reformulation}")
    
    def _setup_file_logger(self) -> logging.Logger:
        """
        Setup file-based logger for RAG pipeline.
        
        Creates a dedicated log file in data/logs/backend/yyyy/mm/dd/ with:
        - Date-based directory structure (data/logs/backend/2026/02/06/)
        - Rotating file handler (max 10MB, 5 backups)
        - Detailed formatting with timestamps and log levels
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        
        # Create logs directory with yyyy/mm/dd structure
        now = datetime.now()
        log_dir = settings.project_root / "data" / "logs" / "backend" / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = now.strftime("%H%M%S")
        log_file = log_dir / f"rag_pipeline_{timestamp}.log"
        
        # Create rotating file handler (max 10MB, keep 5 backup files)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter with detailed information
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.DEBUG)
        logger.info(f"RAG pipeline logger initialized. Log file: {log_file}")
        
        return logger
    
    # =========================================================================
    # Stage 1: Query Reformulation
    # =========================================================================
    def _stage_1_reformulate_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[RAGStageResult, str]:
        """
        Stage 1: Reformulate the query using conversation history.
        
        Args:
            query: Original user query
            conversation_history: Optional conversation history
        
        Returns:
            Tuple of (StageResult, reformulated_query)
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: QUERY REFORMULATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Original Query: {query}")
        
        stage_start = datetime.now()
        
        try:
            if not self.enable_reformulation:
                self.logger.info("Query reformulation disabled, using original query")
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                return (
                    RAGStageResult(
                        stage_name="Query Reformulation",
                        success=True,
                        duration=stage_duration,
                        data={
                            "original_query": query,
                            "reformulated_query": query,
                            "was_reformulated": False
                        }
                    ),
                    query
                )
            
            # Use conversation history from memory if not provided
            if conversation_history is None:
                conversation_history = self.memory.get_history()
            
            # Reformulate the query
            reformulated_query = self.reformulator.reformulate(
                query=query,
                conversation_history=conversation_history
            )
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            was_reformulated = reformulated_query != query
            
            if was_reformulated:
                self.logger.info(f"Reformulated Query: {reformulated_query}")
            else:
                self.logger.info("Query did not require reformulation")
            
            self.logger.info(f"[SUCCESS] Stage 1 completed in {stage_duration:.3f}s")
            
            return (
                RAGStageResult(
                    stage_name="Query Reformulation",
                    success=True,
                    duration=stage_duration,
                    data={
                        "original_query": query,
                        "reformulated_query": reformulated_query,
                        "was_reformulated": was_reformulated,
                        "conversation_turns": len(conversation_history)
                    }
                ),
                reformulated_query
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 1 failed: {str(e)}", exc_info=True)
            
            # Fall back to original query
            return (
                RAGStageResult(
                    stage_name="Query Reformulation",
                    success=False,
                    duration=stage_duration,
                    error=str(e),
                    data={"fallback_to_original": True}
                ),
                query
            )
    
    # =========================================================================
    # Stage 2: Document Retrieval
    # =========================================================================
    def _stage_2_retrieve_documents(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[RAGStageResult, List[RetrievalResult]]:
        """
        Stage 2: Retrieve relevant documents from vector DB.
        
        Args:
            query: Query to use for retrieval
            metadata_filter: Optional metadata filter
        
        Returns:
            Tuple of (StageResult, retrieval_results)
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: DOCUMENT RETRIEVAL")
        self.logger.info("=" * 80)
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Strategy: {self.retrieval_strategy}")
        
        stage_start = datetime.now()
        
        try:
            # Retrieve documents
            results = self.retriever.retrieve(
                query=query,
                strategy=self.retrieval_strategy,
                top_k=self.top_k,
                score_threshold=0.15,  # Lowered to retrieve founding date chunk
                metadata_filter=metadata_filter,
                full_text=True  # Get full text for generation
            )
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            self.logger.info(f"[SUCCESS] Stage 2 completed in {stage_duration:.3f}s")
            self.logger.info(f"  - Documents retrieved: {len(results)}")
            
            if results:
                avg_score = sum(r.score for r in results) / len(results)
                self.logger.info(f"  - Average score: {avg_score:.4f}")
                self.logger.info(f"  - Top score: {results[0].score:.4f}")
            
            return (
                RAGStageResult(
                    stage_name="Document Retrieval",
                    success=True,
                    duration=stage_duration,
                    data={
                        "num_results": len(results),
                        "avg_score": sum(r.score for r in results) / len(results) if results else 0.0,
                        "top_score": results[0].score if results else 0.0,
                        "strategy": self.retrieval_strategy
                    }
                ),
                results
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 2 failed: {str(e)}", exc_info=True)
            
            return (
                RAGStageResult(
                    stage_name="Document Retrieval",
                    success=False,
                    duration=stage_duration,
                    error=str(e)
                ),
                []
            )
    
    # =========================================================================
    # Stage 3: Answer Generation
    # =========================================================================
    def _stage_3_generate_answer(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Tuple[RAGStageResult, GenerationResult]:
        """
        Stage 3: Generate answer using LLM.
        
        Args:
            query: Original user query
            retrieval_results: Retrieved documents (can be empty)
            conversation_history: Optional conversation history
        
        Returns:
            Tuple of (StageResult, generation_result)
        """
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: ANSWER GENERATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Context Documents: {len(retrieval_results)}")
        
        stage_start = datetime.now()
        
        try:
            # Use conversation history from memory if not provided
            if conversation_history is None:
                conversation_history = self.memory.get_history()
            
            # Build the prompt to calculate tokens later
            from app.services.llm.prompt_templates import (
                format_context_from_results,
                format_system_prompt,
                format_user_prompt,
                build_messages
            )
            
            context = format_context_from_results(retrieval_results)
            system_prompt = format_system_prompt(self.generator.company_name, context)
            user_prompt = format_user_prompt(query)
            messages = build_messages(system_prompt, user_prompt, conversation_history)
            
            # Generate streaming response
            print(f"\n{'='*80}")
            print(f"AI RESPONSE (Streaming)")
            print(f"{'='*80}\n")
            
            full_response = ""
            chunk_count = 0
            
            for chunk in self.generator.generate_stream(
                query=query,
                retrieval_results=retrieval_results,
                conversation_history=conversation_history
            ):
                full_response += chunk
                chunk_count += 1
            
            stage_duration = (datetime.now() - stage_start).total_seconds()
            
            # Count tokens using tiktoken
            import tiktoken
            try:
                encoding = tiktoken.encoding_for_model(self.generator.openai_client.model)
                
                # Count prompt tokens (all messages except the response)
                prompt_text = ""
                for msg in messages:
                    prompt_text += msg.get("content", "")
                
                prompt_tokens = len(encoding.encode(prompt_text))
                completion_tokens = len(encoding.encode(full_response))
                total_tokens = prompt_tokens + completion_tokens
                
            except Exception as e:
                self.logger.warning(f"Token counting failed: {e}")
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
            
            # Create a GenerationResult object for compatibility
            result = GenerationResult(
                response=full_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                generation_time=stage_duration,
                model=self.generator.openai_client.model
            )
            
            self.logger.info(f"[SUCCESS] Stage 3 completed in {stage_duration:.3f}s")
            self.logger.info(f"  - Response length: {len(result.response)} characters")
            self.logger.info(f"  - Chunks received: {chunk_count}")
            self.logger.info(f"  - Tokens used: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens})")
            
            return (
                RAGStageResult(
                    stage_name="Answer Generation",
                    success=True,
                    duration=stage_duration,
                    data={
                        "response_length": len(result.response),
                        "total_tokens": total_tokens,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "chunks_received": chunk_count,
                        "streaming": True
                    }
                ),
                result
            )
            
        except Exception as e:
            stage_duration = (datetime.now() - stage_start).total_seconds()
            self.logger.error(f"[FAILED] Stage 3 failed: {str(e)}", exc_info=True)
            import traceback
            traceback.print_exc()
            
            # Create error result
            error_result = GenerationResult(
                response=f"I apologize, but I encountered an error while generating a response: {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                generation_time=stage_duration,
                model=""
            )
            
            return (
                RAGStageResult(
                    stage_name="Answer Generation",
                    success=False,
                    duration=stage_duration,
                    error=str(e)
                ),
                error_result
            )
    
    # =========================================================================
    # Pipeline Execution
    # =========================================================================
    def query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        update_memory: bool = True
    ) -> RAGPipelineResult:
        """
        Execute the complete RAG pipeline for a query.
        
        Args:
            query: User query
            conversation_history: Optional conversation history (uses memory if not provided)
            metadata_filter: Optional metadata filter for retrieval
            update_memory: Whether to update conversation memory
        
        Returns:
            RAGPipelineResult with complete execution details
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info("RAG PIPELINE - START")
        self.logger.info("=" * 80)
        self.logger.info(f"Query: {query}")
        self.logger.info(f"Timestamp: {datetime.now().isoformat()}")
        self.logger.info("=" * 80)
        
        pipeline_start = datetime.now()
        self.stages_completed = []
        
        # Stage 1: Query Reformulation
        stage1_result, reformulated_query = self._stage_1_reformulate_query(
            query=query,
            conversation_history=conversation_history
        )
        self.stages_completed.append(stage1_result)
        
        # Stage 2: Document Retrieval
        stage2_result, retrieval_results = self._stage_2_retrieve_documents(
            query=reformulated_query,
            metadata_filter=metadata_filter
        )
        self.stages_completed.append(stage2_result)
        
        # Stage 3: Answer Generation
        stage3_result, generation_result = self._stage_3_generate_answer(
            query=query,  # Use original query for generation
            retrieval_results=retrieval_results,
            conversation_history=conversation_history
        )
        self.stages_completed.append(stage3_result)
        
        # Calculate total duration
        total_duration = (datetime.now() - pipeline_start).total_seconds()
        
        # Update memory if requested
        if update_memory:
            self.memory.add_user_message(query)
            self.memory.add_assistant_message(generation_result.response)
        
        # Update performance tracking
        self.total_queries_processed += 1
        self.total_processing_time += total_duration
        
        # Determine overall success
        success = all(stage.success for stage in self.stages_completed)
        
        # Create result
        result = RAGPipelineResult(
            success=success,
            query=query,
            reformulated_query=reformulated_query if reformulated_query != query else None,
            response=generation_result.response,
            retrieval_results=retrieval_results,
            total_duration=total_duration,
            stages=self.stages_completed,
            metadata={
                "total_tokens": generation_result.total_tokens,
                "num_sources": len(retrieval_results),
                "has_reformulation": reformulated_query != query,
                "avg_stage_duration": total_duration / len(self.stages_completed)
            }
        )
        
        # Log summary
        self.logger.info("\n" + "=" * 80)
        if success:
            self.logger.info("RAG PIPELINE - [COMPLETED SUCCESSFULLY]")
        else:
            self.logger.info("RAG PIPELINE - [COMPLETED WITH ERRORS]")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Duration: {total_duration:.3f}s")
        self.logger.info(f"Response Length: {len(generation_result.response)} characters")
        self.logger.info(f"Sources Used: {len(retrieval_results)}")
        self.logger.info("=" * 80 + "\n")
        
        return result
    
    # =========================================================================
    # Conversation Memory Management
    # =========================================================================
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()
        self.logger.info("Conversation memory cleared")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history."""
        return self.memory.get_history()
    
    # =========================================================================
    # Performance Monitoring
    # =========================================================================
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_processing_time = (
            self.total_processing_time / self.total_queries_processed
            if self.total_queries_processed > 0
            else 0.0
        )
        
        retriever_stats = self.retriever.get_retrieval_stats()
        
        return {
            "total_queries_processed": self.total_queries_processed,
            "total_processing_time": self.total_processing_time,
            "avg_processing_time": avg_processing_time,
            "retriever_stats": retriever_stats,
            "conversation_turns": len(self.memory.get_history()),
            "max_conversation_turns": self.memory.max_messages
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_queries_processed = 0
        self.total_processing_time = 0.0
        self.logger.info("Performance statistics reset")


# =============================================================================
# Main Entry Point (Interactive Mode)
# =============================================================================
def main():
    """Main entry point for interactive RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="RAG Pipeline - Interactive query-to-answer system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 rag_pipeline.py
  
  # Single query
  python3 rag_pipeline.py --query "What cloud solutions does MHK offer?"
  
  # With custom configuration
  python3 rag_pipeline.py --strategy basic --top-k 10
  
  # Disable query reformulation
  python3 rag_pipeline.py --no-reformulation
        """
    )
    
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to execute (skips interactive mode)'
    )
    
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['mmr', 'basic', 'hybrid'],
        default='hybrid',
        help='Retrieval strategy (default: hybrid)'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        help=f'Number of documents to retrieve (default: {settings.RETRIEVAL_TOP_K})'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        help='Minimum similarity score (default: 0.7)'
    )
    
    parser.add_argument(
        '--no-reformulation',
        action='store_true',
        help='Disable query reformulation'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show performance statistics and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        print("\nInitializing RAG Pipeline...")
        
        pipeline = RAGPipeline(
            retrieval_strategy=args.strategy,
            top_k=args.top_k,
            score_threshold=args.threshold,
            enable_reformulation=not args.no_reformulation
        )
        
        print("[SUCCESS] Pipeline initialized\n")
        
        # Show stats and exit
        if args.stats:
            stats = pipeline.get_performance_stats()
            print("=" * 80)
            print("PERFORMANCE STATISTICS")
            print("=" * 80)
            print(f"Total Queries: {stats['total_queries_processed']}")
            print(f"Avg Processing Time: {stats['avg_processing_time']:.3f}s")
            print(f"Conversation Turns: {stats['conversation_turns']}")
            print("=" * 80 + "\n")
            return 0
        
        # Single query mode
        if args.query:
            result = pipeline.query(args.query)
            
            print("=" * 80)
            print("QUERY RESULT")
            print("=" * 80)
            print(f"\nQuery: {result.query}")
            if result.reformulated_query:
                print(f"Reformulated: {result.reformulated_query}")
            print(f"\n{result.response}\n")
            print("=" * 80)
            print(f"Duration: {result.total_duration:.3f}s | Sources: {len(result.retrieval_results)} | Tokens: {result.metadata['total_tokens']}")
            print("=" * 80 + "\n")
            return 0
        
        # Interactive mode
        print("=" * 80)
        print("INTERACTIVE MODE")
        print("=" * 80)
        print("Type your questions or 'quit' to exit\n")
        
        while True:
            try:
                query = input("Query > ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!\n")
                    break
                
                if query.lower() == 'stats':
                    stats = pipeline.get_performance_stats()
                    print(f"\nQueries: {stats['total_queries_processed']} | ")
                    print(f"Avg Time: {stats['avg_processing_time']:.3f}s | ")
                    print(f"Conversation Turns: {stats['conversation_turns']}\n")
                    continue
                
                if query.lower() == 'clear':
                    pipeline.clear_memory()
                    print("\n[SUCCESS] Conversation memory cleared\n")
                    continue
                
                # Execute query
                print("\nProcessing...\n")
                result = pipeline.query(query)
                
                # Display result
                print("=" * 80)
                print(f"{result.response}\n")
                print("=" * 80)
                print(f"Time: {result.total_duration:.3f}s | Sources: {len(result.retrieval_results)} | Tokens: {result.metadata['total_tokens']}")
                print("=" * 80 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!\n")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}\n")
        
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n[ERROR] {str(e)}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())