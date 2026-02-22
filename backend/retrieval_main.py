"""
Retrieval system CLI tool.
Interactive mode for testing document retrieval.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.rag.retriever import get_retriever_service, RetrieverService
from app.services.rag.generator import get_generator_service, GeneratorService, ConversationMemory
from app.services.rag.query_reformulation import get_query_reformulator, QueryReformulator
from app.core.config import settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# Terminal Colors
# =============================================================================
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# =============================================================================
# Display Functions
# =============================================================================
def print_header():
    """Print application header."""
    print("\n" + "=" * 80)
    print(f"{Colors.BOLD}{Colors.CYAN}MHK-GPT RETRIEVAL SYSTEM{Colors.END}")
    print("=" * 80)
    print(f"Vector DB: {settings.qdrant_url}")
    print(f"Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"Embedding Model: {settings.OPENAI_EMBEDDING_MODEL}")
    print("=" * 80 + "\n")


def print_help():
    """Print help message."""
    print(f"\n{Colors.BOLD}Available Commands:{Colors.END}")
    print(f"  {Colors.GREEN}help{Colors.END}     - Show this help message")
    print(f"  {Colors.GREEN}stats{Colors.END}    - Show retrieval statistics")
    print(f"  {Colors.GREEN}config{Colors.END}   - Show current configuration")
    print(f"  {Colors.GREEN}clear{Colors.END}    - Clear screen")
    print(f"  {Colors.GREEN}quit{Colors.END}     - Exit the program")
    print(f"  {Colors.GREEN}exit{Colors.END}     - Exit the program")
    print(f"\n{Colors.BOLD}Query Options:{Colors.END}")
    print(f"  Just type your question to search!")
    print(f"  Examples:")
    print(f"    - What cloud solutions does MHK offer?")
    print(f"    - Tell me about AI and data science services")
    print(f"    - What is digital transformation?")
    print()


def print_config(retriever: RetrieverService):
    """Print current configuration."""
    print(f"\n{Colors.BOLD}Current Configuration:{Colors.END}")
    print(f"  Strategy: {Colors.CYAN}MMR (Maximal Marginal Relevance){Colors.END}")
    print(f"  Top K: {Colors.CYAN}{retriever.top_k}{Colors.END}")
    print(f"  Fetch K: {Colors.CYAN}{retriever.fetch_k}{Colors.END}")
    print(f"  Lambda (λ): {Colors.CYAN}{retriever.lambda_mult}{Colors.END}")
    print(f"  Score Threshold: {Colors.CYAN}{retriever.score_threshold}{Colors.END}")
    print(f"  Preview Length: {Colors.CYAN}{retriever.preview_length} chars{Colors.END}")
    print(f"  Max Chunks per Doc: {Colors.CYAN}{retriever.MAX_CHUNKS_PER_DOC}{Colors.END}")
    print()


def print_stats(retriever: RetrieverService):
    """Print retrieval statistics."""
    stats = retriever.get_retrieval_stats()
    
    print(f"\n{Colors.BOLD}Retrieval Statistics:{Colors.END}")
    print(f"  Total Queries: {Colors.CYAN}{stats['total_queries']}{Colors.END}")
    print(f"  Total Results: {Colors.CYAN}{stats['total_results']}{Colors.END}")
    print(f"  Avg Results/Query: {Colors.CYAN}{stats['avg_results_per_query']:.2f}{Colors.END}")
    print(f"  Avg Query Time: {Colors.CYAN}{stats['avg_query_time']:.3f}s{Colors.END}")
    
    if stats['strategy_counts']:
        print(f"\n  {Colors.BOLD}Strategy Usage:{Colors.END}")
        for strategy, count in stats['strategy_counts'].items():
            print(f"    {strategy}: {Colors.CYAN}{count}{Colors.END}")
    print()


def print_results(results, query: str, strategy: str = "mmr"):
    """
    Print retrieval results in a formatted way.
    
    Args:
        results: List of RetrievalResult objects
        query: Original query string
        strategy: Search strategy used
    """
    if not results:
        print(f"\n{Colors.YELLOW}No results found for: '{query}'{Colors.END}\n")
        return
    
    print(f"\n{Colors.BOLD}Query:{Colors.END} {Colors.CYAN}{query}{Colors.END}")
    print(f"{Colors.BOLD}Strategy:{Colors.END} {Colors.CYAN}{strategy.upper()}{Colors.END}")
    print(f"{Colors.BOLD}Results:{Colors.END} {Colors.GREEN}{len(results)} documents{Colors.END}")
    print("\n" + "─" * 80 + "\n")
    
    for idx, result in enumerate(results, 1):
        # Header with rank and score
        score_color = Colors.GREEN if result.score >= 0.8 else Colors.YELLOW if result.score >= 0.6 else Colors.RED
        print(f"{Colors.BOLD}[{idx}] Score: {score_color}{result.score:.4f}{Colors.END}")
        
        # Metadata
        file_name = result.metadata.get('file_name', 'Unknown')
        chunk_index = result.metadata.get('chunk_index', 'N/A')
        print(f"{Colors.BOLD}File:{Colors.END} {file_name}")
        print(f"{Colors.BOLD}Chunk:{Colors.END} {chunk_index}")
        
        # Preview
        print(f"\n{Colors.BOLD}Preview:{Colors.END}")
        print(f"{result.preview}")
        
        # Additional metadata (optional)
        content_type = result.metadata.get('content_type')
        if content_type:
            print(f"\n{Colors.BOLD}Type:{Colors.END} {content_type}")
        
        print("\n" + "─" * 80 + "\n")


def display_full_result(result, index: int):
    """
    Display full details of a single result.
    
    Args:
        result: RetrievalResult object
        index: Result index
    """
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.BOLD}Result #{index} - Full Details{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.END}\n")
    
    print(f"{Colors.BOLD}Score:{Colors.END} {result.score:.4f}")
    print(f"{Colors.BOLD}ID:{Colors.END} {result.id}")
    
    print(f"\n{Colors.BOLD}Metadata:{Colors.END}")
    for key, value in result.metadata.items():
        print(f"  {key}: {value}")
    
    print(f"\n{Colors.BOLD}Full Text:{Colors.END}")
    print(f"{result.text}")
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.END}\n")


# =============================================================================
# Interactive Mode
# =============================================================================
def interactive_mode(retriever: RetrieverService):
    """
    Run interactive query mode.
    
    Args:
        retriever: RetrieverService instance
    """
    print_header()
    print(f"{Colors.BOLD}Interactive Mode{Colors.END}")
    print(f"Type '{Colors.GREEN}help{Colors.END}' for commands, '{Colors.GREEN}quit{Colors.END}' to exit\n")
    
    # Initialize generator and conversation memory
    generator = get_generator_service()
    memory = ConversationMemory(max_messages=settings.MAX_CONVERSATION_HISTORY)
    reformulator = get_query_reformulator()
    
    last_results = []
    
    while True:
        try:
            # Get user input
            query = input(f"{Colors.BOLD}{Colors.BLUE}Query >{Colors.END} ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Colors.GREEN}Goodbye!{Colors.END}\n")
                break
            
            elif query.lower() == 'help':
                print_help()
                continue
            
            elif query.lower() == 'stats':
                print_stats(retriever)
                continue
            
            elif query.lower() == 'config':
                print_config(retriever)
                continue
            
            elif query.lower() == 'clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                print_header()
                continue
            
            # Check if user wants to see full result
            if query.lower().startswith('show '):
                try:
                    idx = int(query.split()[1]) - 1
                    if 0 <= idx < len(last_results):
                        display_full_result(last_results[idx], idx + 1)
                    else:
                        print(f"{Colors.RED}Invalid result number{Colors.END}")
                except (ValueError, IndexError):
                    print(f"{Colors.RED}Usage: show <number>{Colors.END}")
                continue
            
            # Reformulate query using conversation history
            reformulated_query = reformulator.reformulate(
                query=query,
                conversation_history=memory.get_history()
            )
            
            # Show reformulation if different from original
            if reformulated_query != query:
                print(f"\n{Colors.CYAN}💡 Reformulated query: {reformulated_query}{Colors.END}")
            
            # Execute retrieval with reformulated query
            print(f"\n{Colors.YELLOW}Searching...{Colors.END}")
            
            results = retriever.retrieve(
                query=reformulated_query,  # Use reformulated query for retrieval
                strategy="mmr",  # Default to MMR
                full_text=False
            )
            
            last_results = results
            
            # Show retrieval results if found
            if results:
                print_results(results, query, strategy="mmr")
            else:
                print(f"\n{Colors.YELLOW}No relevant documents found in knowledge base{Colors.END}")
            
            # Always generate AI answer (with or without context)
            try:
                print(f"\n{Colors.YELLOW}Generating AI answer...{Colors.END}\n")
                
                result = generator.generate(
                    query=query,
                    retrieval_results=results,  # Can be empty list
                    conversation_history=memory.get_history()
                )
                
                # Display AI-generated answer
                print(f"{'='*80}")
                print(f"{Colors.BOLD}{Colors.GREEN}AI-GENERATED ANSWER{Colors.END}")
                print(f"{'='*80}\n")
                print(f"{result.response}\n")
                print(f"{'='*80}")
                print(f"{Colors.CYAN}[Tokens: {result.total_tokens} | Time: {result.generation_time:.2f}s]{Colors.END}\n")
                
                # Update conversation memory
                memory.add_user_message(query)
                memory.add_assistant_message(result.response)
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}", exc_info=True)
                print(f"{Colors.RED}Generation error: {str(e)}{Colors.END}\n")
            
            # Hint for full text (only if there are results)
            if results:
                print(f"{Colors.BOLD}Tip:{Colors.END} Type '{Colors.GREEN}show <number>{Colors.END}' to see full text (e.g., 'show 1')\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}Goodbye!{Colors.END}\n")
            break
        
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}", exc_info=True)
            print(f"\n{Colors.RED}Error: {str(e)}{Colors.END}\n")


# =============================================================================
# Single Query Mode
# =============================================================================
def single_query_mode(
    retriever: RetrieverService,
    query: str,
    strategy: str = "mmr",
    top_k: int = None,
    threshold: float = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    full_text: bool = False
):
    """
    Execute a single query and display results.
    
    Args:
        retriever: RetrieverService instance
        query: Query string
        strategy: Search strategy
        top_k: Number of results
        threshold: Score threshold
        metadata_filter: Metadata filter
        full_text: Show full text
    """
    print_header()
    
    print(f"{Colors.YELLOW}Executing query...{Colors.END}\n")
    
    results = retriever.retrieve(
        query=query,
        strategy=strategy,
        top_k=top_k,
        score_threshold=threshold,
        metadata_filter=metadata_filter,
        full_text=full_text
    )
    
    # Show retrieval results if found
    if results:
        print_results(results, query, strategy=strategy)
    else:
        print(f"{Colors.YELLOW}No relevant documents found in knowledge base{Colors.END}\n")
    
    # Always generate AI answer (with or without context)
    try:
        generator = get_generator_service()
        print(f"\n{Colors.YELLOW}Generating AI answer...{Colors.END}\n")
        
        gen_result = generator.generate(
            query=query,
            retrieval_results=results  # Can be empty list
        )
        
        # Display AI-generated answer
        print(f"\n{'='*80}")
        print(f"{Colors.BOLD}{Colors.GREEN}AI-GENERATED ANSWER{Colors.END}")
        print(f"{'='*80}\n")
        print(f"{gen_result.response}\n")
        print(f"{'='*80}")
        print(f"{Colors.CYAN}[Tokens: {gen_result.total_tokens} | Time: {gen_result.generation_time:.2f}s]{Colors.END}\n")
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        print(f"{Colors.RED}Generation error: {str(e)}{Colors.END}\n")
    
    if full_text and results:
        print(f"\n{Colors.BOLD}Full Results:{Colors.END}\n")
        for idx, result in enumerate(results, 1):
            display_full_result(result, idx)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    """Main entry point for retrieval CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MHK-GPT Retrieval System - Test document retrieval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python3 retrieval_main.py
  
  # Single query
  python3 retrieval_main.py --query "What are MHK's cloud solutions?"
  
  # Custom configuration
  python3 retrieval_main.py --query "AI services" --strategy basic --top-k 10
  
  # With metadata filter
  python3 retrieval_main.py --query "solutions" --filter '{"file_name": "CLOUD Solutions.docx"}'
        """
    )
    
    # Query options
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Single query to execute (skips interactive mode)'
    )
    
    # Search configuration
    parser.add_argument(
        '--strategy', '-s',
        type=str,
        choices=['mmr', 'basic'],
        default='mmr',
        help='Retrieval strategy (default: mmr)'
    )
    
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        help=f'Number of results to return (default: {settings.RETRIEVAL_TOP_K})'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        help='Minimum similarity score (default: 0.7)'
    )
    
    parser.add_argument(
        '--filter', '-f',
        type=str,
        help='Metadata filter as JSON string (e.g., \'{"file_name": "report.docx"}\')'
    )
    
    # Display options
    parser.add_argument(
        '--full-text',
        action='store_true',
        help='Show full document text (not just preview)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize retriever
        print(f"\n{Colors.YELLOW}Initializing retrieval service...{Colors.END}")
        retriever = get_retriever_service()
        print(f"{Colors.GREEN}✓ Retriever initialized{Colors.END}")
        
        # Parse metadata filter if provided
        metadata_filter = None
        if args.filter:
            try:
                metadata_filter = json.loads(args.filter)
            except json.JSONDecodeError:
                print(f"{Colors.RED}Error: Invalid JSON in --filter argument{Colors.END}")
                return 1
        
        # Show stats and exit
        if args.stats:
            print_header()
            print_config(retriever)
            print_stats(retriever)
            return 0
        
        # Single query mode
        if args.query:
            single_query_mode(
                retriever=retriever,
                query=args.query,
                strategy=args.strategy,
                top_k=args.top_k,
                threshold=args.threshold,
                metadata_filter=metadata_filter,
                full_text=args.full_text
            )
        else:
            # Interactive mode
            interactive_mode(retriever)
        
        return 0
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.GREEN}Goodbye!{Colors.END}\n")
        return 0
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        print(f"\n{Colors.RED}Error: {str(e)}{Colors.END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
