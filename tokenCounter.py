"""
Token Counter Script
Counts the total number of tokens across all files in a directory and subdirectories using tiktoken for tokenization
"""
import argparse
import os
import sys

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is not installed. Please install it with 'pip install tiktoken'.")
    sys.exit(1)

# Function to count tokens across files
def count(directory: str, model: str) -> tuple[int, int, int]:
    """
    Recursively count tokens in all files within a directory.
    
    Args:
        directory: Path to the directory to scan
        model: Model name to use for tiktoken encoding
        
    Returns:
        Tuple of (total_tokens, files_processed, files_skipped)
    """
    try:
        encoder = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Model '{model}' not found, using o200k_base encoding")
        encoder = tiktoken.get_encoding("o200k_base")
    
    total_tokens = 0
    files_processed = 0
    files_skipped = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            relative_path = os.path.relpath(filepath, directory)
            
            print(f"\rProcessing: {relative_path[:60]:<60}", end="", flush=True) # Print current file (overwrite same line)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tokens = len(encoder.encode(content))
                    total_tokens += tokens
                    files_processed += 1
            except (UnicodeDecodeError, PermissionError, OSError):
                files_skipped += 1
    
    print("\r" + " " * 80 + "\r", end="") # Clear the processing line
    return total_tokens, files_processed, files_skipped

# Main function to parse arguments and invoke count
def main():
    parser = argparse.ArgumentParser(
        description="Count tokens in all files within a directory using tiktoken"
    )
    parser.add_argument(
        "directory",
        help="Path to the directory to scan"
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Tiktoken model (default: gpt-5)"
    )
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: '{args.directory}' is not a valid directory")
        sys.exit(1)
    
    directory = os.path.abspath(args.directory)
    print(f"Counting tokens in: {directory}")
    print(f"Using encoding for model: {args.model}")
    print("-" * 40)
    
    total_tokens, files_processed, files_skipped = count(
        directory, args.model
    )
    
    # Print results
    print(f"Total tokens: {total_tokens:,}")
    print(f"Files processed: {files_processed:,}")
    print(f"Files skipped: {files_skipped:,}")

main()