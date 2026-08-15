import os
import re
import html
import time
import argparse
import pandas as pd
from tqdm import tqdm

def clean_tweet_text(text: str) -> str:
    """
    Cleans raw tweet text for sentiment analysis:
    1. HTML unescaping (&amp; -> &, &lt; -> <, etc.)
    2. Removes URLs (http/https)
    3. Removes Twitter user mentions (@username)
    4. Normalizes whitespaces and strips leading/trailing spaces
    """
    if not isinstance(text, str):
        return ""
    
    # Unescape HTML entities
    text = html.unescape(text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove @ mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove special control characters but keep punctuation and emojis
    text = re.sub(r'[\r\n\t]+', ' ', text)
    
    # Normalize multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_twcs(
    input_file: str,
    output_file: str,
    target_rows: int = 100000,
    min_words: int = 5,
    inbound_only: bool = True,
    chunksize: int = 100000
):
    """
    Reads twcs.csv in chunks, cleans text, filters low-context & missing rows,
    removes duplicates, and samples to target row count.
    """
    start_time = time.time()
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    print("=" * 65)
    print(" TWCS DATASET PREPROCESSING & REDUCTION PIPELINE ")
    print("=" * 65)
    print(f" Input File        : {input_file}")
    print(f" Output File       : {output_file}")
    print(f" Target Rows       : {target_rows if target_rows > 0 else 'All Cleaned Rows'}")
    print(f" Min Word Count    : {min_words} words")
    print(f" Customer Only     : {'Yes (inbound==True)' if inbound_only else 'No (All tweets)'}")
    print(f" Chunk Size        : {chunksize:,} rows/chunk")
    print("-" * 65)
    
    total_raw_rows = 0
    total_missing_removed = 0
    total_non_inbound_removed = 0
    total_short_removed = 0
    
    cleaned_chunks = []
    processed_clean_rows = 0
    
    # Determine total lines for progress bar if possible
    try:
        with open(input_file, 'rb') as f:
            approx_total_chunks = None
    except Exception:
        approx_total_chunks = None

    print("\n[Step 1/2] Reading and processing CSV in chunks...")
    
    for chunk in tqdm(pd.read_csv(input_file, chunksize=chunksize, low_memory=False), desc="Processing Chunks"):
        chunk_raw_count = len(chunk)
        total_raw_rows += chunk_raw_count
        
        # 1. Filter missing values in 'text'
        initial_count = len(chunk)
        chunk = chunk.dropna(subset=['text']).copy()
        total_missing_removed += (initial_count - len(chunk))
        
        # 2. Filter inbound customer tweets if enabled
        if inbound_only and 'inbound' in chunk.columns:
            initial_count = len(chunk)
            chunk = chunk[chunk['inbound'] == True].copy()
            total_non_inbound_removed += (initial_count - len(chunk))
            
        if chunk.empty:
            continue
            
        # 3. Clean tweet text
        chunk['clean_text'] = chunk['text'].apply(clean_tweet_text)
        
        # 4. Calculate word count of clean text
        chunk['word_count'] = chunk['clean_text'].apply(lambda s: len(s.split()))
        
        # 5. Filter out short comments
        initial_count = len(chunk)
        chunk = chunk[chunk['word_count'] >= min_words].copy()
        total_short_removed += (initial_count - len(chunk))
        
        if chunk.empty:
            continue
            
        cleaned_chunks.append(chunk)
        processed_clean_rows += len(chunk)
        
        # If we have gathered enough rows and target_rows is set, we can pause chunk collection if needed,
        # but to ensure representative sampling, we collect clean candidate rows and sample at end if total is large.
        if target_rows > 0 and processed_clean_rows >= target_rows * 3:
            # Reached a healthy buffer for global deduplication and sampling
            pass

    if not cleaned_chunks:
        print("\nNo valid rows remaining after filtering!")
        return

    print("\n[Step 2/2] Combining chunks, deduplicating, and reducing row count...")
    df_combined = pd.concat(cleaned_chunks, ignore_index=True)
    
    # 6. Remove duplicate clean texts
    initial_clean_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['clean_text']).reset_index(drop=True)
    total_duplicates_removed = initial_clean_len - len(df_combined)
    
    # 7. Downsample to target row count if specified
    if target_rows > 0 and len(df_combined) > target_rows:
        print(f" Reducing from {len(df_combined):,} eligible clean rows to target {target_rows:,} rows...")
        df_combined = df_combined.sample(n=target_rows, random_state=42).reset_index(drop=True)
    
    # Selecting useful columns for sentiment analysis
    output_columns = [col for col in ['tweet_id', 'author_id', 'created_at', 'inbound', 'text', 'clean_text', 'word_count'] if col in df_combined.columns]
    df_final = df_combined[output_columns]
    
    # Save output CSV
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 65)
    print(" PREPROCESSING SUMMARY & RESULTS ")
    print("=" * 65)
    print(f" Total Initial Rows Read    : {total_raw_rows:,}")
    print(f" Missing Text Rows Removed  : {total_missing_removed:,}")
    if inbound_only:
        print(f" Support Bot/Rep Removed    : {total_non_inbound_removed:,}")
    print(f" Short Comments Removed     : {total_short_removed:,} (< {min_words} words)")
    print(f" Duplicate Tweets Removed   : {total_duplicates_removed:,}")
    print(f" Final Dataset Row Count    : {len(df_final):,}")
    print(f" Saved Clean CSV To        : {os.path.abspath(output_file)}")
    print(f" File Size                 : {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    print(f" Total Execution Time      : {elapsed_time:.2f} seconds")
    print("=" * 65)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean and preprocess TWCS dataset for Sentiment Analysis.")
    parser.add_argument("--input", "-i", type=str, default="twcs.csv", help="Path to input twcs.csv file")
    parser.add_argument("--output", "-o", type=str, default="twcs_cleaned.csv", help="Path for cleaned output CSV file")
    parser.add_argument("--target-rows", "-r", type=int, default=100000, help="Target number of output rows (0 for all valid rows)")
    parser.add_argument("--min-words", "-w", type=int, default=5, help="Minimum word count to keep a comment")
    parser.add_argument("--keep-outbound", action="store_true", help="Keep outbound (company response) tweets as well")
    parser.add_argument("--chunk-size", type=int, default=100000, help="Chunk size for processing large CSV")
    
    args = parser.parse_args()
    
    preprocess_twcs(
        input_file=args.input,
        output_file=args.output,
        target_rows=args.target_rows,
        min_words=args.min_words,
        inbound_only=not args.keep_outbound,
        chunksize=args.chunk_size
    )
