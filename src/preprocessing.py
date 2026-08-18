import os
import re
import html
import time
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Common English Contractions Mapping
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bcant\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bwont\b": "will not",
    r"\bn't\b": " not",
    r"\bain't\b": "is not",
    r"\bdon't\b": "do not",
    r"\bdont\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdoesnt\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bdidnt\b": "did not",
    r"\bisn't\b": "is not",
    r"\bisnt\b": "is not",
    r"\baren't\b": "are not",
    r"\barent\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bwasnt\b": "was not",
    r"\bweren't\b": "were not",
    r"\bwerent\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhavent\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhasnt\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bhadnt\b": "had not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bcouldn't\b": "could not",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bwhat's\b": "what is",
    r"\bthere's\b": "there is",
    r"\blet's\b": "let us"
}

# Negation Triggers and Clause Delimiters
NEGATION_TRIGGERS = {
    "not", "no", "never", "cannot", "cant", "n't", "neither", "nor", 
    "without", "hardly", "scarcely", "barely", "rarely", "seldom", 
    "lack", "lacking", "nowhere", "nothing", "none"
}

# Clause Delimiters and Subordinators that Terminate Negation Scope
CLAUSE_DELIMITERS = {
    ".", ",", "!", "?", ";", ":", "-", "--", "(", ")", "[", "]", "{", "}",
    "but", "however", "although", "though", "yet", "except", "while", "nevertheless",
    "instead", "that", "which", "who", "whom", "whose", "because", "since",
    "unless", "whereas", "wherever", "after", "before", "so", "and", "or"
}

# Verbs of removal / cessation / suppression
# When negated (e.g. "could not shake the fear", "cannot stop the anxiety"),
# the verb itself is negated, but the subsequent noun represents an active/retained emotion.
REMOVAL_VERBS = {
    "shake", "stop", "eliminate", "dispel", "prevent", "avoid", "contain",
    "suppress", "quell", "overcome", "control", "hide", "resist", "forget",
    "lose", "calm"
}


def expand_contractions(text: str) -> str:
    """Expands common English contractions for consistent negation recognition."""
    if not isinstance(text, str):
        return ""
    text_lower = text.lower()
    for pattern, replacement in CONTRACTIONS.items():
        text_lower = re.sub(pattern, replacement, text_lower)
    return text_lower


def apply_negation_tagging(text: str, max_window: int = 3) -> str:
    """
    Applies bounded negation scope tagging.
    Appends '_NEG' to tokens within max_window words following a negation trigger,
    terminating immediately upon encountering clause delimiters, subordinating conjunctions,
    relative pronouns, or removal/cessation verbs.
    
    Example:
      "I was not disappointed with the outcome, but I was nervous"
      --> "i was not disappointed_NEG with_NEG the_NEG outcome , but i was nervous"
      
      "I could not shake the fear that everything might disappear"
      --> "i could not shake_NEG the fear that everything might disappear"
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    expanded = expand_contractions(text)
    tokens = re.findall(r"\w+|[^\w\s]", expanded, re.UNICODE)
    
    tagged_tokens = []
    scope_remaining = 0
    
    for token in tokens:
        token_lower = token.lower()
        
        # Punctuation or clause delimiter immediately clears negation scope
        if token_lower in CLAUSE_DELIMITERS or re.match(r"^[.,!?;:\-–—]$", token):
            scope_remaining = 0
            tagged_tokens.append(token)
            continue
            
        # Negation trigger activates scope for max_window words
        if token_lower in NEGATION_TRIGGERS:
            scope_remaining = max_window
            tagged_tokens.append(token)
            continue
            
        if scope_remaining > 0:
            # If the token is a removal verb (e.g., "shake" in "could not shake")
            if token_lower in REMOVAL_VERBS:
                tagged_tokens.append(f"{token}_NEG")
                # Terminate negation scope because the object being shaken/avoided is retained!
                scope_remaining = 0
                continue
                
            if token.isalnum() and not token.isdigit():
                tagged_tokens.append(f"{token}_NEG")
            else:
                tagged_tokens.append(token)
                
            scope_remaining -= 1
        else:
            tagged_tokens.append(token)
            
    return " ".join(tagged_tokens)


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
    input_file: str = None,
    output_file: str = None,
    target_rows: int = 100000,
    min_words: int = 5,
    inbound_only: bool = True,
    chunksize: int = 100000
):
    """
    Reads twcs.csv in chunks, cleans text, filters low-context & missing rows,
    removes duplicates, and samples to target row count.
    """
    if input_file is None:
        input_file = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")
    if output_file is None:
        output_file = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
        
    start_time = time.time()
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
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
    default_in = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")
    default_out = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    
    parser = argparse.ArgumentParser(description="Clean and preprocess TWCS dataset for Sentiment Analysis.")
    parser.add_argument("--input", "-i", type=str, default=default_in, help="Path to input twcs.csv file")
    parser.add_argument("--output", "-o", type=str, default=default_out, help="Path for cleaned output CSV file")
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
