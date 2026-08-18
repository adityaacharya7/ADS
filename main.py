#!/usr/bin/env python
"""
Applied Data Science (ADS) - Twitter Customer Support Emotion & Sentiment Analysis Toolkit
Master Command Line Interface (CLI)

Usage examples:
    python main.py demo
    python main.py predict --text "My flight was delayed 8 hours and nobody helped!"
    python main.py predict --interactive
    python main.py train --sample-size 50000
    python main.py eda --sample-size 50000
    python main.py report
    python main.py preprocess
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure UTF-8 stdout encoding where possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import preprocess_twcs
from src.train import train_and_evaluate
from src.predict import EmotionPredictor, run_demo
from src.eda_analysis import run_experiment_3
from src.generate_report import generate_detailed_pdf


def cmd_preprocess(args):
    print(">>> Running Data Preprocessing Pipeline...")
    preprocess_twcs(
        input_file=args.input,
        output_file=args.output,
        target_rows=args.target_rows,
        min_words=args.min_words,
        inbound_only=not args.keep_outbound,
        chunksize=args.chunk_size
    )


def cmd_train(args):
    print(">>> Running Model Training & Benchmarking Pipeline...")
    train_and_evaluate(
        dataset_path=args.input,
        sample_size=args.sample_size
    )


def cmd_predict(args):
    predictor = EmotionPredictor(args.model)
    if args.interactive:
        print("Starting interactive Emotion Prediction session (type 'exit' or 'quit' to stop):")
        while True:
            try:
                user_input = input("\nEnter tweet text: ").strip()
                if user_input.lower() in ('exit', 'quit', 'q'):
                    break
                if user_input:
                    predictor.print_prediction(user_input)
            except (KeyboardInterrupt, EOFError):
                break
    elif args.text:
        predictor.print_prediction(args.text)
    else:
        run_demo()


def cmd_demo(args):
    run_demo()


def cmd_eda(args):
    print(">>> Running Experiment 3: EDA & Statistical Analysis Pipeline...")
    run_experiment_3(
        input_file=args.input,
        sample_size=args.sample_size
    )


def cmd_report(args):
    print(">>> Compiling Academic PDF Report...")
    generate_detailed_pdf(output_pdf_path=args.output)


def cmd_pipeline(args):
    print("=" * 70)
    print(" RUNNING COMPLETE END-TO-END PIPELINE ")
    print("=" * 70)
    
    clean_csv = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    raw_csv = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")
    
    if not os.path.exists(clean_csv):
        if os.path.exists(raw_csv):
            print("\n[Stage 1/4] Preprocessing raw data...")
            preprocess_twcs(input_file=raw_csv, output_file=clean_csv, target_rows=100000)
        else:
            print(f"Error: Neither '{clean_csv}' nor '{raw_csv}' found!")
            return
            
    print("\n[Stage 2/4] Executing Statistical EDA & Hypothesis Testing...")
    run_experiment_3(input_file=clean_csv, sample_size=args.sample_size or 50000)
    
    print("\n[Stage 3/4] Training Emotion Detection Classifiers...")
    train_and_evaluate(dataset_path=clean_csv, sample_size=args.sample_size or 50000)
    
    print("\n[Stage 4/4] Compiling Formal PDF Report...")
    generate_detailed_pdf()
    
    print("\n" + "=" * 70)
    print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="ADS Twitter Customer Support Emotion & Sentiment Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  preprocess   Clean and reduce raw TWCS dataset
  train        Train ML/NLP emotion detection classifiers
  predict      Predict sentiment & emotion for a tweet (CLI / Interactive)
  demo         Run sample inference tests on customer queries
  eda          Run Experiment 3 EDA, distributions & hypothesis testing
  report       Compile academic PDF report
  pipeline     Execute full end-to-end pipeline
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive prediction shell directly")
    parser.add_argument("--demo", "-d", action="store_true", help="Run prediction demonstration suite")
    parser.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze directly")
    
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")
    
    # 1. Preprocess
    p_preprocess = subparsers.add_parser("preprocess", help="Clean & preprocess raw twcs.csv")
    p_preprocess.add_argument("--input", "-i", type=str, default=str(PROJECT_ROOT / "data" / "raw" / "twcs.csv"), help="Raw CSV path")
    p_preprocess.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv"), help="Clean CSV path")
    p_preprocess.add_argument("--target-rows", "-r", type=int, default=100000, help="Target row count")
    p_preprocess.add_argument("--min-words", "-w", type=int, default=5, help="Minimum word threshold")
    p_preprocess.add_argument("--keep-outbound", action="store_true", help="Include company responses")
    p_preprocess.add_argument("--chunk-size", type=int, default=100000, help="Chunk size")
    p_preprocess.set_defaults(func=cmd_preprocess)
    
    # 2. Train
    p_train = subparsers.add_parser("train", help="Train emotion classifiers (Logistic Regression, LightGBM, XGBoost)")
    p_train.add_argument("--input", "-i", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv"), help="Clean CSV path")
    p_train.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size (0 for all rows)")
    p_train.set_defaults(func=cmd_train)
    
    # 3. Predict
    p_predict = subparsers.add_parser("predict", help="Predict emotion & sentiment for tweets")
    p_predict.add_argument("pos_text", nargs="?", type=str, default=None, help="Tweet text (positional)")
    p_predict.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze")
    p_predict.add_argument("--model", "-m", type=str, default=str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib"), help="Model path")
    p_predict.add_argument("--interactive", action="store_true", help="Interactive prediction shell")
    
    def _predict_wrapper(args):
        args.text = args.text or getattr(args, "pos_text", None)
        cmd_predict(args)
        
    p_predict.set_defaults(func=_predict_wrapper)
    
    # 4. Demo
    p_demo = subparsers.add_parser("demo", help="Run prediction demonstration on sample customer tweets")
    p_demo.set_defaults(func=cmd_demo)
    
    # 5. EDA
    p_eda = subparsers.add_parser("eda", help="Run statistical EDA & hypothesis testing suite")
    p_eda.add_argument("--input", "-i", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv"), help="Clean CSV path")
    p_eda.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size")
    p_eda.set_defaults(func=cmd_eda)
    
    # 6. Report
    p_report = subparsers.add_parser("report", help="Compile academic PDF report")
    p_report.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "reports" / "experiment_3" / "Experiment_3_Report.pdf"), help="Output PDF path")
    p_report.set_defaults(func=cmd_report)
    
    # 7. Pipeline
    p_pipeline = subparsers.add_parser("pipeline", help="Run full end-to-end pipeline")
    p_pipeline.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size for training and EDA")
    p_pipeline.set_defaults(func=cmd_pipeline)
    
    args = parser.parse_args()
    
    if args.interactive:
        args.model = str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib")
        cmd_predict(args)
    elif args.demo:
        cmd_demo(args)
    elif args.text:
        args.model = str(PROJECT_ROOT / "models" / "emotion_pipeline.joblib")
        cmd_predict(args)
    elif args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
