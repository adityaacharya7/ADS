#!/usr/bin/env python
"""
Applied Data Science (ADS) - Master Command Line Interface (CLI)
Organized by Experiments:
  - Experiment 2: Data Preprocessing, Negation-Aware Feature Engineering & Baseline Classifier
  - Experiment 3: Statistical Exploratory Data Analysis, Distribution Fitting & Hypothesis Testing
  - Experiment 4: Multi-Model Machine Learning Benchmarking, Hyperparameter Tuning & MLflow Tracking

Usage examples:
  1. Experiment-based commands:
     python main.py exp2 demo
     python main.py exp2 predict --text "My package was damaged and 3 weeks late!"
     python main.py exp2 train --sample-size 50000
     python main.py exp3 eda --sample-size 50000
     python main.py exp3 report
     python main.py exp4 train --sample-size 25000
     python main.py exp4 report

  2. Legacy direct shortcuts (fully supported for backward compatibility):
     python main.py demo
     python main.py predict --text "Delayed flight!"
     python main.py preprocess
     python main.py train
     python main.py eda
     python main.py report
     python main.py pipeline
"""

import sys
import os
import argparse
import subprocess
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

# Experiment 2 imports
from experiments.experiment_2.src.preprocessing import preprocess_twcs
from experiments.experiment_2.src.train import train_and_evaluate
from experiments.experiment_2.src.predict import EmotionPredictor, run_demo

# Experiment 3 imports
from experiments.experiment_3.src.eda_analysis import run_experiment_3
from experiments.experiment_3.src.generate_report import generate_detailed_pdf

# Experiment 4 imports
from experiments.experiment_4.src.experiment_4_modeling import (
    train_and_track_experiments,
    load_and_prepare_data,
)
from experiments.experiment_4.src.generate_experiment_4_report import (
    generate_experiment_4_report,
    generate_experiment_4_latex,
    generate_experiment_4_pdf,
    load_experiment_metadata as load_exp4_metadata,
)
from experiments.experiment_4.src.generate_experiment_4_docx import generate_experiment_4_docx

# Experiment 5 imports
from experiments.experiment_5.src.experiment_5_xai_fairness import (
    run_experiment_5_pipeline,
    load_experiment_5_summary
)
from experiments.experiment_5.src.generate_experiment_5_report import (
    generate_experiment_5_report
)
from experiments.experiment_5.src.generate_experiment_5_docx import (
    generate_experiment_5_docx
)

# Experiment 6 imports
from experiments.experiment_6.src.test_api import run_api_tests
from experiments.experiment_6.src.generate_experiment_6_report import generate_experiment_6_report
from experiments.experiment_6.src.generate_experiment_6_docx import generate_experiment_6_docx



# ------------------------------------------------------------------------------
# EXPERIMENT 2 HANDLERS
# ------------------------------------------------------------------------------

def cmd_preprocess(args):
    print(">>> [Experiment 2] Running Data Preprocessing Pipeline...")
    preprocess_twcs(
        input_file=args.input,
        output_file=args.output,
        target_rows=args.target_rows,
        min_words=args.min_words,
        inbound_only=not args.keep_outbound,
        chunksize=args.chunk_size
    )


def cmd_train(args):
    print(">>> [Experiment 2] Running Negation-Aware Multi-Label Classifier Training...")
    train_and_evaluate(
        dataset_path=args.input,
        sample_size=args.sample_size
    )


def cmd_predict(args):
    model_path = args.model
    if model_path is None:
        cand1 = PROJECT_ROOT / "experiments" / "experiment_2" / "models" / "emotion_pipeline.joblib"
        cand2 = PROJECT_ROOT / "models" / "emotion_pipeline.joblib"
        model_path = str(cand1 if cand1.exists() else cand2)

    predictor = EmotionPredictor(model_path)
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


# ------------------------------------------------------------------------------
# EXPERIMENT 3 HANDLERS
# ------------------------------------------------------------------------------

def cmd_eda(args):
    print(">>> [Experiment 3] Running Statistical EDA & Hypothesis Testing Pipeline...")
    run_experiment_3(
        input_file=args.input,
        sample_size=args.sample_size
    )


def cmd_report_exp3(args):
    print(">>> [Experiment 3] Compiling Academic PDF Report...")
    output_path = args.output
    if output_path is None:
        output_path = str(PROJECT_ROOT / "experiments" / "experiment_3" / "reports" / "Experiment_3_Report.pdf")
    generate_detailed_pdf(output_pdf_path=output_path)


# ------------------------------------------------------------------------------
# EXPERIMENT 4 HANDLERS
# ------------------------------------------------------------------------------

def cmd_exp4_train(args):
    print(">>> [Experiment 4] Running Model Benchmarking, Tuning & MLflow Tracking...")
    train_and_track_experiments(sample_size=args.sample_size)


def cmd_exp4_report(args):
    print(">>> [Experiment 4] Generating Publication-Quality Academic Reports (LaTeX, PDF, DOCX)...")
    pdf_out = getattr(args, "output", None)
    tex_out = getattr(args, "tex", None)
    if pdf_out is None:
        pdf_out = str(PROJECT_ROOT / "experiments" / "experiment_4" / "reports" / "Experiment_4_Report.pdf")
    if tex_out is None:
        tex_out = str(PROJECT_ROOT / "experiments" / "experiment_4" / "reports" / "Experiment_4_Report.tex")
    generate_experiment_4_report(output_pdf_path=pdf_out, output_tex_path=tex_out)


def cmd_exp4_docx(args):
    print(">>> [Experiment 4] Compiling Academic Word (.docx) Document...")
    docx_out = getattr(args, "output", None)
    if docx_out is None:
        docx_out = str(PROJECT_ROOT / "experiments" / "experiment_4" / "reports" / "Experiment_4_Report.docx")
    metadata, df = load_exp4_metadata()
    generate_experiment_4_docx(metadata, df, output_docx_path=docx_out)


def cmd_exp4_dashboard(args):
    mlruns_dir = PROJECT_ROOT / "experiments" / "experiment_4" / "mlruns"
    sqlite_db = (mlruns_dir / "mlflow.db").as_posix()
    print("=" * 70)
    print(" LAUNCHING MLFLOW INTERACTIVE TRACKING DASHBOARD")
    print("=" * 70)
    print(f" MLflow Database URI: sqlite:///{sqlite_db}")
    print(f" Default Port       : {args.port}")
    print(f"\n To view the dashboard, open your browser to: http://localhost:{args.port}")
    print(" Press Ctrl+C to terminate the dashboard server.\n")
    try:
        subprocess.run([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", f"sqlite:///{sqlite_db}",
            "--port", str(args.port),
            "--host", args.host
        ])
    except KeyboardInterrupt:
        print("\n[+] MLflow UI stopped.")


# ------------------------------------------------------------------------------
# EXPERIMENT 5 HANDLERS
# ------------------------------------------------------------------------------

def cmd_exp5_run(args):
    print(">>> [Experiment 5] Running XAI (SHAP & LIME) and Fairness Auditing Pipeline...")
    run_experiment_5_pipeline()


def cmd_exp5_report(args):
    print(">>> [Experiment 5] Generating Publication-Quality Academic Reports (LaTeX, PDF, DOCX)...")
    pdf_out = getattr(args, "output", None)
    tex_out = getattr(args, "tex", None)
    docx_out = getattr(args, "docx", None)
    if pdf_out is None:
        pdf_out = str(PROJECT_ROOT / "experiments" / "experiment_5" / "reports" / "Experiment_5_Report.pdf")
    if tex_out is None:
        tex_out = str(PROJECT_ROOT / "experiments" / "experiment_5" / "reports" / "Experiment_5_Report.tex")
    if docx_out is None:
        docx_out = str(PROJECT_ROOT / "experiments" / "experiment_5" / "reports" / "Experiment_5_Report.docx")
    generate_experiment_5_report(output_pdf_path=pdf_out, output_tex_path=tex_out)
    generate_experiment_5_docx(output_docx_path=docx_out)


def cmd_exp5_demo(args):
    print(">>> [Experiment 5] Running Interactive Local Explanation & Fairness Demo...")
    summary = load_experiment_5_summary()
    print("\n" + "=" * 70)
    print(" EXPERIMENT 5: XAI & FAIRNESS AUDITING DEMO SUMMARY")
    print("=" * 70)
    bm = summary.get("fairness_and_mitigation", [])
    if bm:
        import pandas as pd
        print("\n--- BIAS MITIGATION BENCHMARK (PERFORMANCE VS FAIRNESS) ---")
        print(pd.DataFrame(bm).to_string(index=False))
    print("\n--- TOP GLOBAL SHAP FEATURES ---")
    top_f = summary.get("shap_explainability", {}).get("top_features_ranked", [])[:5]
    for idx, f in enumerate(top_f, 1):
        print(f"  {idx}. {f['Feature']:20s} (Mean |SHAP| = {f['Mean_Absolute_SHAP']:.4f})")
    print(f"\n[+] Generated plots available in: experiments/experiment_5/plots/")
    print(f"[+] Academic PDF Report available in: experiments/experiment_5/reports/Experiment_5_Report.pdf")
    print(f"[+] Academic Word Report available in: experiments/experiment_5/reports/Experiment_5_Report.docx")


# ------------------------------------------------------------------------------
# EXPERIMENT 6 HANDLERS
# ------------------------------------------------------------------------------

def cmd_exp6_serve(args):
    import uvicorn
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8000)
    reload = getattr(args, "reload", False)
    print("=" * 70)
    print(" LAUNCHING ADS FASTAPI PRODUCTION INFERENCE MICROSERVICE")
    print("=" * 70)
    print(f" Microservice URL        : http://{host}:{port}")
    print(f" Swagger UI Documentation: http://{host}:{port}/docs")
    print(f" ReDoc Documentation     : http://{host}:{port}/redoc")
    print(f" Health Probe Endpoint   : http://{host}:{port}/health")
    print(" Press Ctrl+C to terminate the server.\n")
    uvicorn.run("experiments.experiment_6.src.app:app", host=host, port=port, reload=reload)


def cmd_exp6_test(args):
    print(">>> [Experiment 6] Running Automated API Verification Suite & Benchmarks...")
    run_api_tests()


def cmd_exp6_docker_build(args):
    print("=" * 70)
    print(" BUILDING DOCKER CONTAINER IMAGE (ads-emotion-api:latest)")
    print("=" * 70)
    dockerfile = PROJECT_ROOT / "experiments" / "experiment_6" / "Dockerfile"
    cmd = ["docker", "build", "-t", "ads-emotion-api:latest", "-f", str(dockerfile), str(PROJECT_ROOT)]
    print(f" Executing: {' '.join(cmd)}\n")
    try:
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("\n[+] Docker image 'ads-emotion-api:latest' built successfully!")
        else:
            print("\n[-] Docker build exited with error. Ensure Docker Desktop is running.")
    except Exception as e:
        print(f"[-] Docker invocation failed: {e}")


def cmd_exp6_docker_run(args):
    port = getattr(args, "port", 8000)
    print("=" * 70)
    print(f" RUNNING DOCKER CONTAINER ON PORT {port}")
    print("=" * 70)
    cmd = ["docker", "run", "-d", "--name", "ads-emotion-microservice", "-p", f"{port}:8000", "--rm", "ads-emotion-api:latest"]
    print(f" Executing: {' '.join(cmd)}\n")
    try:
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print(f"\n[+] Container running at http://localhost:{port} (docs: http://localhost:{port}/docs)")
        else:
            print("\n[-] Docker run failed. Ensure Docker Desktop daemon is started.")
    except Exception as e:
        print(f"[-] Docker run invocation failed: {e}")


def cmd_exp6_report(args):
    print(">>> [Experiment 6] Compiling Academic Reports (PDF, Word, LaTeX)...")
    pdf_out = getattr(args, "output", None)
    tex_out = getattr(args, "tex", None)
    docx_out = getattr(args, "docx", None)
    if pdf_out is None:
        pdf_out = str(PROJECT_ROOT / "experiments" / "experiment_6" / "reports" / "Experiment_6_Report.pdf")
    if tex_out is None:
        tex_out = str(PROJECT_ROOT / "experiments" / "experiment_6" / "reports" / "Experiment_6_Report.tex")
    if docx_out is None:
        docx_out = str(PROJECT_ROOT / "experiments" / "experiment_6" / "reports" / "Experiment_6_Report.docx")
    generate_experiment_6_report(output_pdf_path=pdf_out, output_tex_path=tex_out)
    generate_experiment_6_docx(output_docx_path=docx_out)


# ------------------------------------------------------------------------------
# FULL PIPELINE HANDLER
# ------------------------------------------------------------------------------

def cmd_pipeline(args):
    print("=" * 70)
    print(" RUNNING COMPLETE END-TO-END REPOSITORY PIPELINE ")
    print("=" * 70)

    clean_csv = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    raw_csv = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")

    if not os.path.exists(clean_csv):
        if os.path.exists(raw_csv):
            print("\n[Stage 1/7] Preprocessing raw TWCS data (Exp 2)...")
            preprocess_twcs(input_file=raw_csv, output_file=clean_csv, target_rows=100000)
        else:
            print(f"Error: Neither '{clean_csv}' nor '{raw_csv}' found!")
            return

    print("\n[Stage 2/7] Running Statistical EDA & Hypothesis Testing (Exp 3)...")
    run_experiment_3(input_file=clean_csv, sample_size=args.sample_size or 50000)

    print("\n[Stage 3/7] Compiling Experiment 3 Academic PDF Report...")
    exp3_pdf = str(PROJECT_ROOT / "experiments" / "experiment_3" / "reports" / "Experiment_3_Report.pdf")
    generate_detailed_pdf(output_pdf_path=exp3_pdf)

    print("\n[Stage 4/7] Running Multi-Model Training, Tuning & MLflow Tracking (Exp 4)...")
    train_and_track_experiments(sample_size=args.sample_size or 25000)

    print("\n[Stage 5/7] Compiling Experiment 4 Reports (LaTeX, PDF, DOCX)...")
    generate_experiment_4_report()

    print("\n[Stage 6/7] Running Explainable AI & Algorithmic Fairness Pipeline (Exp 5)...")
    run_experiment_5_pipeline()
    generate_experiment_5_report()
    generate_experiment_5_docx()

    print("\n[Stage 7/7] Running API Testing & Container Deployment Verification (Exp 6)...")
    run_api_tests()
    generate_experiment_6_report()
    generate_experiment_6_docx()

    print("\n" + "=" * 70)
    print(" ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
    print("=" * 70)




# ------------------------------------------------------------------------------
# CLI ARGUMENT PARSER SETUP
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ADS Twitter Customer Support Emotion & Sentiment Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands Overview:
  exp2         Experiment 2: Preprocessing, Feature Engineering & Baseline Classifier
  exp3         Experiment 3: Statistical EDA, Distribution Fitting & Hypothesis Testing
  exp4         Experiment 4: ML Modeling, Hyperparameter Tuning & MLflow Tracking
  exp5         Experiment 5: Explainable AI (SHAP & LIME) & Fairness Auditing (Fairlearn)
  exp6         Experiment 6: Containerization & API Deployment (FastAPI & Docker)
  pipeline     Execute complete multi-experiment pipeline end-to-end

Legacy Shortcuts (direct access):
  preprocess   Clean and reduce raw TWCS dataset
  train        Train Experiment 2 emotion classifiers
  predict      Predict sentiment & emotion for a tweet (CLI / Interactive)
  demo         Run sample inference demonstration
  eda          Run Experiment 3 EDA suite
  report       Compile Experiment 3 academic PDF report
        """
    )

    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive prediction shell directly")
    parser.add_argument("--demo", "-d", action="store_true", help="Run prediction demonstration suite")
    parser.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze directly")

    subparsers = parser.add_subparsers(dest="command", help="Command or Experiment to execute")

    default_clean = str(PROJECT_ROOT / "data" / "processed" / "twcs_cleaned.csv")
    default_raw = str(PROJECT_ROOT / "data" / "raw" / "twcs.csv")
    default_exp2_model = str(PROJECT_ROOT / "experiments" / "experiment_2" / "models" / "emotion_pipeline.joblib")
    default_exp3_pdf = str(PROJECT_ROOT / "experiments" / "experiment_3" / "reports" / "Experiment_3_Report.pdf")
    default_exp4_pdf = str(PROJECT_ROOT / "experiments" / "experiment_4" / "reports" / "Experiment_4_Report.pdf")
    default_exp5_pdf = str(PROJECT_ROOT / "experiments" / "experiment_5" / "reports" / "Experiment_5_Report.pdf")
    default_exp5_docx = str(PROJECT_ROOT / "experiments" / "experiment_5" / "reports" / "Experiment_5_Report.docx")
    default_exp6_pdf = str(PROJECT_ROOT / "experiments" / "experiment_6" / "reports" / "Experiment_6_Report.pdf")
    default_exp6_docx = str(PROJECT_ROOT / "experiments" / "experiment_6" / "reports" / "Experiment_6_Report.docx")

    # =========================================================================
    # 1. EXP2 SUBCOMMAND GROUP
    # =========================================================================
    p_exp2 = subparsers.add_parser("exp2", help="Experiment 2: Preprocessing, Features & Emotion Classification")
    exp2_subs = p_exp2.add_subparsers(dest="subcommand", help="Experiment 2 action")

    # exp2 preprocess
    p_e2_prep = exp2_subs.add_parser("preprocess", help="Clean & preprocess raw twcs.csv")
    p_e2_prep.add_argument("--input", "-i", type=str, default=default_raw, help="Raw CSV path")
    p_e2_prep.add_argument("--output", "-o", type=str, default=default_clean, help="Clean CSV path")
    p_e2_prep.add_argument("--target-rows", "-r", type=int, default=100000, help="Target row count")
    p_e2_prep.add_argument("--min-words", "-w", type=int, default=5, help="Minimum word threshold")
    p_e2_prep.add_argument("--keep-outbound", action="store_true", help="Include company responses")
    p_e2_prep.add_argument("--chunk-size", type=int, default=100000, help="Chunk size")
    p_e2_prep.set_defaults(func=cmd_preprocess)

    # exp2 train
    p_e2_train = exp2_subs.add_parser("train", help="Train baseline emotion classifiers")
    p_e2_train.add_argument("--input", "-i", type=str, default=default_clean, help="Clean CSV path")
    p_e2_train.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size (0 for all rows)")
    p_e2_train.set_defaults(func=cmd_train)

    # exp2 predict
    p_e2_pred = exp2_subs.add_parser("predict", help="Predict emotion & sentiment for tweets")
    p_e2_pred.add_argument("pos_text", nargs="?", type=str, default=None, help="Tweet text (positional)")
    p_e2_pred.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze")
    p_e2_pred.add_argument("--model", "-m", type=str, default=default_exp2_model, help="Model path")
    p_e2_pred.add_argument("--interactive", action="store_true", help="Interactive prediction shell")

    def _e2_pred_wrapper(args):
        args.text = args.text or getattr(args, "pos_text", None)
        cmd_predict(args)

    p_e2_pred.set_defaults(func=_e2_pred_wrapper)

    # exp2 demo
    p_e2_demo = exp2_subs.add_parser("demo", help="Run prediction demonstration")
    p_e2_demo.set_defaults(func=cmd_demo)

    # =========================================================================
    # 2. EXP3 SUBCOMMAND GROUP
    # =========================================================================
    p_exp3 = subparsers.add_parser("exp3", help="Experiment 3: Statistical EDA & Hypothesis Testing")
    exp3_subs = p_exp3.add_subparsers(dest="subcommand", help="Experiment 3 action")

    # exp3 eda
    p_e3_eda = exp3_subs.add_parser("eda", help="Run statistical EDA & hypothesis testing suite")
    p_e3_eda.add_argument("--input", "-i", type=str, default=default_clean, help="Clean CSV path")
    p_e3_eda.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size")
    p_e3_eda.set_defaults(func=cmd_eda)

    # exp3 report
    p_e3_rep = exp3_subs.add_parser("report", help="Compile academic PDF report")
    p_e3_rep.add_argument("--output", "-o", type=str, default=default_exp3_pdf, help="Output PDF path")
    p_e3_rep.set_defaults(func=cmd_report_exp3)

    # =========================================================================
    # 3. EXP4 SUBCOMMAND GROUP
    # =========================================================================
    p_exp4 = subparsers.add_parser("exp4", help="Experiment 4: ML Modeling, Tuning & MLflow Tracking")
    exp4_subs = p_exp4.add_subparsers(dest="subcommand", help="Experiment 4 action")

    # exp4 train
    p_e4_train = exp4_subs.add_parser("train", help="Train 5 models, tune hyperparameters, log with MLflow")
    p_e4_train.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size (0 for full 100,000 rows)")
    p_e4_train.set_defaults(func=cmd_exp4_train)

    # exp4 report
    p_e4_rep = exp4_subs.add_parser("report", help="Generate academic PDF & LaTeX reports")
    p_e4_rep.add_argument("--output", "-o", type=str, default=default_exp4_pdf, help="Output PDF path")
    p_e4_rep.add_argument("--tex", type=str, default=None, help="Output LaTeX .tex path")
    p_e4_rep.set_defaults(func=cmd_exp4_report)

    # exp4 docx
    p_e4_docx = exp4_subs.add_parser("docx", help="Generate academic Word (.docx) report")
    p_e4_docx.add_argument("--output", "-o", type=str, default=None, help="Output DOCX path")
    p_e4_docx.set_defaults(func=cmd_exp4_docx)

    # exp4 dashboard
    p_e4_dash = exp4_subs.add_parser("dashboard", help="Launch interactive MLflow tracking UI")
    p_e4_dash.add_argument("--port", "-p", type=int, default=5000, help="Port to bind dashboard server")
    p_e4_dash.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    p_e4_dash.set_defaults(func=cmd_exp4_dashboard)

    # =========================================================================
    # 4. EXP5 SUBCOMMAND GROUP
    # =========================================================================
    p_exp5 = subparsers.add_parser("exp5", help="Experiment 5: Explainable AI (SHAP & LIME) & Fairness (Fairlearn)")
    exp5_subs = p_exp5.add_subparsers(dest="subcommand", help="Experiment 5 action")

    # exp5 run
    p_e5_run = exp5_subs.add_parser("run", help="Run complete XAI and fairness auditing pipeline")
    p_e5_run.set_defaults(func=cmd_exp5_run)

    # exp5 report
    p_e5_rep = exp5_subs.add_parser("report", help="Generate academic PDF, LaTeX & Word reports")
    p_e5_rep.add_argument("--output", "-o", type=str, default=default_exp5_pdf, help="Output PDF path")
    p_e5_rep.add_argument("--tex", type=str, default=None, help="Output LaTeX path")
    p_e5_rep.add_argument("--docx", type=str, default=default_exp5_docx, help="Output DOCX path")
    p_e5_rep.set_defaults(func=cmd_exp5_report)

    # exp5 demo
    p_e5_demo = exp5_subs.add_parser("demo", help="Display XAI and fairness summary demo")
    p_e5_demo.set_defaults(func=cmd_exp5_demo)

    # =========================================================================
    # 5. EXP6 SUBCOMMAND GROUP
    # =========================================================================
    p_exp6 = subparsers.add_parser("exp6", help="Experiment 6: Containerization & API Deployment (FastAPI & Docker)")
    exp6_subs = p_exp6.add_subparsers(dest="subcommand", help="Experiment 6 action")

    # exp6 serve
    p_e6_serve = exp6_subs.add_parser("serve", help="Launch FastAPI ASGI web microservice with Uvicorn")
    p_e6_serve.add_argument("--port", "-p", type=int, default=8000, help="Listening port (default: 8000)")
    p_e6_serve.add_argument("--host", type=str, default="127.0.0.1", help="Binding host address (default: 127.0.0.1)")
    p_e6_serve.add_argument("--reload", action="store_true", help="Enable hot reload on code changes")
    p_e6_serve.set_defaults(func=cmd_exp6_serve)

    # exp6 test
    p_e6_test = exp6_subs.add_parser("test", help="Run automated test client, verify schemas, profile latency")
    p_e6_test.set_defaults(func=cmd_exp6_test)

    # exp6 docker-build
    p_e6_build = exp6_subs.add_parser("docker-build", help="Build production Docker image (ads-emotion-api:latest)")
    p_e6_build.set_defaults(func=cmd_exp6_docker_build)

    # exp6 docker-run
    p_e6_run = exp6_subs.add_parser("docker-run", help="Run Docker container mapping port 8000")
    p_e6_run.add_argument("--port", "-p", type=int, default=8000, help="Host port to map (default: 8000)")
    p_e6_run.set_defaults(func=cmd_exp6_docker_run)

    # exp6 report
    p_e6_rep = exp6_subs.add_parser("report", help="Generate academic PDF, Word (.docx), and LaTeX reports")
    p_e6_rep.add_argument("--output", "-o", type=str, default=default_exp6_pdf, help="Output PDF path")
    p_e6_rep.add_argument("--tex", type=str, default=None, help="Output LaTeX path")
    p_e6_rep.add_argument("--docx", type=str, default=default_exp6_docx, help="Output DOCX path")
    p_e6_rep.set_defaults(func=cmd_exp6_report)

    # =========================================================================
    # 6. LEGACY TOP-LEVEL SHORTCUTS (PRESERVED)
    # =========================================================================
    p_preprocess = subparsers.add_parser("preprocess", help="[Exp 2] Clean & preprocess raw twcs.csv")
    p_preprocess.add_argument("--input", "-i", type=str, default=default_raw, help="Raw CSV path")
    p_preprocess.add_argument("--output", "-o", type=str, default=default_clean, help="Clean CSV path")
    p_preprocess.add_argument("--target-rows", "-r", type=int, default=100000, help="Target row count")
    p_preprocess.add_argument("--min-words", "-w", type=int, default=5, help="Minimum word threshold")
    p_preprocess.add_argument("--keep-outbound", action="store_true", help="Include company responses")
    p_preprocess.add_argument("--chunk-size", type=int, default=100000, help="Chunk size")
    p_preprocess.set_defaults(func=cmd_preprocess)

    p_train = subparsers.add_parser("train", help="[Exp 2] Train emotion classifiers")
    p_train.add_argument("--input", "-i", type=str, default=default_clean, help="Clean CSV path")
    p_train.add_argument("--sample-size", "-s", type=int, default=0, help="Sample size (0 for all rows)")
    p_train.set_defaults(func=cmd_train)

    p_predict = subparsers.add_parser("predict", help="[Exp 2] Predict emotion & sentiment for tweets")
    p_predict.add_argument("pos_text", nargs="?", type=str, default=None, help="Tweet text (positional)")
    p_predict.add_argument("--text", "-t", type=str, default=None, help="Tweet text to analyze")
    p_predict.add_argument("--model", "-m", type=str, default=default_exp2_model, help="Model path")
    p_predict.add_argument("--interactive", action="store_true", help="Interactive prediction shell")
    p_predict.set_defaults(func=_e2_pred_wrapper)

    p_demo = subparsers.add_parser("demo", help="[Exp 2] Run prediction demonstration on sample customer tweets")
    p_demo.set_defaults(func=cmd_demo)

    p_eda = subparsers.add_parser("eda", help="[Exp 3] Run statistical EDA & hypothesis testing suite")
    p_eda.add_argument("--input", "-i", type=str, default=default_clean, help="Clean CSV path")
    p_eda.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size")
    p_eda.set_defaults(func=cmd_eda)

    p_report = subparsers.add_parser("report", help="[Exp 3] Compile academic PDF report")
    p_report.add_argument("--output", "-o", type=str, default=default_exp3_pdf, help="Output PDF path")
    p_report.set_defaults(func=cmd_report_exp3)

    p_pipeline = subparsers.add_parser("pipeline", help="Run full end-to-end multi-experiment pipeline")
    p_pipeline.add_argument("--sample-size", "-s", type=int, default=50000, help="Sample size for pipeline stages")
    p_pipeline.set_defaults(func=cmd_pipeline)

    args = parser.parse_args()

    if args.interactive:
        args.model = default_exp2_model
        cmd_predict(args)
    elif args.demo:
        cmd_demo(args)
    elif args.text:
        args.model = default_exp2_model
        cmd_predict(args)
    elif args.command is None:
        parser.print_help()
    elif hasattr(args, 'func'):
        args.func(args)
    elif args.command in ('exp2', 'exp3', 'exp4', 'exp5', 'exp6') and getattr(args, 'subcommand', None) is None:
        if args.command == 'exp2':
            p_exp2.print_help()
        elif args.command == 'exp3':
            p_exp3.print_help()
        elif args.command == 'exp4':
            p_exp4.print_help()
        elif args.command == 'exp5':
            p_exp5.print_help()
        elif args.command == 'exp6':
            p_exp6.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
