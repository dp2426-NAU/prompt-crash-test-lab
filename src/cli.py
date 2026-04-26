"""Command-line interface for Prompt Crash-Test Lab."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_generate(args):
    """Generate prompt variants."""
    from config.settings import BASE_PROMPTS_DIR, VARIANTS_DIR
    from src.variant_generator import generate_all_variants
    generate_all_variants(BASE_PROMPTS_DIR, VARIANTS_DIR)


def cmd_run(args):
    """Run batch evaluation."""
    from config.settings import VARIANTS_DIR
    from src.batch_runner import run_batch

    variants_file = VARIANTS_DIR / f"{args.task}_variants.jsonl"
    if not variants_file.exists():
        print(f"Variants file not found: {variants_file}")
        print("Run 'python -m src.cli generate' first.")
        return

    run_batch(
        variants_file,
        model_names=args.models,
        max_variants=args.max_variants,
        task_type=args.task,
    )


def cmd_score(args):
    """Score results and generate analysis."""
    from src.analysis import run_full_analysis
    run_full_analysis(args.task)


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    import shutil
    import subprocess

    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    if not dashboard_path.exists():
        print(f"Dashboard file not found: {dashboard_path}")
        return
    if shutil.which("streamlit") is None:
        print("Streamlit is not installed. Run: pip install streamlit")
        return
    subprocess.run(["streamlit", "run", str(dashboard_path)])


def cmd_adversarial(args):
    """Generate adversarial mutation variants."""
    from config.settings import BASE_PROMPTS_DIR, VARIANTS_DIR
    from src.adversarial import generate_all_adversarial
    generate_all_adversarial(BASE_PROMPTS_DIR, VARIANTS_DIR)


def cmd_sensitivity(args):
    """Run parameter sensitivity study."""
    from config.settings import VARIANTS_DIR
    from src.parameter_sensitivity import run_temperature_study, run_system_prompt_study

    variants_file = VARIANTS_DIR / f"{args.task}_variants.jsonl"
    if not variants_file.exists():
        print(f"Variants file not found: {variants_file}")
        print("Run 'python -m src.cli generate' first.")
        return

    if args.study == "temperature":
        run_temperature_study(variants_file, model_name=args.model, max_variants=args.max_variants)
    elif args.study == "system_prompt":
        run_system_prompt_study(variants_file, model_name=args.model, max_variants=args.max_variants)


def cmd_status(args):
    """Show project status."""
    from config.settings import VARIANTS_DIR, RESULTS_DIR, CACHE_DB_PATH
    from src.cache import ResponseCache

    print("Prompt Crash-Test Lab - Status")
    print("=" * 40)

    # Check variants
    for task in ["json_extraction", "grounded_qa"]:
        vf = VARIANTS_DIR / f"{task}_variants.jsonl"
        if vf.exists():
            with open(vf) as f:
                count = sum(1 for _ in f)
            print(f"Variants ({task}): {count}")
        else:
            print(f"Variants ({task}): NOT GENERATED")

    # Check adversarial variants
    for task in ["json_extraction", "grounded_qa"]:
        af = VARIANTS_DIR / f"{task}_adversarial.jsonl"
        if af.exists():
            with open(af) as f:
                count = sum(1 for _ in f)
            print(f"Adversarial ({task}): {count}")

    # Check results
    for task in ["json_extraction", "grounded_qa"]:
        rf = RESULTS_DIR / f"{task}_results.jsonl"
        if rf.exists():
            with open(rf) as f:
                count = sum(1 for _ in f)
            print(f"Results ({task}): {count}")
        else:
            print(f"Results ({task}): NOT RUN")

    # Cache stats
    if CACHE_DB_PATH.exists():
        cache = ResponseCache(CACHE_DB_PATH)
        print(f"Cache: {cache.stats()}")


def main():
    parser = argparse.ArgumentParser(
        prog="crashtest",
        description="Prompt Crash-Test Lab: LLM Robustness Benchmark",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    subparsers.add_parser("generate", help="Generate prompt variants from base prompts")

    # run
    run_parser = subparsers.add_parser("run", help="Run batch evaluation across models")
    run_parser.add_argument("--task", choices=["json_extraction", "grounded_qa"], required=True)
    run_parser.add_argument("--models", nargs="+", default=None)
    run_parser.add_argument("--max-variants", type=int, default=None)

    # score
    score_parser = subparsers.add_parser("score", help="Score results and generate analysis")
    score_parser.add_argument("--task", choices=["json_extraction", "grounded_qa"], required=True)

    # adversarial
    subparsers.add_parser("adversarial", help="Generate adversarial mutation variants")

    # sensitivity
    sens_parser = subparsers.add_parser("sensitivity", help="Run parameter sensitivity study")
    sens_parser.add_argument("--task", choices=["json_extraction", "grounded_qa"], required=True)
    sens_parser.add_argument("--study", choices=["temperature", "system_prompt"], required=True)
    sens_parser.add_argument("--model", default="gpt-4-turbo", help="Model to test")
    sens_parser.add_argument("--max-variants", type=int, default=20, help="Variants to test")

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # status
    subparsers.add_parser("status", help="Show project status")

    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "run": cmd_run,
        "score": cmd_score,
        "adversarial": cmd_adversarial,
        "sensitivity": cmd_sensitivity,
        "dashboard": cmd_dashboard,
        "status": cmd_status,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
