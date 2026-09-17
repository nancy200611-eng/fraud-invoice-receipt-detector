import sys
import argparse
import subprocess
import os

BASE_DIR = r"E:\fraud Invoice&Receipt Detector"

def run_cmd(cmd):
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n>>> Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, env=env, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"Command exited with code {res.returncode}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Receipt & Invoice Fraud Detection Pipeline CLI")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5], help="Run a specific phase (1-5)")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline sequentially")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web dashboard")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    args = parser.parse_args()

    if args.serve:
        print("Starting FastAPI Web Dashboard on http://127.0.0.1:8000 ...")
        run_cmd("python -m uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload")
        return

    if args.test:
        run_cmd("pytest tests/test_system.py -v")
        return

    if args.phase == 1 or args.all:
        print("\n================== PHASE 1: OCR & DATASET LIFT-OFF ==================")
        run_cmd("python src/ocr/dataset_collector.py")
        run_cmd("python src/ocr/ocr_pipeline.py")

    if args.phase == 2 or args.all:
        print("\n================== PHASE 2: MANIPULATION ENGINE ==================")
        run_cmd("python src/manipulation/manipulator.py")

    if args.phase == 3 or args.all:
        print("\n================== PHASE 3: DETECTION & FORENSICS ==================")
        run_cmd("python src/forensics/image_forensics.py")
        run_cmd("python -m src.models.train")
        run_cmd("python -m src.models.gradcam")
        run_cmd("python -m src.models.localizer")

    if args.phase == 4 or args.all:
        print("\n================== PHASE 4: AUDIT RULES & INTEGRATION ==================")
        run_cmd("python -m src.rules.heuristic_engine")

    if args.phase == 5 or args.all:
        print("\n================== PHASE 5: RESILIENCE & SHIELD ==================")
        run_cmd("python -m src.forensics.robustness")

    if not any([args.phase, args.all, args.serve, args.test]):
        parser.print_help()

if __name__ == "__main__":
    main()
