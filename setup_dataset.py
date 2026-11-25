"""
setup.py / Dataset downloader

Downloads the ForeSpeed dataset from GitHub and verifies files.
Also downloads YOLO11n model weights if not present.
"""

import os
import sys
import json
import hashlib
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def check_python_version():
    if sys.version_info < (3, 9):
        print(f"❌ Python 3.9+ required. Current: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")


def install_dependencies():
    print("\n📦 Installing dependencies...")
    req_path = PROJECT_ROOT / "requirements.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("❌ Dependency installation failed.")
        sys.exit(1)
    print("✅ Dependencies installed")


def download_yolo_model():
    print("\n🤖 Checking YOLO11n model...")
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "yolo11n.pt"
    
    from modules.speed_detection.vehicle_detector import VehicleDetector

    VehicleDetector(model_path=str(model_path))
    print(f"YOLO11n loaded and warmed up: {model_path}")


def setup_forespeed_dataset():
    """Acquire and verify the supported ForeSpeed recordings."""
    import cv2

    dataset_dir = PROJECT_ROOT / "dataset" / "forespeed"
    profiles = json.loads((PROJECT_ROOT / "config" / "dataset_config.json").read_text(
        encoding="utf-8"))["forespeed"]["videos"]
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        dataset_dir.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ampedsoftware/forespeed.git",
             str(dataset_dir)], check=True, env=env,
        )

    def verified(profile):
        path = dataset_dir / profile["path"]
        if not path.is_file():
            return False
        with path.open("rb") as video:
            digest = hashlib.sha256()
            for block in iter(lambda: video.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == profile["sha256"]

    missing = [profile["path"] for profile in profiles.values() if not verified(profile)]
    if missing:
        if not (dataset_dir / ".git").exists():
            raise RuntimeError(f"ForeSpeed checkout is missing Git metadata or recordings: {missing}")
        subprocess.run(["git", "lfs", "version"], check=True)
        subprocess.run(["git", "-C", str(dataset_dir), "lfs", "pull",
                        "--include", ",".join(missing), "--exclude", ""], check=True)

    for profile in profiles.values():
        path = dataset_dir / profile["path"]
        if not verified(profile):
            raise RuntimeError(f"Missing or corrupted ForeSpeed video: {path}")
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
        try:
            ok, frame = cap.read()
            if not ok or list(frame.shape[1::-1]) != profile["resolution"]:
                raise RuntimeError(f"ForeSpeed video failed decode or resolution check: {path}")
        finally:
            cap.release()
        print(f"Verified {profile['camera']}: {path}")


def create_output_dirs():
    for d in ["outputs/videos", "outputs/evidence", "outputs/reports"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
    print("✅ Output directories ready")


def verify_structure():
    """Verify the project structure is correct."""
    required = [
        "app.py",
        "requirements.txt",
        "config/settings.json",
        "config/dataset_config.json",
        "modules/speed_detection/__init__.py",
        "modules/speed_detection/video_processor.py",
        "modules/speed_detection/vehicle_detector.py",
        "modules/speed_detection/vehicle_tracker.py",
        "modules/speed_detection/camera_calibration.py",
        "modules/speed_detection/speed_estimator.py",
        "modules/speed_detection/violation_detector.py",
        "modules/speed_detection/evidence_generator.py",
        "modules/speed_detection/report_generator.py",
    ]
    
    missing = []
    for f in required:
        if not (PROJECT_ROOT / f).exists():
            missing.append(f)
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ Project structure verified")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Traffic Violation Detection Setup")
    parser.add_argument("--install-deps", action="store_true", help="Install Python dependencies")
    parser.add_argument("--download-dataset", action="store_true", help="Download ForeSpeed dataset")
    parser.add_argument("--download-model", action="store_true", help="Download YOLO11n model")
    parser.add_argument("--verify", action="store_true", help="Verify project structure")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  AI Traffic Violation Detection - Setup")
    print("=" * 60)
    
    check_python_version()
    create_output_dirs()
    
    if args.all or args.install_deps:
        install_dependencies()
    
    if args.all or args.download_model:
        download_yolo_model()
    
    if args.all or args.download_dataset:
        setup_forespeed_dataset()
    
    if (args.all or args.verify) and not verify_structure():
        sys.exit(1)
    
    if not any(vars(args).values()):
        # Default: run all
        install_dependencies()
        download_yolo_model()
        setup_forespeed_dataset()
        if not verify_structure():
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Setup complete! Launch the application with:")
    print("  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
