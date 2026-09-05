"""Launch nanochat training with checkpoints stored on Google Drive in Colab."""
import os
import runpy
import shutil


def main():
    drive_dir = os.environ.get(
        "NANOCHAT_BASE_DIR",
        "/content/drive/MyDrive/nanochat-runs",
    )
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted. Run `from google.colab import drive; "
            "drive.mount('/content/drive')` in a Colab notebook cell first."
        )
    os.makedirs(drive_dir, exist_ok=True)
    local_dir = "/content/nanochat-cache"
    os.makedirs(local_dir, exist_ok=True)
    for name in ("tokenizer", "base_data_climbmix", "base_data"):
        source = os.path.join(drive_dir, name)
        target = os.path.join(local_dir, name)
        if os.path.exists(source) and not os.path.exists(target):
            print(f"Copying {name} from Drive to local disk...")
            shutil.copytree(source, target)
    # Data/tokenizer are read locally; checkpoints remain on Drive.
    os.environ["NANOCHAT_BASE_DIR"] = local_dir
    if "--checkpoint-base-dir" not in os.sys.argv:
        os.sys.argv.extend(["--checkpoint-base-dir", drive_dir])
    print(f"local data directory: {local_dir}")
    print(f"checkpoint directory: {drive_dir}")
    runpy.run_module("scripts.base_train", run_name="__main__")


if __name__ == "__main__":
    main()
