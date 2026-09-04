"""Launch nanochat training with checkpoints stored on Google Drive in Colab."""
import os
import runpy


def main():
    drive_dir = "/content/drive/MyDrive/nanochat-runs"
    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive is not mounted. Run `from google.colab import drive; "
            "drive.mount('/content/drive')` in a Colab notebook cell first."
        )
    os.makedirs(drive_dir, exist_ok=True)
    # Must be set before scripts.base_train imports the dataset/checkpoint modules.
    os.environ["NANOCHAT_BASE_DIR"] = drive_dir
    print(f"nanochat checkpoint directory: {drive_dir}")
    runpy.run_module("scripts.base_train", run_name="__main__")


if __name__ == "__main__":
    main()
