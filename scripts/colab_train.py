"""Launch nanochat training with checkpoints stored on Google Drive in Colab."""
import os
import runpy


def main():
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This launcher must be run inside Google Colab.") from exc

    drive.mount("/content/drive")
    drive_dir = "/content/drive/MyDrive/nanochat-runs"
    os.makedirs(drive_dir, exist_ok=True)
    # Must be set before scripts.base_train imports the dataset/checkpoint modules.
    os.environ["NANOCHAT_BASE_DIR"] = drive_dir
    print(f"nanochat checkpoint directory: {drive_dir}")
    runpy.run_module("scripts.base_train", run_name="__main__")


if __name__ == "__main__":
    main()
