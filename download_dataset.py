"""
Download TickTockVQA dataset with Windows-safe filename handling.
Skips files with characters illegal on Windows (" < > | : * ?)
These are typically only 1-2 files from the ClockMovies source.
"""

import os
import re
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "jaeha-choi/TickTockVQA"
LOCAL_DIR = os.path.join("data", "dataset")
REPO_TYPE = "dataset"

# Characters illegal in Windows filenames
ILLEGAL_CHARS = re.compile(r'[<>:"|?*]')

def is_windows_safe(path: str) -> bool:
    """Check if a file path is safe for Windows."""
    return not ILLEGAL_CHARS.search(path)

def sanitize_filename(path: str) -> str:
    """Replace illegal Windows characters with underscores."""
    return ILLEGAL_CHARS.sub("_", path)

def main():
    api = HfApi()
    
    print(f"Listing files in {REPO_ID}...")
    all_files = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, recursive=True))
    
    # Filter to actual files (not directories)
    files = [f for f in all_files if hasattr(f, 'rfilename')]
    print(f"Found {len(files)} files in repo.\n")
    
    downloaded = 0
    skipped_exists = 0
    skipped_unsafe = 0
    sanitized = 0
    errors = 0
    
    for i, f in enumerate(files):
        rfilename = f.rfilename
        
        # Check if filename is Windows-safe
        if not is_windows_safe(rfilename):
            safe_name = sanitize_filename(rfilename)
            safe_path = os.path.join(LOCAL_DIR, safe_name)
            
            if os.path.exists(safe_path):
                skipped_exists += 1
                continue
            
            print(f"  [{i+1}/{len(files)}] SANITIZING: {rfilename}")
            print(f"         -> {safe_name}")
            
            try:
                # Download to HF cache first, then copy with safe name
                cached_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=rfilename,
                    repo_type=REPO_TYPE,
                    local_dir=None,  # use default cache
                )
                # Copy to local dir with sanitized name
                os.makedirs(os.path.dirname(safe_path), exist_ok=True)
                import shutil
                shutil.copy2(cached_path, safe_path)
                sanitized += 1
            except Exception as e:
                print(f"         SKIPPED (error): {e}")
                skipped_unsafe += 1
            continue
        
        # Normal file - check if already downloaded
        local_path = os.path.join(LOCAL_DIR, rfilename)
        if os.path.exists(local_path):
            skipped_exists += 1
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(files)}] Progress check... ({skipped_exists} already exist)")
            continue
        
        # Download
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=rfilename,
                repo_type=REPO_TYPE,
                local_dir=LOCAL_DIR,
            )
            downloaded += 1
            if downloaded % 100 == 0:
                print(f"  [{i+1}/{len(files)}] Downloaded {downloaded} new files...")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] ERROR: {rfilename}: {e}")
            errors += 1
    
    print(f"\n{'='*50}")
    print(f"Download complete!")
    print(f"  New downloads:    {downloaded}")
    print(f"  Already existed:  {skipped_exists}")
    print(f"  Sanitized names:  {sanitized}")
    print(f"  Skipped (unsafe): {skipped_unsafe}")
    print(f"  Errors:           {errors}")
    print(f"{'='*50}")
    
    # Verify
    train_dir = os.path.join(LOCAL_DIR, "images", "train")
    test_dir = os.path.join(LOCAL_DIR, "images", "test")
    
    train_count = len([f for f in os.listdir(train_dir) if f.endswith(('.jpg', '.png'))]) if os.path.isdir(train_dir) else 0
    test_count = len([f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]) if os.path.isdir(test_dir) else 0
    
    print(f"\nImage counts:")
    print(f"  Train: {train_count}")
    print(f"  Test:  {test_count}")
    print(f"  Total: {train_count + test_count}")

if __name__ == "__main__":
    main()
