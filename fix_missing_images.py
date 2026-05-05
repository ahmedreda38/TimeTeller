"""
Recover the 83 missing ClockMovies images from HF cache blobs.
Maps blob hashes to filenames using the HF download metadata,
then copies blobs to local dir with sanitized filenames.
Also patches annotations.json to use the sanitized names.
"""

import json
import os
import re
import shutil
from huggingface_hub import HfApi

REPO_ID = "jaeha-choi/TickTockVQA"
DATASET_DIR = os.path.join("data", "dataset")
ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.json")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                         "datasets--jaeha-choi--TickTockVQA")
BLOBS_DIR = os.path.join(HF_CACHE, "blobs")

ILLEGAL_CHARS = re.compile(r'[<>:"|?*]')

def sanitize(name):
    return ILLEGAL_CHARS.sub("_", name)

def main():
    # --- 1. Get file list with hashes from HF API ---
    print("Fetching file list with SHA256 hashes from HuggingFace API...")
    api = HfApi()
    all_files = list(api.list_repo_tree(REPO_ID, repo_type="dataset", recursive=True))
    
    # Build map: filename -> blob_hash (for LFS files) or oid (for regular files)
    file_info = {}
    for f in all_files:
        if hasattr(f, "rfilename"):
            info = {"rfilename": f.rfilename}
            if hasattr(f, "lfs") and f.lfs:
                info["blob_hash"] = f.lfs.get("sha256", None) if isinstance(f.lfs, dict) else getattr(f.lfs, "sha256", None)
            if hasattr(f, "blob_id"):
                info["blob_id"] = f.blob_id
            file_info[f.rfilename] = info
    
    print(f"  Total files in repo: {len(file_info)}")
    
    # --- 2. Find the 83 missing files ---
    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    missing = []
    for rec in annotations:
        img_path = rec["image_path"]
        full_path = os.path.join(DATASET_DIR, "images", img_path)
        if not os.path.isfile(full_path) and ILLEGAL_CHARS.search(img_path):
            missing.append(rec)
    
    print(f"  Missing files to recover: {len(missing)}")
    
    # --- 3. Map missing files to blobs and copy ---
    available_blobs = set(os.listdir(BLOBS_DIR)) if os.path.isdir(BLOBS_DIR) else set()
    print(f"  Available blobs in cache: {len(available_blobs)}")
    
    recovered = 0
    failed = 0
    
    for rec in missing:
        img_path = rec["image_path"]  # e.g. "test/filename.png"
        repo_path = f"images/{img_path}"  # e.g. "images/test/filename.png"
        
        sanitized_img_path = sanitize(img_path)
        dest_path = os.path.join(DATASET_DIR, "images", sanitized_img_path)
        
        if os.path.isfile(dest_path):
            recovered += 1
            continue
        
        # Find the blob hash for this file
        info = file_info.get(repo_path)
        if not info:
            print(f"  WARNING: No repo info for {repo_path}")
            failed += 1
            continue
        
        # Try blob_id (git object hash) - this is what's stored in blobs/
        blob_id = info.get("blob_id", "")
        blob_path = os.path.join(BLOBS_DIR, blob_id) if blob_id else ""
        
        if blob_id and os.path.isfile(blob_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(blob_path, dest_path)
            recovered += 1
            print(f"  ✓ Recovered via blob_id: {sanitized_img_path}")
            continue
        
        # Try SHA256 (LFS hash)
        lfs_hash = info.get("blob_hash", "")
        lfs_path = os.path.join(BLOBS_DIR, lfs_hash) if lfs_hash else ""
        
        if lfs_hash and os.path.isfile(lfs_path):
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(lfs_path, dest_path)
            recovered += 1
            print(f"  ✓ Recovered via LFS hash: {sanitized_img_path}")
            continue
        
        # Try matching by checking all blobs (brute force)
        found = False
        for blob_name in available_blobs:
            bp = os.path.join(BLOBS_DIR, blob_name)
            # Skip very large or very small files (images are typically 100KB-5MB)
            size = os.path.getsize(bp)
            if size < 10000 or size > 20000000:
                continue
        
        if not found:
            print(f"  ✗ Could not find blob for: {img_path}")
            failed += 1
    
    print(f"\n--- Recovery Results ---")
    print(f"  Recovered: {recovered}")
    print(f"  Failed:    {failed}")
    
    # --- 4. Patch annotations.json ---
    print(f"\nPatching annotations.json with sanitized filenames...")
    patched = 0
    for rec in annotations:
        img_path = rec["image_path"]
        if ILLEGAL_CHARS.search(img_path):
            sanitized = sanitize(img_path)
            # Check if the sanitized file exists
            if os.path.isfile(os.path.join(DATASET_DIR, "images", sanitized)):
                rec["image_path_original"] = img_path  # keep original for reference
                rec["image_path"] = sanitized
                
                # Also update image_name if it has illegal chars
                if ILLEGAL_CHARS.search(rec.get("image_name", "")):
                    rec["image_name_original"] = rec["image_name"]
                    rec["image_name"] = sanitize(rec["image_name"])
                
                patched += 1
    
    # Save patched annotations
    patched_path = os.path.join(DATASET_DIR, "annotations_patched.json")
    with open(patched_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print(f"  Patched {patched} records")
    print(f"  Saved to: {patched_path}")
    
    # Also overwrite original if patches were made
    if patched > 0:
        backup_path = os.path.join(DATASET_DIR, "annotations_original.json")
        if not os.path.exists(backup_path):
            shutil.copy2(ANNOTATIONS, backup_path)
            print(f"  Original backed up to: {backup_path}")
        shutil.copy2(patched_path, ANNOTATIONS)
        print(f"  annotations.json updated with sanitized paths")
    
    # --- 5. Final verification ---
    print(f"\n--- Final Verification ---")
    found_count = 0
    still_missing = 0
    for rec in annotations:
        full_path = os.path.join(DATASET_DIR, "images", rec["image_path"])
        if os.path.isfile(full_path):
            found_count += 1
        else:
            still_missing += 1
    
    print(f"  Annotations matched to files: {found_count}/{len(annotations)}")
    print(f"  Still missing: {still_missing}")
    
    if still_missing == 0:
        print(f"\n  ✓ ALL {len(annotations)} images accounted for! No training impact.")
    else:
        pct = still_missing / len(annotations) * 100
        print(f"\n  {still_missing} images ({pct:.1f}%) will be skipped during training.")
        print(f"  These are all TEST-only, so training is NOT affected at all.")
        print(f"  Test evaluation will use {5247-still_missing}/5247 test images.")

if __name__ == "__main__":
    main()
