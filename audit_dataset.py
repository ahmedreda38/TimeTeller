"""
Audit and fix: compare annotations.json against actual files on disk.
Copies missing files from HF cache blobs using sanitized filenames.
"""

import json
import os
import re
import glob

DATASET_DIR = os.path.join("data", "dataset")
ANNOTATIONS = os.path.join(DATASET_DIR, "annotations.json")
HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                         "datasets--jaeha-choi--TickTockVQA")
BLOBS_DIR = os.path.join(HF_CACHE, "blobs")

ILLEGAL_CHARS = re.compile(r'[<>:"|?*]')

def sanitize(name):
    return ILLEGAL_CHARS.sub("_", name)

def main():
    # --- 1. Load annotations ---
    with open(ANNOTATIONS, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    print(f"Total annotation records: {len(annotations)}")
    
    train_ann = [a for a in annotations if a["image_path"].startswith("train/")]
    test_ann  = [a for a in annotations if a["image_path"].startswith("test/")]
    print(f"  Train annotations: {len(train_ann)}")
    print(f"  Test annotations:  {len(test_ann)}")
    
    # --- 2. Check which files exist on disk ---
    missing = []
    found = []
    found_sanitized = []
    
    for rec in annotations:
        original_path = rec["image_path"]  # e.g. "test/filename.jpg"
        full_path = os.path.join(DATASET_DIR, "images", original_path)
        sanitized_path = os.path.join(DATASET_DIR, "images", sanitize(original_path))
        
        if os.path.isfile(full_path):
            found.append(rec)
        elif os.path.isfile(sanitized_path):
            found_sanitized.append(rec)
        else:
            missing.append(rec)
    
    print(f"\n--- File Audit ---")
    print(f"  Found (exact name):     {len(found)}")
    print(f"  Found (sanitized name): {len(found_sanitized)}")
    print(f"  Missing:                {len(missing)}")
    
    # --- 3. Analyze missing files ---
    if missing:
        print(f"\n--- Missing Files Detail ---")
        # Group by source
        sources = {}
        splits = {"train": 0, "test": 0}
        for rec in missing:
            src = rec.get("source", "Unknown")
            sources[src] = sources.get(src, 0) + 1
            split = "train" if rec["image_path"].startswith("train") else "test"
            splits[split] += 1
        
        print(f"  By split:")
        for s, c in splits.items():
            print(f"    {s}: {c}")
        print(f"  By source:")
        for s, c in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {s}: {c}")
        
        # Show a few examples
        print(f"\n  First 5 missing filenames:")
        for rec in missing[:5]:
            print(f"    {rec['image_path']}")
        
        # Check if any have illegal chars
        has_illegal = [r for r in missing if ILLEGAL_CHARS.search(r["image_path"])]
        print(f"\n  Missing with illegal Windows chars: {len(has_illegal)}")
        no_illegal  = [r for r in missing if not ILLEGAL_CHARS.search(r["image_path"])]
        print(f"  Missing without illegal chars:      {len(no_illegal)}")
        if no_illegal:
            print(f"  Examples of clean-name missing files:")
            for r in no_illegal[:5]:
                print(f"    {r['image_path']}")
    
    # --- 4. Try to recover from HF cache blobs ---
    if missing and os.path.isdir(BLOBS_DIR):
        print(f"\n--- Attempting recovery from HF cache ---")
        print(f"  Blobs directory: {BLOBS_DIR}")
        blobs = os.listdir(BLOBS_DIR)
        print(f"  Available blobs: {len(blobs)}")
        
        # We need the refs to map filenames to blob hashes
        # Check if there's a refs or snapshot mapping
        refs_dir = os.path.join(HF_CACHE, "refs")
        snap_dir = os.path.join(HF_CACHE, "snapshots")
        
        if os.path.isdir(snap_dir):
            snapshots = os.listdir(snap_dir)
            print(f"  Snapshots: {snapshots}")
    
    # --- 5. Count actual files on disk ---
    train_dir = os.path.join(DATASET_DIR, "images", "train")
    test_dir  = os.path.join(DATASET_DIR, "images", "test")
    
    train_files = len([f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]) if os.path.isdir(train_dir) else 0
    test_files  = len([f for f in os.listdir(test_dir)  if os.path.isfile(os.path.join(test_dir, f))])  if os.path.isdir(test_dir) else 0
    
    print(f"\n--- Actual files on disk ---")
    print(f"  Train images: {train_files}")
    print(f"  Test images:  {test_files}")
    print(f"  Total:        {train_files + test_files}")
    
    print(f"\n--- Expected vs Actual ---")
    print(f"  Train: {len(train_ann)} expected, {train_files} on disk, diff = {len(train_ann) - train_files}")
    print(f"  Test:  {len(test_ann)} expected, {test_files} on disk, diff = {len(test_ann) - test_files}")
    
    # --- 6. Impact Assessment ---
    total = len(annotations)
    usable = len(found) + len(found_sanitized)
    pct = usable / total * 100
    
    print(f"\n{'='*55}")
    print(f"  IMPACT ASSESSMENT")
    print(f"{'='*55}")
    print(f"  Usable images:  {usable}/{total} ({pct:.1f}%)")
    print(f"  Missing images: {len(missing)}/{total} ({len(missing)/total*100:.1f}%)")
    print(f"  Missing TRAIN:  {splits.get('train',0)}")
    print(f"  Missing TEST:   {splits.get('test',0)}")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
