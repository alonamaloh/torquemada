#!/usr/bin/env python3
"""
Select CWDL files for web deployment using the conjugate trick.

For each conjugate pair of materials (e.g. 2K vs 1K and 1K vs 2K),
keeps only the smaller file. The C++ engine reconstructs the missing
conjugate at probe time via depth-1 search.

Only selects files with ≤7 total pieces.

Usage:
    python3 scripts/select_cwdl_files.py [--source DIR] [--dest DIR] [--dry-run]

Default source: ../damas/
Default dest: ../torquemada-gh-pages/tablebases/
"""

import argparse
import os
import re
import shutil
import json


def parse_material_key(filename):
    """Extract material key (6 digits) from cwdl_XXXXXX.bin filename."""
    m = re.match(r'cwdl_(\d{6})\.bin', os.path.basename(filename))
    if not m:
        return None
    return m.group(1)


def material_piece_count(key):
    """Count total pieces from material key (sum of all 6 digits)."""
    return sum(int(d) for d in key)


def conjugate_key(key):
    """Compute the conjugate material key (swap white and black pieces).

    Material key format: bwp bbp owp obp wq bq
    Conjugate swaps: bwp<->bbp, owp<->obp, wq<->bq
    """
    return key[1] + key[0] + key[3] + key[2] + key[5] + key[4]


def canonical_pair(key):
    """Return the canonical (smaller) key of a conjugate pair."""
    conj = conjugate_key(key)
    return min(key, conj)


def main():
    parser = argparse.ArgumentParser(description='Select CWDL files for web deployment')
    parser.add_argument('--source', default='../damas/',
                        help='Source directory containing cwdl_*.bin files')
    parser.add_argument('--dest', default='../torquemada-gh-pages/tablebases/',
                        help='Destination directory for selected files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only print what would be done, do not copy')
    parser.add_argument('--max-pieces', type=int, default=7,
                        help='Maximum total piece count (default: 7)')
    parser.add_argument('--js-output', default=None,
                        help='Write JS file list to this path')
    args = parser.parse_args()

    source_dir = os.path.abspath(args.source)
    dest_dir = os.path.abspath(args.dest)

    # Find all CWDL files in source
    all_files = {}
    for fname in os.listdir(source_dir):
        key = parse_material_key(fname)
        if key is None:
            continue
        if material_piece_count(key) > args.max_pieces:
            continue
        filepath = os.path.join(source_dir, fname)
        filesize = os.path.getsize(filepath)
        all_files[key] = (filepath, filesize)

    print(f"Found {len(all_files)} CWDL files with ≤{args.max_pieces} pieces")

    # Group by conjugate pairs and select the smaller file
    selected = {}  # key -> (filepath, filesize)
    seen_pairs = set()

    for key in sorted(all_files.keys()):
        canon = canonical_pair(key)
        if canon in seen_pairs:
            continue
        seen_pairs.add(canon)

        conj = conjugate_key(key)
        filepath_a, size_a = all_files[key]

        if conj in all_files and conj != key:
            filepath_b, size_b = all_files[conj]
            # Keep the smaller file
            if size_a <= size_b:
                selected[key] = (filepath_a, size_a)
            else:
                selected[conj] = (filepath_b, size_b)
        else:
            # No conjugate exists (self-conjugate or missing), keep this one
            selected[key] = (filepath_a, size_a)

    total_size = sum(s for _, s in selected.values())
    print(f"Selected {len(selected)} files ({total_size / 1024 / 1024:.1f} MB)")
    print(f"  (from {len(all_files)} total, saved {len(all_files) - len(selected)} via conjugate trick)")

    if args.dry_run:
        print("\nDry run - not copying files")
        for key in sorted(selected.keys()):
            filepath, size = selected[key]
            print(f"  cwdl_{key}.bin ({size / 1024:.1f} KB)")
        return

    # Create destination directory
    os.makedirs(dest_dir, exist_ok=True)

    # Copy selected files
    copied = 0
    for key in sorted(selected.keys()):
        filepath, size = selected[key]
        dest_path = os.path.join(dest_dir, f'cwdl_{key}.bin')
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) != size:
            shutil.copy2(filepath, dest_path)
            copied += 1

    print(f"Copied {copied} new/updated files to {dest_dir}")

    # Generate JS file list
    filenames = sorted(f'cwdl_{key}.bin' for key in selected.keys())

    if args.js_output:
        js_path = os.path.abspath(args.js_output)
    else:
        js_path = None

    # Always print the JS array for inclusion in tablebase-loader.js
    print(f"\n// {len(filenames)} CWDL files for tablebase-loader.js:")
    lines = []
    for i in range(0, len(filenames), 4):
        chunk = filenames[i:i+4]
        lines.append("    " + ", ".join(f"'{f}'" for f in chunk) + ",")
    js_array = "const CWDL_FILES = [\n" + "\n".join(lines) + "\n];\n"

    if js_path:
        with open(js_path, 'w') as f:
            f.write(js_array)
        print(f"Wrote JS file list to {js_path}")
    else:
        print(js_array)


if __name__ == '__main__':
    main()
