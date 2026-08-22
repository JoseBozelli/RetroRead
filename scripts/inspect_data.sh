#!/bin/bash
# Usage: ./inspect_data.sh /path/to/unzipped/folder
# Prints directory tree, file counts by extension, total size, and duplicate files by content hash.

TARGET="${1:-.}"

echo "=================================================="
echo "DIRECTORY TREE (3 levels deep)"
echo "=================================================="
find "$TARGET" -maxdepth 3 -print | sed -e "s|[^-][^/]*/|  |g" -e "s|^|  |"

echo ""
echo "=================================================="
echo "FILE COUNTS BY EXTENSION"
echo "=================================================="
find "$TARGET" -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn

echo ""
echo "=================================================="
echo "TOTAL SIZE"
echo "=================================================="
du -sh "$TARGET"

echo ""
echo "=================================================="
echo "TOP-LEVEL FOLDER SIZES"
echo "=================================================="
du -sh "$TARGET"/* 2>/dev/null | sort -rh

echo ""
echo "=================================================="
echo "DUPLICATE FILES (by content hash, exact duplicates only)"
echo "=================================================="
find "$TARGET" -type f -exec md5sum {} \; | sort | awk '{
    if ($1 == prev_hash) { print prev_line; print $0; dup_found=1 }
    prev_hash = $1; prev_line = $0
} END { if (!dup_found) print "No exact duplicates found." }'

echo ""
echo "Done."