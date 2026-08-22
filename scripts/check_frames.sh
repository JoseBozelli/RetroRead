#!/bin/bash
# Reports total vs. non-empty PNG frame counts per "man*" folder under "4 Test of videos"
TARGET="$1"
echo "man_folder,total_pngs,empty_pngs,usable_pngs"
for d in "$TARGET"/man*; do
  name=$(basename "$d")
  total=$(find "$d" -type f -name "*.png" | wc -l)
  empty=$(find "$d" -type f -name "*.png" -empty | wc -l)
  usable=$((total - empty))
  echo "$name,$total,$empty,$usable"
done