#!/usr/bin/env bash
# Prepares DRIVE and CHASE_DB1 (folder layout HR/, VE/, mask/) and STARE under ./data.
# DRIVE and CHASE_DB1 are taken from the copy distributed with the SGL repository
# (https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation), STARE from its official page.
# The original licence and access conditions of each dataset apply.
set -e
mkdir -p data && cd data
if [ ! -d sgl ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation.git sgl
  git -C sgl sparse-checkout set --no-cone /dataset/DRIVE/ /dataset/DRIVET/ /dataset/CHASE/HR/ /dataset/CHASE/VE/ /dataset/CHASET/
  git -C sgl checkout -q e74eaeac27ce2a483da7717645eb0f6e8e708acd
fi
mkdir -p stare && cd stare
for f in stare-images.tar labels-ah.tar; do
  [ -s "$f" ] || wget -q "https://cecas.clemson.edu/~ahoover/stare/probing/$f" -O "$f" || curl -sSfL -o "$f" "https://cecas.clemson.edu/~ahoover/stare/probing/$f" \
    || { echo "Download $f manually from https://cecas.clemson.edu/~ahoover/stare/probing/ into data/stare/"; rm -f "$f"; }
  [ -s "$f" ] && tar -xf "$f"
done
echo "DRIVE/CHASE_DB1: data/sgl/dataset    STARE: data/stare"
