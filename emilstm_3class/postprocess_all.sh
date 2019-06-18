#!/usr/bin/env bash

source activate tfgpu

list_files=(
            lstmEMI_3class_winlen_256
            lstmEMI_3class_winlen_384
            lstmEMI_3class_winlen_512
            lstmEMI_3class_winlen_640
            lstmEMI_3class_winlen_768
            )

list_hiddensize=(16 32 64)

# Collate outputs first
for l in ${list_files[@]}; do
    sh ../hpc_scripts/4_collate_output_splits.sh $l > $l.out
done

for h in ${list_hiddensize[@]}; do
    echo -e "\n\n\t\t-------------------- Hidden size = $h ---------------------"
    for l in ${list_files[@]}; do
        echo -e "\n\t----- Processing $l -----"
        python3 ../hpc_scripts/5_compute_best_lstmEMI_hiddenfiltered.py $l.out $h
    done
done
