#!/usr/bin/env bash

source activate tfgpu

list_files=(
            twotierEMI_3class_winlen_256
            twotierEMI_3class_winlen_384
            twotierEMI_3class_winlen_512
            twotierEMI_3class_winlen_640
            twotierEMI_3class_winlen_768
            )

list_hiddensize=(16 32 64)

for l in ${list_files[@]}; do
    echo -e "\n\n\t\t-------------------- Processing $l ---------------------"
    sh ../hpc_scripts/4_collate_output_splits.sh $l > $l.out

    for h in ${list_hiddensize[@]}; do
        echo -e "\n\t----- Hidden size = $h -----"
        python3 ../hpc_scripts/5_compute_best_2tierEMI_hiddenfiltered.py $l.out $h
    done
done
