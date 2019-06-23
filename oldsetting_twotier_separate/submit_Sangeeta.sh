#!/usr/bin/env bash

list_files=(
            3_SUBMIT_oldsetting_twotierEMI_3class_winlen_512_jobs.sh
            3_SUBMIT_oldsetting_twotierEMI_3class_winlen_768_jobs.sh
            3_SUBMIT_oldsetting_twotierEMI_3class_winlen_640_jobs.sh
            )

cd ../slurm_hpc

# Submit jobs
for l in ${list_files[@]}; do
    sh $l
    sleep 60
done