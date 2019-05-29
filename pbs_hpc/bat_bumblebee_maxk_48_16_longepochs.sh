#!/usr/bin/env bash

# 2-class
#qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_256_winindex_all/48_16/,O=2,ots=256,k=100,ep=10,it=10,rnd=10 batch_job.pbs
#qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_384_winindex_all/48_16/,O=2,ots=384,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_512_winindex_all/48_16/,O=2,ots=512,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_640_winindex_all/48_16/,O=2,ots=640,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_768_winindex_all/48_16/,O=2,ots=768,k=100,ep=10,it=10,rnd=10 batch_job.pbs

# 3-class
#qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_256_winindex_all/3class_48_16/,O=3,ots=256,k=100,ep=10,it=10,rnd=10 batch_job.pbs
#qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_384_winindex_all/3class_48_16/,O=3,ots=384,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_512_winindex_all/3class_48_16/,O=3,ots=512,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_640_winindex_all/3class_48_16/,O=3,ots=640,k=100,ep=10,it=10,rnd=10 batch_job.pbs
qsub -V -l walltime=96:00:00 -v Dat=/fs/project/PAS1090/radar/Bumblebee/bb_3class_winlen_768_winindex_all/3class_48_16/,O=3,ots=768,k=100,ep=10,it=10,rnd=10 batch_job.pbs