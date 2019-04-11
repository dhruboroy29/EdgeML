#!/usr/bin/env bash

qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=2,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=5,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=8,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=10,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=12,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=15,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=17,ep=3,it=4,rnd=10 batch_job.pbs
qsub -V -l walltime=24:00:00 -v Dat=/fs/project/PAS1090/mil/BumbleBee/winlen_384_stride_128/48_16/,O=2,ots=384,k=20,ep=3,it=4,rnd=10 batch_job.pbs