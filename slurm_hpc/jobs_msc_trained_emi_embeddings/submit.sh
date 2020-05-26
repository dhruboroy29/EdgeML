sbatch -t 1-0 --export=filename=msc_trained_emi_embeddings_H=64_winlen=256.sh batch_job.sbatch
sleep 10
sbatch -t 1-0 --export=filename=msc_trained_emi_embeddings_H=64_winlen=384.sh batch_job.sbatch
sleep 10
sbatch -t 1-0 --export=filename=msc_trained_emi_embeddings_H=64_winlen=512.sh batch_job.sbatch