list_files=(
            dispdetEMI_2class_winlen_256
            dispdetEMI_2class_winlen_384
            dispdetEMI_2class_winlen_512
            #dispdetEMI_2class_winlen_640
            #dispdetEMI_2class_winlen_768
            )

restartFile='3_SUBMIT_dispdetEMI_2class_remaining_jobs.sh'
if [ -f "$restartFile" ]; then
    rm $restartFile
fi


for filePrefix in ${list_files[@]}; do
    for i in `seq 01 1 24`; do
	if [[ ${#i} -lt 2 ]] ; then
	    i="0${i}"
	fi
	outFile=${filePrefix}_${i}_spl.out
	shFile=${filePrefix}_${i}_spl.sh

	if [ ! -f "$outFile" ]; then
	    echo "Adding $outFile to remaining jobs!"
	    echo "sbatch -t 1-0 --export=filename=../slurm_hpc/${shFile} batch_job.sbatch" >> $restartFile
	    echo "sleep 1" >> $restartFile
	fi
    done
done
