source activate l3embedding-tf-12-gpu
#source activate tfgpu
# echo "FILENAME: " ${filename}
module purge
module load cudnn/9.0v7.3.0.29

outname=test_out.out
python3 ../tf/examples/EMI-RNN/step2_2tier_joint_tvt.py -O 3 -gN sigmoid -uN tanh -bs 64 -H 16 -Dat /scratch/sk7898/buildsys_paper_data/bb_3class_winlen_256_winindex_all/3class_48_16 -rnd 5 -it 10 -ep 50 -ots 256 -k 10 -out $outname
