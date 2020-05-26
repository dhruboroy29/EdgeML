outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../tf/examples/EMI-RNN/step2_2tier_joint_save_emi_embeddings_tvt.py -O 3 -gN sigmoid -uN tanh -bs 64 -H 64 -Dat /scratch/sk7898/buildsys_paper_data/bb_3class_winlen_256_winindex_all/3class_48_16 -rnd 1 -it 1 -ep 1 -ots 256 -k 10 -out $outname
