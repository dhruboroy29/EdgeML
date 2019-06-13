outname=`echo $0 | sed "s/.sh/.out/g"`
python3 ../tf/examples/EMI-RNN/step2_emi_fastrgnn_disp_det.py -O 2 -gN sigmoid -uN sigmoid -bs 64 -H 64 -Dat /scratch/dr2915/Bumblebee/bb_3class_winlen_384_winindex_all/48_16 -rnd 10 -it 5 -ep 10 -k 100 -out $outname
