import numpy as np
import os
import glob


def extract_windows(indirs, outdir, class_label, stride, winlen, samprate=256, minlen_secs=1):
    """
    Ref: https://github.com/dhruboroy29/MATLAB_Scripts/blob/neel/Scripts/extract_target_windows.m
    Extract sliding windows out of input data
    :param samprate: sampling rate
    :param indirs: input directory list of data
    :param outdir: output directory of windowed data
    :param class_label: data label (Target, Noise, etc.)
    :param stride: stride length in samples
    :param winlen: window length in samples
    :param minlen_secs: minimum cut length in seconds
    """

    assert isinstance(indirs, (str, list))
    assert isinstance(outdir, str)

    # If single directory given, create list
    if isinstance(indirs, str):
        indirs = [indirs]

    # Make output directory
    outdir = os.path.join(outdir, 'winlen_' + str(winlen) + '_stride_' + str(stride), class_label)
    # Silently create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    for indir in indirs:
        assert isinstance(indir, str)

        # Find data files
        list_files = glob.glob(os.path.join(indir,'*.data'))

        max_walk_secs = 0
        for cur_file in list_files:
            # Get filename without extension
            cur_file_name = os.path.basename(os.path.splitext(cur_file)[0])

            # Read IQ samples
            comp = np.fromfile(cur_file, dtype=np.uint16)
            I = comp[::2]
            Q = comp[1::2]
            assert len(I) == len(Q)
            L = len(I)
            cur_walk_secs = L / samprate

            # Print data column-wise
            # print(*comp.tolist(),sep='\n')

            # Ignore very short walks
            if cur_walk_secs < minlen_secs:
                continue

            # Get max walklength encountered so far
            if cur_walk_secs > max_walk_secs:
                max_walk_secs = cur_walk_secs

            # Extract windows
            for k1 in range(0, L - winlen, stride):
                temp_I = I[k1:k1 + winlen]
                temp_Q = Q[k1:k1 + winlen]

                # uint16 cut file array
                Data_cut = np.zeros(2 * winlen, dtype=np.uint16)
                Data_cut[::2] = temp_I
                Data_cut[1::2] = temp_Q

                # Print cut column-wise
                # print(*Data_cut.astype(int), sep='\n')

                # Output filenames follow MATLAB array indexing convention
                outfilename = os.path.join(outdir,
                                           cur_file_name + '_' + str(k1 + 1) + '_to_' + str(k1 + winlen + 1) + '.data')

                # Save to output file
                Data_cut.tofile(outfilename, )

    print('All done!')


# Test
if __name__=='__main__':
    austere_base_folder = '/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/Research/Deep_Learning_Radar/Data/Austere/'
    extract_windows(indirs=[
        austere_base_folder + 'Bora_New_Detector/Bora_new_det_aus_M_30_N_96_win_res_last_w_padded_with_signal_lookahead/austere_386_human',
        austere_base_folder + 'Bora_New_Detector/Bora_new_det_aus_M_30_N_96_win_res_last_w_padded_with_signal_lookahead/austere_310_cow'],
                    outdir='/home/Roy.174/Desktop/Test',
                    class_label='Target_Python',
                    stride=171,
                    winlen=256)

    extract_windows(indirs=austere_base_folder + 'Noise',
                    outdir='/home/Roy.174/Desktop/Test',
                    class_label='Noise_Python',
                    stride=171,
                    winlen=256)
