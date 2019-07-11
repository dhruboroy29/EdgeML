import os
import csv

path_prefix = '/scratch/sk7898/pedbike'
data_dirs = ['final_bike_radial_full_cuts', 'final_human_radial_full_cuts']

for dirs in data_dirs:
    file_list = []
    label_list = []
    output_path = os.path.join(path_prefix, dirs + '_labels.csv')
    data_path = os.path.join(path_prefix, dirs)
    print(data_path)
    for root, dir, f_list in os.walk(data_path):
        for fname in f_list:
            if fname[-4:] == 'data':
                count_str = fname.split('_')[-2]
                label = int(count_str.split('p')[0])
                label_list.append(label)
                file_list.append(fname)

    with open(output_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(zip(file_list, label_list))
