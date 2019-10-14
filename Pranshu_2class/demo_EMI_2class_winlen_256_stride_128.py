import sys
import os

sys.path.append('../')

######################### ONLY MODIFY THESE VALUES #########################
# Number of splits of hyperparam file
winlen = 256
stride = 128
num_splits='12'

# Base path of data
prefix = 'demo_EMI_2class_winlen_' + str(winlen) + '_stride_' + str(stride)

base='/mnt/6b93b438-a3d4-40d2-9f3d-d8cdbb850183/' \
     'Research/Deep_Learning_Radar/Data/Austere/BuildSys_Demo/' \
     'Windowed/winlen_' + str(winlen) + '_stride_' + str(stride) + '/12_8'

# Batch system
bat_sys='pbs'

# List of values of hyperparam k
k=[5, 6, 7]

# Running time
walltime='24:00:00'

######################### KEEP THE REST INTACT #########################
# Folder where jobs are saved
jobfolder = '../'+ bat_sys +'_hpc/'

#Init args
init_argv=sys.argv

# Enter hpc_scripts folder
os.chdir('../hpc_scripts')

# Prepare data
#print('###### Scripts/processing_data #####')
#sys.argv=init_argv+['-type', prefix, '-base', base]
#import Scripts.create_train_val_test_split

# Generate gridsearch
print('###### hpc_scripts/gridsearch #####')
sys.argv=init_argv+['-bat', bat_sys, '-type', prefix, '-base',
                    base, '-O', str(2), '-ots', str(winlen) ,'-k']+[str(param) for param in k]
import hpc_scripts.gridsearch_0_EMI

# Split hyperparam file
print('###### hpc_scripts/split_hyp_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'.sh',num_splits]
import hpc_scripts.split_hyp_wrapper_1

# Create batch job
print('###### hpc_scripts/create_batch_wrapper #####')
sys.argv=init_argv+[jobfolder+prefix+'_',walltime,bat_sys]
import hpc_scripts.create_batch_wrapper_2

# Submit
print("\nNow submit " + bat_sys + "_hpc/3_SUBMIT_"+prefix+"_jobs.sh on server")
