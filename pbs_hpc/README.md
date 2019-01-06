Before running the HPC scripts, create virtual environment with Python 3.6.3 and install the requirements.

1. Install Miniconda

    `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`  
    `sh Miniconda3-latest-Linux-x86_64.sh`
    
2. Create virtual environment **using Python 3.6**

    `conda create -n tfgpu python=3.6.7`
    
3. Activate virtual environment, install remaining requirements

   `source activate tfgpu`  
   `pip install -r requirements.txt`  
   `source deactivate tfgpu`