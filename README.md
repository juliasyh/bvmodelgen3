# BVModelGen
Scripts to generate a 3D geometry for cardiac MRI

## Setting up in BigBlue
If you have access to the bigblue server, you do not need to install it, just add the correct python to your path. You can set it up so you can use the scripts from both the terminal and in VScode
### VScode
1. Open VScode in Bigblue
2. Open the VScode options and search Python
3. Select the one that says Python: Select Interpreter
4. Click on Enter interpreter path
5. Copy and paste this path: '/home/jilberto/.conda/envs/bvgen3/bin/python' and press enter.
6. On the bottom right you should see a text indicating the environment bvgen3 is active.
7. Make sure to reopen any terminals to make sure the correct environment is loaded. 

### Terminal
1. Open your '.bashrc' file with your favorite editor.
```
nano ~/.bashrc
```
2. At the end copy and paste the following,
```
alias pybvgen='/home/jilberto/.conda/envs/bvgen3/bin/python'
```
3. To use any of the scripts or modules of the repository, in a terminal, do,
```
pybvgen your_script.py
```

### Installation
1. Create a conda environment and activate it,
```
conda create -n bvgen3 python=3.13.2
```
2. Install necessary packages and modules,
```
python -m pip install -e .
```
3. Install NNUnetv2 following the steps [here](https://github.com/javijv4/CMR-nnUNet).
4. Install cheart-python-io. Follow the instructions [here](https://gitlab.eecs.umich.edu/jilberto/cheart-python-io) (you will need access to Gitlab).



### TODO list
* Smooth valve position
* Do a spatial smoothing of points
    * You need to generate a tet template mesh
* Calculate chamber and ventricle volumes.