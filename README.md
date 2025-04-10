# BVModelGen
Scripts to generate a 3D geometry for cardiac MRI

# Setting up in BigBlue
If you have access to the bigblue server, you do not need to install the scripts, just set up the correct paths and python interpreter. 

## Setting up the NN paths
1. Open your `~/.bashrc` file (you can do `nano ~/.bashrc` if in the terminal or `code ~/.bashrc` if in VScode).
2. At the end of the `~/.bashrc` file copy the following (exactly as they are!),
   ```
   export nnUNet_raw="/home/jilberto/software/nnUNetv2/nnUNet_raw"
   export nnUNet_preprocessed="/home/jilberto/software/nnUNetv2/nnUNet_preprocessed"
   export nnUNet_results="/home/jilberto/software/nnUNetv2/nnUNet_results"
   ```
3. Save the `~/.bashrc` file and close and reopen any terminals. 

## Setting up the Python interpreter
You can set it up so you can use the scripts from both the terminal and VScode.
### VScode
1. Open VScode in Bigblue
2. Open the VScode options (Ctrl+Shift+P) and search Python.
3. Select the one that says Python: Select Interpreter
4. Click on Enter interpreter path
5. Copy and paste this path: `/home/jilberto/.conda/envs/bvgen3/bin/python` and press enter.
6. On the bottom right you should see a text indicating the environment bvgen3 is active.
7. Make sure to reopen any terminals to make sure the correct environment is loaded. 

### Terminal
1. Open your `.bashrc` file with your favorite editor.
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

# Installation
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


# Inputs
The inputs of the `main.py` script are the short axis, four chamber, three chamber, and two chamber (SA, LA_4CH, LA_3CH, LA_2CH) cine images and the valve segmentation on the first frame.

For the valve segmentation, select points using label 1 for MV, 2 for TV, 3 for AV. *If there is more than one slice for the 3CH view, only add points in one!*

### TODO list (in order of urgency)
* Fix plotting function
* Create function to interpolate frames
* Make valve extrapolation always use user inputs
* Make parallelizable process parallel
* Smooth valve position?
