# DeepExplore_vs_VFI
A repository for reproducing the results in my Bachelor Thesis: "Examining a Method from Deep Reinforcement Learning as an Alternative
to Value Function Iteration"

University of Copenhagen
Department of Economics
Supervisior: Jeppe Druedahl

## Files

<li>results</li>
<li>shared_modules</li>
<li>VFI_modules</li>
<li>ValueFunctionIteration.py</li>
<li>DeepExplore.py</li>
<li>ResultsFromThesis.ipynb</li>
<li>VisualizeResults.ipynb</li>


The solutions are mainly based on the modules ValueFunctionIteration.py and DeepExplore.py

These two modules both use the modules in the directory "shared_modules". The reason for this is that drawing the same random distributions for the two different solutions methods across NumPy and PyTorch meant that I had to write a shared file for this, implemented in NumPy, which converts the results to torch.tensors if requested. Besides, it was a good chance to gather the different functionalities in .py files.

If you want to reproduce my results presented in my Thesis, I recommend running the scripts in the ResultsFromThesis.ipynb. If you want to recreate the figures in the thesis, use the notebook VisualizeResults.ipynb. This notebook uses that many of my results are located in the results directory. If the specific result is there, the notebook automatically loads the relevant tensors, arrays and nns and visualizes them like I have.

You might get an error when running functions from the load_and_dump module, because it was originally implemented on GPUs (this should not happen)

This repository features many more figures in VisualizeResults.ipynb than what I could include in my Thesis.

Also, some results (like policy functions for ValueFunctionIteration and state grids, which I originally had in my results directory), are too big for uploading. Therefore I have uploaded the following in the different subdirectories in results:
<li>arrays: Solution time and objective function evaluation of simulation of VFI</li>
<li>nns: 14 trained Deep Neural Networks. 13 of them are part of my extensions presented in the thesis, while the last is the baseline nn</li>
<li>tensors: Out-of-sample (evaluation iteration) evaluations, denoted OSL for all different nns in the training phase + Timestamp along the training phase for nns + Objective function evaluations for 2 nns that have been simulated afterwards </li>

Note that while the name and placement of most files is intuitive, some files have a _x, where x is a number, often close to 1000. This is because the early stopping threshold was enforced and the training stopped in evaluation iteration x instead of 1000 (0-indexed: 999)

If you want to compute euler errors or simulate using the nns, do this:
1) Load nn using the load_and_dump module
2) Call simulation or euler errors computation functions in DeepExplore module
3) Now you should have all files used for visualization. You might need to change the paths such that the script in VisualizeResults.ipynb can locate the arrays/tensors

A final comment: The train_nn_evals function in the DeepExplore module is very versatile for different training scheudles of Deep Neural Networks in the DeepExplore framework, why it is recommended to use. Also, the documentation should be good in DeepExplore.py.
