# DeepExplore_vs_VFI
A repository for reproducing the results in my undergraduate thesis "Examining a Method from Deep Reinforcement Learning as an Alternative
to Value Function Iteration".

## Files

<li>results</li>
<li>shared_modules</li>
<li>VFI_modules</li>
<li>ValueFunctionIteration.py</li>
<li>DeepExplore.py</li>
<li>ResultsFromThesis.ipynb</li>
<li>VisualizeResults.ipynb</li>


The solutions are mainly based on the modules ValueFunctionIteration.py and DeepExplore.py

These two modules do however both use the modules in the directory "shared_modules". The reason for this is that drawing the same random distributions for the two different solutions methods across NumPy and PyTorch meant that I had to write a shared file for this, implemented in NumPy, which converts the results to torch.tensors if requested. Besides, it was a good chance to gather the different functionalities in .py files.

If you want to reproduce my results presented in my Thesis (please reach out if you want a copy of the pdf ;-) ), I recommend running the scripts in the ResultsFromThesis.ipynb. If you want to recreate the figures in the thesis, use the notebook VisualizeResults.ipynb. This notebook uses the fact that all my results are gathered in the results directory. The notebook automatically loads the relevant tensors, arrays and nns and visualizes them like I have. You might get an error when running functions from the load_and_dump module, because it was originally implemented on GPUs. This should however not happen.

This repository features many more figures than what I could include in my Thesis.

A final comment: The train_nn_evals function in the DeepExplore module is very versatile for different training scheudles of Deep Neural Networks in the DeepExplore framework, why it is recommended to use. Also, the documentation should be good in DeepExplore.py.
