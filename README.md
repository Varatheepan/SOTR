# SOTR ([see in portfolio](https://varatheepan.github.io/myportfolio/FYP.html))
SOTR(Self Organizing Task Representation) is a method we worked on in the continual learning paradigm.
This repo includes the code-base of the work. 

### Following acronyms, abbreviations and other key words used in the file names.
* GWR - Grow when required
* NC50 - Fifty incremental new class scenario
* Multi-head - Multi-headed deep learning  model architecture
* CNN - Convolutional neural network
* SI - Synaptic intelligence
* AP -Average precision
* SE - Squeeze and Excitation
* FC - Fully connected deep neural network
* Task - A set of data provided to the network at once during the incremental training

## SOTR Architecture
![SOTR Architecture](/core50_hybrid/benchmarks/sonn/SOTR_architecture.png)

### The following files are used in final results generation.
Fifty incremental new class scenario training and testing
* Here each category of the core50 dataset are incrementally fed to the network.
* File path: core50_hybrid/benchmarks/sonn/NC50_incremental_test.py

Five incremental new class scenario training and testing
* Here all the fifty individual classes of core50 dataset are incrementally fed to the network.
* File path: core50_hybrid/benchmarks/sonn/NC5_incremental_test.py

GWR neural network
* This has different implementations of GWR using networkx module and PyTorch(only one used at each implementation). 
* File path: core50_hybrid/benchmarks/sonn/gwr.py
