# Neural-Network-Matrix-Completion
This is Tensorflow implementation of the paper "A Neural Net Framework for Accumulative Feature-based Matrix Completion". Please cite the paper if you find the source code useful for your research.

# Model
The framework impute missing values of a given matrix. The values are assumed to be real valued. The framework introduces separate neural network for each feature. Update is performed at each iteration by exploiting the model right after all networks are trained. 

# Dataset
Protein dataset is provided for demonstration.
