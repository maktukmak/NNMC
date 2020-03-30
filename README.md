# Neural-Network-Matrix-Completion
This is Tensorflow implementation of the paper "A Neural Net Framework for Accumulative Feature-based Matrix Completion". Please cite the paper in your publication if you find the source code useful for your research.

# Model
The framework impute missing values of given matrix. The values are assumed to be real valued. The framework introduces seperate neural network for each feature column. Update is performed at each iteration by exploiting the model right after all networks are trained. 

# Dataset
Protein dataset is provided as an example usage.
