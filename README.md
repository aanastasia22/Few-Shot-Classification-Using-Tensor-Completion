# Few-Shot-Classification-Using-Tensor-Completion

This repository contains Python codes and scripts designed for the classification problem that is formulated as a tensor completion problem combining classification and tensor decomposition techniques, as it is presented in the paper ["Few-Shot Classification Using Tensor Completion" (A. Aidini, G. Tsagkatakis, N. Sidiropoulos, P. Tsakalides)](https://users.ics.forth.gr/~tsakalid/PAPERS/CNFRS/2023-Asilomar1.pdf). In the  proposed method, a tensor of scores of the samples is learned, where the sample values correspond to indices in the tensor, containing the scores for each class. Then, given new data points, we only need to take the score values of the learned tensor in the position indicated by the sample to classify it into the class with the highest score. Since only a small fraction of tensor entries can be obtained from a given training set of samples, we complete the score tensor, taking into account the discrete nature of the predicted class variable by combining the hinge loss function with Tucker decomposition.

## Requirements
### Datasets
The efficacy of the proposed classification method is evaluated on several real-world classification tasks using datasets obtained from the [UCI machine learning repository](https://archive.ics.uci.edu/ml/datasets.php). 
For each dataset, we expressed the data and the corresponding targets as integers and saved them in a mat file, which is also given in the Data folder of this repository.

### Framework
We use the library [TensorLy](http://tensorly.org/): Tensor Learning in Python to execute tensor operations.

## Contents
`main.py`: The primary script that loads the data, performs the classification using the proposed tensor-based approach, and provides the results. Note that for each dataset, the appropriate dimensions of the label tensor in the variable dim_label_tensor have to be adjusted accordingly, as well as some of the parameters.

`label_tensor.py`: Create the label tensor, in which the tuple of input variables corresponds to a cell multi-index and the cell content is the corresponding label.

`label_tensor_completion.py`: Perform the label tensor completion method.

`compute_accuracy.py`: Find the predicted samples from the recovered label tensor and compute accuracy (%).

## Reference
If you use our method, please cite:
```
@inproceedings{aidini2023few,
  title={Few-Shot Classification Using Tensor Completion},
  author={Aidini, Anastasia and Tsagkatakis, Grigorios and Sidiropoulos, Nicholas D and Tsakalides, Panagiotis},
  booktitle={2023 57th Asilomar Conference on Signals, Systems, and Computers},
  pages={1283--1287},
  year={2023},
  organization={IEEE}
}
```
