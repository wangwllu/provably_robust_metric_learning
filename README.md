# Provably Robust Metric Learning

Implementation of the algorithm ARML (Adversarially Robust Metric Learning) proposed in the paper Provably Robust Metric Learning

## Requirements

Our program is tested on Python 3.7.7. The required packages are

* pytorch 1.5.0
* metric-learn 0.5.0 (implementations of the compared methods)
* foolbox 3.0.0 (implementation of the Boundary attack)

## Examples

The configuration files are in the folder `config`. Please make sure the path to the dataset (`dataset_dir`) is correctly set.

Some examples are shown below:

* Learn and save the positive semi-definite matrix (the Mahalanobis distance) of the proposed algorithm ARML:

```bash
python main_mahalanobis.py --section arml 
```

* Compute and save the norms of the exact minimal adversarial perturbations of 1-NN with respect to the proposed algorithm ARML (please learn and save the PSD matrix first):

```bash
python main_exact_perturbation_norms.py --section arml
```

* Compute and save the results of K-NN verification with respect to the proposed algorithm ARML (please learn and save the PSD matrix first):

```bash
python main_knn_verify.py --section arml
```

* Compute and save the results of K-NN attack with respect to the proposed algorithm ARML (please learn and save the PSD matrix first):

```bash
python main_boundary_attack.py --section arml
```
