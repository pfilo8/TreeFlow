# TreeFlow: Going Beyond Tree-based Parametric Probabilistic Regression

This repository contains the code and resources related to the research paper titled **"TreeFlow: Going Beyond Tree-based Parametric Probabilistic Regression"** by **Patryk Wielopolski** and **Maciej Zięba**. The paper is published in **26th European Conference on Artificial Intelligence ECAI 2023**, and it can be accessed at **[Link](https://arxiv.org/abs/2206.04140)**.

## Abstract

The tree-based ensembles are known for their outstanding performance in classification and regression problems characterized by feature vectors represented by mixed-type variables from various ranges and domains. However, considering regression problems, they are primarily designed to provide deterministic responses or model the uncertainty of the output with Gaussian or parametric distribution. In this work, we introduce TreeFlow, the tree-based approach that combines the benefits of using tree ensembles with the capabilities of modeling flexible probability distributions using normalizing flows. The main idea of the solution is to use a tree-based model as a feature extractor and combine it with a conditional variant of normalizing flow. Consequently, our approach is capable of modeling complex distributions for the regression outputs. We evaluate the proposed method on challenging regression benchmarks with varying volume, feature characteristics, and target dimensionality. We obtain the SOTA results for both probabilistic and deterministic metrics on datasets with multi-modal target distributions and competitive results on unimodal ones compared to tree-based regression baselines.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Code Structure](#code-structure)
- [Data](#data)
- [Experiments](#experiments)
- [Citation](#citation)
- [Contact](#contact)

## Introduction

We present TreeFlow, a novel tree-based approach for modeling probabilistic regression. Our method combines the advantages of tree-based structures as feature extractors with the power of normalizing flows to model flexible data distributions. We introduce a unique concept of merging forest structures with a conditional flow variant to effectively model uncertainty in regression output.

The key benefits of TreeFlow include:

  * 📈 Pioneering the use of tree-based models for non-parametric probabilistic regression, covering both uni- and multivariate predictions.
  * 🌳 A novel approach that combines tree-based models with conditional flows through a binary representation of the forest structure.
  * 🌟 Achieving state-of-the-art results in both probabilistic (NLL, CRPS) and deterministic (RMSE) metrics on datasets with multi-modal target distributions, while also delivering competitive results on unimodal datasets when compared to tree-based regression baselines.

## Prerequisites

Clone the repository and use prepared Makefile and Dockerfile. The whole project is build using the Kedro framework in version 0.17.5. 

```shell
# Clone the repository
git clone https://github.com/pfilo8/TreeFlow
cd TreeFlow
```

## Getting Started

```python
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from src.probabilistic_flow_boosting.tfboost import TreeFlow
from src.probabilistic_flow_boosting.tfboost.tree import EmbeddableCatBoostPriorNormal
from src.probabilistic_flow_boosting.tfboost.flow import ContinuousNormalizingFlow

tree = EmbeddableCatBoostPriorNormal(loss_function="RMSEWithUncertainty", depth=2, num_trees=100)
flow = ContinuousNormalizingFlow(input_dim=1, hidden_dims=(16, 16), num_blocks=2, context_dim=16, conditional=True)

treeflow = TreeFlow(tree, flow, embedding_size=16)

x, y = load_diabetes(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train)

treeflow.fit(x_train, y_train, x_val, y_val, n_epochs=50, batch_size=1024, verbose=True)
samples = treeflow.sample(x_test, num_samples=1000)  # Shape (X number of samples, num_samples, Y dimension / input_dim)
samples = samples.squeeze(axis=-1)

# Plot KDE plot for the first observation, i.e. index = 0.
plt.axvline(x=y_test[0], color='r', label='True value')
sns.kdeplot(samples[0, :], color='blue', label='TreeFlow')
plt.legend()
plt.show()
```

## Code Structure

```
|── conf/                 # Configuration files
├── data/                 # Datasets and trained models 
├── images/               # Images for documentation purposes
├── logs/                 # Logs generated during experiments execution
├── notebooks/            # Jupyter notebooks for models analysis
├── src/                  # Source code
├── README.md             # This document
└── ...
```

Source code in `src/probabilistic_flow_boosting` directory contains the following:
  * `cnf` - Continuous Normalizing Flow implementation in the sklearn style.
  * `extras` - Kedro wrappers for the TreeFlow model and UCI Datasets.
  * `indepedendent_multivariate_boosting` - Implementation of the multivariate independent tree-based models.
  * `pgbm` - PGBM implementation taken from the official repository of [PGBM](https://github.com/elephaint/pgbm).
  * `pipelines` - Kedro pipelines for the experiments.
  * `tfboost` - Implementation of the TreeFlow.

Side note regarding codebase: originally TreeFlow appears in the implementation as TFBoost or the whole repository mentions `probabilistic_flow_boosting` which was the previous name of the method. To do not break things we have decided to leave the names as it is.

## Data

The full data folder can be found under the following link: [Link](https://drive.google.com/file/d/1c95eJeJS0P8Ts24G6hfQgx7RPKl9AF3D/view?usp=sharing).
More details regarding the datasets can be found in the paper in the appendix directory.

## Experiments

Experiments are created using Kedro framework, and it's definition is in the `src/probabilistic_flow_boosting/pipeline_registry.py` file. 
The most important configuration file is in the `conf/base/parameters/modeling/treeflow.yml` file. 
Here you define the hyperparameters for the grid search of the parameters. It needs to be adjusted for each dataset separately. 

### How to build docker?
```shell
make build
```

### How to run the container in the background?
```shell
make run
```

### How to get to the Docker container?
```shell
docker exec -it "${username}-${project_name}" /bin/bash
```

### How to run the Kedro project?
Inside the container run the following command.
```shell
python -m kedro run --pipeline <pipeline_name>
```
where pipeline_name is the name of the pipeline defined in the `src/probabilistic_flow_boosting/pipeline_registry.py` file.

### How to stop the container?
```shell
make rm
```

## Citation
If you use this code or the research findings in your work, please cite our paper:

```
@article{wielopolski2023treeflow,
      title={TreeFlow: Going beyond Tree-based Gaussian Probabilistic Regression}, 
      author={Patryk Wielopolski and Maciej Zięba},
      year={2023},
      eprint={2206.04140},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact
In case of questions or comments please contact using LinkedIn: [Patryk Wielopolski](https://www.linkedin.com/in/patryk-wielopolski/)
