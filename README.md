# Google Summer of Code 2026: DeepLense

[![GSoC 2026](https://img.shields.io/badge/GSoC-2026-blue.svg)](https://summerofcode.withgoogle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)

**Organization:** Universidad de Guadalajara 
**Author:** Ortega Rivera Leonardo Fabyan

---

## 📌 Project Overview

This repository contains the code, experiments, and documentation developed during the Google Summer of Code 2026 for the **Physics Guided Machine Learning on Real Lensing Images** project, under the Machine Learning for Science (DeepLense) organization.

The primary focus of this research lies in the development of a Physics-Informed Neural Network (PINN) framework designed to analyze real strong gravitational lensing datasets and study dark matter distribution. Strong gravitational lensing occurs when a massive galaxy or cluster bends light from a background source, creating arcs or Einstein rings. Since traditional algorithms often struggle or fail entirely when processing real data due to observational complexities and noise, this project directly integrates physical laws into the model's optimization process.

The purpose of this repository is to provide a reproducible experimental environment that allows for:
1. Designing and building physics-informed neural network architectures capable of operating on a wide variety of real lensing images.
2. Applying these models to study dark matter in various methodological contexts, such as classification, regression, and anomaly detection.
3. Enhancing the accuracy and interpretability of mass distribution estimates, extracting critical underlying insights into lensing systems and their sub-structures.

For comprehensive details regarding the methodology and theoretical framework, please refer to the [Original GSoC Proposal]([https://DeepLense.org/gsoc/2026/proposal_DEEPLENSE5.html](https://ml4sci.org/gsoc/2026/proposal_DEEPLENSE5.html)).

---

## 📓 Experimental Notebooks (Jupyter)

The progression of the empirical research is documented through the following experimental notebooks. Each notebook has been designed to be fully reproducible.

### Exploratory Analysis and Assessment of Spatial Separability in Dark Matter Substructure Dataset
* **Objective:** Conduct a comprehensive Exploratory Data Analysis (EDA) on a synthetic dark matter strong gravitational lensing dataset to evaluate spatial separability. This involves assessing class representativeness, spatial consistency, and analyzing the discriminative capacity of global intensity patterns and linear projections.
* **Results:** The dataset exhibited perfect structural homogeneity, featuring an exact class balance with a relative frequency of approximately 0.33 per class. Furthermore, all samples showed a uniform 150x150 pixel spatial resolution in a single grayscale channel, completely eliminating the need for morphological preprocessing like cropping or padding. However, linear subspace analysis utilizing the top 25 Principal Components revealed a near-complete overlap of class distributions. Clustering metrics, specifically a Silhouette Score of -0.0022 and a Davies-Bouldin Index of 109.92, proved that samples frequently cross decision boundaries in the linear manifold. This lack of linear separability formally justifies the requirement of spatially-aware deep learning architectures, such as CNNs, over shallow learning techniques.

### Task 1. Gravitational Lens Substructure Classification via Modified ResNet-18.pdf
* **Objective:** Define and initialize a modified ResNet-18 architecture from scratch. The base architecture was adapted by modifying the initial convolutional layer to accept single-channel input and replacing the final fully connected layer to output three-class logits. Prior to training, a structured suite of six sanity checks (e.g., single-batch memorization, gradient stability validation) was executed, followed by Bayesian hyperparameter optimization utilizing the Optuna framework.
* **Results:** The model was trained for 100 epochs using the AdamW optimizer. The optimal checkpoint (Epoch 86) achieved a validation accuracy of **93.76%** and a macro-averaged F1-score of **0.9370**. Evaluation on the held-out validation set demonstrated a mean AUC of **0.9908** under a One-vs-Rest ROC scheme. Confusion matrix analysis indicated that the SubHalo (Sphere) class presented the greatest discriminative challenge, exhibiting the most inter-class confusion.

---

## 🛠️ Installation and Setup

To reproduce the experiments, cloning the repository and setting up a virtual environment using `conda` or `venv` is recommended.

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[YourUsername]/[RepoName].git
cd [RepoName]

# Install dependencies
pip install -r requirements.txt
```

---

## 🖼️ Dataset

The empirical evaluations and model training pipelines documented in this repository rely on the dataset shared by **DeepLense**. The complete dataset can be accessed and downloaded via the following link:

🔗 **[Download the dataset here](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)**

**Configuration Requirement:** To ensure the seamless reproducibility of the experimental notebooks, the downloaded dataset must be extracted and placed directly into the root directory of this repository. The data loaders and execution scripts are strictly configured to expect the following hierarchical structure:

```text
[RepoName]/
├── dataset/ <-- Place the extracted dataset here
│   ├── train/
│   └── val/
├── notebooks/
├── src/
├── README.md
└── requirements.txt
```
