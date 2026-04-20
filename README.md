# LELA60342_14196358

## Overview 

This project implements a full sentiment classification pipeline, comparing a 
logistic regression baseline against a multi-layer perceptron (MLP). The models are evaluated using standard metrics (F1, ROC, AUC) and further compared using bootstrap resampling to assess statistical significance.

This repository contains code for the Research Methods 2 module task. The task includes: implementing a logistic regression classifier from CL1 using PyTorch, making additions to improve the initial model (thereby producing model 2), calculate AUC for both classifiers, and implement bootstrapping to generate a p-value for classifier comparison, testing whether AUC for model 2 is significantly higher than model 1. The underlying task (CL1) was to build a classifier to determine whether a review of a product is positive or negative (binary classification). The dataset consists of Amazon reviews.

Each *.py script is divided into sections. Broadly, the scripts follow this order: import packages, download data, preprocess data, manually train-test split, convert data into torch tensors, set up model, train model, evaluate. 

The jobscript needs to be adjusted for running each script by inserting the name of the script in the dedicated space. Number of nodes and GPUs used can be altered to increase speed of training. ```watch -n 1 tail -c 2048 slurm-JOBID.out``` can be used to monitor the job. 

Data is manually split into train-test, much like in the CL1 submission, with 80% of the dataset belonging to train and remaining 20% belonging to test. ```replace = False``` is used to ensure that all datapoints in a sample are unique. 

## Repository Structure

```text
LELA60342_14196358/
├── Combined_script/
│   ├── combined_script.py
│   └── slurm-12568409.out
│
├── Model1/
│   ├── RMmodel1.py
│   └── slurm-12224880.out
│
├── Model2/
│   ├── RMmodel2.py
│   └── slurm-12226744.out
│
├── jobscript.slurm
├── requirements.txt
└── README.md
```
## Dependencies 

- numpy
- torch
- matplotlib
- scikit-learn

Install dependencies:

```
pip install -r requirements.txt
pip install torch
```

## Models

Model 1 is a simple single-layer model using the linear function and BCE (binary cross-entropy loss) with logits, combining a sigmoid layer and the classic BCE loss. This is because combining them in one class is more numerically stable. For optimisation, it uses SGD (stochastic gradient descent), which uses a fixed learning rate. Learning does not happen in batches. It is a mathematical equivalent to this binary classifier from CL1:

```python
num_features= M_train.shape[1]
y = y_int_sentiment_train.reshape(-1, 1)
weights = np.random.rand(num_features, 1)
num_samples=len(y)
bias=np.random.rand(1) 
n_iters = #desired num_iters
lr= #desired lr
logistic_loss=[]
eps= 1e-7
for i in range(n_iters):
  z = (M_train.dot(weights)+bias)
  q = 1/(1+np.exp(-z))
  loss = -np.mean(y * np.log(q + eps) + (1 - y) * np.log(1 - q + eps))
  logistic_loss.append(loss)

  dw = M_train.T.dot(q - y) / num_samples
  db = np.mean(q - y)
  weights = weights - lr*dw
  bias = bias - lr*db
```
Model 2 is an MLP model with two hidden layers with ReLU activation and dropout (0.3). For optimisation, it uses Adam, which adjusts learning rate and generally yields faster convergence, using a default learning rate of 1e-3 and weight decay of 1e-4 to penalise large weights. The training loop uses mini-batch training (with batch size of 256) via DataLoader, with shuffling on. 

## Evaluation 

Models were evaluated using macro F1 score, AUC, further evaluated with bootstrapping. The combined script contains ```def bootstrap_auc_diff``` for comparison. It returns observed difference (AUC1 - AUC2 on test set), bootstrapped mean difference, 95% confidence interval of difference, and the p-value. Samples with only one class are excluded as AUC becomes undefined. The alpha for the p-value is 0.05. 

## Results 

Model 1 individual run:
```shell
Macro F1: 0.7914763817577608
AUC: 0.8757831300689385
Using bootstrapping...
Bootstrap F1: 0.7916  95% CI [0.7823, 0.8009]
```
Model 2 individual run:
```shell
Macro F1: 0.8202806642655683
AUC: 0.9049801097044914
Using bootstrapping...
Bootstrap F1: 0.8203  95% CI [0.8111, 0.8289]
```
Combined script run:

```shell
Training Model 1: Logistic Regression
Model 1 — Macro F1: 0.7981  |  AUC: 0.8825
Training Model 2: MLP
Model 2 — Macro F1: 0.8063  |  AUC: 0.8952
Bootstrap AUC Comparison...

Model 1 AUC: 0.8825
Model 2 AUC: 0.8952
Observed AUC difference (M1 - M2): -0.0128
Bootstrap mean difference: -0.0129
95% CI of difference: [-0.0186, -0.0069]
Two-sided p-value: 0.0000

Conclusion: The AUC difference is statistically significant (p < 0.05).
  Model 2 has a significantly higher AUC.
```
### Interpretation 

The MLP classifier yields marginally better but statistially significant performance in accordance with the AUC test. 

### Other

The main scripts also issue and save a .png to visualise the loss. 

## Limitations

The evaluation methods could be more robust to target other questions regarding the performance of classifiers. For the purpose of the task evaluation was limited according to task specifications. 
