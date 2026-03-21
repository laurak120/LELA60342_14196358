# LELA60342_14196358

## Overview 

This repository contains code for the Research Methods 2 module task. The task includes: implementing a logistic regression classifier from CL1 using PyTorch, making additions to improve the initial model (thereby producing model 2), calculate AUC for both classifiers, and implement bootstrapping to generate a p-value for classifier comparison, testing whether AUC for model 2 is significantly higher than model 1. 

The underlying task (CL1) was to build a classifier to determine whether a review of a product is positive or negative (binary classification). The dataset consists of Amazon reviews.

The repository contains: 
- Model 1 script (data preprocessing, model, training, evaluation) in *.py format
- Model 2 script (data preprocessing, model, training, evaluation) in *.py format
- Combined script with both models to perform comparison in evaluation in *.py format
- slurm script to run on CSF
- slurm.out files to demonstrate results after running the scripts on the CSF

## Models

Model 1 is a simple single-layer model using the linear function and BCE (binary cross-entropy loss) with logits, combining a sigmoid layer and the classic BCE loss. This is because combining them in one class is more numerically stable. For optimisation, it uses SGD (stochastic gradient descent), which uses a fixed learning rate. Learning does not happen in batches. It is a mathematical equivalent to this binary classifier from CL1:

```python
num_features= M_train.shape[1]
y = y_int_sentiment_train.reshape(-1, 1)
weights = np.random.rand(num_features, 1)
num_samples=len(y)
bias=np.random.rand(1) 
n_iters = 000
lr= 0
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

Model 2 is an MLP model with two hidden layers with ReLU activation and dropout (0.3). For optimisation, it uses Adam, which adjusts learning rate and generally yields faster convergence, using a default learning rate of 1e-3 and weight decay of 1e-4 to penalise large weights. These choices were made to prevent overfitting. The training loop uses mini-batch training (with batch size of 256) via DataLoader, with shuffling on. 

## Evaluation 

Models were evaluated using macro F1 score, AUC, further evaluated with bootstrapping. 

## Results 

