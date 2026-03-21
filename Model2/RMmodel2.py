import urllib.request
from collections import Counter
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
import math
import random

print("Downloading data...")

url = "https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt"
filename = 'Compiled_Reviews.txt'
urllib.request.urlretrieve(url, filename)

print("Preprocessing data...")

reviews_raw_text = []
sentiment_ratings = []
product_types = []
helpfulness_ratings = []

with open("Compiled_Reviews.txt") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews_raw_text.append(fields[0])
        sentiment_ratings.append(fields[1])
        product_types.append(fields[2])
        helpfulness_ratings.append(fields[3])

token_def = re.compile("[^ ]+")

tokenized_sents = [token_def.findall(txt) for txt in reviews_raw_text]

tokens=[]
for s in tokenized_sents:
      tokens.extend(s)

counts = Counter(tokens)
so = sorted(counts.items(), key = lambda item: item[1], reverse = True)
so = list(zip(*so))[0]
type_list = so[0:1000]

M = np.zeros((len(reviews_raw_text), len(type_list)))

for i, rev in enumerate(reviews_raw_text):
    tokens = token_def.findall(rev)
    for j,t in enumerate(type_list):
        if t in tokens:
            M[i,j] = 1

print("Splitting data...")

train_ints = np.random.choice(len(reviews_raw_text), int(len(reviews_raw_text)*0.8), replace = False)
test_ints=list(set(range(0,len(reviews_raw_text))) - set(train_ints))
M_train = M[train_ints,]
M_test = M[test_ints,]

sentiment_ratings_train = [sentiment_ratings[i] for i in train_ints]
sentiment_ratings_test = [sentiment_ratings[i] for i in test_ints]

sentiment_classes = sorted(set(sentiment_ratings_train))
class_to_idx_sentiment = {cls: i for i, cls in enumerate(sentiment_classes)}

y_int_sentiment = np.array([class_to_idx_sentiment[c] for c in sentiment_ratings_train])

y_int_sentiment_train = np.array([class_to_idx_sentiment[c] for c in sentiment_ratings_train])
y_int_sentiment_test = np.array([class_to_idx_sentiment[c] for c in sentiment_ratings_test])

print(type(M_train), M_train.dtype, M_train.shape)

print("Specyfying device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X = torch.tensor(M_train.toarray() if hasattr(M_train, 'toarray') else M_train, dtype = torch.float32)
y = torch.tensor(y_int_sentiment_train, dtype = torch.float32).reshape(-1,1)
X_test = torch.tensor(M_test.toarray() if hasattr(M_test, 'toarray') else M_test, dtype = torch.float32)
y_int_sentiment = torch.tensor(y_int_sentiment_test, dtype = torch.float32).reshape(-1, 1)

num_features = X.shape[1]

model = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64,1)
) #hidden layer (MLP)
optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4) #Adam instead of 
#SGD
criterion = nn.BCEWithLogitsLoss()

num_epochs = 2000
logistic_loss = []

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size = 256, shuffle = True)

for i in range(num_epochs):
    for X_batch, y_batch in loader: #using mini batches 
        optimiser.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimiser.step()
    
    logistic_loss.append(loss.item())

plt.plot(range(1, num_epochs), logistic_loss[1:])
plt.xlabel("num_epochs")
plt.ylabel("loss")
plt.savefig("lossmodel2.png", dpi=300)
plt.show()

model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.sigmoid(logits).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

y_test = y_int_sentiment_test

fscore = precision_recall_fscore_support(y_test, preds, average="macro")[2]
print("Macro F1:", fscore)

fpr, tpr, thresholds = roc_curve(y_test, probs)
print("AUC:", auc(fpr, tpr))

import random
import math

print("Using bootstrapping...")

def draw_bootstrap_sample(data):
    n = len(data[0])
    indices = random.choices(range(n), k=n)
    indices = torch.tensor(indices)
    return (data[0][indices], data[1][indices])

def bootstrap_fscore(data, model, num_samples):
    scores = []
    model.eval()
    with torch.no_grad():                 
        for i in range(num_samples):
            X_bs, y_bs = draw_bootstrap_sample((data[0], data[1]))
            logits = model(X_bs)
            preds= (torch.sigmoid(logits) >= 0.5).int().numpy().flatten()
            y_bs_np = y_bs.flatten()

            if len(np.unique(y_bs_np)) < 2:
                continue

            score = precision_recall_fscore_support(y_bs_np, preds, average="macro")[2]
            scores.append(score)

    scores = np.sort(scores)
    return (
        np.mean(scores),
        scores[math.floor(len(scores) * 0.025)],  
        scores[math.floor(len(scores) * 0.975)] 
    )

mean_f1, lower_ci, upper_ci = bootstrap_fscore((X_test, y_test), model, 1000)
print(f"Bootstrap F1: {mean_f1:.4f}  95% CI [{lower_ci:.4f}, {upper_ci:.4f}]")
