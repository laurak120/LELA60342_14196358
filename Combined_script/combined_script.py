import urllib.request
from collections import Counter
import re
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset

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

tokens = []
for s in tokenized_sents:
    tokens.extend(s)

counts = Counter(tokens)
so = sorted(counts.items(), key=lambda item: item[1], reverse=True)
so = list(zip(*so))[0]
type_list = so[0:1000]

M = np.zeros((len(reviews_raw_text), len(type_list)))
for i, rev in enumerate(reviews_raw_text):
    tokens = token_def.findall(rev)
    for j, t in enumerate(type_list):
        if t in tokens:
            M[i, j] = 1

print("Splitting data...")
train_ints = np.random.choice(len(reviews_raw_text), int(len(reviews_raw_text) * 0.8), replace=False)
test_ints = list(set(range(0, len(reviews_raw_text))) - set(train_ints))

M_train = M[train_ints, ]
M_test = M[test_ints, ]

sentiment_ratings_train= [sentiment_ratings[i] for i in train_ints]
sentiment_ratings_test = [sentiment_ratings[i] for i in test_ints]

sentiment_classes= sorted(set(sentiment_ratings_train))
class_to_idx= {cls: i for i, cls in enumerate(sentiment_classes)}

y_int_train = np.array([class_to_idx[c] for c in sentiment_ratings_train])
y_int_test= np.array([class_to_idx[c] for c in sentiment_ratings_test])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X= torch.tensor(M_train, dtype=torch.float32)
y= torch.tensor(y_int_train, dtype=torch.float32).reshape(-1, 1)
X_test=torch.tensor(M_test,  dtype=torch.float32)
y_test =y_int_test   # kept as numpy for sklearn metrics


print("Training Model 1: Logistic Regression")
model1 = nn.Linear(X.shape[1], 1)
optimiser1 = torch.optim.SGD(model1.parameters(), lr=1)
criterion   = nn.BCEWithLogitsLoss()

num_epochs = 2000
loss_history_m1 = []

for i in range(num_epochs):
    model1.train()
    optimiser1.zero_grad()
    z= model1(X)
    loss = criterion(z, y)
    loss.backward()
    optimiser1.step()
    loss_history_m1.append(loss.item())

plt.figure()
plt.plot(range(1, num_epochs), loss_history_m1[1:])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 1 Training Loss")
plt.savefig("loss_model1.png", dpi=300)
plt.close()

model1.eval()
with torch.no_grad():
    logits1 = model1(X_test)
    probs1= torch.sigmoid(logits1).numpy().flatten()
    preds1= (probs1 >= 0.5).astype(int)

fpr1, tpr1, _ = roc_curve(y_test, probs1)
auc1 = auc(fpr1, tpr1)
f1_1 = precision_recall_fscore_support(y_test, preds1, average="macro")[2]
print(f"Model 1 — Macro F1: {f1_1:.4f}  |  AUC: {auc1:.4f}")

print("Training Model 2: MLP")
num_features = X.shape[1]

model2 = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
)
optimiser2 = torch.optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-4)

dataset = TensorDataset(X, y)
loader= DataLoader(dataset, batch_size=256, shuffle=True)

loss_history_m2 = []
for i in range(num_epochs):
    model2.train()
    for X_batch, y_batch in loader:
        optimiser2.zero_grad()
        preds_batch = model2(X_batch)
        loss        = criterion(preds_batch, y_batch)
        loss.backward()
        optimiser2.step()
    loss_history_m2.append(loss.item())

plt.figure()
plt.plot(range(1, num_epochs), loss_history_m2[1:])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model 2 Training Loss")
plt.savefig("loss_model2.png", dpi=300)
plt.close()

model2.eval()
with torch.no_grad():
    logits2 = model2(X_test)
    probs2  = torch.sigmoid(logits2).numpy().flatten()
    preds2  = (probs2 >= 0.5).astype(int)

fpr2, tpr2, _ = roc_curve(y_test, probs2)
auc2 = auc(fpr2, tpr2)
f1_2 = precision_recall_fscore_support(y_test, preds2, average="macro")[2]
print(f"Model 2 — Macro F1: {f1_2:.4f}  |  AUC: {auc2:.4f}")

print("Bootstrap AUC Comparison...")

def bootstrap_auc_diff(probs_a, probs_b, y_true, num_samples=1000, seed=42):
    """
    Bootstrap the difference in AUC between two models (model_a - model_b).

    Returns
    -------
    observed_diff: AUC(a) - AUC(b) on the original test set
    mean_diff: mean bootstrapped difference
    ci_lower: 2.5th percentile of bootstrapped differences
    ci_upper:97.5th percentile of bootstrapped differences
    p_value: two-sided p-value (H0: auc difference = 0)
    auc_diffs:full distribution of bootstrapped differences
    """
    rng = np.random.default_rng(seed)
    n= len(y_true)

    fpr_a, tpr_a, _ = roc_curve(y_true, probs_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, probs_b)
    observed_diff = auc(fpr_a, tpr_a) - auc(fpr_b, tpr_b)

    auc_diffs = []
    for _ in range(num_samples):
        idx = rng.choice(n, size=n, replace=True)
        y_bs, pa_bs, pb_bs = y_true[idx], probs_a[idx], probs_b[idx]

        #skip samples with only one class (AUC undefined)
        if len(np.unique(y_bs)) < 2:
            continue

        fpr_a_bs, tpr_a_bs, _ = roc_curve(y_bs, pa_bs)
        fpr_b_bs, tpr_b_bs, _ = roc_curve(y_bs, pb_bs)
        auc_diffs.append(auc(fpr_a_bs, tpr_a_bs) - auc(fpr_b_bs, tpr_b_bs))

    auc_diffs = np.array(auc_diffs)

    shifted = auc_diffs - np.mean(auc_diffs)
    p_value = np.mean(np.abs(shifted) >= np.abs(observed_diff))

    ci_lower= np.percentile(auc_diffs, 2.5)
    ci_upper = np.percentile(auc_diffs, 97.5)

    return observed_diff, np.mean(auc_diffs), ci_lower, ci_upper, p_value, auc_diffs


obs_diff, mean_diff, ci_lo, ci_hi, p_val, diff_dist = bootstrap_auc_diff(
    probs1, probs2, y_test, num_samples=1000
)

print(f"\nModel 1 AUC: {auc1:.4f}")
print(f"Model 2 AUC: {auc2:.4f}")
print(f"Observed AUC difference (M1 - M2): {obs_diff:+.4f}")
print(f"Bootstrap mean difference: {mean_diff:+.4f}")
print(f"95% CI of difference: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"Two-sided p-value: {p_val:.4f}")

alpha = 0.05
if p_val < alpha:
    print(f"\nConclusion: The AUC difference is statistically significant (p < {alpha}).")
    winner= "Model 1" if obs_diff > 0 else "Model 2"
    print(f"  {winner} has a significantly higher AUC.")
else:
    print(f"\nConclusion: No statistically significant difference in AUC (p >= {alpha}).")

