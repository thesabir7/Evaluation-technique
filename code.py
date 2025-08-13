from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# For ROC, we’ll make it a binary classification problem (digit '0' vs rest)
y_binary = (y == 0).astype(int)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# 3. Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Cross-validation
cv_scores = cross_val_score(model, X, y_binary, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Average CV score:", np.mean(cv_scores))

# 5. Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. ROC-AUC
y_prob = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
