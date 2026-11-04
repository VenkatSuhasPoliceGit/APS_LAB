import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

data=load_breast_cancer()
print(data)

X=pd.DataFrame(
    data.data,
    columns=data.feature_names
)
print(X.head())
y=pd.Series(
    (data.target==0).astype(int),
    name="malignant"
)

print(y.value_counts())
print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("class names:", data.target_names)

class_counts=y.value_counts().sort_index()
class_distribution=pd.DataFrame({
    "Class":data.target_names,
    "Count":class_counts.values,
    "Probability": class_counts.values/len(y)
})

class_distribution.plot(
    x="Class",
    y="Count",
    kind="bar",
    legend=False,
    color=["tomato", "steelblue"]
)

plt.ylabel("Number of observatins")
plt.title("Class Distributions")
plt.xticks(rotation=0)
plt.show()

X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training size:", len(y_train))
print("Testing size:", len(y_test))

print("Training proportions:")
print(y_train.value_counts(normalize=True).sort_index())

print("Testing proportions:")
print(y_test.value_counts(normalize=True).sort_index())


from sklearn.linear_model import LogisticRegression
model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)


probabilities=model.predict_proba(X_test)

y_pred=model.predict(X_test)

print(probabilities[:5].sum(axis=1))

results=pd.DataFrame({
    "Actual Class": y_test.values,
    "P_malignant": probabilities[:, 0],
    "P_benign": probabilities[:, 1]
    
})

print(results.head(10))


threshold=0.5

results["Predicted_malignant"]=(
    results["P_malignant"]>=threshold
).astype(int)

results["predicted_label"]=results[
    "Predicted_malignant"
].map({
    0: "Malignant",
    1:"Benign"
})

print(results[
    [
        "Actual Class",
        "P_malignant",
        "predicted_label"
    ]
].head(10))

for threshold in [0.30, 0.50, 0.70]:
  predictions=(
      results["P_malignant"]>=threshold
  ).astype(int)

  print(
      f"Threshold= {threshold} : "
      f"Predicted Malignant cases: {predictions.sum()}")

actual_malignant=(y_test.values==0).astype(int)

for threshold in [0.30, 0.50, 0.70]:
  predicted_malignant=(
      probabilities[:, 0]>=threshold
  ).astype(int)

  cm=confusion_matrix(
      actual_malignant,
      predicted_malignant
  )

  print(f"\n Threshold={threshold}")
  print(cm)

from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score
)

metric_results=[]

for threshold in [0.10, 0.30, 0.50, 0.70, 0.90]:
  y_pred_threshold=(
      probabilities[:, 1]>=threshold
  ).astype(int)

  tn, fp, fn, tp=confusion_matrix(
      y_test,
      y_pred_threshold,
      labels=[0,1]
  ).ravel()

  accuracy=(tp+tn)/(tp+tn+fp+fn)
  precision=tp/(tp+fp) if (tp+fp)>0 else 0
  sensitivity=tp/(tp+fn) if (tp+fn)>0 else 0
  specificity=tn/(tn+fp) if (tn+fp)>0 else 0

  f1=(
      2*precision*sensitivity/
      (precision+sensitivity)
      if (precision+sensitivity)>0
      else 0
  )

  metric_results.append({
      "Threshold": threshold,
      "TN":tn, "FP":fp, "FN":fn, "TP":tp, "Accuracy":accuracy, "Precision":precision, "Sensitivity":sensitivity, "Specificity":specificity, "F1_score":f1
  })

threshold_metrics=pd.DataFrame(metric_results)
print(threshold_metrics.round(2))

fpr, tpr, thresholds=roc_curve(
    y_test, probabilities[:, 1]
)

auc_score=roc_auc_score(
    y_test, probabilities[:, 1]
)

roc_results=pd.DataFrame({
    "Threshold": thresholds,
    "False positive rates": fpr,
    "True positive rates": tpr
})

print("roc values:\n ", roc_results)

plt.plot(fpr, tpr, label=f"Logistic Regression (AUC ={auc_score:.4f})")
plt.plot( [0, 1], [0, 1], linestyle="--", label="Random Classifier" )

plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve - Customer Churn Prediction") 
plt.legend() 
plt.grid() 
plt.show()

print(f"AUC Score: {auc_score:.4f}")