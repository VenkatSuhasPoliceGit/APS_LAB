import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve


dataset=pd.read_csv(r"C:\Users\PRANEETH\aps_logistic_reg\Telco-Customer-Churn.csv")
print(dataset)


print("Dataset missing information:", dataset.isna().sum())

print("Shape:", dataset.shape)
print("Columns:", dataset.columns)

#convert total charges to numeric

dataset["TotalCharges"]=pd.to_numeric(dataset["TotalCharges"], errors="coerce")
print("TotalCharges missing values:", dataset["TotalCharges"].isna().sum())

#drop missing values

dataset=dataset.dropna()
print("Shape of dataset:", dataset.shape)

#removing custId as it is not an imp feature
dataset=dataset.drop("customerID", axis=1)

dataset["Churn"]=(dataset["Churn"]=="Yes").astype(int)
print("Churn values:", dataset["Churn"].value_counts())

X=dataset.drop("Churn", axis=1)
print(X)

y=dataset["Churn"]
print(y.head())


print("X shape:", X.shape)
print("y shape:", y.shape)

X=pd.get_dummies(X, drop_first=True, dtype=int)
print("Encoded X:", X.head())
print("Encoded X shape:", X.shape)

class_counts=y.value_counts().sort_index()
print(class_counts)

#Class distribution

class_distribution=pd.DataFrame({
    "Class":["No churn", "churn"],
    "Count": class_counts.values,
    "Probability": class_counts.values/len(y)
})

print("class distribution: ",class_distribution)

class_distribution.plot(x="Class", y="Count", kind="bar", legend=False, color=["steelblue", "tomato"])
plt.ylabel("Number of customers")
plt.xlabel("Customer class")
plt.title("Customer churn class distribution")
plt.xticks(rotation=0)
plt.show()


# TRAIN-TEST-SPLIT
X_train, X_test, y_train, y_test=train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training size:", len(y_train))
print("Testing sixe:", len(y_test))

print("Training proportions:", y_train.value_counts(normalize=True).sort_index())
print("Testing proportions:", y_test.value_counts(normalize=True).sort_index())

from sklearn.linear_model import LogisticRegression

model=make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)

)

model.fit(X_train, y_train)

probabilities=model.predict_proba(X_test)
print(probabilities[0:5])

y_pred=model.predict(X_test)

results=pd.DataFrame({
    "Actual values": y_test.values,
    "P_no_churn": probabilities[:, 0],
    "P_churn": probabilities[:, 1]
})

print("predictions:")
print(results.head(10))

#prediction with threshold

for threshold in [0.30, 0.50, 0.70]: 
    predictions = ( results["P_churn"] >= threshold ).astype(int) 
    print( f"\nThreshold = {threshold}" ) 
    print( "Predicted Churn customers:", predictions.sum() )

#confusion matrix
for threshold in [0.30, 0.50, 0.70]: 
    predicted_churn = ( probabilities[:, 1] >= threshold ).astype(int) 
    cm = confusion_matrix( y_test, predicted_churn ) 
    print( f"\nConfusion Matrix - Threshold {threshold}" ) 
    print(cm)


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
accuracy = accuracy_score( y_test, y_pred )
print("Accuracy: ", accuracy)

precision = precision_score( y_test, y_pred )
print("precision: ", precision)

recall=recall_score( y_test, y_pred )
print("Recall: ", recall)

f1 = f1_score( y_test, y_pred ) 
print("f1 score: ", f1)


cm=confusion_matrix(
    y_test, y_pred
)
print("Confusion matrix: ", cm)

metric_results=[]

for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    y_pred_threshold=(probabilities[:, 1]>=threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred_threshold, labels=[0,1]

    ).ravel()

    accuracy=(tp+tn)/(tp+tn+fp+fn)
    precison=tp/(tp+fp) if (tp+fp)>0 else 0
    sensitivity=tp/(tp+fn) if (tp+fn)>0 else 0
    specificity=tn/(tn+fp) if (tn+fp)>0 else 0
    f1 = ( 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0 )
    metric_results.append({ "Threshold": threshold, "TN": tn, "FP": fp, "FN": fn, "TP": tp, "Accuracy": accuracy, "Precision": precision, "Sensitivity": sensitivity, "Specificity": specificity, "F1_score": f1 })

threshold_metrics = pd.DataFrame( metric_results ) 
print("\nThreshold Metrics:") 
print( threshold_metrics.round(2) )
print()


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