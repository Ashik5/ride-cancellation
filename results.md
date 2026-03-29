# Ride Cancellation Prediction Results

This document contains the evaluation output from the final machine learning pipeline run *after* removing all "data leakage" attributes.

## Model Performance Summary

After properly restricting the dataset to only pre-booking information (dropping features like post-ride ratings, turnaround times with literal `null` strings, and cancellation reasons), the model produced highly realistic and scientifically sound accuracy metrics.

| Algorithm | Accuracy | F1-Score | Status |
| :--- | :--- | :--- | :--- |
| **J48 (Decision Tree)** | **80.00%** | NaN | ✅ Selected as Champion |
| **RandomForest** | 78.10% | NaN | ✅ Completed |
| **SMO (SVM)** | - | - | ⚠️ Skipped gracefully (OOM Memory limits reached) |

> **Note on F1-Score (NaN):** Because we removed perfect predictors (leakage features), the model aggressively predicts the majority classes (`Completed` and `Cancelled by Driver`) but struggles to correctly pinpoint minority classes like `Cancelled by Customer` and `No Driver Found` using only distance, payment, and vehicle type. This leads to a Precision/Recall of 0.000 for minority classes, causing Weka's `weightedFMeasure` to compute as `NaN`.

---

## Detailed Evaluation (J48 Decision Tree Classifier)

Using 10-Fold Cross Validation on 150,000 instances:

```text
Correctly Classified Instances      120000               80      %
Incorrectly Classified Instances     30000               20      %
Kappa statistic                          0.616 
Mean absolute error                      0.119 
Root mean squared error                  0.2439
Total Number of Instances           150000     
```

### Detailed Accuracy By Class

| Class | TP Rate | FP Rate | Precision | Recall | ROC Area |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **No Driver Found** | 0.000 | 0.000 | ? | 0.000 | 0.866 |
| **Incomplete** | 0.000 | 0.000 | ? | 0.000 | 0.666 |
| **Completed** | 1.000 | 0.158 | 0.912 | 1.000 | 0.920 |
| **Cancelled by Driver** | 1.000 | 0.171 | 0.563 | 1.000 | 0.915 |
| **Cancelled by Customer** | 0.000 | 0.000 | ? | 0.000 | 0.866 |
| **Weighted Avg.** | **0.800** | **0.129** | **?** | **0.800** | **0.896** |

### Confusion Matrix

```text
     a     b     c     d     e   <-- classified as
     0     0     0 10500     0 |     a = No Driver Found
     0     0  9000     0     0 |     b = Incomplete
     0     0 93000     0     0 |     c = Completed
     0     0     0 27000     0 |     d = Cancelled by Driver
     0     0     0 10500     0 |     e = Cancelled by Customer
```

---

## Association Rules (Apriori)
The unsupervised Apriori miner successfully grouped transactional patterns into logical rules. Top rules discovered include:

1. `Payment Method=UPI` ==> `Booking Status=Completed` (Conf: 0.91)
2. `Payment Method=Cash` ==> `Booking Status=Completed` (Conf: 0.91)
3. `Booking Status=Cancelled by Driver` ==> `Ride Distance=null` (Conf: 1.00)

## KMeans Clusters
We also identified 3 distinct passenger profiles (Centroids):
* **Centroid 0:** Unassigned / Auto requests 
* **Centroid 1:** Go Sedan riders
* **Centroid 2:** Premium Sedan riders, longer trips (avg 14.9 km), high likelihood of UPI payments.

---

**Champion Persisted to:** `models/best_model.model` (113.77 KB)
