# Project Summary: Ride Cancellation Prediction Pipeline

This document serves as a comprehensive overview of everything we have built for the **Ride Cancellation Prediction Pipeline**. It outlines the core objectives, the dataset features used, and a step-by-step breakdown of the machine learning pipeline implemented from scratch using Java and the WEKA API.

---

## 🎯 1. What We Are Actually Doing

The goal of this project is to build an end-to-end data mining and machine learning application to predict **ride cancellations** (i.e., whether a booked ride will be completed or canceled).

Instead of using Python (which is common for prototyping), we built a robust, production-style, heavily object-oriented **Java pipeline** utilizing the **WEKA Machine Learning API**. The pipeline is modular, cross-platform (managed via Maven), and handles everything from raw CSV ingestion to model persistence.

### Key Objectives Achieved:
1. **End-to-End Pipeline**: Fully automated steps spanning from data ingestion to model evaluation.
2. **Multiple Algorithms**: Implementing both Unsupervised (K-Means), Association Rule Mining (Apriori), and Supervised Learning (J48, Random Forest, SMO).
3. **Data Integrity & Memory Management**: Handling malformed CSV data, preventing "data leakage", and dealing with Java Heap Space (`OutOfMemoryError`) issues when processing 150,000+ records via cross-validation.

---

## 📊 2. Our Dataset & Features

We utilized a Kaggle dataset (`rides.csv`) containing 150,000 rows of ride-booking information. A major part of the data mining process involves selecting the *right* features and aggressively discarding bad or "leaky" features.

### A. The Target Variable
- **`Booking Status`**: The class label we are predicting.

### B. Features Kept for Prediction
These are the core attributes driving the predictions.
- **`Vehicle Type`**: (Categorical) E.g., Auto, Premier Sedan, Go Mini.
- **`Avg VTAT` / `Avg CTAT`**: (Numeric) Time-based metrics for the vehicle/customer.
- **`Booking Value` / `Ride Distance`**: (Numeric) The pricing and length of the requested trip.
- **`Driver Ratings` / `Customer Rating`**: (Numeric) Historical satisfaction scores.
- **`Payment Method`**: (Categorical) UPI, Cash, Credit Card, etc.

### C. Features Dropped (Data Leakage & Noise)
We aggressively filter out attributes using the WEKA `Remove` filter to prevent cheating (data leakage) or crashing the computer with high cardinality.
- **Data Leakage (Known only *after* cancellation):** `Reason for cancelling by Customer`, `Cancelled Rides by Customer`, `Driver Cancellation Reason`, `Incomplete Rides`, `Incomplete Rides Reason`. (If the model sees these, it is "cheating").
- **High Cardinality / Noise:** `Booking ID`, `Customer ID`, `Time`, `Pickup Location`, `Drop Location`. (Too many unique values will cause a massive memory explosion during one-hot encoding).

---

## ⚙️ 3. Step-by-Step Pipeline Execution

We split the logic into heavily modularized Java classes orchestrated by a central `Main.java` file. 

### Step 1: Data Loading (`DataLoader.java`)
- Loads the raw 150,000-row `rides.csv` dataset into WEKA `Instances`.
- Cleans and formats the data, then exports a standardized `rides.arff` file to be used directly by the WEKA ecosystem.

### Step 2: Data Preprocessing (`Preprocessor.java`)
- Programmatically drops the leakage and identifier columns mentioned above.
- Handles missing values by imputing them (WEKA's `ReplaceMissingValues` filter applies mean for numeric and mode for nominal data).
- Explicitly establishes the `Booking Status` column as the dataset's class index.

### Step 3: Exploratory Data Analysis (`Explorer.java`)
- Loops through the clean instances and prints a statistical summary to the terminal.
- Calculates and visualizes Distinct values, Missing percentages, Means, StdDevs, and value distributions for categorical columns.

### Step 4: Unsupervised Learning - Clustering (`Clusterer.java`)
- Uses the **K-Means Algorithm** (`SimpleKMeans` in WEKA) to discover hidden groupings in the passenger data without looking at the booking status.
- Evaluated over different values of K (K=2, 3, 4) while tracking the Sum of Squared Errors (SSE) to find optimal centroids.

### Step 5: Association Rule Mining (`Associator.java`)
- Uses the **Apriori Algorithm**.
- Automatically converts numeric variables into bins using WEKA's `Discretize` filter since Apriori only accepts nominal data.
- Discovers hidden causal rules (e.g., "If *Payment Method* is X and *Vehicle Type* is Y, then *Booking Status* is likely Z").

### Step 6: Supervised Learning & Cross-Validation (`Classifier.java`)
- Compares three distinct algorithms to predict cancellations:
  1. **J48**: A standard decision tree architecture.
  2. **Random Forest**: An ensemble method combining multiple decision trees. *(Note: Configured specifically with 5 trees and depth 5 to avoid out-of-memory crashes on 150k rows).*
  3. **SMO (Support Vector Machine)**: A powerful linear delimiter. Wraps execution in try-catch logic to gracefully skip if the algorithm exhausts JVM Heap Space.
- Evaluates them all rigorously using **10-Fold Cross-Validation** (averaging results across 10 randomized 90/10 data splits).

### Step 7: Final Evaluation & Output (`Evaluator.java`)
- Generates a cleanly formatted terminal table summarizing: **Accuracy, Precision, Recall, and F1-Score**.
- Isolates the "Champion" model (the algorithm that scored the highest Weighted F1-Score).
- Serializes and saves the champion model to `models/best_model.model` so it can be loaded later without retraining.

---

## 🚀 4. Technical Challenges Overcome

1. **Massive Memory Footprints (OOM Errors):**
   Trying to train a Random Forest on 150,000 records inside Java quickly hits limits. We solved this by:
   - Limiting the Depth and Iterations of the Forest.
   - Dropping high-cardinality nominal columns (locations, IDs).
   - Upgrading our Maven execution from an in-process build (`exec:java`) to a Forked JVM (`exec:exec`) with 4 gigabytes of dedicated RAM (`MAVEN_OPTS="-Xmx4g"`).
2. **Resilient Engineering:**
   We wrapped algorithms in dynamic `try-catch` blocks and called `System.gc()` ensuring that if one algorithm (like `SMO`) fails due to complexity, the pipeline gracefully recovers and prints results for the ones that succeeded (`J48`, `Random Forest`).
3. **Automated Git Management:**
   Configured a standard `.gitignore` but ensured the primary dataset (`rides.csv`) and `.model` binaries were actively committed so anyone pulling the repository can test the code immediately.
