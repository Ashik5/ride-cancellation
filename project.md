# Predictive Analysis of Ride Cancellation Pipeline

## 1. Clean Java Project Structure

```text
ride-cancellation-weka/
├── data/
│   ├── rides.csv               # Original Kaggle dataset
│   └── rides.arff              # Converted ARFF dataset ready for WEKA
├── lib/
│   └── weka.jar                # WEKA library dependency
├── models/
│   └── best_model.model        # Persisted trained model
├── src/
│   └── org/project/
│       ├── Main.java           # Main execution workflow
│       ├── DataLoader.java     # CSV to ARFF conversion logic
│       ├── Preprocessor.java   # Feature selection and filtering
│       ├── Explorer.java       # Basic statistical exploration
│       ├── Classifier.java     # Supervised algorithms execution
│       ├── Clusterer.java      # KMeans execution
│       ├── Associator.java     # Apriori execution
│       └── Evaluator.java      # Evaluation and Serialization logic
└── README.md
```

## 2. Step-by-Step Pipeline

### Step 1: Data Loading
* **Purpose**: Load the raw CSV dataset into WEKA's native `Instances` format.
* **WEKA Class**: `weka.core.converters.CSVLoader` and `weka.core.converters.ArffSaver`.
* **Input/Output**: Input is `rides.csv`. Output is an `Instances` object in memory (and optionally saved as `rides.arff`).

### Step 2: Data Preprocessing
* **Purpose**: Clean data, remove attributes causing data leakage (like `cancellation reason`), handle missing values, and convert numeric fields to nominal if required by specific algorithms.
* **WEKA Class**: `weka.filters.unsupervised.attribute.Remove` (to drop columns), `weka.filters.unsupervised.attribute.NumericToNominal` (for Apriori).
* **Input/Output**: Input is raw `Instances`. Output is preprocessed `Instances`.

### Step 3: Exploratory Analysis
* **Purpose**: Extract basic statistics for individual attributes to understand data distribution.
* **WEKA Class**: `weka.core.AttributeStats` and `weka.core.Instances.attributeStats()`.
* **Input/Output**: Input is preprocessed `Instances`. Output consists of printed summary statistics (e.g., mean, count, nominal frequencies).

### Step 4: Supervised Learning
* **Purpose**: Train predictive classification models.
* **WEKA Class**: `weka.classifiers.trees.J48`, `weka.classifiers.trees.RandomForest`, `weka.classifiers.functions.SMO`.
* **Input/Output**: Input is preprocessed `Instances` (with class index set). Output is a trained model.

### Step 5: Unsupervised Learning
* **Purpose**: Discover natural groupings in the ride data without using the class label.
* **WEKA Class**: `weka.clusterers.SimpleKMeans`.
* **Input/Output**: Input is preprocessed `Instances` (class attribute ignored). Output is cluster assignments and centroid definitions.

### Step 6: Association Rule Mining
* **Purpose**: Discover frequent patterns and relationships between factors (e.g., rainy + far distance -> highest cancellation).
* **WEKA Class**: `weka.associations.Apriori`.
* **Input/Output**: Input is preprocessed `Instances` (all numeric attributes discretized). Output is a list of top association rules.

### Step 7: Model Evaluation
* **Purpose**: Assess the performance of classification models using hold-out or cross-validation.
* **WEKA Class**: `weka.classifiers.Evaluation`.
* **Input/Output**: Input is a trained classifier and test `Instances`. Output is evaluation metrics.

### Step 8: Model Saving and Reloading
* **Purpose**: Persist the trained model to disk so it can be deployed without retraining.
* **WEKA Class**: `weka.core.SerializationHelper`.
* **Input/Output**: Input is your trained WEKA classifier. Output is a `.model` binary file on disk.

## 3. Recommended Features for Classification

* **Target Variable**: `booking status` (e.g., Cancelled vs. Completed). Ensure the class index is correctly set to this attribute.
* **Features to Keep**: `ride distance`, `driver rating`, `customer rating`, `waiting time`, `payment method`, `vehicle type`.
* **Features to Drop (CRITICAL)**:
  * `cancellation reason`: Direct target leakage (this is only known *after* a ride is cancelled).
  * `booking timestamps`: Raw timestamps confuse standard classifiers. Can be kept only if transformed into `hour of day` or `day of week`.

## 4. Algorithms Explained

* **J48 (C4.5 Decision Tree)**: Creates a highly interpretable, tree-based set of rules. Ideal for a student project as you can print the tree and directly explain exactly "why" rides are classified as cancelled.
* **RandomForest**: An ensemble of decision trees. Tends to have the highest accuracy and handles overfitting better than J48. Good for showing contrast between interpretable (J48) and performant (RandomForest) models.
* **SMO (Sequential Minimal Optimization / SVM)**: A robust algorithm that draws a hyperplane to separate completions from cancellations. Excellent for handling categorical and numerical mixes if appropriately normalized.
* **KMeans**: The standard clustering baseline. It's useful to see if the algorithm natively groups rides into recognizable profiles (e.g., "long wait cash rides" vs "short distance card rides").
* **Apriori**: Finds if-then rules. Extremely useful for actionable business insights (e.g., `payment method = cash` AND `waiting time = High` => `booking status = Cancelled`). Note: Apriori requires all attributes to be Nominal, so `NumericToNominal` or `Discretize` filters must be applied first.

## 5. Simple Workflow Diagram Description

1. **Start**: Initialize `CSVLoader` to load `rides.csv` and convert it to WEKA `Instances`.
2. **Filter Data**: Apply `Remove` filter to drop `cancellation reason` and timestamps.
3. **Set Class Index**: Set the target variable to `booking status`.
4. **Branch 1 (EDA)**: Iterate through attributes using `AttributeStats` to print summary data.
5. **Branch 2 (Clustering)**: Remove class index temporarily to run `SimpleKMeans`, printing cluster centroids.
6. **Branch 3 (Rules)**: Apply numeric discretization filter, then run `Apriori` to print top 10 rules.
7. **Branch 4 (Classification)**: Split data into training (80%) and test (20%), or use 10-fold Cross Validation.
8. **Train & Evaluate**: Train `J48`, `RandomForest`, and `SMO`. Pass them to the `Evaluation` class.
9. **End**: Save the best performing classifier (e.g., `RandomForest`) using `SerializationHelper`.

## 6. Evaluation Metrics in WEKA

Using the `weka.classifiers.Evaluation` class, you should extract and report:
* **Accuracy (`eval.pctCorrect()`)**: Overall percentage of correct predictions.
* **Precision (`eval.precision(classIndex)`)**: When the model predicts "Cancelled", how often is it right?
* **Recall (`eval.recall(classIndex)`)**: Out of all actual "Cancelled" rides, how many did the model find?
* **F-Measure (`eval.fMeasure(classIndex)`)**: Harmonic mean of Precision and Recall.
* **Confusion Matrix (`eval.toMatrixString()`)**: Shows True Positives, False Positives, etc.

## 7. Persisting and Reloading the Model

To avoid retraining every time, persist the trained classifier:

**Saving the Model:**
```java
Classifier model = new RandomForest();
model.buildClassifier(trainingData);
SerializationHelper.write("models/best_model.model", model);
```

**Reloading the Model:**
```java
Classifier loadedModel = (Classifier) SerializationHelper.read("models/best_model.model");
// immediate use: loadedModel.classifyInstance(newInstance);
```
