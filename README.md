# Ride Cancellation Prediction Pipeline

WEKA-based data mining pipeline for predictive analysis of ride cancellations.  
Runs **J48**, **RandomForest**, **SMO**, **KMeans**, and **Apriori** on a ride-booking dataset.

---

## Prerequisites

| Tool | Version | Download |
|------|---------|----------|
| Java JDK | 17+ | [adoptium.net](https://adoptium.net/) |
| Apache Maven | 3.8+ | [maven.apache.org](https://maven.apache.org/download.cgi) |

### Windows Setup

1. **Install JDK 17**  
   Download and install from [Adoptium](https://adoptium.net/).  
   After install, verify:
   ```cmd
   java -version
   ```

2. **Install Maven**  
   - Download the **Binary zip** from [maven.apache.org](https://maven.apache.org/download.cgi)
   - Extract to `C:\apache-maven-3.9.x`
   - Add to **System PATH**:
     - Open **System Properties → Environment Variables**
     - Under **System Variables**, edit `Path` and add: `C:\apache-maven-3.9.x\bin`
   - Verify:
     ```cmd
     mvn -version
     ```

3. **Set JAVA_HOME** (if not auto-detected)
   - Add a new System Variable:
     - Name: `JAVA_HOME`
     - Value: `C:\Program Files\Eclipse Adoptium\jdk-17.x.x` (your JDK path)

---

## Running the Pipeline

### Step 1: Clone the repo
```cmd
git clone https://github.com/Ashik5/ride-cancellation.git
cd ride-cancellation
```

### Step 2: Place the dataset
Download `rides.csv` from Kaggle and place it in the `data/` folder:
```
ride-cancellation/
└── data/
    └── rides.csv
```

### Step 3: Build and Run

**On Windows (CMD):**
```cmd
mvn clean compile exec:exec
```

**On macOS / Linux:**
```bash
mvn clean compile exec:exec
```

> The `exec:exec` goal forks a dedicated JVM with **4 GB heap** (`-Xmx4g`) to handle the 150k-row dataset.

### Step 4: Check Output
After the pipeline completes, you will see:
- **Console**: Full pipeline output (EDA stats, cluster centroids, association rules, classification results table)
- **`data/rides.arff`**: Converted ARFF dataset
- **`models/best_model.model`**: The best-performing serialized classifier

---

## Project Structure

```
ride-cancellation/
├── data/
│   └── rides.csv               # Original Kaggle dataset (place here)
├── models/
│   └── best_model.model        # Persisted trained model (generated)
├── src/main/java/org/project/
│   ├── Main.java               # Pipeline orchestrator
│   ├── DataLoader.java         # CSV → ARFF conversion
│   ├── Preprocessor.java       # Feature selection & filtering
│   ├── Explorer.java           # Statistical exploration
│   ├── Classifier.java         # J48, RandomForest, SMO
│   ├── Clusterer.java          # KMeans clustering
│   ├── Associator.java         # Apriori rule mining
│   └── Evaluator.java          # Evaluation & model persistence
├── pom.xml                     # Maven build config
└── project.md                  # Pipeline design document
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `mvn` not recognized | Add Maven's `bin/` folder to your system `PATH` |
| `java` not recognized | Install JDK 17 and set `JAVA_HOME` |
| `OutOfMemoryError` | The `exec:exec` goal already allocates 4 GB. If still failing, edit `pom.xml` and change `-Xmx4g` to `-Xmx6g` |
| `Dataset not found` | Make sure `rides.csv` is inside the `data/` folder |
