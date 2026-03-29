package org.project;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * Classifier — Step 4: Supervised Learning.
 * Trains J48, RandomForest, SMO with 10-fold CV.
 */
public class Classifier {

    private final Map<String, weka.classifiers.Classifier> trainedModels = new LinkedHashMap<>();
    private final Map<String, Evaluation> evaluations = new LinkedHashMap<>();

    public void runAll(Instances data) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 4 ▸ SUPERVISED LEARNING  (10-Fold CV)");
        System.out.println("═══════════════════════════════════════════════════════");

        RandomForest rf = new RandomForest();
        rf.setNumIterations(5);  // 5 trees (default 100 causes OOM on 150k rows)
        rf.setMaxDepth(5);       // Shallow trees to control memory usage

        weka.classifiers.Classifier[] algos = { new J48(), rf, new SMO() };
        String[] labels = { "J48  (Decision Tree)", "RandomForest", "SMO  (SVM)" };

        for (int i = 0; i < algos.length; i++) {
            System.out.printf("%n  ▸ Training %s …%n", labels[i]);
            try {
                algos[i].buildClassifier(data);

                Evaluation eval = new Evaluation(data);
                eval.crossValidateModel(algos[i], data, 10, new Random(42));

                trainedModels.put(labels[i], algos[i]);
                evaluations.put(labels[i], eval);

                System.out.printf("    Accuracy: %.2f%%  |  F1: %.4f%n",
                        eval.pctCorrect(), eval.weightedFMeasure());
            } catch (OutOfMemoryError | Exception e) {
                System.out.printf("    ⚠ SKIPPED — %s (%s)%n", e.getClass().getSimpleName(), e.getMessage());
                System.gc(); // Reclaim memory for next classifier
            }
        }
        printResultsTable();
    }

    private void printResultsTable() {
        System.out.println();
        System.out.println("  ┌────────────────────────┬──────────┬──────────┬──────────┬──────────┐");
        System.out.println("  │ Algorithm              │ Accuracy │ Precisn  │ Recall   │ F1-Score │");
        System.out.println("  ├────────────────────────┼──────────┼──────────┼──────────┼──────────┤");
        for (var entry : evaluations.entrySet()) {
            try {
                Evaluation e = entry.getValue();
                System.out.printf(
                    "  │ %-22s │ %6.2f%% │ %7.4f │ %7.4f │ %7.4f │%n",
                    entry.getKey(), e.pctCorrect(), e.weightedPrecision(),
                    e.weightedRecall(), e.weightedFMeasure());
            } catch (OutOfMemoryError | Exception ex) {
                System.out.printf("  │ %-22s │  error   │  error   │  error   │  error   │%n", entry.getKey());
            }
        }
        System.out.println("  └────────────────────────┴──────────┴──────────┴──────────┴──────────┘");
        System.out.println();
    }

    public Map<String, weka.classifiers.Classifier> getTrainedModels() { return trainedModels; }
    public Map<String, Evaluation> getEvaluations() { return evaluations; }

    /** Returns the classifier with the highest weighted F1-Score. */
    public Map.Entry<String, weka.classifiers.Classifier> getChampion() {
        String bestName = null;
        double bestF1 = -1;
        for (var entry : evaluations.entrySet()) {
            double f1 = entry.getValue().weightedFMeasure();
            if (f1 > bestF1) { bestF1 = f1; bestName = entry.getKey(); }
        }
        return Map.entry(bestName, trainedModels.get(bestName));
    }
}
