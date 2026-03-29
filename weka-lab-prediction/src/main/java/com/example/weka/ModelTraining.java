package com.example.weka;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.clusterers.SimpleKMeans;
import weka.associations.Apriori;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.Random;

public class ModelTraining {

    // ── Supervised Learning ──────────────────────────────
    public static Evaluation trainAndEvaluate(Classifier classifier, Instances data, String name) throws Exception {
        System.out.println("\n=== Classifier: " + name + " ===");
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        System.out.printf("Accuracy     : %.2f%%%n", eval.pctCorrect());
        System.out.printf("Precision    : %.4f%n", eval.weightedPrecision());
        System.out.printf("Recall       : %.4f%n", eval.weightedRecall());
        System.out.printf("F1-Score     : %.4f%n", eval.weightedFMeasure());
        System.out.printf("ROC AUC      : %.4f%n", eval.weightedAreaUnderROC());
        System.out.println("Confusion Matrix:");
        double[][] cm = eval.confusionMatrix();
        for (double[] row : cm) {
            for (double v : row) System.out.printf("%8.0f", v);
            System.out.println();
        }
        return eval;
    }

    public static Classifier getBestClassifier(Instances data) throws Exception {
    J48 j48 = new J48();
    RandomForest rf = new RandomForest();
    NaiveBayes nb = new NaiveBayes();

    Evaluation e1 = trainAndEvaluate(j48, data, "J48 Decision Tree");
    Evaluation e2 = trainAndEvaluate(rf, data, "Random Forest");
    Evaluation e3 = trainAndEvaluate(nb, data, "Naive Bayes");

    double best = Math.max(e1.pctCorrect(), Math.max(e2.pctCorrect(), e3.pctCorrect()));
    if (best == e2.pctCorrect()) { rf.buildClassifier(data); System.out.println("\nBest: Random Forest"); return rf; }
    if (best == e3.pctCorrect()) { nb.buildClassifier(data); System.out.println("\nBest: Naive Bayes"); return nb; }
    j48.buildClassifier(data); System.out.println("\nBest: J48"); return j48;
}

    // ── Unsupervised Learning ─────────────────────────────
    public static void runKMeans(Instances data) throws Exception {
        System.out.println("\n=== K-Means Clustering ===");
        // Remove class attribute for clustering
        Remove remove = new Remove();
        remove.setAttributeIndices("" + (data.classIndex() + 1));
        remove.setInputFormat(data);
        Instances clusterData = Filter.useFilter(data, remove);

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(3);
        kmeans.setSeed(10);
        kmeans.buildClusterer(clusterData);
        System.out.println(kmeans);
    }

    // ── Association Rule Mining ───────────────────────────
    public static void runApriori(Instances data) throws Exception {
        System.out.println("\n=== Apriori Association Rules ===");
        // Apriori needs all nominal - remove numeric class, use nominal only
        Apriori apriori = new Apriori();
        apriori.setNumRules(10);
        apriori.buildAssociations(data);
        System.out.println(apriori);
    }
}