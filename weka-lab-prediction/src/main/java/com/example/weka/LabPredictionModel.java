package com.example.weka;

import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

public class LabPredictionModel {

    public static void main(String[] args) throws Exception {

        String csvPath   = "data/lab_data.csv";
        String arffPath  = "data/lab_data.arff";
        String modelPath = "models/best_model.model";

        // ── 1. Load & Preprocess ──────────────────────────
        Instances data = DataProcessing.loadAndPreprocess(csvPath, arffPath);

        // ── 2. Set class attribute ────────────────────────
        // Change "Booking_Status" to whatever your target column is named
        data = DataProcessing.setClassAttribute(data, "Booking Status");

        // ── 3. Exploratory Analysis ───────────────────────
        System.out.println("\n=== Exploratory Analysis ===");
        System.out.println("Total instances : " + data.numInstances());
        System.out.println("Total attributes: " + data.numAttributes());
        System.out.println("Class attribute : " + data.classAttribute().name());
        System.out.println("Class distribution:");
        for (int i = 0; i < data.classAttribute().numValues(); i++) {
            System.out.println("  " + data.classAttribute().value(i) + ": " + data.attributeStats(data.classIndex()).nominalCounts[i]);
        }

        // ── 4. Supervised Learning ────────────────────────
        Classifier best = ModelTraining.getBestClassifier(data);

        // ── 5. Unsupervised Learning ──────────────────────
        ModelTraining.runKMeans(data);

        // ── 6. Association Rules ──────────────────────────
        // Discretize for Apriori
        Discretize disc = new Discretize();
        disc.setInputFormat(data);
        Instances discData = Filter.useFilter(data, disc);
        ModelTraining.runApriori(discData);

        // ── 7. Save & Reload Model ────────────────────────
        SerializationHelperWrapper.saveModel(best, modelPath);
        Classifier reloaded = SerializationHelperWrapper.loadModel(modelPath);
        System.out.println("\nReloaded model type: " + reloaded.getClass().getSimpleName());

        System.out.println("\n✅ All steps completed successfully!");
    }
}