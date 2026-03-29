package org.project;

import weka.core.Instances;
import java.util.Map;
import weka.classifiers.Evaluation;

public class Main {
    public static void main(String[] args) {
        try {
            // STEP 1: Data Loading
            Instances rawData = DataLoader.loadCSV("data/rides.csv");
            DataLoader.saveARFF(rawData, "data/rides.arff");
            
            // STEP 2: Data Preprocessing
            Instances cleanData = Preprocessor.preprocess(rawData);
            
            // STEP 3: Exploratory Analysis
            Explorer.explore(cleanData);
            
            // STEP 5: Unsupervised Learning
            Clusterer.run(cleanData);
            
            // STEP 6: Association Rule Mining
            Associator.run(cleanData);
            
            // STEP 4: Supervised Learning
            Classifier classifier = new Classifier();
            classifier.runAll(cleanData);
            
            // STEP 7: Model Evaluation
            Map<String, Evaluation> evaluations = classifier.getEvaluations();
            Evaluator.printDetailedEvaluation(evaluations);
            
            // STEP 8: Model Saving and Reloading
            Map.Entry<String, weka.classifiers.Classifier> champion = classifier.getChampion();
            if (champion != null) {
                Evaluator.saveModel(champion.getValue(), champion.getKey(), "models/best_model.model");
            } else {
                System.out.println("⚠ No champion model to save.");
            }
            
        } catch (Exception e) {
            System.err.println("Pipeline failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
