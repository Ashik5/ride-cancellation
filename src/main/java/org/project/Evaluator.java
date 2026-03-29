package org.project;

import weka.classifiers.Evaluation;
import weka.core.SerializationHelper;

import java.io.File;
import java.util.Map;

/**
 * Evaluator — Steps 7 & 8: Detailed evaluation and model persistence.
 */
public class Evaluator {

    /** Prints per-class precision/recall/F1 and confusion matrices. */
    public static void printDetailedEvaluation(Map<String, Evaluation> evaluations) {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 7 ▸ DETAILED EVALUATION");
        System.out.println("═══════════════════════════════════════════════════════");

        for (var entry : evaluations.entrySet()) {
            System.out.printf("%n  ▸ %s%n", entry.getKey());
            System.out.println("  ─────────────────────────────────────────────");
            try {
                System.out.println(entry.getValue().toSummaryString("    ", false));
                System.out.println(entry.getValue().toClassDetailsString("    "));
                System.out.println("    Confusion Matrix:");
                System.out.println(entry.getValue().toMatrixString("    "));
            } catch (Exception e) {
                System.out.println("    [Error: " + e.getMessage() + "]");
            }
        }
    }

    /** Saves the champion classifier to disk via SerializationHelper. */
    public static void saveModel(weka.classifiers.Classifier model,
                                 String name, String path) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 8 ▸ MODEL PERSISTENCE");
        System.out.println("═══════════════════════════════════════════════════════");

        new File(path).getParentFile().mkdirs();
        SerializationHelper.write(path, model);

        System.out.printf("  ✓ Champion   : %s%n", name);
        System.out.printf("  ✓ Saved to   : %s%n", path);
        System.out.printf("  ✓ File size  : %.2f KB%n", new File(path).length() / 1024.0);
        System.out.println();
    }
}
