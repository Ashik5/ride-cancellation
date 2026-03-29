package org.project;

import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Clusterer — Step 5: Unsupervised Learning.
 * Runs SimpleKMeans with k = {2, 3, 4} and prints cluster profiles.
 */
public class Clusterer {

    public static void run(Instances data) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 5 ▸ UNSUPERVISED LEARNING  (KMeans)");
        System.out.println("═══════════════════════════════════════════════════════");

        // Remove class attribute for unsupervised learning
        Instances unlabeled = new Instances(data);
        if (unlabeled.classIndex() >= 0) {
            Remove rm = new Remove();
            rm.setAttributeIndices(String.valueOf(unlabeled.classIndex() + 1));
            rm.setInputFormat(unlabeled);
            unlabeled = Filter.useFilter(unlabeled, rm);
        }

        // Try multiple k values
        int[] kValues = {2, 3, 4};
        for (int k : kValues) {
            SimpleKMeans km = new SimpleKMeans();
            km.setNumClusters(k);
            km.setSeed(42);
            km.setPreserveInstancesOrder(true);
            km.buildClusterer(unlabeled);

            System.out.printf("%n  ▸ K = %d  |  SSE = %.4f%n", k, km.getSquaredError());
            int[] assignments = km.getAssignments();
            int[] sizes = new int[k];
            for (int a : assignments) sizes[a]++;
            for (int c = 0; c < k; c++) {
                System.out.printf("    Cluster %d : %,d instances (%.1f%%)%n",
                        c, sizes[c], 100.0 * sizes[c] / assignments.length);
            }
        }

        // Print centroids for k=3
        SimpleKMeans primary = new SimpleKMeans();
        primary.setNumClusters(3);
        primary.setSeed(42);
        primary.buildClusterer(unlabeled);
        System.out.println("\n  ▸ Cluster Centroids (k=3):");
        Instances centroids = primary.getClusterCentroids();
        for (int c = 0; c < centroids.numInstances(); c++) {
            System.out.printf("    Centroid %d: %s%n", c, centroids.instance(c));
        }
        System.out.println();
    }
}
