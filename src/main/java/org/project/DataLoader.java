package org.project;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;

import java.io.File;

/**
 * DataLoader — Step 1 of the pipeline.
 *
 * Loads the raw CSV dataset into WEKA's native {@link Instances} format
 * and optionally persists it as an ARFF file for reproducibility.
 */
public class DataLoader {

    /**
     * Loads a CSV file and returns it as a WEKA Instances object.
     *
     * @param csvPath path to the input CSV (e.g. "data/rides.csv")
     * @return loaded Instances
     * @throws Exception if the file is missing or malformed
     */
    public static Instances loadCSV(String csvPath) throws Exception {
        System.out.println("═══════════════════════════════════════════════════════");
        System.out.println("  STEP 1 ▸ DATA LOADING");
        System.out.println("═══════════════════════════════════════════════════════");

        File csvFile = new File(csvPath);
        if (!csvFile.exists()) {
            throw new RuntimeException(
                "Dataset not found: " + csvPath +
                "\n  → Place your rides.csv inside the data/ directory and re-run.");
        }

        CSVLoader loader = new CSVLoader();
        loader.setSource(csvFile);
        Instances data = loader.getDataSet();

        System.out.printf("  ✓ Loaded  : %s%n", csvFile.getName());
        System.out.printf("  ✓ Instances: %,d%n", data.numInstances());
        System.out.printf("  ✓ Attributes: %d  →  [", data.numAttributes());
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i > 0) System.out.print(", ");
            System.out.print(data.attribute(i).name());
        }
        System.out.println("]");
        System.out.println();

        return data;
    }

    /**
     * Saves a processed Instances object as an ARFF file.
     *
     * @param data     the Instances to persist
     * @param arffPath output path (e.g. "data/rides.arff")
     * @throws Exception on I/O failure
     */
    public static void saveARFF(Instances data, String arffPath) throws Exception {
        File outFile = new File(arffPath);
        outFile.getParentFile().mkdirs();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(outFile);
        saver.writeBatch();

        System.out.printf("  ✓ ARFF saved to: %s (%.1f KB)%n%n", arffPath,
                outFile.length() / 1024.0);
    }
}
