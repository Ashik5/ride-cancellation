package com.example.weka;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.StringToNominal;

import java.io.*;
import java.nio.file.*;

public class DataProcessing {

    public static Instances loadAndPreprocess(String csvPath, String arffPath) throws Exception {

        // ── Pre-clean CSV: strip extra quotes ──────────────
        System.out.println("=== STEP 0: Pre-cleaning CSV ===");
        String cleanedPath = csvPath.replace(".csv", "_cleaned.csv");
        cleanCsv(csvPath, cleanedPath);

        // ── Load cleaned CSV ───────────────────────────────
        System.out.println("=== STEP 1: Loading CSV ===");
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(cleanedPath));
        Instances data = loader.getDataSet();
        System.out.println("Loaded " + data.numInstances() + " instances, "
                + data.numAttributes() + " attributes.");

        // ── Replace missing values ─────────────────────────
        System.out.println("=== STEP 2: Replacing Missing Values ===");
        ReplaceMissingValues rmv = new ReplaceMissingValues();
        rmv.setInputFormat(data);
        data = Filter.useFilter(data, rmv);

        // ── StringToNominal ────────────────────────────────
        System.out.println("=== STEP 3: StringToNominal ===");
        StringToNominal stn = new StringToNominal();
        stn.setAttributeRange("first-last");
        stn.setInputFormat(data);
        data = Filter.useFilter(data, stn);

        // ── Remove ID/Date columns (1=Date,2=Time,3=BookingID,5=CustomerID) ──
        System.out.println("=== STEP 4: Removing irrelevant columns ===");
        Remove remove = new Remove();
        remove.setAttributeIndices("1,2,3,5");
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);
        System.out.println("Remaining attributes: " + data.numAttributes());

        // ── Save ARFF ──────────────────────────────────────
        System.out.println("=== STEP 5: Saving ARFF ===");
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(arffPath));
        saver.writeBatch();
        System.out.println("ARFF saved to: " + arffPath);

        return data;
    }

    private static void cleanCsv(String inputPath, String outputPath) throws Exception {
        BufferedReader reader = new BufferedReader(new FileReader(inputPath));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath));
        String line;
        while ((line = reader.readLine()) != null) {
            // Remove triple quotes and extra quotes around fields
            line = line.replaceAll("\"\"\"", "");
            line = line.replaceAll("\"\"", "");
            writer.write(line);
            writer.newLine();
        }
        reader.close();
        writer.close();
        System.out.println("Cleaned CSV saved to: " + outputPath);
    }

    public static Instances setClassAttribute(Instances data, String className) throws Exception {
        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).name().equalsIgnoreCase(className)) {
                data.setClassIndex(i);
                System.out.println("Class attribute set to: " + className);
                return data;
            }
        }
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Class attribute defaulted to last: "
                + data.classAttribute().name());
        return data;
    }
}