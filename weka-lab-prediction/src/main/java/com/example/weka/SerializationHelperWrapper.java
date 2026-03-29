package com.example.weka;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

public class SerializationHelperWrapper {

    public static void saveModel(Classifier model, String path) throws Exception {
        SerializationHelper.write(path, model);
        System.out.println("Model saved to: " + path);
    }

    public static Classifier loadModel(String path) throws Exception {
        Classifier model = (Classifier) SerializationHelper.read(path);
        System.out.println("Model loaded from: " + path);
        return model;
    }
}