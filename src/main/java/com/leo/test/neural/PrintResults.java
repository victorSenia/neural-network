package com.leo.test.neural;

import com.leo.test.neural.network.NeuralNetwork;

/**
 * Created by Senchenko Victor on 11/5/2016.
 */
public abstract class PrintResults {
    public static void printResults(NeuralNetwork network, String type) {
        System.out.println("Testing " + network.getName());
        printResult(network, type, new double[]{0, 0});
        printResult(network, type, new double[]{0, 1});
        printResult(network, type, new double[]{1, 0});
        printResult(network, type, new double[]{1, 1});
    }

    private static void printResult(NeuralNetwork network, String type, double[] doubles) {
        network.setInputs(doubles);
        System.out.format("%.0f %s %.0f: %f%n", doubles[0], type, doubles[1], (network.getOutput()[0]));
    }
}
