package com.leo.test.neural.training;

import java.util.Random;

/**
 * Created by Senchenko Victor on 04.11.2016.
 */
public abstract class AbstractTrainingDataGenerator implements TrainingDataGenerator {
    public abstract double[][] getInputs();

    public abstract double[][] getOutputs();

    public abstract int[] getInputIndices();

    public TrainingData getTrainingData() {
        double[][] randomizedInputs = new double[4][2];
        double[][] randomizedOutputs = new double[4][1];
        int[] inputIndices = shuffle(getInputIndices());
        for (int i = 0; i < inputIndices.length; i++) {
            randomizedInputs[i] = getInputs()[inputIndices[i]];
            randomizedOutputs[i] = getOutputs()[inputIndices[i]];
        }
        return new TrainingData(randomizedInputs, randomizedOutputs);
    }

    private int[] shuffle(int[] array) {
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--) {
            int index = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[index];
            array[index] = temp;
        }
        return array;
    }
}
