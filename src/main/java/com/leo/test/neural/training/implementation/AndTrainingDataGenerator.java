package com.leo.test.neural.training.implementation;

import com.leo.test.neural.training.AbstractTrainingDataGenerator;

public class AndTrainingDataGenerator extends AbstractTrainingDataGenerator {

    private final double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    private final double[][] outputs = {{0}, {0}, {0}, {1}};

    @Override
    public double[][] getInputs() {
        return inputs;
    }

    @Override
    public double[][] getOutputs() {
        return outputs;
    }
}
