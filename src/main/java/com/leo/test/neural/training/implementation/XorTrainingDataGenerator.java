package com.leo.test.neural.training.implementation;

import com.leo.test.neural.training.AbstractTrainingDataGenerator;

public class XorTrainingDataGenerator extends AbstractTrainingDataGenerator {

    @Override
    public double[][] getInputs() {
        return inputs;
    }

    @Override
    public double[][] getOutputs() {
        return outputs;
    }

    private final double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    private final double[][] outputs = {{0}, {1}, {1}, {0}};
}
