package com.leo.test.neural.training.implementation;

import com.leo.test.neural.training.DataInterface;

/**
 * Created by Senchenko Victor on 07.11.2016.
 */
public class Data implements DataInterface {

    protected double[] input;

    protected double[] output;

    public Data(double[] input, double[] output) {
        this.input = input;
        this.output = output;
    }

    @Override
    public double[] getInput() {
        return input;
    }

    @Override
    public double[] getOutput() {
        return output;
    }
}
