package com.leo.test.neural.training;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Senchenko Victor on 04.11.2016.
 */
public abstract class AbstractTrainingDataGenerator implements TrainingDataGenerator {
    protected abstract double[][] getInputs();

    protected abstract double[][] getOutputs();

    public TrainingData getTrainingData() {
        return new TrainingData(getData());
    }

    protected List<double[][]> getData() {
        List<double[][]> list = new ArrayList<>();
        for (int i = 0; i < getInputs().length; i++)
            list.add(new double[][]{getInputs()[i], getOutputs()[i]});
        return list;
    }
}
