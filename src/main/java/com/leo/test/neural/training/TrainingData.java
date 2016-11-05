package com.leo.test.neural.training;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class TrainingData {

    private static final Random RANDOM = new Random();

    private final List<double[][]> data;

    public TrainingData(List<double[][]> data) {
        this.data = data;
    }

    public List<double[][]> getData() {
        Collections.shuffle(data, RANDOM);
        return data;
    }
}
