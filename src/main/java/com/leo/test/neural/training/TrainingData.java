package com.leo.test.neural.training;

import com.leo.test.neural.training.implementation.Data;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class TrainingData {

    private static final Random RANDOM = new Random();

    private final List<? extends DataInterface> data;

    public TrainingData(List<? extends DataInterface> data) {
        this.data = data;
    }

    public List<? extends DataInterface> getData() {
        Collections.shuffle(data, RANDOM);
        return data;
    }
}
