package com.leo.test.neural.network.activators;

import java.io.Serializable;

public class LinearActivationStrategy implements ActivationStrategy, Serializable {
    private static final long serialVersionUID = 9178412649401489837L;

    public double activate(double weightedSum) {
        return weightedSum;
    }

    public double derivative(double weightedSum) {
        return 1;
    }

    public ActivationStrategy copy() {
        return new LinearActivationStrategy();
    }
}
