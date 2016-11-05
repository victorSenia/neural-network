package com.leo.test.neural.network.activators;

import java.io.Serializable;

public class ThresholdActivationStrategy implements ActivationStrategy, Serializable {

    private static final long serialVersionUID = 1781726411565869364L;

    private final double threshold;

    public ThresholdActivationStrategy(double threshold) {
        this.threshold = threshold;
    }

    public double activate(double weightedSum) {
        return weightedSum > threshold ? 1 : 0;
    }

    public double derivative(double weightedSum) {
        return 0;
    }

    public ThresholdActivationStrategy copy() {
        return new ThresholdActivationStrategy(threshold);
    }
}
