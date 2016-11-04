package com.leo.test.neural.network.activators;

import java.io.Serializable;

public class SigmoidActivationStrategy implements ActivationStrategy, Serializable {
    private static final long serialVersionUID = -5503757542015080905L;

    public double activate(double weightedSum) {
        return 1.0 / (1 + Math.exp(-1.0 * weightedSum));
    }

    public double derivative(double weightedSum) {
        return weightedSum * (1.0 - weightedSum);
    }

    public SigmoidActivationStrategy copy() {
        return new SigmoidActivationStrategy();
    }
}
