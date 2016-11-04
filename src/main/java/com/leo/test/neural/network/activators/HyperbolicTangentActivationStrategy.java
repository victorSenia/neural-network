package com.leo.test.neural.network.activators;

import java.io.Serializable;

public class HyperbolicTangentActivationStrategy implements ActivationStrategy, Serializable {
    private static final long serialVersionUID = -268299068707852082L;

    public double activate(double weightedSum) {
        double a = Math.exp(weightedSum);
        double b = Math.exp(-weightedSum);
        return ((a - b) / (a + b));
    }

    public double derivative(double weightedSum) {
        return 1 - Math.pow(activate(weightedSum), 2.0);
    }

    public HyperbolicTangentActivationStrategy copy() {
        return new HyperbolicTangentActivationStrategy();
    }
}
