package com.leo.test.neural.network.activators;

/**
 * Created by IntelliJ IDEA.
 * Date: 11/5/11
 * Time: 3:04 PM
 */
public interface ActivationStrategy {
    double activate(double weightedSum);

    double derivative(double weightedSum);

    ActivationStrategy copy();
}
