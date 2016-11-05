package com.leo.test.neural.network;

import com.leo.test.neural.network.activators.ActivationStrategy;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Neuron implements Serializable {

    private static final long serialVersionUID = -8946973915773955854L;

    private final List<Synapse> inputSynapse;

    transient private final List<Synapse> outputSynapse;

    private final ActivationStrategy activationStrategy;

    private double output;

    transient private double derivative;

    transient private double weightedSum;

    transient private double error;

    public Neuron(ActivationStrategy activationStrategy) {
        inputSynapse = new ArrayList<>();
        outputSynapse = new ArrayList<>();
        this.activationStrategy = activationStrategy;
    }

    public Neuron(ActivationStrategy activationStrategy, double output) {
        this(activationStrategy);
        this.output = output;
    }

    public void addInput(Synapse input) {
        inputSynapse.add(input);
    }

    public void addOutput(Synapse output) {
        outputSynapse.add(output);
    }

    public List<Synapse> getInputSynapse() {
        return this.inputSynapse;
    }

    private void calculateWeightedSum() {
        weightedSum = 0;
        for (Synapse synapse : inputSynapse) {
            weightedSum += synapse.getWeight() * synapse.getSourceNeuron().getOutput();
        }
    }

    public void activate() {
        calculateWeightedSum();
        output = activationStrategy.activate(weightedSum);
        derivative = activationStrategy.derivative(output);
    }

    public double getOutput() {
        return this.output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDerivative() {
        return this.derivative;
    }

    public double getError() {
        return error;
    }

    public void setError(double error) {
        this.error = error;
    }

    public List<Synapse> getOutputSynapse() {
        return outputSynapse;
    }
}
