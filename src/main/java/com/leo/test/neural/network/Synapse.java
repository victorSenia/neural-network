package com.leo.test.neural.network;

import java.io.Serializable;

public class Synapse implements Serializable {

    private static final long serialVersionUID = 4872067492386852635L;

    private final Neuron sourceNeuron;

    transient private final Neuron sinkNeuron;

    private double weight;

    public Synapse(Neuron sourceNeuron, Neuron sinkNeuron, double weight) {
        this.sourceNeuron = sourceNeuron;
        this.sinkNeuron = sinkNeuron;
        this.weight = weight;
    }

    public Neuron getSourceNeuron() {
        return sourceNeuron;
    }

    public double getWeight() {
        return weight;
    }

    public void setWeight(double weight) {
        this.weight = weight;
    }

    public Neuron getSinkNeuron() {
        return sinkNeuron;
    }
}
