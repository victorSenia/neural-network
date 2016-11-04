package com.leo.test.neural.network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Layer implements Serializable {

    private static final long serialVersionUID = 4547186786807920459L;

    private final List<Neuron> neurons;

    private Layer previousLayer;

    private Layer nextLayer;

    private Neuron bias;

    public Layer() {
        neurons = new ArrayList<>();
    }

    public Layer(Layer previousLayer) {
        this();
        this.previousLayer = previousLayer;
    }

    public Layer(Layer previousLayer, Neuron bias) {
        this(previousLayer);
        this.bias = bias;
        neurons.add(bias);
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    public void addNeuron(Neuron neuron) {
        neurons.add(neuron);
        if (previousLayer != null) {
            Synapse synapse;
            for (Neuron previousLayerNeuron : previousLayer.getNeurons()) {
                synapse = new Synapse(previousLayerNeuron, neuron, (Math.random() * 1) - 0.5);//initialize with a random weight between -1 and 1
                neuron.addInput(synapse);
                previousLayerNeuron.addOutput(synapse);
            }
        }
    }

    public void addNeuron(Neuron neuron, double[] weights) {
        neurons.add(neuron);
        if (previousLayer != null) {
            if (previousLayer.getNeurons().size() != weights.length) {
                throw new IllegalArgumentException("The number of weights supplied must be equal to the number of neurons in the previous layer");
            } else {
                List<Neuron> previousLayerNeurons = previousLayer.getNeurons();
                Synapse synapse;
                for (int i = 0; i < previousLayerNeurons.size(); i++) {
                    synapse = new Synapse(previousLayerNeurons.get(i), neuron, weights[i]);
                    neuron.addInput(synapse);
                    previousLayerNeurons.get(i).addOutput(synapse);
                }
            }
        }
    }

    public void feedForward() {
        int biasCount = hasBias() ? 1 : 0;
        for (int i = biasCount; i < neurons.size(); i++) {
            neurons.get(i).activate();
        }
    }

//    public Layer getPreviousLayer() {
//        return previousLayer;
//    }
//
//    void setPreviousLayer(Layer previousLayer) {
//        this.previousLayer = previousLayer;
//    }
//
//    public Layer getNextLayer() {
//        return nextLayer;
//    }

    public boolean hasBias() {
        return bias != null;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }
}
