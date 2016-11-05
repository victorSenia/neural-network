package com.leo.test.neural.network;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Layer implements Serializable {

    private static final long serialVersionUID = 4547186786807920459L;

    private final List<Neuron> neurons;

    final transient private Layer previousLayer;

    transient private Layer nextLayer;

    private boolean hasBias;

    public Layer(Layer previousLayer) {
        neurons = new ArrayList<>();
        this.previousLayer = previousLayer;
    }

    public Layer(Layer previousLayer, Neuron bias) {
        this(previousLayer);
        if (bias != null) {
            hasBias = true;
            neurons.add(bias);
        }
    }

    public List<Neuron> getNeurons() {
        return this.neurons;
    }

    public void addNeuron(Neuron neuron) {
        neurons.add(neuron);
        if (previousLayer != null) {
            Synapse synapse;
            for (Neuron previousLayerNeuron : previousLayer.getNeurons()) {
                //initialize with a random weight between -1 and 1
                synapse = new Synapse(previousLayerNeuron, neuron, (Math.random() * 1) - 0.5);
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
        for (int i = hasBias ? 1 : 0; i < neurons.size(); i++) {
            neurons.get(i).activate();
        }
    }

    public boolean hasBias() {
        return hasBias;
    }

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }
}
