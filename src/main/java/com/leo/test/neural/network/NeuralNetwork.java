package com.leo.test.neural.network;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class NeuralNetwork implements Serializable {

    private static final long serialVersionUID = 6802005297649476972L;

    private final String name;

    private final List<Layer> layers;

    private Layer input;

    private Layer output;

    public NeuralNetwork(String name) {
        this.name = name;
        layers = new ArrayList<>();
    }

    public static NeuralNetwork readFromFile(String fileName) {
        System.out.println("Reading trained neural network from file " + fileName);
        try (ObjectInputStream objectInputStream = new ObjectInputStream(new FileInputStream(fileName))) {
            return (NeuralNetwork) objectInputStream.readObject();
        } catch (IOException e) {
            System.out.println("Could not read from file: " + fileName);
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            System.out.println("Could not find class");
            e.printStackTrace();
        }
        return null;
    }

//    public NeuralNetwork copy() {
//        NeuralNetwork copy = new NeuralNetwork(this.name);
//        Layer previousLayer = null;
//        for (Layer layer : layers) {
//            Layer layerCopy;
//            if (layer.hasBias()) {
//                Neuron bias = layer.getNeurons().get(0);
//                Neuron biasCopy = new Neuron(bias.getActivationStrategy().copy());
//                biasCopy.setOutput(bias.getOutput());
//                layerCopy = new Layer(previousLayer, biasCopy);
//            } else {
//                layerCopy = new Layer(previousLayer);
//            }
//            int biasCount = layerCopy.hasBias() ? 1 : 0;
//            Neuron neuron, neuronCopy;
//            for (int i = biasCount; i < layer.getNeurons().size(); i++) {
//                neuron = layer.getNeurons().get(i);
//                neuronCopy = new Neuron(neuron.getActivationStrategy().copy());
//                neuronCopy.setOutput(neuron.getOutput());
//                neuronCopy.setError(neuron.getError());
//                if (neuron.getInputSynapse().size() == 0) {
//                    layerCopy.addNeuron(neuronCopy);
//                } else {
//                    double[] weights = neuron.getWeights();
//                    layerCopy.addNeuron(neuronCopy, weights);
//                }
//            }
//            copy.addLayer(layerCopy);
//            previousLayer = layerCopy;
//        }
//        return copy;
//    }

    public void addLayer(Layer layer) {
        if (layers.isEmpty())
            input = layer;
        else
            layers.get(layers.size() - 1).setNextLayer(layer);
        output = layer;
        layers.add(layer);
    }

    public void setInputs(double[] inputs) {
        if (input != null) {
            int biasCount = input.hasBias() ? 1 : 0;
            if (input.getNeurons().size() - biasCount != inputs.length) {
                throw new IllegalArgumentException("The number of inputs must equal the number of neurons in the input layer");
            } else {
                List<Neuron> neurons = input.getNeurons();
                for (int i = biasCount; i < neurons.size(); i++) {
                    neurons.get(i).setOutput(inputs[i - biasCount]);
                }
            }
        }
    }

    public String getName() {
        return name;
    }

    public double[] getOutput() {
        double[] outputs = new double[output.getNeurons().size()];
        for (int i = 1; i < layers.size(); i++) {
            layers.get(i).feedForward();
        }
        int i = 0;
        for (Neuron neuron : output.getNeurons()) {
            outputs[i] = neuron.getOutput();
            i++;
        }
        return outputs;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void reset() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                for (Synapse synapse : neuron.getInputSynapse()) {
                    synapse.setWeight((Math.random() * 1) - 0.5);
                }
            }
        }
    }

//    public double[] getWeights() {
//        List<Double> weights = new ArrayList<>();
//        for (Layer layer : layers) {
//            for (Neuron neuron : layer.getNeurons()) {
//                for (Synapse synapse : neuron.getInputSynapse()) {
//                    weights.add(synapse.getWeight());
//                }
//            }
//        }
//        return weights.stream().mapToDouble(Double::doubleValue).toArray();
//    }

//    public void copyWeightsFrom(NeuralNetwork sourceNeuralNetwork) {
//        if (layers.size() != sourceNeuralNetwork.layers.size()) {
//            throw new IllegalArgumentException("Cannot copy weights. Number of layers do not match (" + sourceNeuralNetwork.layers.size() + " in source versus " + layers.size() + " in destination)");
//        }
//        int layerId = 0;
//        for (Layer sourceLayer : sourceNeuralNetwork.layers) {
//            Layer destinationLayer = layers.get(layerId);
//            if (destinationLayer.getNeurons().size() != sourceLayer.getNeurons().size()) {
//                throw new IllegalArgumentException("Number of neurons do not match in layer " + (layerId + 1) + "(" + sourceLayer.getNeurons().size() + " in source versus " + destinationLayer.getNeurons().size() + " in destination)");
//            }
//            int neuronId = 0;
//            for (Neuron sourceNeuron : sourceLayer.getNeurons()) {
//                Neuron destinationNeuron = destinationLayer.getNeurons().get(neuronId);
//                if (destinationNeuron.getInputSynapse().size() != sourceNeuron.getInputSynapse().size()) {
//                    throw new IllegalArgumentException("Number of inputs to neuron " + (neuronId + 1) + " in layer " + (layerId + 1) + " do not match (" + sourceNeuron.getInputSynapse().size() + " in source versus " + destinationNeuron.getInputSynapse().size() + " in destination)");
//                }
//                int synapseId = 0;
//                for (Synapse sourceSynapse : sourceNeuron.getInputSynapse()) {
//                    Synapse destinationSynapse = destinationNeuron.getInputSynapse().get(synapseId);
//                    destinationSynapse.setWeight(sourceSynapse.getWeight());
//                    synapseId++;
//                }
//                neuronId++;
//            }
//            layerId++;
//        }
//    }

    public void save() {
        save(name.replaceAll(" ", "") + "-" + new Date().getTime() + ".net");
    }

    public void save(String fileName) {
        System.out.println("Writing trained neural network to file " + fileName);
        try (ObjectOutputStream objectOutputStream = new ObjectOutputStream(new FileOutputStream(fileName))) {
            objectOutputStream.writeObject(this);
        } catch (IOException e) {
            System.out.println("Could not write to file: " + fileName);
            e.printStackTrace();
        }
    }
}
