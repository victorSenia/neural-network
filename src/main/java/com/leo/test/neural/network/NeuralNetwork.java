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

    public static NeuralNetwork copy(NeuralNetwork network) {
        byte[] bytes = null;
        try (ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream(2048);
             ObjectOutputStream outputStream = new ObjectOutputStream(byteArrayOutputStream)) {
            outputStream.writeObject(network);
            bytes = byteArrayOutputStream.toByteArray();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try (ObjectInputStream inputStream = new ObjectInputStream(new ByteArrayInputStream(bytes))) {
            return (NeuralNetwork) inputStream.readObject();
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void addLayer(Layer layer) {
        if (layers.isEmpty())
            input = layer;
        else
            output.setNextLayer(layer);
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
