package com.leo.test.neural.teacher;

import com.leo.test.neural.network.Layer;
import com.leo.test.neural.network.NeuralNetwork;
import com.leo.test.neural.network.Neuron;
import com.leo.test.neural.network.Synapse;
import com.leo.test.neural.training.TrainingData;
import com.leo.test.neural.training.TrainingDataGenerator;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Backpropagator {

    private final NeuralNetwork neuralNetwork;

    private final double learningRate;

    private final double momentum;

    private final double characteristicTime;

    private final int samples;

    private long currentEpoch;

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate, double momentum, double characteristicTime, int samples) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.characteristicTime = characteristicTime;
        this.samples = samples;
    }

    public Backpropagator(NeuralNetwork neuralNetwork) {
        this(neuralNetwork, 0.3, 0.9, 0, 25);
    }

    public void train(TrainingDataGenerator generator, double errorThreshold) {
        double error;
        double sum = 0.0;
        double average = 1;
        int index;
        double[] errors = new double[samples];
        do {
            TrainingData trainingData = generator.getTrainingData();
            error = backpropagate(trainingData.getData());
            index = (int) (currentEpoch) % samples;
            sum = sum - errors[index] + error;
            errors[index] = error;
            if (currentEpoch > samples) {
                average = sum / samples;
            }
            if (currentEpoch % 10000 == 10)
                System.out.println("Error for epoch " + currentEpoch + ": " + error + ". Average: " + average + " Learning rate: " + (characteristicTime > 0 ? learningRate / (1 + (currentEpoch / characteristicTime)) : learningRate));
            currentEpoch++;
        } while (average > errorThreshold);
        System.out.println(currentEpoch);
    }

    private double backpropagate(List<double[][]> data) {
        double error = 0;
        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<>();
        for (double[][] dataDoubles : data) {
            double[] input = dataDoubles[0];
            double[] expectedOutput = dataDoubles[1];
            neuralNetwork.setInputs(input);
            double[] output = neuralNetwork.getOutput();
            List<Layer> layers = neuralNetwork.getLayers();
            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up to the first hidden layer
            backpropagateErrors(layers, output, expectedOutput);

            //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the network
            updateWeights(layers, synapseNeuronDeltaMap);

            output = neuralNetwork.getOutput();
            error += error(output, expectedOutput);
        }
        return error;
    }

    private void updateWeights(List<Layer> layers, Map<Synapse, Double> synapseNeuronDeltaMap) {
        for (int layerId = layers.size() - 1; layerId > 0; layerId--) {
            Layer layer = layers.get(layerId);
            for (Neuron neuron : layer.getNeurons()) {
                for (Synapse synapse : neuron.getInputSynapse()) {
                    double newLearningRate = characteristicTime > 0 ? learningRate / (1 + (currentEpoch / characteristicTime)) : learningRate;
                    double delta = newLearningRate * neuron.getError() * synapse.getSourceNeuron().getOutput();
                    if (synapseNeuronDeltaMap.get(synapse) != null) {
                        double previousDelta = synapseNeuronDeltaMap.get(synapse);
                        delta += momentum * previousDelta;
                    }
                    synapseNeuronDeltaMap.put(synapse, delta);
                    synapse.setWeight(synapse.getWeight() - delta);
                }
            }
        }
    }

    private void backpropagateErrors(List<Layer> layers, double[] output, double[] expectedOutput) {
        for (int layerId = layers.size() - 1; layerId > 0; layerId--) {
            Layer layer = layers.get(layerId);
            for (int neuronId = 0; neuronId < layer.getNeurons().size(); neuronId++) {
                Neuron neuron = layer.getNeurons().get(neuronId);
                double neuronError;
                if (layer.getNextLayer() == null) {
                    //the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta if we have expected - output we add the delta.
                    neuronError = (output[neuronId] - expectedOutput[neuronId]);
                } else {
                    neuronError = 0;
                    for (Synapse synapse : neuron.getOutputSynapse()) {
                        neuronError += (synapse.getWeight() * synapse.getSinkNeuron().getError());
                    }
                }
                neuron.setError(neuronError * neuron.getDerivative());
            }
        }
    }

    private double error(double[] actual, double[] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }
        return sum / 2;
    }

    public void reset() {
        neuralNetwork.reset();
        currentEpoch = 0;
    }
}