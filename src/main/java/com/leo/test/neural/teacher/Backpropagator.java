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

    private NeuralNetwork neuralNetwork;

    private double learningRate;

    private double momentum;

    private double characteristicTime;

    private long currentEpoch;

    public Backpropagator(NeuralNetwork neuralNetwork, double learningRate, double momentum, double characteristicTime) {
        this.neuralNetwork = neuralNetwork;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.characteristicTime = characteristicTime;
    }

    public void train(TrainingDataGenerator generator, double errorThreshold) {
        double error;
        double sum = 0.0;
        double average = 25;
        int samples = 25;
        int index;
        double[] errors = new double[samples];
        do {
            TrainingData trainingData = generator.getTrainingData();
            error = backpropagate(trainingData.getInputs(), trainingData.getOutputs());

            index = (int) (currentEpoch) % samples;
            sum -= errors[index];
            errors[index] = error;
            sum += errors[index];

            if (currentEpoch > samples) {
                average = sum / samples;
            }
            if (currentEpoch % 10000 == 0)
                System.out.println("Error for epoch " + currentEpoch + ": " + error + ". Average: " + average + (characteristicTime > 0 ? " Learning rate: " + learningRate / (1 + (currentEpoch / characteristicTime)) : ""));
            currentEpoch++;
        } while (average > errorThreshold);
        System.out.println(currentEpoch);
    }

    public double backpropagate(double[][] inputs, double[][] expectedOutputs) {
        double error = 0;
        Map<Synapse, Double> synapseNeuronDeltaMap = new HashMap<>();
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] expectedOutput = expectedOutputs[i];
            List<Layer> layers = neuralNetwork.getLayers();
            neuralNetwork.setInputs(input);
            double[] output = neuralNetwork.getOutput();
            //First step of the backpropagation algorithm. Backpropagate errors from the output layer all the way up
            //to the first hidden layer
            for (int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);
                for (int k = 0; k < layer.getNeurons().size(); k++) {
                    Neuron neuron = layer.getNeurons().get(k);
                    double neuronError;
                    if (layer.getNextLayer() == null) {
                        //the order of output and expected determines the sign of the delta. if we have output - expected, we subtract the delta
                        //if we have expected - output we add the delta.
                        neuronError = neuron.getDerivative() * (output[k] - expectedOutput[k]);
                    } else {
                        neuronError = neuron.getDerivative();
                        double sum = 0;
                        for (Synapse synapse : neuron.getOutputSynapse()) {
                            sum += (synapse.getWeight() * synapse.getSinkNeuron().getError());
                        }
                        neuronError *= sum;
                    }
                    neuron.setError(neuronError);
                }
            }

            //Second step of the backpropagation algorithm. Using the errors calculated above, update the weights of the
            //network
            for (int j = layers.size() - 1; j > 0; j--) {
                Layer layer = layers.get(j);
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
            output = neuralNetwork.getOutput();
            error += error(output, expectedOutput);
        }
        return error;
    }

    private String explode(double[] array) {
        StringBuilder string = new StringBuilder("[");
        for (double number : array) {
            string.append(number).append(", ");
        }
        string.setLength(string.length() - 3);
        return string.append("]").toString();
    }

    public double error(double[] actual, double[] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException("The lengths of the actual and expected value arrays must be equal");
        }
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i] - actual[i], 2);
        }
        return sum / 2;
    }
}