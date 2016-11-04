package com.leo.test.neural;

import com.leo.test.neural.network.Layer;
import com.leo.test.neural.network.NeuralNetwork;
import com.leo.test.neural.network.Neuron;
import com.leo.test.neural.network.activators.ThresholdActivationStrategy;

/**
 * Created by Senchenko Victor on 03.11.2016.
 */
public class RigidNetwork extends PrintResults {
    public static void main(String... args) {
        printResults(createAndNeuralNetwork(), "AND");
        printResults(createOrNeuralNetwork(), "OR");
        printResults(createXorNeuralNetwork(), "XOR");
    }

    private static NeuralNetwork createAndNeuralNetwork() {
        NeuralNetwork andNeuralNetwork = new NeuralNetwork("AND Network");

        Layer inputLayer = new Layer(null);

        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));
        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));

        Layer outputLayer = new Layer(inputLayer);
        // for both input neurons weight = 1, border can be anything more 1 but less 2
        outputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1.9)), new double[]{1, 1});

        andNeuralNetwork.addLayer(inputLayer);
        andNeuralNetwork.addLayer(outputLayer);

        return andNeuralNetwork;
    }

    private static NeuralNetwork createOrNeuralNetwork() {
        NeuralNetwork orNeuralNetwork = new NeuralNetwork("OR Network");

        Layer inputLayer = new Layer(null);

        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));
        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));

        Layer outputLayer = new Layer(inputLayer);
        // for both input neurons weight = 1, border can be anything more 0 but less 1
        outputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(0.9)), new double[]{1, 1});

        orNeuralNetwork.addLayer(inputLayer);
        orNeuralNetwork.addLayer(outputLayer);

        return orNeuralNetwork;
    }

    private static NeuralNetwork createXorNeuralNetwork() {
        NeuralNetwork xorNeuralNetwork = new NeuralNetwork("XOR Network");

        Layer inputLayer = new Layer(null);

        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));
        inputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1)));

        Layer hiddenLayer = new Layer(inputLayer);
        // for both input neurons weight = 1, border can be anything more 1 but less 2
        hiddenLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(1.9)), new double[]{1, 1});
        // for both input neurons weight = 1, border can be anything more 0 but less 1
        hiddenLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(0.9)), new double[]{1, 1});

        Layer outputLayer = new Layer(hiddenLayer);
        // for first hidden neurons weight = -1 (it means that if two input true, this will decrease the result),
        // of second weight = 1 (if at least one true - increase the result),
        // border can be anything more 0 but less 1
        outputLayer.addNeuron(new Neuron(new ThresholdActivationStrategy(0.9)), new double[]{-1, 1});

        xorNeuralNetwork.addLayer(inputLayer);
        xorNeuralNetwork.addLayer(hiddenLayer);
        xorNeuralNetwork.addLayer(outputLayer);

        return xorNeuralNetwork;
    }
}
