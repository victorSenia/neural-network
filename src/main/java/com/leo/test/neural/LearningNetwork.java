package com.leo.test.neural;

import com.leo.test.neural.network.Layer;
import com.leo.test.neural.network.NeuralNetwork;
import com.leo.test.neural.network.Neuron;
import com.leo.test.neural.network.activators.ActivationStrategy;
import com.leo.test.neural.network.activators.SigmoidActivationStrategy;
import com.leo.test.neural.teacher.Backpropagator;
import com.leo.test.neural.training.implementation.AndTrainingDataGenerator;
import com.leo.test.neural.training.implementation.OrTrainingDataGenerator;
import com.leo.test.neural.training.implementation.XorTrainingDataGenerator;

/**
 * Created by User on 04.11.2016.
 */
public class LearningNetwork extends PrintResults {
    public static void main(String... args) {
        trainSave();
//        readFromFile();
    }

    private static void trainSave() {
        NeuralNetwork untrained;
        untrained = createUntrainedXorNeuralNetwork(new SigmoidActivationStrategy());
        Backpropagator backpropagator = new Backpropagator(untrained);
        //                Date instant = new Date();
        backpropagator.train(new XorTrainingDataGenerator(), 0.0001);
        //                System.out.println("Train took " + (new Date().getTime() - instant.getTime()));
//        untrained.save("trained_XOR");
        printResults(untrained, "XOR");
//        untrained = createUntrainedOrAndNeuralNetwork(new SigmoidActivationStrategy());
//        backpropagator = new Backpropagator(untrained);
        backpropagator.reset();
        backpropagator.train(new OrTrainingDataGenerator(), 0.0001);
//        untrained.save("trained_OR");
        printResults(untrained, "OR");
        backpropagator.reset();
        backpropagator.train(new AndTrainingDataGenerator(), 0.0001);
//        untrained.save("trained_AND");
        printResults(untrained, "AND");
    }

    private static void readFromFile() {
        NeuralNetwork untrained;
        untrained = NeuralNetwork.readFromFile("trained_XOR");
        printResults(untrained, "XOR");
        untrained = NeuralNetwork.readFromFile("trained_OR");
        printResults(untrained, "OR");
        untrained = NeuralNetwork.readFromFile("trained_AND");
        printResults(untrained, "AND");
    }

    private static NeuralNetwork createUntrainedXorNeuralNetwork(ActivationStrategy activationStrategy) {
        NeuralNetwork xorNeuralNetwork = new NeuralNetwork("Trained XOR Network");
        createLayer(xorNeuralNetwork, activationStrategy, 2, true, true);
        createLayer(xorNeuralNetwork, activationStrategy, 2, false, false);
        createLayer(xorNeuralNetwork, activationStrategy, 1, false, false);
        return xorNeuralNetwork;
    }

    private static NeuralNetwork createUntrainedOrAndNeuralNetwork(ActivationStrategy activationStrategy) {
        NeuralNetwork orNeuralNetwork = new NeuralNetwork("Trained OR Network");
        createLayer(orNeuralNetwork, activationStrategy, 2, true, true);
        createLayer(orNeuralNetwork, activationStrategy, 1, false, false);
        return orNeuralNetwork;
    }

    private static void createLayer(NeuralNetwork network, ActivationStrategy activationStrategy, int neuronQuantity, boolean hasBias, boolean isInput) {
        Neuron bias = null;
        if (hasBias)
            bias = new Neuron(activationStrategy.copy(), 1);
        Layer inputLayer;
        if (isInput)
            inputLayer = null;
        else
            inputLayer = network.getLayers().get(network.getLayers().size() - 1);
        Layer layer = new Layer(inputLayer, bias);
        while (neuronQuantity-- > 0)
            layer.addNeuron(new Neuron(activationStrategy.copy()));
        network.addLayer(layer);
    }
}
