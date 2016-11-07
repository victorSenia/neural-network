package com.leo.test.neural;

import com.leo.test.neural.network.NeuralNetwork;
import com.leo.test.neural.network.activators.SigmoidActivationStrategy;
import com.leo.test.neural.service.DigitImageLoadingService;
import com.leo.test.neural.teacher.Backpropagator;
import com.leo.test.neural.training.DataInterface;
import com.leo.test.neural.training.implementation.DigitImage;
import com.leo.test.neural.training.implementation.DigitTrainingDataGenerator;

import java.io.IOException;
import java.util.List;

public class DigitRecognizing {

    public static void main(String[] args) throws IOException {

        DigitImageLoadingService trainingService = new DigitImageLoadingService("/train/train-labels-idx1-ubyte.dat", "/train/train-images-idx3-ubyte.dat");
        DigitImageLoadingService testService = new DigitImageLoadingService("/test/t10k-labels-idx1-ubyte.dat", "/test/t10k-images-idx3-ubyte.dat");

                NeuralNetwork neuralNetwork = createAndTeachNeuralNetwork(trainingService);
//        NeuralNetwork neuralNetwork = NeuralNetwork.readFromFile("DigitRecognizingNeuralNetwork-1478618533924.net");

        DigitTrainingDataGenerator testDataGenerator = new DigitTrainingDataGenerator(testService.loadDigitImages(), 0);
        List<? extends DataInterface> testData = testDataGenerator.getTrainingData().getData();

        print(neuralNetwork, testData);
    }

    private static NeuralNetwork createAndTeachNeuralNetwork(DigitImageLoadingService trainingService) throws IOException {
        NeuralNetwork neuralNetwork = new NeuralNetwork("Digit Recognizing Neural Network");

        LearningNetwork.createLayer(neuralNetwork, new SigmoidActivationStrategy(), DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS, true, true);
                LearningNetwork.createLayer(neuralNetwork, new SigmoidActivationStrategy(), (int) Math.round((2.0 / 3.0) * (DigitImageLoadingService.ROWS * DigitImageLoadingService.COLUMNS) + 10), true, false);
//        LearningNetwork.createLayer(neuralNetwork, new SigmoidActivationStrategy(), 50, true, false);
        LearningNetwork.createLayer(neuralNetwork, new SigmoidActivationStrategy(), 10, false, false);

        DigitTrainingDataGenerator trainingDataGenerator = new DigitTrainingDataGenerator(trainingService.loadDigitImages(), 0);

        Backpropagator backpropagator = new Backpropagator(neuralNetwork, 0.3, 0.9, 0, 1);
        backpropagator.train(trainingDataGenerator, 50);
        neuralNetwork.save("trained_Digit");
        return neuralNetwork;
    }

    private static void print(NeuralNetwork neuralNetwork, List<? extends DataInterface> testData) {
        int wrong = 0;
        for (DataInterface dataInterface : testData) {
            DigitImage image = (DigitImage) dataInterface;

            neuralNetwork.setInputs(dataInterface.getInput());
            double[] receivedOutput = neuralNetwork.getOutput();

            double max = receivedOutput[0];
            double max2 = 0;
            int recognizedDigit = 0;
            int recognizedDigit2 = 0;
            for (int j = 1; j < receivedOutput.length; j++) {
                if (receivedOutput[j] > max) {
                    max2 = max;
                    max = receivedOutput[j];
                    recognizedDigit2 = recognizedDigit;
                    recognizedDigit = j;
                }
            }
            if (image.getLabel() != recognizedDigit)
                wrong++;
            System.out.format("Recognized %d as %d (output value was %.5f)? second %d with value %.5f%n", image.getLabel(), recognizedDigit, max, recognizedDigit2, max2);
        }
        System.out.format("Wrong recognised %d numbers from %d%n", wrong, testData.size());
    }
}
