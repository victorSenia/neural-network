package com.leo.test.neural.training.implementation;

import com.leo.test.neural.training.AbstractTrainingDataGenerator;
import com.leo.test.neural.training.DataInterface;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class DigitTrainingDataGenerator extends AbstractTrainingDataGenerator {

    private List<? extends DataInterface> list;

    private int quantity;

    public DigitTrainingDataGenerator(List<DigitImage> digitImages, int quantity) {
        list = digitImages;
        this.quantity = quantity;
    }

    protected List<? extends DataInterface> getData() {
        if (quantity == 0)
            return this.list;
        List<DigitImage> list = new ArrayList<>();
        DigitImage image;
        Random random = new Random();
        int i = 0;
        while (list.size() != 10 * quantity) {
            image = (DigitImage) this.list.get(random.nextInt(this.list.size()));
            if (image.getLabel() == i) {
                list.add(image);
                i = i == 9 ? 0 : i + 1;
            }
        }
        return list;
    }

    @Override
    protected double[][] getInputs() {
        throw new UnsupportedOperationException();
    }

    @Override
    protected double[][] getOutputs() {
        throw new UnsupportedOperationException();
    }
}
