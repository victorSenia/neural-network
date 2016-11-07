package com.leo.test.neural.training.implementation;

import com.leo.test.neural.training.DataInterface;

public class DigitImage implements DataInterface {

    private double[] input;

    private int label;

    private double[] output = new double[10];

    public DigitImage(int label, byte[] data) {
        this.label = label;
        output[label] = 1;
        this.input = new double[data.length];
        for (int i = 0; i < this.input.length; i++) {
            this.input[i] = data[i] & 0xFF; //convert to unsigned
        }
        otsu();
    }

    //Uses Otsu's Threshold algorithm to convert from grayscale to black and white
    private void otsu() {
        int[] histogram = new int[256];
        for (double datum : input) {
            histogram[(int) datum]++;
        }
        double sum = 0;
        for (int i = 0; i < histogram.length; i++) {
            sum += i * histogram[i];
        }
        double sumB = 0;
        int wB = 0;
        int wF;
        double maxVariance = 0;
        int threshold = 0;
        int i = 0;
        boolean found = false;
        while (i < histogram.length && !found) {
            wB += histogram[i];
            if (wB != 0) {
                wF = input.length - wB;
                if (wF != 0) {
                    sumB += (i * histogram[i]);
                    double mB = sumB / wB;
                    double mF = (sum - sumB) / wF;
                    double varianceBetween = wB * Math.pow((mB - mF), 2);
                    if (varianceBetween > maxVariance) {
                        maxVariance = varianceBetween;
                        threshold = i;
                    }
                } else {
                    found = true;
                }
            }
            i++;
        }

/*        System.out.println(label + ": threshold is " + threshold);
        for(i = 0; i < input.length; i++) {
            if(i % 28 == 0) {
                System.out.println("<br />");
            }
            System.out.print("<span style='color:rgb(" + (int) (255 - input[i]) + ", " + (int) (255 - input[i]) + ", " + (int) (255 - input[i]) + ")'>&#9608;</span>");
        } */

        for (i = 0; i < input.length; i++) {
            input[i] = input[i] <= threshold ? 0 : 1;
        }
/*
        if(label == 7 || label == 9) {
            for(i = 0; i < input.length; i++) {
                if(i % 28 == 0) {
                    System.out.println("");
                }
                System.out.print(input[i] == 1 ? "#" : " ");
            }
        }*/
    }

    @Override
    public double[] getInput() {
        return input;
    }

    @Override
    public double[] getOutput() {
        return output;
    }

    public int getLabel() {
        return label;
    }
}
