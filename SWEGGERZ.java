/*
 * ECS 170 Programming Assignment 3
 * Facial Recognition Neural Network
 * Date: March 15, 2017
 * 
 * Team Name: SWEGGERZ
 * 
 * Programmers:
 *   Susie Chac (912004424)
 *   Melody Chang (912110826)
*/

import java.io.File;
import javax.swing.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.Random;
import java.lang.Math;
import java.util.Collections;

public class SWEGGERZ {
    public static class Image
    {
        ArrayList<Double> pixels;
        int gender = 0;
        String fname;
        Image(ArrayList<Double> pixels, int gender, String name)
        {
            this.pixels = pixels;
            this.gender = gender;
            this.fname = name;
        }
    }

    public static class NeuralNetwork 
    {
        // only one hidden layer with 3 hidden nodes
        int numHidnNodes = 3;
        // img size stays constant
        int imgSize = 128*120;
        // synapse 1
        double[][] hiddenWeights = new double[numHidnNodes][imgSize];
        // calculate sum of input*weight
        double[] hiddenUnits = new double[numHidnNodes];
        // synapse 2
        double[] outputWeight = new double[numHidnNodes];
        // male or female?
        double output;
        // learning rate of algorithm
        double eta = .05;

        NeuralNetwork() 
        {
            Random rand = new Random();
            for(int i = 0; i < numHidnNodes; ++i) 
            {
                //initialize the weights to random numbers between 0 and 0.2
                for (int j = 0; j < imgSize; ++j) 
                {
                    this.hiddenWeights[i][j] = -.1 + rand.nextDouble()*.2 ;
                }
                this.hiddenUnits[i] = 0;
                this.outputWeight[i] =  -.1 + rand.nextDouble()*.2;
            }
        }

        // train the algorithm
        void train(Image img) 
        {
            ArrayList<Double> inputNodes = img.pixels; // input data
            // Each val is sum of a node * all edge weights into hidden layer
            // node has edge to each hidden node in the next layer
            ArrayList<Double> hiddenSums = new ArrayList<Double>();
            // input layer to the hidden layer
            for (int i = 0; i < numHidnNodes; ++i) 
            {
                hiddenUnits[i] = 0;
                for (int j = 0; j < inputNodes.size(); ++j) 
                {
                    // add node * edge weight (matrix multiplication)
                    hiddenUnits[i] += (inputNodes.get(j)*hiddenWeights[i][j]);
                }
                // add entire sum for node to hiddenSums
                hiddenSums.add(hiddenUnits[i]);
                // run sigmoid function to get the hidden nodes (the next layer)
                hiddenUnits[i] = sigmoid(hiddenUnits[i]);
            }

            // Hidden layer to output layer.
            double sum = 0.0;
            for (int i = 0; i < numHidnNodes; ++i) 
            {
                // add node * edge weight (matrix multiplication)
                sum += hiddenUnits[i]*outputWeight[i];
            }
            // run sigmoid function to get the output nodes (the next layer)
            output = sigmoid(sum);

            //backpropagation
            if (output != img.gender) 
            {
                double delta_a = img.gender - output;
                double delta_b[] = new double[numHidnNodes];
                //update weights = gradient descent
                for(int i = 0; i < numHidnNodes; ++i) 
                {
                    delta_b[i] = outputWeight[i]*delta_a;
                    for (int j = 0; j < inputNodes.size(); ++j) 
                    {
                        hiddenWeights[i][j] += eta*delta_b[i]*inputNodes.get(j)*sigmoidPrime(hiddenSums.get(i));
                    }
                    outputWeight[i] += eta*delta_a*hiddenUnits[i]*sigmoidPrime(sum);
                }
            }
        }

        double CalculateConfidence(double value) {
            return Math.abs(.5 - value) * 2;
        }

        // run through neural network to test algorithm
        int test(Image img) 
        {
            // input to the hidden layer
            ArrayList<Double> inputNodes = img.pixels;
            
            for (int i = 0; i < numHidnNodes; ++i) 
            {
                hiddenUnits[i] = 0;
                for (int j = 0; j < inputNodes.size(); ++j) 
                {
                    hiddenUnits[i] += (inputNodes.get(j)*hiddenWeights[i][j]);
                }
                hiddenUnits[i] = sigmoid(hiddenUnits[i]);
            }
            // Hidden layer to output layer.
            double sum = 0.0;
            for (int i = 0; i < numHidnNodes; ++i) {
                // add node * edge weight (matrix multiplication)
                sum += hiddenUnits[i]*outputWeight[i];
            }
            // run sigmoid function to get the output nodes (the next layer)
            output = sigmoid(sum);
            if (output >= 0.50) // 0.50-1.00 is MALE
            {
                System.out.println(img.fname + " MALE " + CalculateConfidence(output) + ".");
                return 1;
            } 
            else // 0.00- 0.49 is FEMALE
            {
                System.out.println(img.fname + " FEMALE " + CalculateConfidence(output) + ".");
                return 0;
            }
        }
    }

    // sigmoid function
    public static double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }

    // derivative of sigmoid function
    public static double sigmoidPrime(double x) {
        return Math.pow(Math.E,-1*x)/Math.pow(1+Math.pow(Math.E,-1*x), 2);
    }

    // load test directory
    public static ArrayList<Image> Load(final String testDir) 
    {
        final ArrayList<Image> testFiles = new ArrayList<Image>();
        File[] files = new File(testDir).listFiles();
        for (File file : files) 
        {
            try
            {
                Scanner sc1 = new Scanner(new File(testDir + "/" + file.getName()));
                // puts all numbers in file into a list as doubles
                ArrayList<Double> listDouble = new ArrayList<Double>();
                while(sc1.hasNextLine()) 
                {
                    Scanner sc2 = new Scanner(sc1.nextLine());
                    while(sc2.hasNext()) 
                    { 
                        // making all values doubles less than 1 (divide by 255)
                        listDouble.add(Double.parseDouble(sc2.next())/255.0);
                    }
                }
                try
                {
                    testFiles.add(new Image(listDouble, -1, file.getName()));
                } catch (Exception e) 
                {
                    e.printStackTrace();
                }
            } catch (Exception e ) 
            {
                e.printStackTrace();
            }
        }
        return testFiles;
    }

    // load female and male directories
    public static ArrayList<Image> Load(final String femaleDir, final String maleDir) 
    {
        final ArrayList<Image> fmFiles = new ArrayList<Image>();
        File[] females = new File(femaleDir).listFiles();
        File[] males = new File(maleDir).listFiles();
        for (File file : females) 
        {
            try
            {
                Scanner sc1 = new Scanner(new File(femaleDir + "/" + file.getName()));
                // puts all numbers in file into a list as doubles
                ArrayList<Double> listInt = new ArrayList<Double>();
                while(sc1.hasNextLine()) 
                {
                    Scanner sc2 = new Scanner(sc1.nextLine());
                    while(sc2.hasNext()) 
                    {
                        // making all values doubles less than 1 (divide by 255)
                        listInt.add(Double.parseDouble(sc2.next())/255.0);
                    }
                }
                try
                {
                    fmFiles.add(new Image(listInt, 0, file.getName()));
                } 
                catch (Exception e) 
                {
                    e.printStackTrace();
                }
            } 
            catch (Exception e ) 
            {
                e.printStackTrace();
            }
        }

        for (File file : males) 
        {
            try
            {
                Scanner sc1 = new Scanner(new File(maleDir + "/" + file.getName()));
                // puts all numbers in file into a list as doubles
                ArrayList<Double> listInt = new ArrayList<Double>();
                while(sc1.hasNextLine()) 
                {
                    Scanner sc2 = new Scanner(sc1.nextLine());
                    while(sc2.hasNext()) 
                    {
                        // making all values doubles less than 1 (divide by 255)
                        listInt.add(Double.parseDouble(sc2.next())/255.0);
                    }
                }
                try
                {
                    fmFiles.add(new Image(listInt, 1, file.getName()));
                } 
                catch (Exception e) 
                {
                    e.printStackTrace();
                }
            } 
            catch (Exception e ) 
            {
                e.printStackTrace();
            }
        }
        Collections.shuffle(fmFiles);
        return fmFiles;
    }


    public static void main(String[] args) 
    {
        int it = 0;
        Boolean train = false;
        Boolean test = false;
        String femaleDir = "";
        String maleDir = "";
        String testDir = "";

        // looping through command line arguments
        while(it < args.length) 
        {
            // When running the command, it would look something like
            // -train female male
            if(args[it].equalsIgnoreCase("-train")) 
            {
                // Point to the female directory
                train = true;
                femaleDir = args[it+1];
                // Point to the male directory
                maleDir = args[it+2];
                it += 3;
            } 
            else if (args[it].equalsIgnoreCase("-test")) 
            {
                // Point to the test directory
                testDir = args[it+1];
                test = true;
                it += 2;
            } 
            else 
            {
                it += 1;
            }
        } // Looping through command line arguments.
        if (train && test) 
        {
            ArrayList<Image> fmFiles = Load(femaleDir, maleDir);
            NeuralNetwork nn = new NeuralNetwork();
            // Training
            int iteration = 0;
            while(iteration < 32) 
            {
                for (Image pic : fmFiles) 
                {
                    nn.train(pic);
                }
                iteration++;
            }

            // Testing the data
            ArrayList<Image> testFiles = Load(testDir);
            for (Image img : testFiles) 
            {
                nn.test(img);
            }
        } 
        else if (train) 
        {
            ArrayList<Image> fmFiles = Load(femaleDir, maleDir);
            NeuralNetwork nn = new NeuralNetwork();
            // Split the files into 5 folds to do cross validation.
            ArrayList<ArrayList<Image>> folds = new ArrayList<ArrayList<Image>>();
            int split = fmFiles.size()/5;
            for (int i = 0; i < 5; ++i) 
            {
                ArrayList<Image> tempList = new ArrayList<Image>(fmFiles.subList(i*fmFiles.size()/5, fmFiles.size()/5*(i+1)));
                folds.add(tempList);
            }
            Double average = 0.0;
            ArrayList<Double> perCorr = new ArrayList<Double>();
            // 5 Layer Cross Validation
            for (int i = 0; i < 5; ++i) 
            {
                nn = new NeuralNetwork();
                int numCorrect = 0;
                // 25 loops through to train the neural network
                int iteration = 0;
                while(iteration < 32) 
                {
                    for (int k = 0; k < 5; ++k) 
                    {
                        for (int j = 0; j < folds.get(k).size(); ++j) 
                        {
                            if (k != i) 
                            {
                                nn.train(folds.get(k).get(j));
                            }
                        }
                    }
                    iteration++;
                }

                // Testing our neural network.
                for (int j = 0; j < folds.get(i).size(); ++j) 
                {
                    if (nn.test(folds.get(i).get(j)) == folds.get(i).get(j).gender) 
                    {
                        numCorrect++;
                    }
                }
                perCorr.add(numCorrect/(folds.get(i).size()+0.0));
                System.out.println("Percent Correct: " + numCorrect/(folds.get(i).size()+0.0));
                average += numCorrect/(folds.get(i).size()+0.0);
            }
            System.out.println("AVERAGE : " + average/5.0);
            double sum = 0.0;
            for(int i = 0; i < perCorr.size(); i++)
            {
                sum += Math.pow(perCorr.get(i)-average/5.0, 2);
                System.out.println("SUM: " + sum);
            }
            System.out.println("STANDARD DEVIATION : " + Math.sqrt(sum/5));
        } 
        else 
        {
            System.out.println("Please train before you test.");
        }
        return;
    }
}
