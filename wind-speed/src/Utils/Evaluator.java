package Utils;

import Processing.Preprocess;
import Processing.IPreprocess;
import Model.ModelBase;

import weka.core.*;
import weka.core.Instances;

import java.util.Random;

public class Evaluator {
    public Evaluator() {}

    public void k_folds_validation(ModelBase model, Instances data, int folds) {
        Preprocess preprocess = new Preprocess();
        k_folds_validation(model, data, folds, preprocess);
    }

    public void k_folds_validation(ModelBase model, Instances data, int folds, IPreprocess pre_process) {
        double[] accuracy = new double[folds];
        double[] mse_accuracy = new double[folds];
        double[] mae_accuracy = new double[folds];
        long train_elapsed_time = 0;
        try {
            for (int fold = 0; fold < folds; fold++) {
                Instances train = pre_process.apply(data.trainCV(folds, fold));
                Instances test = data.testCV(folds, fold);

                long start_time = System.currentTimeMillis();
                ModelBase copy_model = model.copy();
                copy_model.buildClassifier(train);
                long elapsed_time = System.currentTimeMillis() - start_time;
                train_elapsed_time += elapsed_time;

                int match = 0;
                double mse = 0.0;
                double mae = 0.0;

                for (Instance instance : test) {
                    double predicted = copy_model.classifyInstance(instance);
                    double actual = instance.classValue();
                    mse += Math.pow(predicted - actual, 2);
                    mae += Math.abs(predicted - actual);
                    if (predicted == actual) match++;
                }

                accuracy[fold] = (double) match / test.numInstances();
                mse_accuracy[fold] = mse / test.numInstances();
                mae_accuracy[fold] = mae / test.numInstances();
            }

            System.out.println("--------------------------------- Total average model train time: " + train_elapsed_time/folds + " ms\n");

            printInformation(accuracy, folds, "accuracy");
            printInformation(mse_accuracy, folds, "mse");
            printInformation(mae_accuracy, folds, "mae");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void printInformation(double[] accuracy, double folds, String type) {
        System.out.println(">>>>>>>>> 1-fold to " + folds + "-folds " + type + " :");

        double sum_accuracy = 0;
        for (int i = 0; i < folds; i++) {
            sum_accuracy += accuracy[i];
            System.out.print(accuracy[i] + ", ");
        }

        System.out.println("\n>>>>>>>>> Average " + type + " : " + sum_accuracy / folds + "\n");
    }
}