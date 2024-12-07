import Processing.DataProcess;
import Model.*;
import Utils.Evaluator;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.classifiers.functions.MultilayerPerceptron;
import java.util.ArrayList;

import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomTree;

import static java.lang.System.out;

public class Main {
    public static void main(String[] args) throws Exception {

        Instances data = DataProcess.PreProcess("dataset/wind_dataset_new.csv");

        DataProcess.SummarizeData(data);
        evaluate_models(data);

    }

    private static ArrayList<Classifier> getClassifiers() {

        RandomTree randomTree = new RandomTree();

        ArrayList<Classifier> classifiers = new ArrayList<>();

        classifiers.add(randomTree);
        return classifiers;
    }

    public static void evaluate_models(Instances data) throws Exception {
        int k_num = 10;
        Evaluator evaluator = new Evaluator();
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<ModelBase> models = new ArrayList<>();

        for (Classifier classifier : classifiers) {

            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        out.println("---------------------------------");
        out.println(k_num + "-folds Validation");

        for (ModelBase model : models) {
            long start_time = System.currentTimeMillis();
            out.println("---------------------------------");
            out.println("Algorithm: " + model.modelName());
            evaluator.k_folds_validation(model, data, k_num);
            long end_time = System.currentTimeMillis();
            long elapsed_time = end_time - start_time;
            out.println("Total run time: " + elapsed_time + " ms");
        }
    }
}
