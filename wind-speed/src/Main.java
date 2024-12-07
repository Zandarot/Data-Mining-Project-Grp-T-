import Processing.DataProcess;
import Model.*;
import Utils.Evaluator;
import Processing.Smote;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomTree;

import javax.xml.crypto.Data;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class Main {
    public static void main(String[] args) throws Exception {
        Instances data = DataProcess.PreProcess("dataset/wind_dataset_new.csv");

        data.setClassIndex(0);
        data.randomize(new Random(507));
        data = DataProcess.normalize(data);

        DataProcess.SummarizeData(data);
        evaluateModels(data);

        evaluateWithSMOTE();
    }

    private static ArrayList<Classifier> getClassifiers() {
        RandomTree randomTree = new RandomTree();
        IBk ibk = new IBk(5);

        ArrayList<Classifier> classifiers = new ArrayList<>();
        classifiers.add(randomTree);
        classifiers.add(ibk);
        return classifiers;
    }

    public static void evaluateModels(Instances data) throws Exception {
        int k = 10;

        Evaluator evaluator = new Evaluator();
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<ModelBase> models = new ArrayList<>();

        for (Classifier classifier : classifiers) {
            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        System.out.println("***************** Original Methods *****************");
        for (ModelBase model : models) {
            long start_time = System.currentTimeMillis();
            System.out.println("--------------------------------- Algorithm " + model.modelName());
            evaluator.k_folds_validation(model, data, k);
            long elapsed_time = System.currentTimeMillis() - start_time;
            System.out.println("--------------------------------- Total runtime: " + elapsed_time + " ms\n");
        }
    }

    public static void evaluateWithSMOTE() throws Exception {
        Instances data = new DataSource("dataset/wind_dataset_pre_processed_smote.arff").getDataSet();

        data.setClass(data.attribute("IND"));
        data.randomize(new Random(507));
        data = DataProcess.normalize(data);

        /*
        Class 0.00: 5352 instances (majority class).
        Class 0.46: 61 instances.
        Class 1.00: 525 instances.
        Class 2.00: 21 instances.
        Class 3.00: 2 instances (minority class).
        Class 4.00: 613 instances.
        */
        HashMap<String, Integer> samplingStrategy = new HashMap<>();
        samplingStrategy.put("0.0", 5352);
        samplingStrategy.put("0.46", 1000);
        samplingStrategy.put("1.0", 1000);
        samplingStrategy.put("2.0", 1000);
        samplingStrategy.put("3.0", 1000);
        samplingStrategy.put("4.0", 1000);

        int k = 10;
        Evaluator evaluator = new Evaluator();
        ArrayList<Classifier> classifiers = getClassifiers();
        ArrayList<ModelBase> models = new ArrayList<>();

        for (Classifier classifier : classifiers) {
            ModelBase model = new WekaModel(AbstractClassifier.makeCopy(classifier));
            models.add(model);
        }

        System.out.println("***************** Methods with SMOTE *****************");
        for (ModelBase model : models) {
            Random rand = new Random();
            Smote smote = new Smote(samplingStrategy, k, rand);

            long start_time = System.currentTimeMillis();
            System.out.println("--------------------------------- Algorithm " + model.modelName() + " with SMOTE pre-processing:");
            evaluator.k_folds_validation(model, data, k, smote);
            long elapsed_time = System.currentTimeMillis() - start_time;
            System.out.println("--------------------------------- Total runtime: " + elapsed_time + " ms\n");
        }
    }
}
