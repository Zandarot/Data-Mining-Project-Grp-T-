package Processing;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToNominal;
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class DataProcess {
    public DataProcess() {}

    public static void SummarizeData(Instances data) throws Exception {
        System.out.println("################# Total number of attributes: " + data.numAttributes() + " #################");
        Enumeration<Attribute> total_attributes = data.enumerateAttributes();
        while (total_attributes.hasMoreElements()) {
            Attribute attribute = (Attribute) total_attributes.nextElement();
            System.out.println(attribute);
        }
        System.out.println("################# Total number of instances: " + data.numInstances() + " #################\n");
    }

    public static Instances PreProcess(String data_path) throws Exception {
        Instances data = read_csv_dataset("dataset/wind_dataset_new.csv");
        DataProcess.save_csv2arff("dataset/wind_dataset_new.csv", "dataset/wind_dataset_original.arff");
        data = removeColumn(data, 0); // remove DATE attributes
        data = fixMissingValues(data);
        save_instances2arff(data, "dataset/wind_dataset_pre_processed.arff");
        return data;
    }

    public static Instances read_arff_dataset(String data_path) throws Exception {
        return new DataSource(data_path).getDataSet();
    }

    public static Instances read_csv_dataset(String data_path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(data_path));
        return loader.getDataSet();
    }

    public static void save_csv2arff(String csv_path, String arff_path) throws Exception {
        /*
         * Convert csv file to arff file
         * Params:
         *  csv_path: path to read csv file
         * arff_path: path to save arff file
         */
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(csv_path));
        Instances data = loader.getDataSet();
        save_instances2arff(data, arff_path);
    }

    public static void save_instances2arff(Instances instances, String arffFilePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setFile(new File(arffFilePath));
        saver.writeBatch();
    }

    public static Instances removeColumn(Instances data, int index) {
        Instances newData = new Instances(data);
        newData.deleteAttributeAt(index);
        return newData;
    }

    public static Instances removeColumn(Instances data, String name) {
        Instances newData = new Instances(data);
        newData.deleteAttributeAt(data.attribute(name).index());
        return newData;
    }

    public static Instances fixMissingValues(Instances data) {
        for (int att_idx = 0; att_idx < data.numAttributes(); att_idx++) {
            if (att_idx == data.classIndex()) continue;
            ArrayList<Double> arrayList = new ArrayList<>();
            double sum = 0;
            int count = 0;
            for (Instance instance : data) {
                if (instance.isMissing(att_idx)) continue;
                double value;
                try {
                    value = instance.value(att_idx);
                } catch (NumberFormatException e) {
                    // Treat non-numeric values as missing
                    instance.setMissing(att_idx);
                    continue;
                }
                arrayList.add(value);
                sum += value;
                count++;
            }
            if (count == 0) continue; // No valid values found for this attribute
            double mean = sum / count;
            DecimalFormat df = new DecimalFormat("#.##");
            double roundedMean = Double.parseDouble(df.format(mean));

            for (Instance instance : data) {
                if (instance.isMissing(att_idx)) {
                    // Replace missing values with the mean
                    instance.setValue(att_idx, roundedMean);
                }
            }
        }
        return data;
    }

    public static Instances numericToNominal(Instances data, String column_index) throws Exception {
        NumericToNominal numericToNominal = new NumericToNominal();
        numericToNominal.setAttributeIndices(column_index);
        numericToNominal.setInputFormat(data);
        data = NominalToBinary.useFilter(data, numericToNominal);
        return data;
    }

    public static Instances normalize(Instances data) {
        for (int i = 0 ; i < data.numAttributes(); i++) {
            if (data.classIndex() == i) continue;

            if (data.attribute(i).isNumeric()) {
                double mx = 0.0, mn = 1e9;

                // Find the maximum and minimum values of the attribute
                for (int j = 0 ; j < data.numInstances(); j++) {
                    mx = max(mx, data.instance(j).value(i));
                    mn = min(mn, data.instance(j).value(i));
                }

                // Scale the attribute values to the range [0, 1]
                for (int j = 0 ; j < data.numInstances(); j++) {
                    double val = data.instance(j).value(i);
                    data.instance(j).setValue(i, (val - mn/ (mx - mn)));
                }
            }
        }
        return data;
    }
    public static Instances newStandard(Instances data) {
        Instances newData = new Instances(data);
        newData.renameAttributeValue(0, 1, "1");
        newData.renameAttributeValue(0, 2, "1");
        newData.renameAttributeValue(0, 3, "1");
        newData.renameAttributeValue(0, 0, "0");
        return  newData;
    }
    public static void analyze_data(Instances data) {
        HashMap<String, Integer> _count = new HashMap<>();
        for (Instance instance : data) {
            String label = instance.stringValue(data.classIndex());
            if (_count.containsKey(label)) {
                _count.put(label, _count.get(label) + 1);
            } else {
                _count.put(label, 1);
            }
        }
        System.out.println(_count);
    }
}
