package Processing;

import java.util.*;

import weka.core.Instances;
import weka.core.Instance;
import weka.core.DenseInstance;

public class Smote implements IPreprocess {
    private int K = 10;
    public Random rand;
    public HashMap<String, Integer> samplingStrategy;

    public Smote(HashMap<String, Integer> samplingStrategy, int k, Random rand) {
        this.samplingStrategy = samplingStrategy;
        this.K = k;
        this.rand = rand;
    }

    public Instances applySmote(Instances data) {
        // Find minority class instances
        HashMap<String, ArrayList<Instance>> classInstances = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            Instance instance = data.instance(i);
            String classValue = instance.stringValue(5);
            classInstances.computeIfAbsent(classValue, k -> new ArrayList<>()).add(instance);
        }

        // Create balanced data
        Instances balancedData = new Instances(data);
        for (String classValue : classInstances.keySet()) {
            ArrayList<Instance> currentInstances = classInstances.get(classValue);

            int requiredInstances = this.samplingStrategy.getOrDefault(classValue, currentInstances.size());
            if (currentInstances.size() >= requiredInstances) continue;

            // Generate synthetic instances
            int numSynthetic = requiredInstances - currentInstances.size();
            for (int i = 0; i < numSynthetic; i++) {
                Instance targetInstance = currentInstances.get(this.rand.nextInt(currentInstances.size()));
                ArrayList<Instance> nearestNeighbors = findKNearestNeighbors(currentInstances, targetInstance, this.K);
                Instance syntheticInstance = generateSyntheticInstance(targetInstance, nearestNeighbors, this.rand);
                balancedData.add(syntheticInstance);
            }
        }
        return balancedData;
    }

    private ArrayList<Instance> findKNearestNeighbors(ArrayList<Instance> instances, Instance targetInstance, int k) {
        /*
        @params:
            instances: Dataset
            targetInstance: Instance to find k nearest neighbors
            k: Number of nearest neighbors
        @return:
            kNearestNeighbors: List of k nearest neighbors of targetInstance
         */
        PriorityQueue<Instance> queue = new PriorityQueue<>(Comparator.comparingDouble(i -> calculateDistance(targetInstance, i)));
        // ArrayList<Instance> kNearestInstances = new ArrayList<>();
        for (Instance instance : instances) {
            if (!instance.equals(targetInstance)) {
                queue.add(instance);
                if (queue.size() > k) queue.poll();
            }
        }
        return new ArrayList<>(queue);
    }

    public Instance generateSyntheticInstance(Instance instance, ArrayList<Instance> neighbors, Random rand) {
        /*
        @params:
            instance: Instance to be synthesized
            neighbor: Instance in the neighborhood of instance
            random: Random number in range [0, 1]
        @return:
            syntheticInstance: Synthetic instance generated from instance and neighbor

        * Generate synthetic instance =   instance + random * (neighbor - instance)
        * If attribute not numeric, synthetic instance's attribute value is the same as instance's
        */
        Instance neighbor = neighbors.get(rand.nextInt(neighbors.size()));
        DenseInstance syntheticInstance = new DenseInstance(instance);
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (instance.attribute(i).isNumeric()) {
                double diff = neighbor.value(i) - instance.value(i);
                syntheticInstance.setValue(i, instance.value(i) + rand.nextDouble() * diff);
            }
        }
        return syntheticInstance;
    }

    protected double calculateDistance(Instance a, Instance b) {
        /*
        @params:
            a: Instance a
            b: Instance b
        @return:
            distance: Distance between a and b
         */
        double distance = 0;
        for (int i = 0; i < a.numAttributes(); i++) {
            if (a.attribute(i).isNumeric()) {
                distance += Math.pow(a.value(i) - b.value(i), 2);
            }
        }
        return Math.sqrt(distance);
    }

    @Override
    public Instances apply(Instances data) {
        return this.applySmote(data);
    }
}
