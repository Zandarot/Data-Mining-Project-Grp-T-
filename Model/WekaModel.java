package Model;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class WekaModel implements ModelBase {
    public Classifier classifier;

    public WekaModel(Classifier classifier) {
        this.classifier = classifier;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        this.classifier.buildClassifier(instances);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return this.classifier.classifyInstance(instance);
    }

    @Override
    public WekaModel copy() throws Exception {
        AbstractClassifier classifier = (AbstractClassifier) AbstractClassifier.makeCopy(this.classifier);
        return new WekaModel(classifier);
    }

    @Override
    public String modelName() {
        return this.classifier.getClass().getSimpleName();
    }
}
