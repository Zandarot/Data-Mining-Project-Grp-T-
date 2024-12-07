package Model;

import weka.core.Instance;
import weka.core.Instances;

public interface ModelBase {
    public void buildClassifier(Instances instances) throws Exception;
    public double classifyInstance(Instance instance) throws Exception;
    public ModelBase copy() throws Exception;
    public String modelName();
}
