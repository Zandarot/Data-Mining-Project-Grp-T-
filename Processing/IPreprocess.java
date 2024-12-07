package Processing;

import weka.core.Instances;

public interface IPreprocess {
    public abstract Instances apply(Instances data) throws Exception;
}
