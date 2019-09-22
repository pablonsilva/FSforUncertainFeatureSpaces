package thesis.ageing.uncertain;

import thesis.ageing.classifiers.NaiveBayesLazyV2;
import thesis.ageing.featureselection.attributeeval.FStatisticAttributeEval;

import thesis.ageing.utils.MathUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.util.ArrayList;

public class NBLazyFStat extends AbstractClassifier {

    private AbstractClassifier _cls;
    private double th;
    private int numInst;
    private int sumFeatSelected;
    private boolean keepGO;
    private boolean keep_all_gos;
    private Instances _instances;
    private FStatisticAttributeEval eval;
    private double threshold_ttest;

    public NBLazyFStat()
    {
        this(0,true);
    }

    public NBLazyFStat(double t)
    {
        this(t,true);
    }

    public NBLazyFStat(double t, boolean gos)
    {
        th = t;
        keep_all_gos = gos;
    }

    public void setTH(double t)
    {
        th = t;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        _cls = new NaiveBayesLazyV2();
        _cls.buildClassifier(instances);
        _instances = new Instances(instances);

        eval = new FStatisticAttributeEval();
        eval.buildEvaluator(instances);

        ArrayList<Double> values = new ArrayList<Double>();
        for(int i = 0 ; i < instances.numAttributes()-1;i++)
        {
            if (!instances.attribute(i).name().contains("GO:")) {
                double val = eval.evaluateAttribute(i) / (instances.numAttributes() - 1);
                if(!Double.isNaN(val)) {
                    threshold_ttest += val / (instances.numAttributes() - 1);
                    values.add(val);
                }
            }
        }
        double[] values2 = new double[values.size()];
        for(int i = 0 ; i < values.size();i++)
            values2[i] = values.get(i);
        threshold_ttest= MathUtils.average(values2);
        sumFeatSelected=0;
        numInst=0;
    }

    public double getAvgSelectedFeatures()
    {
        return ((double)sumFeatSelected)/numInst;
    }

    public ArrayList<Integer> getSelectedFeatures()
    {
        return selectedFeatures;
    }

    private ArrayList<Integer> selectedFeatures = new ArrayList<>();
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] threshold = {.150,0.400,0.700,0.900};
        int i = 0;

        StringBuilder sb = new StringBuilder();
        for(int a = 0 ;a < instance.numAttributes(); a++)
        {
            if (instance.attribute(a).name().contains("GO:")) {
                sb.append((instance.attribute(a).index() + 1) + ",");
                selectedFeatures.add(a);
            } else if (instance.value(a) >= th && !instance.attribute(a).name().contains("GO:")
                    && eval.evaluateAttribute(a) > threshold_ttest) {
                sb.append((instance.attribute(a).index() + 1) + ",");
                selectedFeatures.add(a);
            }
        }

        sb.append(instance.classIndex()+1);

        Remove rmv = new Remove();
        rmv.setAttributeIndices(sb.toString());
        rmv.setInvertSelection(true);
        rmv.setInputFormat(instance.dataset());

        Instances data_aux = new Instances(instance.dataset(),0);
        data_aux.add(instance);
        Instances data_train = Filter.useFilter(_instances,rmv);
        Instances data_rmv = Filter.useFilter(data_aux,rmv);

        sumFeatSelected += data_rmv.numAttributes()-1;
        numInst++;

        return _cls.distributionForInstance(data_rmv.firstInstance());
    }
}
