package thesis.ageing.uncertain;

import thesis.ageing.experiments.basics.EvaluationAdapted;
import weka.classifiers.AbstractClassifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class JaccardLazyICVFStat extends AbstractClassifier {

    private AbstractClassifier _cls;
    private int best_th;
    private boolean keep_all_gos;

    @Override
    public void buildClassifier(Instances data) throws Exception {

        double[] threshold = {.150,0.400,0.700,0.900};
        double best = 0;
        best_th = 0;

        for (int i = 0; i < threshold.length; i++) {
            JaccardLazyFStat nb = new JaccardLazyFStat(threshold[i]);
            nb.buildClassifier(data);

            EvaluationAdapted eval = new EvaluationAdapted(data);
            eval.crossValidateModel(nb, data, 5, new Random(1));

            double gm = eval.gMean();

            if (gm > best) {
                best_th = i;
                best = gm;
            }
        }

        _cls = new JaccardLazyFStat(threshold[best_th],keep_all_gos);
        _cls.buildClassifier(data);
    }

    public ArrayList<Integer> getSelectedFeatures()
    {
        return ((JaccardLazyFStat)_cls).getSelectedFeatures();
    }

    public void set_select_allGOS(boolean x)
    {
        keep_all_gos = x;
    }

    public double getAvgSelectedFeatures()
    {
        return ((JaccardLazyFStat)_cls).getAvgSelectedFeatures();
    }

    public double getBestThreshold()
    {
        double[] threshold = {.150,0.400,0.700,0.900};
        return threshold[best_th];
    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        return _cls.distributionForInstance(instance);
    }
}
