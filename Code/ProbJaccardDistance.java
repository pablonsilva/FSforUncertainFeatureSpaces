package thesis.ageing.classifiers.distance;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.PerformanceStats;

import java.io.Serializable;
import java.util.Enumeration;

/**
 * Created by pablonsilva on 4/28/17.
 */
public class ProbJaccardDistance implements DistanceFunction, Serializable {

    Instances _data;

    @Override
    public Instances getInstances() {
        return null;
    }

    @Override
    public void setInstances(Instances insts) {
        _data = insts;
    }

    @Override
    public String getAttributeIndices() {
        return null;
    }

    @Override
    public void setAttributeIndices(String value) {

    }

    @Override
    public boolean getInvertSelection() {
        return false;
    }

    @Override
    public void setInvertSelection(boolean value) {

    }

    @Override
    public double distance(Instance first, Instance second) {
        double m_11 = 0;
        double m_diff = 0;

        for (int i = 0; i < first.numAttributes() - 1; i++) {
            double v1 = first.value(i);
            double v2 = second.value(i);

            m_11 += v1*v2;
            m_diff += v1*(1-v2) + (1-v1)*v2;
        }

        double jaccard = m_11 / (m_11 + m_diff);

        return (1 - jaccard);
    }

    @Override
    public double distance(Instance first, Instance second, PerformanceStats stats) throws Exception {
        return distance(first, second);
    }

    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
        return distance(first, second);
    }

    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {
        return distance(first, second);
    }

    @Override
    public void postProcessDistances(double[] distances) {

    }

    @Override
    public void update(Instance ins) {

    }

    @Override
    public void clean() {

    }

    @Override
    public Enumeration listOptions() {
        return null;
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }
}

