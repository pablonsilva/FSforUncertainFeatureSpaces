package thesis.ageing.featureselection.attributeeval;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;

import java.util.Enumeration;

/**
 * Created by pablonsilva on 09/02/2018.
 */
public class FStatisticAttributeEval extends ASEvaluation implements AttributeEvaluator, OptionHandler {
    private double[] m_fStatistics;

    public void buildEvaluator(Instances data) throws Exception {
        final int numClasses = 2;
        final int numInstances = data.numInstances();
        final int numFeatures = data.numAttributes() - 1; // discard the class
        int numInstances0 = 0;          // count number of instances of class 0
        int numInstances1 = 0;          // count number of instances of class 1

        double[][] mean = new double[numFeatures][numClasses + 1];
        double[][] variance = new double[numFeatures][numClasses];

        for (int j = 0; j < numFeatures; j++) {
            mean[j][0] = 0;             // mean of attribute j in instances of class 0
            mean[j][1] = 0;             // mean of attribute j in instances of class 1
            mean[j][2] = 0;             // mean of attribute j in all instances
            variance[j][0] = 0;         // variance of attribute j in instances of class 0
            variance[j][1] = 0;         // variance of attribute j in instances of class 1
        }

        for (int i = 0; i < numInstances; i++) {
            int c = (int)(data.instance(i).classValue());
            if (c == 0) {
                numInstances0++;
            } else {
                numInstances1++;
            }
            for (int j = 0; j < numFeatures; j++) {
                mean[j][c] += data.instance(i).value(j);
                mean[j][2] += data.instance(i).value(j);
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            mean[j][0] /= numInstances0;
            mean[j][1] /= numInstances1;
            mean[j][2] /= numInstances;
        }

        for (int i = 0; i < numInstances; i++) {
            for (int j = 0; j < numFeatures; j++) {
                int c = (int)(data.instance(i).classValue());
                double v = data.instance(i).value(j);
                double meanV = mean[j][c];
                variance[j][c] += (v-meanV)*(v-meanV);
            }
        }
        for (int j = 0; j < numFeatures; j++) {
            variance[j][0] /= numInstances0 - 1;
            variance[j][1] /= numInstances1 - 1;
        }

        this.m_fStatistics = new double[numFeatures];

        for(int j = 0; j < numFeatures; j++) {
            double numerator0 = (numInstances0/(numClasses-1.0)) * (mean[j][0] - mean[j][2]) * (mean[j][0] - mean[j][2]);
            double numerator1 = (numInstances1/(numClasses-1.0)) * (mean[j][1] - mean[j][2]) * (mean[j][1] - mean[j][2]);
            double numerator = numerator0 + numerator1;
            double denominator0 = (numInstances0 - 1.0) * variance[j][0];
            double denominator1 = (numInstances1 - 1.0) * variance[j][1];
            double denominator = (1.0 / (numInstances - numClasses)) * (denominator0 + denominator1);
            this.m_fStatistics[j] = numerator/denominator;
        }
    }

    protected void resetOptions() {
        this.m_fStatistics = null;
    }

    public double evaluateAttribute(int attribute) throws Exception {
        return this.m_fStatistics[attribute];
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {

    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}