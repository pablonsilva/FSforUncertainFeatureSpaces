package thesis.ageing.classifiers;

import thesis.ageing.featureselection.general.TestClassifier;
import thesis.ageing.featureselection.hierarchy.published.HIP;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.estimators.DiscreteEstimator;
import weka.estimators.Estimator;
import weka.estimators.KernelEstimator;
import weka.estimators.NormalEstimator;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.Enumeration;
import java.util.Vector;

/**
 * Created by pablonsilva on 5/5/16.
 */

public class NaiveBayesLazy extends AbstractClassifier implements OptionHandler, WeightedInstancesHandler, TechnicalInformationHandler {
    protected static final double DEFAULT_NUM_PRECISION = 0.01D;
    static final long serialVersionUID = 5995231201785697655L;
    protected Estimator[][] m_Distributions;
    protected Estimator m_ClassDistribution;
    protected boolean m_UseKernelEstimator = false;
    protected boolean m_UseDiscretization = false;
    protected int m_NumClasses;
    protected Instances m_Instances;
    protected Discretize m_Disc = null;
    protected boolean m_displayModelInOldFormat = false;

    public NaiveBayesLazy() {
    }

    public static void main(String[] argv) {
        TestClassifier tc = new TestClassifier();

        tc.test(new HIP(new NaiveBayesLazy()));
    }

    public String globalInfo() {
        return "Class for a Naive Bayes classifier using estimator classes. Numeric estimator precision values are chosen based on analysis of the  training data. For this reason, the classifier is not an UpdateableClassifier (which in typical usage are initialized with zero training instances) -- if you need the UpdateableClassifier functionality, use the NaiveBayesUpdateable classifier. The NaiveBayesUpdateable classifier will  use a default precision of 0.1 for numeric attributes when buildClassifier is called with zero training instances.\n\nFor more information on Naive Bayes classifiers, see\n\n" + this.getTechnicalInformation().toString();
    }

    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "George H. John and Pat Langley");
        result.setValue(TechnicalInformation.Field.TITLE, "Estimating Continuous Distributions in Bayesian Classifiers");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Eleventh Conference on Uncertainty in Artificial Intelligence");
        result.setValue(TechnicalInformation.Field.YEAR, "1995");
        result.setValue(TechnicalInformation.Field.PAGES, "338-345");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Morgan Kaufmann");
        result.setValue(TechnicalInformation.Field.ADDRESS, "San Mateo");
        return result;
    }

    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capabilities.Capability.MISSING_VALUES);
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);
        result.setMinimumNumberInstances(0);
        return result;
    }

    public void buildClassifier(Instances instances) throws Exception {
        this.getCapabilities().testWithFail(instances);
        instances = new Instances(instances);
        instances.deleteWithMissingClass();
        this.m_NumClasses = instances.numClasses();
        this.m_Instances = new Instances(instances);
        if (this.m_UseDiscretization) {
            this.m_Disc = new Discretize();
            this.m_Disc.setInputFormat(this.m_Instances);
            this.m_Instances = Filter.useFilter(this.m_Instances, this.m_Disc);
        } else {
            this.m_Disc = null;
        }

        this.m_Distributions = new Estimator[this.m_Instances.numAttributes() - 1][this.m_Instances.numClasses()];
        this.m_ClassDistribution = new DiscreteEstimator(this.m_Instances.numClasses(), true);
        int attIndex = 0;

        for (Enumeration enu = this.m_Instances.enumerateAttributes(); enu.hasMoreElements(); ++attIndex) {
            Attribute enumInsts = (Attribute) enu.nextElement();
            double instance = 0.01D;
            if (enumInsts.type() == 0) {
                this.m_Instances.sort(enumInsts);
                if (this.m_Instances.numInstances() > 0 && !this.m_Instances.instance(0).isMissing(enumInsts)) {
                    double j = this.m_Instances.instance(0).value(enumInsts);
                    double deltaSum = 0.0D;
                    int distinct = 0;

                    for (int i = 1; i < this.m_Instances.numInstances(); ++i) {
                        Instance currentInst = this.m_Instances.instance(i);
                        if (currentInst.isMissing(enumInsts)) {
                            break;
                        }

                        double currentVal = currentInst.value(enumInsts);
                        if (currentVal != j) {
                            deltaSum += currentVal - j;
                            j = currentVal;
                            ++distinct;
                        }
                    }

                    if (distinct > 0) {
                        instance = deltaSum / (double) distinct;
                    }
                }
            }

            for (int var18 = 0; var18 < this.m_Instances.numClasses(); ++var18) {
                switch (enumInsts.type()) {
                    case 0:
                        if (this.m_UseKernelEstimator) {
                            this.m_Distributions[attIndex][var18] = new KernelEstimator(instance);
                        } else {
                            this.m_Distributions[attIndex][var18] = new NormalEstimator(instance);
                        }
                        break;
                    case 1:
                        this.m_Distributions[attIndex][var18] = new DiscreteEstimator(enumInsts.numValues(), true);
                        break;
                    default:
                        throw new Exception("Attribute type unknown to NaiveBayes");
                }
            }
        }

        Enumeration var16 = this.m_Instances.enumerateInstances();

        while (var16.hasMoreElements()) {
            Instance var17 = (Instance) var16.nextElement();
            this.updateClassifier(var17);
        }
        this.m_Instances = new Instances(this.m_Instances, 0);
    }

    public void updateClassifier(Instance instance) throws Exception {
        if (!instance.classIsMissing()) {
            Enumeration enumAtts = this.m_Instances.enumerateAttributes();

            for (int attIndex = 0; enumAtts.hasMoreElements(); ++attIndex) {
                Attribute attribute = (Attribute) enumAtts.nextElement();
                if (!instance.isMissing(attribute)) {
                    this.m_Distributions[attIndex][(int) instance.classValue()].addValue(instance.value(attribute), instance.weight());
                }
            }

            this.m_ClassDistribution.addValue(instance.classValue(), instance.weight());
        }

    }

    public double[] distributionForInstance(Instance instance) throws Exception {
        if (this.m_UseDiscretization) {
            this.m_Disc.input(instance);
            instance = this.m_Disc.output();
        }

        ///REMOVER
        // double[][] cls = new double[this.m_NumClasses][instance.dataset().numAttributes()];
        // int c = 0;
        ///

        double[] probs = new double[this.m_NumClasses];

        for (int enumAtts = 0; enumAtts < this.m_NumClasses; ++enumAtts) {
            probs[enumAtts] = this.m_ClassDistribution.getProbability((double) enumAtts);
            ///REMOVER
            //cls[enumAtts][0] =  probs[enumAtts];
        }

        Enumeration var11 = instance.enumerateAttributes();

        boolean useLazy = true;

        //Se a instancia nao for lazy usar o NB normal.
        if (instance.numAttributes() == m_Instances.numAttributes()) {
            useLazy = false;
        }

        int attIndexCorrected = 0;
        for (int attIndex = 0; var11.hasMoreElements(); ++attIndex) {
            Attribute attribute = (Attribute) var11.nextElement();
            if (!instance.isMissing(attribute)) {
                double max = 0.0D;

                if (useLazy)
                    attIndexCorrected = m_Instances.attribute(attribute.name()).index();
                else
                    attIndexCorrected = attIndex;

                int j;
                for (j = 0; j < this.m_NumClasses; ++j) {
                    double temp = Math.max(1.0E-75D, Math.pow(this.m_Distributions[attIndexCorrected][j].getProbability(instance.value(attribute)), this.m_Instances.attribute(attIndexCorrected).weight()));
                    //cls[j][c+1] = temp;
                    probs[j] *= temp;
                    if (probs[j] > max) {
                        max = probs[j];
                    }

                    if (Double.isNaN(probs[j])) {
                        throw new Exception("NaN returned from estimator for attribute " + attribute.name() + ":\n" + this.m_Distributions[attIndexCorrected][j].toString());
                    }
                }

                if (max > 0.0D && max < 1.0E-75D) {
                    for (j = 0; j < this.m_NumClasses; ++j) {
                        probs[j] *= 1.0E75D;
                    }
                }
            }
            //c++;
        }


        Utils.normalize(probs);

//        System.out.println(Utils.arrayToString(cls[0]));
//        System.out.println(Utils.arrayToString(cls[1]));
//        System.out.println(Utils.arrayToString(probs));
//        if(probs[0] > probs[1])
//            System.out.println(instance.classValue() + " - " + 0);
//        else
//            System.out.println(instance.classValue() + " - " + 1);


        return probs;
    }

    public Enumeration listOptions() {
        Vector newVector = new Vector(3);
        newVector.addElement(new Option("\tUse kernel density estimator rather than normal\n\tdistribution for numeric attributes", "K", 0, "-K"));
        newVector.addElement(new Option("\tUse supervised discretization to process numeric attributes\n", "D", 0, "-D"));
        newVector.addElement(new Option("\tDisplay model in old format (good when there are many classes)\n", "O", 0, "-O"));
        return newVector.elements();
    }

    public String[] getOptions() {
        String[] options = new String[3];
        int current = 0;
        if (this.m_UseKernelEstimator) {
            options[current++] = "-K";
        }

        if (this.m_UseDiscretization) {
            options[current++] = "-D";
        }

        if (this.m_displayModelInOldFormat) {
            options[current++] = "-O";
        }

        while (current < options.length) {
            options[current++] = "";
        }

        return options;
    }

    public void setOptions(String[] options) throws Exception {
        boolean k = Utils.getFlag('K', options);
        boolean d = Utils.getFlag('D', options);
        if (k && d) {
            throw new IllegalArgumentException("Can\'t use both kernel density estimation and discretization!");
        } else {
            this.setUseSupervisedDiscretization(d);
            this.setUseKernelEstimator(k);
            this.setDisplayModelInOldFormat(Utils.getFlag('O', options));
            Utils.checkForRemainingOptions(options);
        }
    }

    public String toString() {
        if (this.m_displayModelInOldFormat) {
            return this.toStringOriginal();
        } else {
            StringBuffer temp = new StringBuffer();
            temp.append("Naive Bayes Classifier");
            if (this.m_Instances == null) {
                temp.append(": No model built yet.");
            } else {
                int maxWidth = 0;
                int maxAttWidth = 0;
                boolean containsKernel = false;

                int counter;
                for (counter = 0; counter < this.m_Instances.numClasses(); ++counter) {
                    if (this.m_Instances.classAttribute().value(counter).length() > maxWidth) {
                        maxWidth = this.m_Instances.classAttribute().value(counter).length();
                    }
                }

                String kL;
                for (counter = 0; counter < this.m_Instances.numAttributes(); ++counter) {
                    if (counter != this.m_Instances.classIndex()) {
                        Attribute i = this.m_Instances.attribute(counter);
                        if (i.name().length() > maxAttWidth) {
                            maxAttWidth = this.m_Instances.attribute(counter).name().length();
                        }

                        if (i.isNominal()) {
                            for (int attName = 0; attName < i.numValues(); ++attName) {
                                kL = i.value(attName) + "  ";
                                if (kL.length() > maxAttWidth) {
                                    maxAttWidth = kL.length();
                                }
                            }
                        }
                    }
                }

                String stdDevL;
                int j;
                String meanL;
                int var19;
                for (counter = 0; counter < this.m_Distributions.length; ++counter) {
                    for (var19 = 0; var19 < this.m_Instances.numClasses(); ++var19) {
                        if (this.m_Distributions[counter][0] instanceof NormalEstimator) {
                            NormalEstimator var23 = (NormalEstimator) this.m_Distributions[counter][var19];
                            double var26 = Math.log(Math.abs(var23.getMean())) / Math.log(10.0D);
                            double var28 = Math.log(Math.abs(var23.getPrecision())) / Math.log(10.0D);
                            double var36 = var26 > var28 ? var26 : var28;
                            if (var36 < 0.0D) {
                                var36 = 1.0D;
                            }

                            var36 += 6.0D;
                            if ((int) var36 > maxWidth) {
                                maxWidth = (int) var36;
                            }
                        } else {
                            int var24;
                            if (this.m_Distributions[counter][0] instanceof KernelEstimator) {
                                containsKernel = true;
                                KernelEstimator var22 = (KernelEstimator) this.m_Distributions[counter][var19];
                                var24 = var22.getNumKernels();
                                stdDevL = "K" + var24 + ": mean (weight)";
                                if (maxAttWidth < stdDevL.length()) {
                                    maxAttWidth = stdDevL.length();
                                }

                                if (var22.getNumKernels() > 0) {
                                    double[] precL = var22.getMeans();
                                    double[] maxK = var22.getWeights();

                                    for (j = 0; j < var22.getNumKernels(); ++j) {
                                        meanL = Utils.doubleToString(precL[j], maxWidth, 4).trim();
                                        meanL = meanL + " (" + Utils.doubleToString(maxK[j], maxWidth, 1).trim() + ")";
                                        if (maxWidth < meanL.length()) {
                                            maxWidth = meanL.length();
                                        }
                                    }
                                }
                            } else if (this.m_Distributions[counter][0] instanceof DiscreteEstimator) {
                                DiscreteEstimator var21 = (DiscreteEstimator) this.m_Distributions[counter][var19];

                                for (var24 = 0; var24 < var21.getNumSymbols(); ++var24) {
                                    stdDevL = "" + var21.getCount((double) var24);
                                    if (stdDevL.length() > maxWidth) {
                                        maxWidth = stdDevL.length();
                                    }
                                }

                                var24 = ("" + var21.getSumOfCounts()).length();
                                if (var24 > maxWidth) {
                                    maxWidth = var24;
                                }
                            }
                        }
                    }
                }

                String var20;
                for (counter = 0; counter < this.m_Instances.numClasses(); ++counter) {
                    var20 = this.m_Instances.classAttribute().value(counter);
                    if (var20.length() > maxWidth) {
                        maxWidth = var20.length();
                    }
                }

                for (counter = 0; counter < this.m_Instances.numClasses(); ++counter) {
                    var20 = Utils.doubleToString(this.m_ClassDistribution.getProbability((double) counter), maxWidth, 2).trim();
                    var20 = "(" + var20 + ")";
                    if (var20.length() > maxWidth) {
                        maxWidth = var20.length();
                    }
                }

                if (maxAttWidth < "Attribute".length()) {
                    maxAttWidth = "Attribute".length();
                }

                if (maxAttWidth < "  weight sum".length()) {
                    maxAttWidth = "  weight sum".length();
                }

                if (containsKernel && maxAttWidth < "  [precision]".length()) {
                    maxAttWidth = "  [precision]".length();
                }

                maxAttWidth += 2;
                temp.append("\n\n");
                temp.append(this.pad("Class", " ", maxAttWidth + maxWidth + 1 - "Class".length(), true));
                temp.append("\n");
                temp.append(this.pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

                for (counter = 0; counter < this.m_Instances.numClasses(); ++counter) {
                    var20 = this.m_Instances.classAttribute().value(counter);
                    temp.append(this.pad(var20, " ", maxWidth + 1 - var20.length(), true));
                }

                temp.append("\n");
                temp.append(this.pad("", " ", maxAttWidth, true));

                for (counter = 0; counter < this.m_Instances.numClasses(); ++counter) {
                    var20 = Utils.doubleToString(this.m_ClassDistribution.getProbability((double) counter), maxWidth, 2).trim();
                    var20 = "(" + var20 + ")";
                    temp.append(this.pad(var20, " ", maxWidth + 1 - var20.length(), true));
                }

                temp.append("\n");
                temp.append(this.pad("", "=", maxAttWidth + maxWidth * this.m_Instances.numClasses() + this.m_Instances.numClasses() + 1, true));
                temp.append("\n");
                counter = 0;

                for (var19 = 0; var19 < this.m_Instances.numAttributes(); ++var19) {
                    if (var19 != this.m_Instances.classIndex()) {
                        String var25 = this.m_Instances.attribute(var19).name();
                        temp.append(var25 + "\n");
                        int var27;
                        int var31;
                        String var32;
                        String var33;
                        int var35;
                        String var39;
                        if (this.m_Distributions[counter][0] instanceof NormalEstimator) {
                            kL = "  mean";
                            temp.append(this.pad(kL, " ", maxAttWidth + 1 - kL.length(), false));

                            for (var27 = 0; var27 < this.m_Instances.numClasses(); ++var27) {
                                NormalEstimator var30 = (NormalEstimator) this.m_Distributions[counter][var27];
                                var32 = Utils.doubleToString(var30.getMean(), maxWidth, 4).trim();
                                temp.append(this.pad(var32, " ", maxWidth + 1 - var32.length(), true));
                            }

                            temp.append("\n");
                            stdDevL = "  std. dev.";
                            temp.append(this.pad(stdDevL, " ", maxAttWidth + 1 - stdDevL.length(), false));

                            for (var31 = 0; var31 < this.m_Instances.numClasses(); ++var31) {
                                NormalEstimator var34 = (NormalEstimator) this.m_Distributions[counter][var31];
                                var39 = Utils.doubleToString(var34.getStdDev(), maxWidth, 4).trim();
                                temp.append(this.pad(var39, " ", maxWidth + 1 - var39.length(), true));
                            }

                            temp.append("\n");
                            var33 = "  weight sum";
                            temp.append(this.pad(var33, " ", maxAttWidth + 1 - var33.length(), false));

                            for (var35 = 0; var35 < this.m_Instances.numClasses(); ++var35) {
                                NormalEstimator var41 = (NormalEstimator) this.m_Distributions[counter][var35];
                                meanL = Utils.doubleToString(var41.getSumOfWeights(), maxWidth, 4).trim();
                                temp.append(this.pad(meanL, " ", maxWidth + 1 - meanL.length(), true));
                            }

                            temp.append("\n");
                            var32 = "  precision";
                            temp.append(this.pad(var32, " ", maxAttWidth + 1 - var32.length(), false));

                            for (j = 0; j < this.m_Instances.numClasses(); ++j) {
                                NormalEstimator var40 = (NormalEstimator) this.m_Distributions[counter][j];
                                String k = Utils.doubleToString(var40.getPrecision(), maxWidth, 4).trim();
                                temp.append(this.pad(k, " ", maxWidth + 1 - k.length(), true));
                            }

                            temp.append("\n\n");
                        } else if (this.m_Distributions[counter][0] instanceof DiscreteEstimator) {
                            Attribute var29 = this.m_Instances.attribute(var19);

                            for (var27 = 0; var27 < var29.numValues(); ++var27) {
                                var33 = "  " + var29.value(var27);
                                temp.append(this.pad(var33, " ", maxAttWidth + 1 - var33.length(), false));

                                for (var35 = 0; var35 < this.m_Instances.numClasses(); ++var35) {
                                    DiscreteEstimator var45 = (DiscreteEstimator) this.m_Distributions[counter][var35];
                                    meanL = "" + var45.getCount((double) var27);
                                    temp.append(this.pad(meanL, " ", maxWidth + 1 - meanL.length(), true));
                                }

                                temp.append("\n");
                            }

                            stdDevL = "  [total]";
                            temp.append(this.pad(stdDevL, " ", maxAttWidth + 1 - stdDevL.length(), false));

                            for (var31 = 0; var31 < this.m_Instances.numClasses(); ++var31) {
                                DiscreteEstimator var38 = (DiscreteEstimator) this.m_Distributions[counter][var31];
                                var39 = "" + var38.getSumOfCounts();
                                temp.append(this.pad(var39, " ", maxWidth + 1 - var39.length(), true));
                            }

                            temp.append("\n\n");
                        } else if (this.m_Distributions[counter][0] instanceof KernelEstimator) {
                            kL = "  [# kernels]";
                            temp.append(this.pad(kL, " ", maxAttWidth + 1 - kL.length(), false));

                            for (var27 = 0; var27 < this.m_Instances.numClasses(); ++var27) {
                                KernelEstimator var37 = (KernelEstimator) this.m_Distributions[counter][var27];
                                var32 = "" + var37.getNumKernels();
                                temp.append(this.pad(var32, " ", maxWidth + 1 - var32.length(), true));
                            }

                            temp.append("\n");
                            stdDevL = "  [std. dev]";
                            temp.append(this.pad(stdDevL, " ", maxAttWidth + 1 - stdDevL.length(), false));

                            for (var31 = 0; var31 < this.m_Instances.numClasses(); ++var31) {
                                KernelEstimator var42 = (KernelEstimator) this.m_Distributions[counter][var31];
                                var39 = Utils.doubleToString(var42.getStdDev(), maxWidth, 4).trim();
                                temp.append(this.pad(var39, " ", maxWidth + 1 - var39.length(), true));
                            }

                            temp.append("\n");
                            var33 = "  [precision]";
                            temp.append(this.pad(var33, " ", maxAttWidth + 1 - var33.length(), false));

                            for (var35 = 0; var35 < this.m_Instances.numClasses(); ++var35) {
                                KernelEstimator var46 = (KernelEstimator) this.m_Distributions[counter][var35];
                                meanL = Utils.doubleToString(var46.getPrecision(), maxWidth, 4).trim();
                                temp.append(this.pad(meanL, " ", maxWidth + 1 - meanL.length(), true));
                            }

                            temp.append("\n");
                            var35 = 0;

                            for (j = 0; j < this.m_Instances.numClasses(); ++j) {
                                KernelEstimator var44 = (KernelEstimator) this.m_Distributions[counter][j];
                                if (var44.getNumKernels() > var35) {
                                    var35 = var44.getNumKernels();
                                }
                            }

                            for (j = 0; j < var35; ++j) {
                                meanL = "  K" + (j + 1) + ": mean (weight)";
                                temp.append(this.pad(meanL, " ", maxAttWidth + 1 - meanL.length(), false));

                                for (int var43 = 0; var43 < this.m_Instances.numClasses(); ++var43) {
                                    KernelEstimator ke = (KernelEstimator) this.m_Distributions[counter][var43];
                                    double[] means = ke.getMeans();
                                    double[] weights = ke.getWeights();
                                    String m = "--";
                                    if (ke.getNumKernels() == 0) {
                                        m = "0";
                                    } else if (j < ke.getNumKernels()) {
                                        m = Utils.doubleToString(means[j], maxWidth, 4).trim();
                                        m = m + " (" + Utils.doubleToString(weights[j], maxWidth, 1).trim() + ")";
                                    }

                                    temp.append(this.pad(m, " ", maxWidth + 1 - m.length(), true));
                                }

                                temp.append("\n");
                            }

                            temp.append("\n");
                        }

                        ++counter;
                    }
                }
            }

            return temp.toString();
        }
    }

    protected String toStringOriginal() {
        StringBuffer text = new StringBuffer();
        text.append("Naive Bayes Classifier");
        if (this.m_Instances == null) {
            text.append(": No model built yet.");
        } else {
            try {
                for (int ex = 0; ex < this.m_Distributions[0].length; ++ex) {
                    text.append("\n\nClass " + this.m_Instances.classAttribute().value(ex) + ": Prior probability = " + Utils.doubleToString(this.m_ClassDistribution.getProbability((double) ex), 4, 2) + "\n\n");
                    Enumeration enumAtts = this.m_Instances.enumerateAttributes();

                    for (int attIndex = 0; enumAtts.hasMoreElements(); ++attIndex) {
                        Attribute attribute = (Attribute) enumAtts.nextElement();
                        if (attribute.weight() > 0.0D) {
                            text.append(attribute.name() + ":  " + this.m_Distributions[attIndex][ex]);
                        }
                    }
                }
            } catch (Exception var6) {
                text.append(var6.getMessage());
            }
        }

        return text.toString();
    }

    private String pad(String source, String padChar, int length, boolean leftPad) {
        StringBuffer temp = new StringBuffer();
        int i;
        if (leftPad) {
            for (i = 0; i < length; ++i) {
                temp.append(padChar);
            }

            temp.append(source);
        } else {
            temp.append(source);

            for (i = 0; i < length; ++i) {
                temp.append(padChar);
            }
        }

        return temp.toString();
    }

    public String useKernelEstimatorTipText() {
        return "Use a kernel estimator for numeric attributes rather than a normal distribution.";
    }

    public boolean getUseKernelEstimator() {
        return this.m_UseKernelEstimator;
    }

    public void setUseKernelEstimator(boolean v) {
        this.m_UseKernelEstimator = v;
        if (v) {
            this.setUseSupervisedDiscretization(false);
        }

    }

    public String useSupervisedDiscretizationTipText() {
        return "Use supervised discretization to convert numeric attributes to nominal ones.";
    }

    public boolean getUseSupervisedDiscretization() {
        return this.m_UseDiscretization;
    }

    public void setUseSupervisedDiscretization(boolean newblah) {
        this.m_UseDiscretization = newblah;
        if (newblah) {
            this.setUseKernelEstimator(false);
        }
    }

    public String displayModelInOldFormatTipText() {
        return "Use old format for model output. The old format is better when there are many class values. The new format is better when there are fewer classes and many attributes.";
    }

    public boolean getDisplayModelInOldFormat() {
        return this.m_displayModelInOldFormat;
    }

    public void setDisplayModelInOldFormat(boolean d) {
        this.m_displayModelInOldFormat = d;
    }

    public String getRevision() {
        return RevisionUtils.extract("$Revision: 5516 $");
    }
}