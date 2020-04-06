package hex.util;

import hex.Model;
import hex.ScoreKeeper;
import hex.genmodel.utils.DistributionFamily;
import hex.tree.SharedTreeModel;


public class EffectiveParametersUtils {
    
    public static void initFoldAssignment(
        Model.Parameters params,
        Model.Parameters effectiveParams
    ) {
        if (params._fold_assignment == Model.Parameters.FoldAssignmentScheme.AUTO) {
            if (params._nfolds > 0 && params._fold_column == null){
                effectiveParams._fold_assignment = Model.Parameters.FoldAssignmentScheme.Random;
            } else {
                effectiveParams._fold_assignment = null;
            }
        }
    }
    
    public static void initHistogramType(
            SharedTreeModel.SharedTreeParameters params,
            SharedTreeModel.SharedTreeParameters effectiveParams
    ) {
        if (params._histogram_type == SharedTreeModel.SharedTreeParameters.HistogramType.AUTO) {
            effectiveParams._histogram_type = SharedTreeModel.SharedTreeParameters.HistogramType.UniformAdaptive;
        }
    }
    
    public static void initStoppingMetric(
            Model.Parameters params,
            Model.Parameters effectiveParams,
            boolean isClassifier,
            boolean isAutoencoder
    ) {
        if (params._stopping_metric == ScoreKeeper.StoppingMetric.AUTO) {
            if (params._stopping_rounds == 0) {
                effectiveParams._stopping_metric = null;
            } else {
                if (isClassifier) {
                    effectiveParams._stopping_metric = ScoreKeeper.StoppingMetric.logloss;
                } else if (isAutoencoder) {
                    effectiveParams._stopping_metric = ScoreKeeper.StoppingMetric.MSE;
                } else {
                    effectiveParams._stopping_metric = ScoreKeeper.StoppingMetric.deviance;
                }
            }
        }
    }
    
    public static void initDistribution(
            Model.Parameters params,
            Model.Parameters effectiveParams,
            int nclasses
    ) {
        if (params._distribution == DistributionFamily.AUTO) {
            if (nclasses == 1) {
                effectiveParams._distribution = DistributionFamily.gaussian;}
            if (nclasses == 2) {
                effectiveParams._distribution = DistributionFamily.bernoulli;}
            if (nclasses >= 3) {
                effectiveParams._distribution = DistributionFamily.multinomial;}
        }
    }

    public static void initCategoricalEncoding(
            Model.Parameters params,
            Model.Parameters effectiveParams,
            int nclasses,
            Model.Parameters.CategoricalEncodingScheme scheme
    ) {
        if (params._categorical_encoding == Model.Parameters.CategoricalEncodingScheme.AUTO) {
            if (nclasses == 1)
                effectiveParams._categorical_encoding = null;
            else
                effectiveParams._categorical_encoding = scheme;
        }
        
    }
}
