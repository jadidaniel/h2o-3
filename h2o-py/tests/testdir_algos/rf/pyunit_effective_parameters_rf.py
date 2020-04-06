from __future__ import print_function
import sys
import h2o
import numpy as np
sys.path.insert(1,"../../../")
from tests import pyunit_utils
from h2o.estimators.random_forest import H2ORandomForestEstimator

#testing default setup of following parameters:
#distribution (available in Deep Learning, XGBoost, GBM):
#stopping_metric (available in: GBM, DRF, Deep Learning, AutoML, XGBoost, Isolation Forest):
#histogram_type (available in: GBM, DRF)
#solver (available in: GLM) already done in hex.glm.GLM.defaultSolver()
#categorical_encoding (available in: GBM, DRF, Deep Learning, K-Means, Aggregator, XGBoost, Isolation Forest)
#fold_assignment (available in: GBM, DRF, Deep Learning, GLM, Naïve-Bayes, K-Means, XGBoost)


def test_random_forrest_effective_parameters():
    frame = h2o.import_file(path=pyunit_utils.locate("smalldata/gbm_test/ecology_model.csv"))
    frame["Angaus"] = frame["Angaus"].asfactor()
    frame["Weights"] = h2o.H2OFrame.from_python(abs(np.random.randn(frame.nrow, 1)).tolist())[0]
    train, calib = frame.split_frame(ratios=[.8], destination_frames=["eco_train", "eco_calib"], seed=42)

    rf1 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights",
                                   stopping_rounds = 3, calibrate_model=True, calibration_frame=calib, seed = 1234)
    rf1.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    rf2 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights",
                                   stopping_rounds = 3, stopping_metric='logloss', calibrate_model=True, calibration_frame=calib,
                                   seed = 1234, categorical_encoding = 'Enum')
    rf2.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    assert rf1.effective_params['stopping_metric'] == rf2.actual_params['stopping_metric']
    assert rf1.logloss() == rf2.logloss()
    assert rf1.effective_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.effective_params['categorical_encoding'] == rf2.effective_params['categorical_encoding']
    assert rf1.effective_params['fold_assignment'] == None

    rf1 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights",
                                   nfolds = 5, calibrate_model=True, calibration_frame=calib, seed = 1234)
    rf1.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    rf2 = H2ORandomForestEstimator(ntrees=100, distribution="bernoulli", min_rows=10, max_depth=5, weights_column="Weights",
                                   nfolds=5, fold_assignment='Random', calibrate_model=True, calibration_frame=calib, seed = 1234,
                                   categorical_encoding = 'Enum')
    rf2.train(x=list(range(2, train.ncol)), y="Angaus", training_frame=train)

    assert rf1.effective_params['stopping_metric'] is None
    assert rf1.logloss() == rf2.logloss()
    assert rf1.effective_params['distribution'] == rf2.actual_params['distribution']
    assert rf1.effective_params['fold_assignment'] == rf2.actual_params['fold_assignment']
    assert rf1.effective_params['categorical_encoding'] == rf2.actual_params['categorical_encoding']


if __name__ == "__main__":
  pyunit_utils.standalone_test(test_random_forrest_effective_parameters)
else:
    test_random_forrest_effective_parameters()
