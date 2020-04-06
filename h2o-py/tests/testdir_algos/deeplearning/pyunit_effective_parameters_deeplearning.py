from __future__ import print_function
from builtins import range
import sys
sys.path.insert(1,"../../../")
import h2o
from tests import pyunit_utils
import tests
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

#testing default setup of following parameters:
#distribution (available in Deep Learning, XGBoost, GBM):
#stopping_metric (available in: GBM, DRF, Deep Learning, AutoML, XGBoost, Isolation Forest):
#histogram_type (available in: GBM, DRF)
#solver (available in: GLM) already done in hex.glm.GLM.defaultSolver()
#categorical_encoding (available in: GBM, DRF, Deep Learning, K-Means, Aggregator, XGBoost, Isolation Forest)
#fold_assignment (available in: GBM, DRF, Deep Learning, GLM, Naïve-Bayes, K-Means, XGBoost)

def test_deep_learning_effective_parameters():
    train_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_model.csv"))
    train_data = train_data.drop('Site')
    train_data['Angaus'] = train_data['Angaus'].asfactor()

    test_data = h2o.import_file(path=tests.locate("smalldata/gbm_test/ecology_eval.csv"))
    test_data['Angaus'] = test_data['Angaus'].asfactor()

    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, stopping_rounds=5)
    dl1.train(x=list(range(1, train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True,
                                   distribution="bernoulli", categorical_encoding="OneHotInternal", stopping_rounds=5)
    dl2.train(x=list(range(1, train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    assert dl1.effective_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.effective_params['stopping_metric'] == dl2.effective_params['stopping_metric']
    assert dl1.effective_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.effective_params['fold_assignment'] is None

    dl1 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True, nfolds=5)
    dl1.train(x=list(range(1,train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    dl2 = H2ODeepLearningEstimator(loss="CrossEntropy", epochs=1000, hidden=[20,20,20], seed=1234, reproducible=True,
                                   distribution="bernoulli", categorical_encoding="OneHotInternal", nfolds=5, fold_assignment="Random")
    dl2.train(x=list(range(1,train_data.ncol)), y="Angaus", training_frame=train_data, validation_frame=test_data)

    assert dl1.effective_params['distribution'] == dl2.actual_params['distribution']
    assert dl1.logloss() == dl2.logloss()
    assert dl1.effective_params['stopping_metric'] is None
    assert dl1.effective_params['categorical_encoding'] == dl2.actual_params['categorical_encoding']
    assert dl1.effective_params['fold_assignment'] == dl2.actual_params['fold_assignment']

if __name__ == "__main__":
  pyunit_utils.standalone_test(test_deep_learning_effective_parameters)
else:
    test_deep_learning_effective_parameters()
