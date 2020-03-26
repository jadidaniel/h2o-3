package hex.tree.xgboost;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.gbm.GBTree;
import biz.k11i.xgboost.gbm.GradBooster;
import biz.k11i.xgboost.tree.RegTree;
import biz.k11i.xgboost.tree.RegTreeNode;
import hex.*;
import hex.genmodel.algos.tree.*;
import hex.genmodel.algos.xgboost.XGBoostJavaMojoModel;
import hex.genmodel.algos.xgboost.XGBoostMojoModel;
import hex.genmodel.utils.DistributionFamily;
import hex.tree.PlattScalingHelper;
import hex.tree.xgboost.predict.*;
import hex.tree.xgboost.util.BoosterHelper;
import hex.tree.xgboost.util.PredictConfiguration;
import ml.dmlc.xgboost4j.java.*;
import water.*;
import water.codegen.CodeGeneratorPipeline;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.JCodeGen;
import water.util.Log;
import water.util.SBPrintStream;

import java.io.*;
import java.util.*;
import java.util.stream.Stream;

import static hex.genmodel.algos.xgboost.XGBoostMojoModel.ObjectiveType;
import static hex.tree.xgboost.XGBoost.makeDataInfo;
import static water.H2O.OptArgs.SYSTEM_PROP_PREFIX;

public class XGBoostModel extends Model<XGBoostModel, XGBoostModel.XGBoostParameters, XGBoostOutput> 
        implements SharedTreeGraphConverter, Model.LeafNodeAssignment, Model.Contributions {

  private static final String PROP_VERBOSITY = H2O.OptArgs.SYSTEM_PROP_PREFIX + ".xgboost.verbosity";
  private static final String PROP_NTHREAD = SYSTEM_PROP_PREFIX + "xgboost.nthreadMax";

  private XGBoostModelInfo model_info;

  public XGBoostModelInfo model_info() { return model_info; }

  public static class XGBoostParameters extends Model.Parameters implements Model.GetNTrees, PlattScalingHelper.ParamsWithCalibration {
    public enum TreeMethod {
      auto, exact, approx, hist
    }
    public enum GrowPolicy {
      depthwise, lossguide
    }
    public enum Booster {
      gbtree, gblinear, dart
    }
    public enum DartSampleType {
      uniform, weighted
    }
    public enum DartNormalizeType {
      tree, forest
    }
    public enum DMatrixType {
      auto, dense, sparse
    }
    public enum Backend {
      auto, gpu, cpu
    }

    // H2O GBM options
    public boolean _quiet_mode = true;

    public int _ntrees = 50; // Number of trees in the final model. Grid Search, comma sep values:50,100,150,200
    /**
     * @deprecated will be removed in 3.30.0.1, use _ntrees
     */
    public int _n_estimators; // This doesn't seem to be used anywhere... (not in clients)

    public int _max_depth = 6; // Maximum tree depth. Grid Search, comma sep values:5,7

    public double _min_rows = 1;
    public double _min_child_weight = 1;

    public double _learn_rate = 0.3;
    public double _eta = 0.3;

    public double _learn_rate_annealing = 1;

    public double _sample_rate = 1.0;
    public double _subsample = 1.0;

    public double _col_sample_rate = 1.0;
    public double _colsample_bylevel = 1.0;

    public double _col_sample_rate_per_tree = 1.0; //fraction of columns to sample for each tree
    public double _colsample_bytree = 1.0;

    public KeyValue[] _monotone_constraints;

    public float _max_abs_leafnode_pred = 0;
    public float _max_delta_step = 0;

    public int _score_tree_interval = 0; // score every so many trees (no matter what)
    public int _initial_score_interval = 4000; //Adding this parameter to take away the hard coded value of 4000 for scoring the first  4 secs
    public int _score_interval = 4000; //Adding this parameter to take away the hard coded value of 4000 for scoring each iteration every 4 secs
    public float _min_split_improvement = 0;
    public float _gamma;

    // Runtime options
    public int _nthread = -1;
    public String _save_matrix_directory; // dump the xgboost matrix to this directory
    public boolean _build_tree_one_node = false; // force to run on single node

    // LightGBM specific (only for grow_policy == lossguide)
    public int _max_bins = 256;
    public int _max_leaves = 0;
    public float _min_sum_hessian_in_leaf = 100;
    public float _min_data_in_leaf = 0;

    // XGBoost specific options
    public TreeMethod _tree_method = TreeMethod.auto;
    public GrowPolicy _grow_policy = GrowPolicy.depthwise;
    public Booster _booster = Booster.gbtree;
    public DMatrixType _dmatrix_type = DMatrixType.auto;
    public float _reg_lambda = 1;
    public float _reg_alpha = 0;
    
    // Platt scaling
    public boolean _calibrate_model;
    public Key<Frame> _calibration_frame;

    // Dart specific (booster == dart)
    public DartSampleType _sample_type = DartSampleType.uniform;
    public DartNormalizeType _normalize_type = DartNormalizeType.tree;
    public float _rate_drop = 0;
    public boolean _one_drop = false;
    public float _skip_drop = 0;
    public int _gpu_id = 0; // which GPU to use
    public Backend _backend = Backend.auto;

    public String algoName() { return "XGBoost"; }
    public String fullName() { return "XGBoost"; }
    public String javaName() { return XGBoostModel.class.getName(); }

    @Override
    public long progressUnits() {
      return _ntrees;
    }

    /**
     * Finds parameter settings that are not available on GPU backend.
     * In this case the CPU backend should be used instead of GPU.
     * @return map of parameter name -> parameter value
     */
    Map<String, Object> gpuIncompatibleParams() {
      Map<String, Object> incompat = new HashMap<>();
      if (!(TreeMethod.auto == _tree_method || TreeMethod.hist == _tree_method) && Booster.gblinear != _booster) {
        incompat.put("tree_method", "Only auto and hist are supported tree_method on GPU backend.");
      } 
      if (_max_depth > 15 || _max_depth < 1) {
        incompat.put("max_depth",  _max_depth + " . Max depth must be greater than 0 and lower than 16 for GPU backend.");
      }
      if (_grow_policy == GrowPolicy.lossguide)
        incompat.put("grow_policy", GrowPolicy.lossguide); // See PUBDEV-5302 (param.grow_policy != TrainParam::kLossGuide Loss guided growth policy not supported. Use CPU algorithm.)
      return incompat;
    }

    Map<String, Integer> monotoneConstraints() {
      if (_monotone_constraints == null || _monotone_constraints.length == 0) {
        return Collections.emptyMap();
      }
      Map<String, Integer> constraints = new HashMap<>(_monotone_constraints.length);
      for (KeyValue constraint : _monotone_constraints) {
        final double val = constraint.getValue();
        if (val == 0) {
          continue;
        }
        if (constraints.containsKey(constraint.getKey())) {
          throw new IllegalStateException("Duplicate definition of constraint for feature '" + constraint.getKey() + "'.");
        }
        final int direction = val < 0 ? -1 : 1;
        constraints.put(constraint.getKey(), direction);
      }
      return constraints;
    }

    @Override
    public int getNTrees() {
      return _ntrees;
    }

    @Override
    public Frame getCalibrationFrame() {
      return _calibration_frame != null ? _calibration_frame.get() : null;
    }

    @Override
    public boolean calibrateModel() {
      return _calibrate_model;
    }

    @Override
    public Parameters getParams() {
      return this;
    }

    static String[] CHECKPOINT_NON_MODIFIABLE_FIELDS = { 
        "_tree_method", "_grow_policy", "_booster", "_sample_rate", "_max_depth", "_min_rows" 
    };

  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    switch(_output.getModelCategory()) {
      case Binomial:    return new ModelMetricsBinomial.MetricBuilderBinomial(domain);
      case Multinomial: return new ModelMetricsMultinomial.MetricBuilderMultinomial(_output.nclasses(),domain);
      case Regression:  return new ModelMetricsRegression.MetricBuilderRegression();
      default: throw H2O.unimpl();
    }
  }

  public XGBoostModel(Key<XGBoostModel> selfKey, XGBoostParameters parms, XGBoostOutput output, Frame train, Frame valid) {
    super(selfKey,parms,output);
    final DataInfo dinfo = makeDataInfo(train, valid, _parms, output.nclasses());
    DKV.put(dinfo);
    setDataInfoToOutput(dinfo);
    model_info = new XGBoostModelInfo(parms, dinfo);
  }

  // useful for debugging
  @SuppressWarnings("unused")
  public void dump(String format) {
    File fmFile = null;
    try {
      Booster b = BoosterHelper.loadModel(new ByteArrayInputStream(this.model_info._boosterBytes));
      fmFile = File.createTempFile("xgboost-feature-map", ".bin");
      FileOutputStream os = new FileOutputStream(fmFile);
      os.write(this.model_info._featureMap.getBytes());
      os.close();
      String fmFilePath = fmFile.getAbsolutePath();
      String[] d = b.getModelDump(fmFilePath, true, format);
      for (String l : d) {
        System.out.println(l);
      }
    } catch (Exception e) {
      Log.err(e);
    } finally {
      if (fmFile != null) {
        fmFile.delete();
      }
    }
  }
  
  public static XGBoostParameters.Backend getActualBackend(XGBoostParameters p) {
    if ( p._backend == XGBoostParameters.Backend.auto || p._backend == XGBoostParameters.Backend.gpu ) {
      if (H2O.getCloudSize() > 1) {
        Log.info("GPU backend not supported in distributed mode. Using CPU backend.");
        return XGBoostParameters.Backend.cpu;
      } else if (! p.gpuIncompatibleParams().isEmpty()) {
        Log.info("GPU backend not supported for the choice of parameters (" + p.gpuIncompatibleParams() + "). Using CPU backend.");
        return XGBoostParameters.Backend.cpu;
      } else if (XGBoost.hasGPU(H2O.CLOUD.members()[0], p._gpu_id)) {
        Log.info("Using GPU backend (gpu_id: " + p._gpu_id + ").");
        return XGBoostParameters.Backend.gpu;
      } else {
        Log.info("No GPU (gpu_id: " + p._gpu_id + ") found. Using CPU backend.");
        return XGBoostParameters.Backend.cpu;
      }
    } else {
      Log.info("Using CPU backend.");
      return XGBoostParameters.Backend.cpu;
    }
  }
  
  public static Map<String, Object> createParamsMap(XGBoostParameters p, int nClasses, String[] coefNames) {
    Map<String, Object> params = new HashMap<>();

    // Common parameters with H2O GBM
    if (p._n_estimators != 0) {
      Log.info("Using user-provided parameter n_estimators instead of ntrees.");
      params.put("nround", p._n_estimators);
      p._ntrees = p._n_estimators;
    } else {
      params.put("nround", p._ntrees);
      p._n_estimators = p._ntrees;
    }
    if (p._eta != 0.3) {
      Log.info("Using user-provided parameter eta instead of learn_rate.");
      params.put("eta", p._eta);
      p._learn_rate = p._eta;
    } else {
      params.put("eta", p._learn_rate);
      p._eta = p._learn_rate;
    }
    params.put("max_depth", p._max_depth);
    if (System.getProperty(PROP_VERBOSITY) != null) {
      params.put("verbosity", System.getProperty(PROP_VERBOSITY));
    } else {
      params.put("silent", p._quiet_mode);
    }
    if (p._subsample != 1.0) {
      Log.info("Using user-provided parameter subsample instead of sample_rate.");
      params.put("subsample", p._subsample);
      p._sample_rate = p._subsample;
    } else {
      params.put("subsample", p._sample_rate);
      p._subsample = p._sample_rate;
    }
    if (p._colsample_bytree != 1.0) {
      Log.info("Using user-provided parameter colsample_bytree instead of col_sample_rate_per_tree.");
      params.put("colsample_bytree", p._colsample_bytree);
      p._col_sample_rate_per_tree = p._colsample_bytree;
    } else {
      params.put("colsample_bytree", p._col_sample_rate_per_tree);
      p._colsample_bytree = p._col_sample_rate_per_tree;
    }
    if (p._colsample_bylevel != 1.0) {
      Log.info("Using user-provided parameter colsample_bylevel instead of col_sample_rate.");
      params.put("colsample_bylevel", p._colsample_bylevel);
      p._col_sample_rate = p._colsample_bylevel;
    } else {
      params.put("colsample_bylevel", p._col_sample_rate);
      p._colsample_bylevel = p._col_sample_rate;
    }
    if (p._max_delta_step != 0) {
      Log.info("Using user-provided parameter max_delta_step instead of max_abs_leafnode_pred.");
      params.put("max_delta_step", p._max_delta_step);
      p._max_abs_leafnode_pred = p._max_delta_step;
    } else {
      params.put("max_delta_step", p._max_abs_leafnode_pred);
      p._max_delta_step = p._max_abs_leafnode_pred;
    }
    params.put("seed", (int)(p._seed % Integer.MAX_VALUE));

    // XGBoost specific options
    params.put("grow_policy", p._grow_policy.toString());
    if (p._grow_policy== XGBoostParameters.GrowPolicy.lossguide) {
      params.put("max_bins", p._max_bins);
      params.put("max_leaves", p._max_leaves);
      params.put("min_sum_hessian_in_leaf", p._min_sum_hessian_in_leaf);
      params.put("min_data_in_leaf", p._min_data_in_leaf);
    }
    params.put("booster", p._booster.toString());
    if (p._booster== XGBoostParameters.Booster.dart) {
      params.put("sample_type", p._sample_type.toString());
      params.put("normalize_type", p._normalize_type.toString());
      params.put("rate_drop", p._rate_drop);
      params.put("one_drop", p._one_drop ? "1" : "0");
      params.put("skip_drop", p._skip_drop);
    }
    XGBoostParameters.Backend actualBackend = getActualBackend(p);
    if (actualBackend == XGBoostParameters.Backend.gpu) {
      params.put("gpu_id", p._gpu_id);
      // we are setting updater rather than tree_method here to keep CPU predictor, which is faster
      if (p._booster == XGBoostParameters.Booster.gblinear) {
        Log.info("Using gpu_coord_descent updater."); 
        params.put("updater", "gpu_coord_descent");
      } else {
        Log.info("Using gpu_hist tree method.");
        params.put("max_bin", p._max_bins);
        params.put("updater", "grow_gpu_hist");
      }
    } else if (p._booster == XGBoostParameters.Booster.gblinear) {
      Log.info("Using coord_descent updater.");
      params.put("updater", "coord_descent");
    } else if (H2O.CLOUD.size() > 1 && p._tree_method == XGBoostParameters.TreeMethod.auto &&
        p._monotone_constraints != null) {
      Log.info("Using hist tree method for distributed computation with monotone_constraints.");
      params.put("tree_method", XGBoostParameters.TreeMethod.hist.toString());
      params.put("max_bin", p._max_bins);
    } else {
      Log.info("Using " + p._tree_method.toString() + " tree method.");
      params.put("tree_method", p._tree_method.toString());
      if (p._tree_method == XGBoostParameters.TreeMethod.hist) {
        params.put("max_bin", p._max_bins);
      }
    }
    if (p._min_child_weight != 1) {
      Log.info("Using user-provided parameter min_child_weight instead of min_rows.");
      params.put("min_child_weight", p._min_child_weight);
      p._min_rows = p._min_child_weight;
    } else {
      params.put("min_child_weight", p._min_rows);
      p._min_child_weight = p._min_rows;
    }
    if (p._gamma != 0) {
      Log.info("Using user-provided parameter gamma instead of min_split_improvement.");
      params.put("gamma", p._gamma);
      p._min_split_improvement = p._gamma;
    } else {
      params.put("gamma", p._min_split_improvement);
      p._gamma = p._min_split_improvement;
    }

    params.put("lambda", p._reg_lambda);
    params.put("alpha", p._reg_alpha);

    if (nClasses==2) {
      params.put("objective", ObjectiveType.BINARY_LOGISTIC.getId());
    } else if (nClasses==1) {
      if (p._distribution == DistributionFamily.gamma) {
        params.put("objective", ObjectiveType.REG_GAMMA.getId());
      } else if (p._distribution == DistributionFamily.tweedie) {
        params.put("objective", ObjectiveType.REG_TWEEDIE.getId());
        params.put("tweedie_variance_power", p._tweedie_power);
      } else if (p._distribution == DistributionFamily.poisson) {
        params.put("objective", ObjectiveType.COUNT_POISSON.getId());
      } else if (p._distribution == DistributionFamily.gaussian || p._distribution == DistributionFamily.AUTO) {
        params.put("objective", ObjectiveType.REG_SQUAREDERROR.getId());
      } else {
        throw new UnsupportedOperationException("No support for distribution=" + p._distribution.toString());
      }
    } else {
      params.put("objective", ObjectiveType.MULTI_SOFTPROB.getId());
      params.put("num_class", nClasses);
    }
    assert ObjectiveType.fromXGBoost((String) params.get("objective")) != null;

    final int nthreadMax = getMaxNThread();
    final int nthread = p._nthread != -1 ? Math.min(p._nthread, nthreadMax) : nthreadMax;
    if (nthread < p._nthread) {
      Log.warn("Requested nthread=" + p._nthread + " but the cluster has only " + nthreadMax + " available." +
              "Training will use nthread=" + nthread + " instead of the user specified value.");
    }
    params.put("nthread", nthread);

    Map<String, Integer> monotoneConstraints = p.monotoneConstraints();
    if (! monotoneConstraints.isEmpty()) {
      int constraintsUsed = 0;
      StringBuilder sb = new StringBuilder();
      sb.append("(");
      for (String coef : coefNames) {
        final String direction;
        if (monotoneConstraints.containsKey(coef)) {
          direction = monotoneConstraints.get(coef).toString();
          constraintsUsed++;
        } else {
          direction = "0";
        }
        sb.append(direction);
        sb.append(",");
      }
      sb.replace(sb.length()-1, sb.length(), ")");
      params.put("monotone_constraints", sb.toString());
      assert constraintsUsed == monotoneConstraints.size();
    }

    Log.info("XGBoost Parameters:");
    for (Map.Entry<String,Object> s : params.entrySet()) {
      Log.info(" " + s.getKey() + " = " + s.getValue());
    }
    Log.info("");
    return Collections.unmodifiableMap(params);
  }

  public static BoosterParms createParams(XGBoostParameters p, int nClasses, String[] coefNames) {
    return BoosterParms.fromMap(createParamsMap(p, nClasses, coefNames));
  }

  /** Performs deep clone of given model.  */
  protected XGBoostModel deepClone(Key<XGBoostModel> result) {
    XGBoostModel newModel = IcedUtils.deepCopy(this);
    newModel._key = result;
    // Do not clone model metrics
    newModel._output.clearModelMetrics(false);
    newModel._output._training_metrics = null;
    newModel._output._validation_metrics = null;
    return newModel;
  }
  
  static int getMaxNThread() {
    if (System.getProperty(PROP_NTHREAD) != null) {
      return Integer.getInteger(PROP_NTHREAD);
    } else {
      int maxNodesPerHost = 1;
      Set<String> checkedNodes = new HashSet<>();
      for (H2ONode node : H2O.CLOUD.members()) {
        String nodeHost = node.getIp();
        if (!checkedNodes.contains(nodeHost)) {
          checkedNodes.add(nodeHost);
          long cnt = Stream.of(H2O.CLOUD.members()).filter(h -> h.getIp().equals(nodeHost)).count();
          if (cnt > maxNodesPerHost) {
            maxNodesPerHost = (int) cnt;
          }
        }
      }
      return Math.max(1, H2O.ARGS.nthreads / maxNodesPerHost);
    }
  }

  @Override protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    ab.putKey(model_info.getDataInfoKey());
    return super.writeAll_impl(ab);
  }

  @Override protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    ab.getKey(model_info.getDataInfoKey(), fs);
    return super.readAll_impl(ab, fs);
  }

  @Override
  public XGBoostMojoWriter getMojo() {
    return new XGBoostMojoWriter(this);
  }

  private ModelMetrics makeMetrics(Frame data, Frame originalData, boolean isTrain, String description) {
    Log.debug("Making metrics: " + description);
    return new XGBoostModelMetrics(_output, data, originalData, isTrain, this).compute();
  }

  /**
   * Score an XGBoost model on training and validation data (optional)
   * Note: every row is scored, all observation weights are assumed to be equal
   * @param _train training data in the form of matrix
   * @param _valid validation data (optional, can be null)
   */
  final void doScoring(Frame _train, Frame _trainOrig, Frame _valid, Frame _validOrig) {
    ModelMetrics mm = makeMetrics(_train, _trainOrig, true, "Metrics reported on training frame");
    _output._training_metrics = mm;
    _output._scored_train[_output._ntrees].fillFrom(mm);
    addModelMetrics(mm);
    // Optional validation part
    if (_valid!=null) {
      mm = makeMetrics(_valid, _validOrig, false, "Metrics reported on validation frame");
      _output._validation_metrics = mm;
      _output._scored_valid[_output._ntrees].fillFrom(mm);
      addModelMetrics(mm);
    }
  }

  @Override
  protected Frame postProcessPredictions(Frame adaptedFrame, Frame predictFr, Job j) {
    return PlattScalingHelper.postProcessPredictions(predictFr, j, _output);
  }

  @Override
  protected double[] score0(double[] data, double[] preds) {
    return score0(data, preds, 0.0);
  }

  @Override // per row scoring is slow and should be avoided!
  public double[] score0(final double[] data, final double[] preds, final double offset) {
    final DataInfo di = model_info.dataInfo();
    assert di != null;
    MutableOneHotEncoderFVec row = new MutableOneHotEncoderFVec(di, _output._sparse);
    row.setInput(data);
    Predictor predictor = PredictorFactory.makePredictor(model_info._boosterBytes);
    float[] out;
    if (_output.hasOffset()) {
      out = predictor.predict(row, (float) offset);
    } else if (offset != 0) {
      throw new UnsupportedOperationException("Unsupported: offset != 0");
    } else {
      out = predictor.predict(row);
    }
    return XGBoostMojoModel.toPreds(data, out, preds, _output.nclasses(), _output._priorClassDist, defaultThreshold());
  }

  @Override
  protected XGBoostBigScorePredict setupBigScorePredict(BigScore bs) {
    return setupBigScorePredict(false);
  }

  public XGBoostBigScorePredict setupBigScorePredict(boolean isTrain) {
    DataInfo di = model_info().scoringInfo(isTrain); // always for validation scoring info for scoring (we are not in the training phase)
    return PredictConfiguration.useJavaScoring() ? setupBigScorePredictJava(di) : setupBigScorePredictNative(di);
  }

  private XGBoostBigScorePredict setupBigScorePredictNative(DataInfo di) {
    BoosterParms boosterParms = XGBoostModel.createParams(_parms, _output.nclasses(), di.coefNames());
    return new XGBoostNativeBigScorePredict(model_info, _parms, _output, di, boosterParms, defaultThreshold());
  }

  private XGBoostBigScorePredict setupBigScorePredictJava(DataInfo di) {
    return new XGBoostJavaBigScorePredict(model_info, _output, di, _parms, defaultThreshold());
  }

  @Override
  public Frame scoreContributions(Frame frame, Key<Frame> destination_key) {
    return scoreContributions(frame, destination_key, false);
  }

  public Frame scoreContributions(Frame frame, Key<Frame> destination_key, boolean approx) {
    Frame adaptFrm = new Frame(frame);
    adaptTestForTrain(adaptFrm, true, false);

    DataInfo di = model_info().dataInfo();
    assert di != null;
    final String[] outputNames = ArrayUtils.append(di.coefNames(), "BiasTerm");

    return makePredictContribTask(di, approx)
            .doAll(outputNames.length, Vec.T_NUM, adaptFrm)
            .outputFrame(destination_key, outputNames, null);
  }
  
  private MRTask<?> makePredictContribTask(DataInfo di, boolean approx) {
    return approx ? new PredictContribApproxTask(_parms, model_info, _output, di) : new PredictTreeSHAPTask(di, model_info(), _output);
  }

  @Override
  public Frame scoreLeafNodeAssignment(
      Frame frame, LeafNodeAssignmentType type, Key<Frame> destination_key
  ) {
    AssignLeafNodeTask task = AssignLeafNodeTask.make(model_info.scoringInfo(false), _output, model_info._boosterBytes, type);
    Frame adaptFrm = new Frame(frame);
    adaptTestForTrain(adaptFrm, true, false);
    return task.execute(adaptFrm, destination_key);
  }

  private void setDataInfoToOutput(DataInfo dinfo) {
    _output.setNames(dinfo._adaptedFrame.names(), dinfo._adaptedFrame.typesStr());
    _output._domains = dinfo._adaptedFrame.domains();
    _output._nums = dinfo._nums;
    _output._cats = dinfo._cats;
    _output._catOffsets = dinfo._catOffsets;
    _output._useAllFactorLevels = dinfo._useAllFactorLevels;
  }

  @Override
  protected Futures remove_impl(Futures fs, boolean cascade) {
    DataInfo di = model_info().dataInfo();
    if (di != null) {
      di.remove(fs);
    }
    if (_output._calib_model != null)
      _output._calib_model.remove(fs);
    return super.remove_impl(fs, cascade);
  }

  @Override
  public SharedTreeGraph convert(final int treeNumber, final String treeClassName) {
    GradBooster booster = XGBoostJavaMojoModel.makePredictor(model_info._boosterBytes).getBooster();
    if (!(booster instanceof GBTree)) {
      throw new IllegalArgumentException("XGBoost model is not backed by a tree-based booster. Booster class is " + 
              booster.getClass().getCanonicalName());
    }

    final RegTree[][] groupedTrees = ((GBTree) booster).getGroupedTrees();
    final int treeClass = getXGBoostClassIndex(treeClassName);
    if (treeClass >= groupedTrees.length) {
      throw new IllegalArgumentException(String.format("Given XGBoost model does not have given class '%s'.", treeClassName));
    }

    final RegTree[] treesInGroup = groupedTrees[treeClass];

    if (treeNumber >= treesInGroup.length || treeNumber < 0) {
      throw new IllegalArgumentException(String.format("There is no such tree number for given class. Total number of trees is %d.", treesInGroup.length));
    }

    final RegTreeNode[] treeNodes = treesInGroup[treeNumber].getNodes();
    assert treeNodes.length >= 1;

    SharedTreeGraph sharedTreeGraph = new SharedTreeGraph();
    final SharedTreeSubgraph sharedTreeSubgraph = sharedTreeGraph.makeSubgraph(_output._training_metrics._description);

    final XGBoostUtils.FeatureProperties featureProperties = XGBoostUtils.assembleFeatureNames(model_info.dataInfo()); // XGBoost's usage of one-hot encoding assumed
    constructSubgraph(treeNodes, sharedTreeSubgraph.makeRootNode(), 0, sharedTreeSubgraph, featureProperties, true); // Root node is at index 0
    return sharedTreeGraph;
  }

  private static void constructSubgraph(final RegTreeNode[] xgBoostNodes, final SharedTreeNode sharedTreeNode,
                                        final int nodeIndex, final SharedTreeSubgraph sharedTreeSubgraph,
                                        final XGBoostUtils.FeatureProperties featureProperties, boolean inclusiveNA) {
    final RegTreeNode xgBoostNode = xgBoostNodes[nodeIndex];
    // Not testing for NaNs, as SharedTreeNode uses NaNs as default values.
    //No domain set, as the structure mimics XGBoost's tree, which is numeric-only
    if (featureProperties._oneHotEncoded[xgBoostNode.getSplitIndex()]) {
      //Shared tree model uses < to the left and >= to the right. Transforiming one-hot encoded categoricals
      // from 0 to 1 makes it fit the current split description logic
      sharedTreeNode.setSplitValue(1.0F);
    } else {
      sharedTreeNode.setSplitValue(xgBoostNode.getSplitCondition());
    }
    sharedTreeNode.setPredValue(xgBoostNode.getLeafValue());
    sharedTreeNode.setInclusiveNa(inclusiveNA);
    sharedTreeNode.setNodeNumber(nodeIndex);
    if (!xgBoostNode.isLeaf()) {
      sharedTreeNode.setCol(xgBoostNode.getSplitIndex(), featureProperties._names[xgBoostNode.getSplitIndex()]);
      constructSubgraph(xgBoostNodes, sharedTreeSubgraph.makeLeftChildNode(sharedTreeNode),
              xgBoostNode.getLeftChildIndex(), sharedTreeSubgraph, featureProperties, xgBoostNode.default_left());
      constructSubgraph(xgBoostNodes, sharedTreeSubgraph.makeRightChildNode(sharedTreeNode),
          xgBoostNode.getRightChildIndex(), sharedTreeSubgraph, featureProperties, !xgBoostNode.default_left());
    }
  }

  @Override
  public SharedTreeGraph convert(int treeNumber, String treeClass, ConvertTreeOptions options) {
    return convert(treeNumber, treeClass); // options are currently not applicable to in-H2O conversion
  }

  private int getXGBoostClassIndex(final String treeClass) {
    final ModelCategory modelCategory = _output.getModelCategory();
    if(ModelCategory.Regression.equals(modelCategory) && (treeClass != null && !treeClass.isEmpty())){
      throw new IllegalArgumentException("There should be no tree class specified for regression.");
    }
    if ((treeClass == null || treeClass.isEmpty())) {
      // Binomial & regression problems do not require tree class to be specified, as there is only one available.
      // Such class is selected automatically for the user.
      switch (modelCategory) {
        case Binomial:
        case Regression:
          return 0;
        default:
          // If the user does not specify tree class explicitely and there are multiple options to choose from,
          // throw an error.
          throw new IllegalArgumentException(String.format("Model category '%s' requires tree class to be specified.",
                  modelCategory));
      }
    }

    final String[] domain = _output._domains[_output._domains.length - 1];
    final int treeClassIndex = ArrayUtils.find(domain, treeClass);

    if (ModelCategory.Binomial.equals(modelCategory) && treeClassIndex != 0) {
      throw new IllegalArgumentException(String.format("For binomial XGBoost model, only one tree for class %s has been built.", domain[0]));
    } else if (treeClassIndex < 0) {
      throw new IllegalArgumentException(String.format("No such class '%s' in tree.", treeClass));
    }

    return treeClassIndex;
  }

  //--------------------------------------------------------------------------------------------------------------------
  // Serialization into a POJO
  //--------------------------------------------------------------------------------------------------------------------

  @Override
  protected boolean toJavaCheckTooBig() {
    return _output == null || _output._ntrees * _parms._max_depth > 1000;
  }

  @Override protected SBPrintStream toJavaInit(SBPrintStream sb, CodeGeneratorPipeline fileCtx) {
    sb.nl();
    sb.ip("public boolean isSupervised() { return true; }").nl();
    sb.ip("public int nclasses() { return ").p(_output.nclasses()).p("; }").nl();
    return sb;
  }
  
  @Override
  protected void toJavaPredictBody(
      SBPrintStream sb, CodeGeneratorPipeline classCtx, CodeGeneratorPipeline fileCtx, boolean verboseCode
  ) {
    final String namePrefix = JCodeGen.toJavaId(_key.toString());
    Predictor p = PredictorFactory.makePredictor(model_info._boosterBytes, false);
    XGBoostPojoWriter.make(p, namePrefix, _output, defaultThreshold()).renderJavaPredictBody(sb, fileCtx);
  }

}
