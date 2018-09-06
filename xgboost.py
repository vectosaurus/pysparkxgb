#
# Copyright (c) 2018 by Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from pyspark import SparkContext, keyword_only
from pyspark.ml.common import _java2py
from pyspark.ml.param import Param
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasPredictionCol, HasWeightCol, HasCheckpointInterval
from pyspark.ml.util import JavaMLWritable, JavaPredictionModel
from pyspark.ml.wrapper import JavaEstimator, JavaModel, JavaWrapper
from .util import XGBoostReadable


class JavaParamsOverrides(object):
    """
    Mixin for overriding methods derived from JavaParams.
    """
    # Define a fix similar to SPARK-10931 (For Spark <2.3)
    def _create_params_from_java(self):
        """
        Create params that are defined in the Java obj but not here
        """
        java_params = list(self._java_obj.params())
        from pyspark.ml.param import Param
        for java_param in java_params:
            java_param_name = java_param.name()
            if not hasattr(self, java_param_name):
                param = Param(self, java_param_name, java_param.doc())
                setattr(param, "created_from_java_param", True)
                setattr(self, java_param_name, param)
                self._params = None  # need to reset so self.params will discover new params
    # Backport SPARK-10931 (For Spark <2.3)
    def _transfer_params_from_java(self):
        """
        Transforms the embedded params from the companion Java object.
        """
        sc = SparkContext._active_spark_context
        for param in self.params:
            if self._java_obj.hasParam(param.name):
                java_param = self._java_obj.getParam(param.name)
                # SPARK-14931: Only check set params back to avoid default params mismatch.
                if self._java_obj.isSet(java_param):
                    value = _java2py(sc, self._java_obj.getOrDefault(java_param))
                    self._set(**{param.name: value})
                # SPARK-10931: Temporary fix for params that have a default in Java
                if self._java_obj.hasDefault(java_param) and not self.isDefined(param):
                    value = _java2py(sc, self._java_obj.getDefault(java_param)).get()
                    self._setDefault(**{param.name: value})
    # Override the "_from_java" method, so we can read our objects.
    @classmethod
    def _from_java(cls, java_stage):
        """
        Given a Java object, create and return a Python wrapper of it.
        """

        # Create a new instance of this stage.
        py_stage = cls()

        # Load information from java_stage to the instance.
        py_stage._java_obj = java_stage
        py_stage._create_params_from_java()
        py_stage._resetUid(java_stage.uid())
        py_stage._transfer_params_from_java()

        return py_stage


class XGBoostRegressor(JavaParamsOverrides, JavaEstimator, HasCheckpointInterval, HasFeaturesCol, HasLabelCol,
                       HasPredictionCol, HasWeightCol, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor.
    """
    @keyword_only
    def __init__(self, alpha=0.0, baseMarginCol="baseMargin", baseScore=0.5, checkpointInterval=-1, checkpointPath="", colsampleBylevel=1.0, colsampleBytree=1.0, eta=0.3, evalMetric="error", featuresCol="features", gamma=0.0, growPolicy="depthwise", labelCol="label", regLambda=0.0, lambdaBias=0.0, maxBins=256, maxDeltaStep=0.0, maxDepth=6, minChildWeight=1.0, normalizeType="tree", numRound=10, numWorkers=1, objective="binary:logistic", predictionCol="prediction", rateDrop=0.0, sampleType="uniform", scalePosWeight=1.0, seed=42, silent=1, sketchEps=0.03, skipDrop=0.0, subsample=1.0, treeMethod="auto", useExternalMemory=False):
        super(XGBoostRegressor, self).__init__()
        self._java_obj = self._new_java_obj("ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor", self.uid)
        self._create_params_from_java()
        self._setDefault(
            # Column Params
            featuresCol="features", labelCol="label", predictionCol="prediction",
            # Booster Params
            objective="binary:logistic", evalMetric="error", numRound=2)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    @keyword_only
    def setParams(self, alpha=0.0, baseScore=0.5, checkpointInterval=-1, checkpointPath="", colsampleBylevel=1.0, colsampleBytree=1.0, eta=0.3, evalMetric="error", featuresCol="features", gamma=0.0, growPolicy="depthwise", labelCol="label", regLambda=0.0, lambdaBias=0.0, maxBins=256, maxDeltaStep=0.0, maxDepth=6, minChildWeight=1.0, normalizeType="tree", numClass=2, numRound=10, numWorkers=1, objective="regression:error", predictionCol="prediction", probabilityCol="probability", rateDrop=0.0, rawPredictionCol="rawPrediction", sampleType="uniform", scalePosWeight=1.0, seed=42, silent=1, sketchEps=0.03, skipDrop=0.0, subsample=1.0, treeMethod="auto", useExternalMemory=False, weightCol="weightCol"):

        kwargs = self._input_kwargs_processed()
        return self._set(**kwargs)
    def _input_kwargs_processed(self):
        """
        Until consensus on parameter names can be achieved, we must rename kwargs which would break python.
        """

        kwargs = self._input_kwargs
        if "regLambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("regLambda")

        return kwargs
    def _create_model(self, java_model):
        """
        Create the correct python object for the model type.
        """

        java_package = java_model.getClass().getName()
        java_class = java_package.split('.')[-1]

        if java_class == 'XGBoostClassificationModel':
            return XGBoostClassificationModel(java_model)
        elif java_class == 'XGBoostRegressionModel':
            return XGBoostRegressionModel(java_model)
        else:
            raise NotImplementedError("This XGBoost model type cannot loaded into Python currently: %r"
                                      % java_class)


class XGBoostClassifier(JavaParamsOverrides, JavaEstimator, HasCheckpointInterval, HasFeaturesCol, HasLabelCol,
                       HasPredictionCol, HasWeightCol, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor.
    """
    @keyword_only
    def __init__(self, alpha=0.0, baseScore=0.5, checkpointInterval=-1, checkpointPath="", colsampleBylevel=1.0, colsampleBytree=1.0, eta=0.3, evalMetric="error", featuresCol="features", gamma=0.0, growPolicy="depthwise", labelCol="label", regLambda=0.0, lambdaBias=0.0, maxBins=256, maxDeltaStep=0.0, maxDepth=6, minChildWeight=1.0, normalizeType="tree", numClass=2, numRound=10, numWorkers=1, objective="binary:logistic", predictionCol="prediction", probabilityCol="probability", rateDrop=0.0, rawPredictionCol="rawPrediction", sampleType="uniform", scalePosWeight=1.0, seed=42, silent=1, sketchEps=0.03, skipDrop=0.0, subsample=1.0, treeMethod="auto", useExternalMemory=False):

        super(XGBoostClassifier, self).__init__()
        self._java_obj = self._new_java_obj("ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier", self.uid)
        self._create_params_from_java()
        self._setDefault(
            # Column Params
            featuresCol="features", labelCol="label", predictionCol="prediction",
            # Booster Params
            objective="binary:logistic", evalMetric="error", numRound=2)

        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    @keyword_only
    def setParams(self, alpha=0.0, baseScore=0.5, checkpointInterval=-1, checkpointPath="", colsampleBylevel=1.0, colsampleBytree=1.0, eta=0.3, evalMetric="error", featuresCol="features", gamma=0.0, growPolicy="depthwise", labelCol="label", regLambda=0.0, lambdaBias=0.0, maxBins=256, maxDeltaStep=0.0, maxDepth=6, minChildWeight=1.0, normalizeType="tree", numClass=2, numRound=10, numWorkers=1, objective="regression:error", predictionCol="prediction", probabilityCol="probability", rateDrop=0.0, rawPredictionCol="rawPrediction", sampleType="uniform", scalePosWeight=1.0, seed=42, silent=1, sketchEps=0.03, skipDrop=0.0, subsample=1.0, treeMethod="auto", useExternalMemory=False, weightCol="weightCol"):

        kwargs = self._input_kwargs_processed()
        return self._set(**kwargs)
    def _input_kwargs_processed(self):
        """
        Until consensus on parameter names can be achieved, we must rename kwargs which would break python.
        """

        kwargs = self._input_kwargs
        if "regLambda" in kwargs:
            kwargs["lambda"] = kwargs.pop("regLambda")

        return kwargs
    def _create_model(self, java_model):
        """
        Create the correct python object for the model type.
        """

        java_package = java_model.getClass().getName()
        java_class = java_package.split('.')[-1]

        if java_class == 'XGBoostClassificationModel':
            return XGBoostClassificationModel(java_model)
        elif java_class == 'XGBoostRegressionModel':
            return XGBoostRegressionModel(java_model)
        else:
            raise NotImplementedError("This XGBoost model type cannot loaded into Python currently: %r"
                                      % java_class)



class XGBoostClassificationModel(JavaParamsOverrides, JavaModel, JavaPredictionModel, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel.
    """
    def __init__(self, java_model=None):
        """
        Override the __init__ from JavaModel.
        """

        super(XGBoostClassificationModel, self).__init__(java_model)
        if java_model is not None:
            # Get parameters only present in the model object.
            self._create_params_from_java()

            self._resetUid(java_model.uid())

            # Transfer parameter values from java object.
            self._transfer_params_from_java()
    @property
    def numClasses(self):
        """
        Number of classes (values which the label can take).
        """
        return self._call_java("numClasses")
    def setThresholds(self, value):
        """
        Sets the value of :py:attr:`thresholds`.
        """
        return self._set(thresholds=value)
    def getThresholds(self):
        """
        Gets the value of thresholds or its default value.
        """
        return self.getOrDefault(self.thresholds)
    def setRawPredictionCol(self, value):
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)
    def getRawPredictionCol(self):
        """
        Gets the value of rawPredictionCol or its default value.
        """
        return self.getOrDefault(self.rawPredictionCol)
    def getBooster(self):
        return self._call_java("booster")
    def get_booster(self):
        return self.getBooster()
    def get_dump(self,fmap="",with_stats=True,format= "text"):
        jxgb = JavaWrapper(self.getBooster())
        return jxgb._call_java("getModelDump",fmap,with_stats,format)
    def saveBooster(self, save_path):
        jxgb = JavaWrapper(self.getBooster())
        jxgb._call_java("saveModel", save_path)
    def fscore(self,fmap=""):
        jxgb = JavaWrapper(self.getBooster())
        return jxgb._call_java("getFeatureScore",fmap)


class XGBoostRegressionModel(JavaParamsOverrides, JavaModel, JavaPredictionModel, JavaMLWritable, XGBoostReadable):
    """
    A PySpark implementation of ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel.
    """
    def __init__(self, java_model=None):
        """
        Override the __init__ from JavaModel.
        """
        super(XGBoostRegressionModel, self).__init__(java_model)
        if java_model is not None:
            # Get parameters only present in the model object.
            self._create_params_from_java()
            self._resetUid(java_model.uid())
            # Transfer parameter values from java object.
            self._transfer_params_from_java()

    def setRawPredictionCol(self, value):
        """
        Sets the value of :py:attr:`rawPredictionCol`.
        """
        return self._set(rawPredictionCol=value)
    def getRawPredictionCol(self):
        """
        Gets the value of rawPredictionCol or its default value.
        """
        return self.getOrDefault(self.rawPredictionCol)
    def getBooster(self):
        return self._call_java("booster")
    def get_booster(self):
        return self.getBooster()
    def get_dump(self,fmap="",with_stats=True,format= "text"):
        jxgb = JavaWrapper(self.getBooster())
        return jxgb._call_java("getModelDump",fmap,with_stats,format)
    def saveBooster(self, save_path):
        jxgb = JavaWrapper(self.getBooster())
        jxgb._call_java("saveModel", save_path)
    def fscore(self,fmap=""):
        jxgb = JavaWrapper(self.getBooster())
        return jxgb._call_java("getFeatureScore",fmap)


