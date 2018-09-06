
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

spark = SparkSession\
        .builder\
        .appName("PySpark XGBOOST Titanic")\
        .getOrCreate()


from sparkxgb.xgboost import XGBoostClassifier

schema = StructType(
  [StructField("PassengerId", DoubleType()),
    StructField("Survival", DoubleType()),
    StructField("Pclass", DoubleType()),
    StructField("Name", StringType()),
    StructField("Sex", StringType()),
    StructField("Age", DoubleType()),
    StructField("SibSp", DoubleType()),
    StructField("Parch", DoubleType()),
    StructField("Ticket", StringType()),
    StructField("Fare", DoubleType()),
    StructField("Cabin", StringType()),
    StructField("Embarked", StringType())
  ])

df_raw = spark\
  .read\
  .option("header", "true")\
  .schema(schema)\
  .csv("/home/siddarth/Desktop/Stuff/trainvorg.csv")


df = df_raw.na.fill(0)

sexIndexer = StringIndexer() \
    .setInputCol("Sex") \
    .setOutputCol("SexIndex") \
    .setHandleInvalid("keep")

cabinIndexer = StringIndexer() \
    .setInputCol("Cabin") \
    .setOutputCol("CabinIndex") \
    .setHandleInvalid("keep")

embarkedIndexer = StringIndexer() \
    .setInputCol("Embarked") \
    .setOutputCol("EmbarkedIndex") \
    .setHandleInvalid("keep")

vectorAssembler  = VectorAssembler()\
  .setInputCols(["Pclass", "SexIndex", "Age", "SibSp", "Parch", "Fare", "CabinIndex", "EmbarkedIndex"])\
  .setOutputCol("features")

xgboost = XGBoostClassifier(
    featuresCol="features",
    labelCol="Survival",
    predictionCol="prediction"
)

pipeline = Pipeline().setStages([sexIndexer, cabinIndexer, embarkedIndexer, vectorAssembler, xgboost])
trainDF, testDF = df.randomSplit([0.8, 0.2], seed=24)

model  =pipeline.fit(trainDF)

# print(trainDF.schema)
