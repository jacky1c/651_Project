import argparse
from datetime import datetime

# import numpy as np
# from tqdm import tqdm

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from xgboost.spark import SparkXGBRegressor

"""
spark-submit --deploy-mode cluster s3://uwaterloo-cs651-project/train_xgboost.py \
    --proj_root s3://uwaterloo-cs651-project

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proj_root', help="The URI for project root folder.")
    args = parser.parse_args()

    proj_root = args.proj_root
    print(f"proj_root = {proj_root}")

    now = datetime.now()

    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    print(f"timestamp = {timestamp}")

    with SparkSession.builder.appName("train_xgboost.py").getOrCreate() as spark:
        xy_df = spark.read.parquet(f"{proj_root}/data/x_y_df.parquet")
        # train test split
        splits = xy_df.randomSplit([0.7, 0.3], 651)
        train_df = splits[0]
        test_df = splits[1]

        
        xgb_regressor = SparkXGBRegressor(
            features_col="features",
            label_col="label",
            num_workers=10,
        )
        # train and return the model
        xgb_model = xgb_regressor.fit(train_df)

        # save the model
        xgb_model.save(f"{proj_root}/models/xgboost_{timestamp}")
        # # load the model
        # model2 = SparkXGBRankerModel.load("/tmp/xgboost-pyspark-model")

        predictions = xgb_model.transform(test_df)
        predictions.write.mode('overwrite').parquet(f"{proj_root}/predictions/{timestamp}.parquet")

        # Select (prediction, true label) and compute test error
        evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction")
        r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})
        print(f"R2 = {r2}")
        rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
        print(f"RMSE = {rmse}")
        mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
        print(f"MAE = {mae}")