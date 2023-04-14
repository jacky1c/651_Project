import argparse

# import numpy as np
# from tqdm import tqdm

from pyspark import SparkContext
from pyspark.sql import SparkSession, Window
# from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml import Pipeline

"""
spark-submit --deploy-mode cluster s3://uwaterloo-cs651-project/feature_engineering.py \
    --proj_root s3://uwaterloo-cs651-project
    --input_file sample0.1.parquet

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proj_root', help="The URI for project root folder.")
    parser.add_argument(
        '--input_file', help="Input parquet file name.")
    args = parser.parse_args()

    proj_root = args.proj_root
    input_file = args.input_file

    with SparkSession.builder.appName("feature_engineering.py").getOrCreate() as spark:
        df = spark.read.parquet(f"{proj_root}/data/{input_file}")

        categorical_cols = [
            'body_type'
            # ,'franchise_make'
            ,'make_name'
            ,'transmission_display'
            # ,'exterior_color'
            # ,'interior_color'
            ,'listing_color'
            ,'state_id'
            # ,'franchise_make'
        ]

        df = df.fillna(value="N/A",subset=categorical_cols)

        # Mileage Fill NA
        # get mean mileage by state
        mean_mileage_df = spark.read.parquet(f"{proj_root}/data/mean_mileage_by_state.parquet")
        # if a car is labeled as new, fill mileage with a small numer
        filled_df = df.withColumn("mileage", F.when(df.is_new == 1, 5))

        # join by state ID
        filled_df = filled_df.join(mean_mileage_df, filled_df.state_id == mean_mileage_df.state_id2, 'left').drop("state_id2")
        # create a new mileage column and fill na with state mean mileage
        filled_df = filled_df.withColumn('new_mileage',F.when(filled_df.mileage.isNull(), filled_df.mean_mileage).otherwise(filled_df.mileage))

        # drop interim columns
        filled_df = filled_df.drop("mean_mileage")\
            .withColumnRenamed("mileage", "old_mileage")\
            .withColumnRenamed("new_mileage", "mileage")
        
        # Horse power Fill NA
        # find mean mileage by state
        mean_hp_by_model = spark.read.parquet(f"{proj_root}/data/mean_hp_by_model.parquet")
        # join by make, model and year
        filled_df = filled_df\
            .join(mean_hp_by_model, (filled_df.make_name == mean_hp_by_model.make_name2) & (filled_df.model_name == mean_hp_by_model.model_name2), 'left')\
            .drop(*["model_name2", "make_name2"])
        # create a new mileage column and fill na with state mean mileage
        filled_df = filled_df.withColumn('new_horsepower', F.when(filled_df.horsepower.isNull(), filled_df.mean_horsepower).otherwise(filled_df.horsepower))\
            .withColumn('new_engine_cylinders', F.when(filled_df.engine_cylinders.isNull(), filled_df.mean_engine_cylinders).otherwise(filled_df.engine_cylinders))\
            .withColumn('new_fuel_tank_volume', F.when(filled_df.fuel_tank_volume.isNull(), filled_df.mean_fuel_tank_volume).otherwise(filled_df.fuel_tank_volume))\
            .withColumn('new_back_legroom', F.when(filled_df.back_legroom.isNull(), filled_df.mean_back_legroom).otherwise(filled_df.back_legroom))\
            .withColumn('new_height', F.when(filled_df.height.isNull(), filled_df.mean_height).otherwise(filled_df.height))\
            .withColumn('new_width', F.when(filled_df.width.isNull(), filled_df.mean_width).otherwise(filled_df.width))\
            .withColumn('new_length', F.when(filled_df.length.isNull(), filled_df.mean_length).otherwise(filled_df.length))\
            .withColumn('new_city_fuel_economy', F.when(filled_df.city_fuel_economy.isNull(), filled_df.mean_city_fuel_economy).otherwise(filled_df.city_fuel_economy))\
            .withColumn('new_highway_fuel_economy', F.when(filled_df.highway_fuel_economy.isNull(), filled_df.mean_highway_fuel_economy).otherwise(filled_df.highway_fuel_economy))
        
        # drop interim columns
        filled_df = filled_df\
            .drop("horsepower")\
            .withColumnRenamed("new_horsepower", "horsepower")\
            .drop("engine_cylinders")\
            .withColumnRenamed("new_engine_cylinders", "engine_cylinders")\
            .drop("fuel_tank_volume")\
            .withColumnRenamed("new_fuel_tank_volume", "fuel_tank_volume")\
            .drop("back_legroom")\
            .withColumnRenamed("new_back_legroom", "back_legroom")\
            .drop("height")\
            .withColumnRenamed("new_height", "height")\
            .drop("width")\
            .withColumnRenamed("new_width", "width")\
            .drop("length")\
            .withColumnRenamed("new_length", "length")\
            .drop("city_fuel_economy")\
            .withColumnRenamed("new_city_fuel_economy", "city_fuel_economy")\
            .drop("highway_fuel_economy")\
            .withColumnRenamed("new_highway_fuel_economy", "highway_fuel_economy")
        
        # Feature engineering
        numeric_cols = [x + '_numeric' for x in categorical_cols]
        onehot_cols = [x + '_onehot' for x in categorical_cols]
        # apply StringIndexer to convert the features into the numerical format
        str_indexer = StringIndexer(inputCols=categorical_cols, outputCols=numeric_cols)
        # apply OneHotEncoder
        one_hot_encoder = OneHotEncoder(inputCols=numeric_cols,outputCols=onehot_cols)#.fit(indexed_df)


        # create pipeline
        pipeline = Pipeline(stages=[str_indexer,
                                    one_hot_encoder])
        # transform dataset
        df_transformed = pipeline.fit(filled_df).transform(filled_df)

        # # combine categorical features with numerica features
        model_cols = onehot_cols + ['year', 'mileage', 
                                    'horsepower', 'engine_cylinders', 'fuel_tank_volume',
                                    "back_legroom", "height", "width", "length", 
                                    # "city_fuel_economy", "highway_fuel_economy"
                                    ]
        
        df_transformed = df_transformed.filter(F.col("horsepower").isNotNull() & F.col("engine_cylinders").isNotNull() & F.col("fuel_tank_volume").isNotNull()\
                                       & F.col("back_legroom").isNotNull() & F.col("height").isNotNull() & F.col("width").isNotNull() & F.col("length").isNotNull()\
                                       & F.col("city_fuel_economy").isNotNull() & F.col("highway_fuel_economy").isNotNull()
                                       )

        # vectorize features
        vectorAssembler = VectorAssembler(inputCols = model_cols, outputCol = 'features')
        vectorized_df = vectorAssembler.transform(df_transformed)
        vectorized_df = vectorized_df.select(['features', 'price'])
        vectorized_df.show(truncate=False)


        xy_df = vectorized_df.select(F.col("features"), F.col("price"))\
            .withColumnRenamed("price", "label")
        
        xy_df.write.mode('overwrite').parquet(f"{proj_root}/data/x_y_df.parquet")