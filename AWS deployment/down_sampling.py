import argparse

# import numpy as np
# from tqdm import tqdm

from pyspark import SparkContext
from pyspark.sql import SparkSession
# from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer

"""
spark-submit --deploy-mode cluster s3://uwaterloo-cs651-project/down_sampling.py \
    --proj_root s3://uwaterloo-cs651-project

"""

# general transformation
def to_lower_str(df, col_name):
  return df.withColumn(col_name, F.lower(F.col(col_name)))
def cast_to_double(df, col_name):
  return df.withColumn(col_name, F.col(col_name).cast('double'))
def cast_to_int(df, col_name):
  return df.withColumn(col_name, F.col(col_name).cast('int'))
def cast_to_boolean(df, col_name):
  return df.withColumn(col_name, 
                       F.when(F.upper(F.col(col_name)) == 'TRUE', 1) \
                       .when(F.upper(F.col(col_name)) == 'FALSE', 0) \
                       .otherwise(None))
def cast_to_date(df, col_name):
  return df.withColumn(col_name, F.to_date(F.col(col_name), "yyyy-MM-dd"))


def parse_double_before_space(df, col_name):
  return df.withColumn(col_name, F.split(F.col(col_name), ' ').getItem(0).cast('double'))

# column specific transformation
def clean_engine_cylinders(df):
  return df.withColumn("engine_cylinders", F.regexp_replace(df.engine_cylinders, '[a-zA-Z]', '').cast('int'))
def clean_engine_type(df):
  return df.withColumn("engine_type", F.lower(F.regexp_replace(df.engine_type, '[0-9]', '')))
def clean_city(df, col_name):
  return df.withColumn(col_name, F.regexp_replace(F.regexp_replace(F.regexp_replace(F.lower(F.col(col_name)), '^fort ', 'ft '), '^saint ', 'st '), '[^a-zA-Z]', ''))
def clean_model_name(df):
  return df.withColumn("model_name", F.regexp_replace(F.lower(df.model_name), '[^a-z0-9]', ''))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proj_root', help="The URI for project root folder.")
    args = parser.parse_args()

    proj_root = args.proj_root

    with SparkSession.builder.appName("down_sampling.py").getOrCreate() as spark:
        df = spark\
            .read\
            .option("header", "true")\
            .option("escape", '"')\
            .option("multiline", "true")\
            .csv(f"{proj_root}/data/used_cars_data.csv")
        # create surrogate key
        df = df.withColumn("rowno", F.monotonically_increasing_id()+1)

        # Drop columns that have little data or useless to our analysis
        df_dropped = df.drop("bed","bed_height","bed_length", "cabin", "combine_fuel_economy", \
                     "is_certified", "main_picture_url", "description", "us_city", \
                     "is_oemcpo", "is_cpo", "vehicle_damage_category", \
                     "vin", "listing_id", "sp_id", "trimId")
        
        # transform columns
        df_dropped_clean = df_dropped \
            .transform(parse_double_before_space, "back_legroom") \
            .transform(to_lower_str, "body_type") \
            .transform(clean_city, "city") \
            .transform(cast_to_double, "city_fuel_economy") \
            .transform(cast_to_int, "daysonmarket") \
            .transform(cast_to_int, "dealer_zip") \
            .transform(clean_engine_cylinders) \
            .transform(cast_to_double, "engine_displacement") \
            .transform(clean_engine_type) \
            .transform(to_lower_str, "exterior_color") \
            .transform(cast_to_boolean, "fleet") \
            .transform(cast_to_boolean, "frame_damaged") \
            .transform(cast_to_boolean, "franchise_dealer") \
            .transform(to_lower_str, "franchise_make") \
            .transform(parse_double_before_space, "front_legroom") \
            .transform(parse_double_before_space, "fuel_tank_volume") \
            .transform(to_lower_str, "fuel_type") \
            .transform(cast_to_boolean, "has_accidents") \
            .transform(parse_double_before_space, "height") \
            .transform(cast_to_double, "highway_fuel_economy") \
            .transform(cast_to_double, "horsepower") \
            .transform(to_lower_str, "interior_color") \
            .transform(cast_to_boolean, "isCab") \
            .transform(cast_to_boolean, "is_new") \
            .transform(cast_to_double, "latitude") \
            .transform(parse_double_before_space, "length") \
            .transform(cast_to_date, "listed_date") \
            .transform(to_lower_str, "listing_color") \
            .transform(cast_to_double, "longitude") \
            .transform(to_lower_str, "major_options") \
            .transform(to_lower_str, "make_name") \
            .transform(parse_double_before_space, "maximum_seating") \
            .transform(cast_to_double, "mileage") \
            .transform(clean_model_name) \
            .transform(cast_to_int, "owner_count") \
            .transform(parse_double_before_space, "power") \
            .transform(cast_to_double, "price") \
            .transform(cast_to_boolean, "salvage") \
            .transform(cast_to_double, "savings_amount") \
            .transform(cast_to_double, "seller_rating") \
            .transform(to_lower_str, "sp_name") \
            .transform(cast_to_boolean, "theft_title") \
            .transform(parse_double_before_space, "torque") \
            .transform(to_lower_str, "transmission") \
            .transform(to_lower_str, "transmission_display") \
            .transform(to_lower_str, "trim_name") \
            .transform(to_lower_str, "wheel_system") \
            .transform(to_lower_str, "wheel_system_display") \
            .transform(parse_double_before_space, "wheelbase") \
            .transform(parse_double_before_space, "width") \
            .transform(cast_to_int, "year")
        

        # Join with US cities dataset to find state
        # read US cities from CSV file
        cities_df = spark\
        .read\
        .option("header", "true")\
        .option("escape", '"')\
        .option("multiline", "true")\
        .csv(f"{proj_root}/data/uscities.csv")

        # some cities occur many times in our dataset but are missing in CSV file
        # high frequency cities are manually added here
        more_cities = [\
        ("Van Nuys","Van Nuys","CA","California"), \
        ("Wexford","Wexford","PA","Pennsylvania"), \
        ("Clinton Township","Clinton Township","MI","Michigan"), \
        ("North Hollywood","North Hollywood","CA","California"), \
        ("Freehold","Freehold","NJ","New Jersey"), \
        ("Toms River","Toms River","NJ","New Jersey"), \
        ("Westborough","Westborough","MA","Massachusetts"), \
        ("Egg Harbor Township","Egg Harbor Township","NJ","New Jersey"), \
        ("Braintree","Braintree,","MA","Massachusetts"), \
        ("Long Island City","Long Island City","NY","New York"), \
        ("Maple Shade","Maple Shade","NJ","New Jersey"), \
        ("East Hartford","East Hartford","CT","Connecticut"), \
        # ("","","",""),\
        ("Orchard Park","Orchard Park","NY","New York"),\
        ("City of Industry","City of Industry","CA","California"),\
        ("Lynnfield","Lynnfield","MA","Massachusetts"),\
        ("Riverhead","Riverhead","NY","New York"),\
        ("Sheffield Village","Sheffield Village","OH","Ohio"),\
        ("Ft Myers","Ft Myers","FL","Florida"),\
        ("New Hudson","New Hudson","NY","New York"),\
        ("Mt Pleasant","Mt Pleasant","MI","Michigan"),\
        ("Old Bridge","Old Bridge","NJ","New Jersey") \
        
        ]

        more_cities_cols = ["city","city_ascii","state_id","state_name"]
        more_cities_df = spark.createDataFrame(data = more_cities, schema = more_cities_cols)


        cities_df_small = cities_df\
            .transform(clean_city, "city")\
            .transform(clean_city, "city_ascii")\
            .withColumnRenamed("city_ascii","us_city").select(F.col("us_city"), F.col("state_id"), F.col("state_name"))

        more_cities_df = more_cities_df\
            .transform(clean_city, "city")\
            .transform(clean_city, "city_ascii")\
            .withColumnRenamed("city_ascii","us_city").select(F.col("us_city"), F.col("state_id"), F.col("state_name"))
        
        cities_df_small = cities_df_small.union(more_cities_df)
        df_dropped_clean = df_dropped_clean.join(cities_df_small, df_dropped_clean.city == cities_df_small.us_city, "left")

        df_dropped_clean.write.mode('overwrite').parquet(f"{proj_root}/data/used_cars_data.parquet")


        splits = df_dropped_clean.randomSplit([0.9, 0.1], 651)
        splits[1].write.mode('overwrite').parquet(f"{proj_root}/data/sample0.1.parquet")