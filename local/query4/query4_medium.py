# la_crime_analysis.py

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, round as spark_round,
    radians, sin, cos, sqrt, asin
)
from pyspark.sql.functions import min as spark_min

# ========================
# CONFIGURABLE PARAMETERS
# ========================

# Spark configuration
SPARK_CONFIG = {
    "spark.executor.instances": "2",
    "spark.executor.cores": "2",
    "spark.executor.memory": "2g",
    "spark.driver.memory": "4g"
}

# File paths
POLICE_STATIONS_FILE = "../LA_Data/LA_Police_Stations.csv"
CRIME_DATA_FILES = [
    "../LA_Data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv",
    "../LA_Data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
]


# ========================
# FUNCTION DEFINITIONS
# ========================

def haversine_expr(lat1, lon1, lat2, lon2):
    """Haversine distance formula in Spark SQL expressions (km)."""
    return 2 * 6371.0 * asin(
        sqrt(
            sin((radians(lat2) - radians(lat1)) / 2) ** 2 +
            cos(radians(lat1)) * cos(radians(lat2)) *
            sin((radians(lon2) - radians(lon1)) / 2) ** 2
        )
    )

def run_query_4():
    """Compute average distance of crimes to nearest police station by division."""
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("LA Crime Distance Analysis") \
        .config("spark.executor.instances", SPARK_CONFIG["spark.executor.instances"]) \
        .config("spark.executor.cores", SPARK_CONFIG["spark.executor.cores"]) \
        .config("spark.executor.memory", SPARK_CONFIG["spark.executor.memory"]) \
        .config("spark.driver.memory", SPARK_CONFIG["spark.driver.memory"]) \
        .getOrCreate()

    # Load and prepare police stations
    stations = spark.read.option("header", True).csv(POLICE_STATIONS_FILE) \
        .withColumnRenamed("DIVISION", "division") \
        .withColumnRenamed("X", "station_lon") \
        .withColumnRenamed("Y", "station_lat") \
        .withColumn("station_lon", col("station_lon").cast("double")) \
        .withColumn("station_lat", col("station_lat").cast("double"))

    # Load and combine crime data
    crime_dfs = [spark.read.option("header", True).csv(f) for f in CRIME_DATA_FILES]
    combined_crime = crime_dfs[0].unionByName(crime_dfs[1])

    # Filter out null or zero coordinates
    crime_coords = combined_crime.select(
        col("DR_NO"),
        col("LAT").cast("double").alias("crime_lat"),
        col("LON").cast("double").alias("crime_lon")
    ).filter(
        (col("crime_lat").isNotNull()) &
        (col("crime_lon").isNotNull()) &
        ~((col("crime_lat") == 0) & (col("crime_lon") == 0))
    )

    # Compute distances using cross join
    crime_station = crime_coords.crossJoin(stations) \
        .withColumn(
            "distance",
            haversine_expr(
                col("crime_lat"), col("crime_lon"),
                col("station_lat"), col("station_lon")
            )
        )

    # Find nearest station per crime
    crime_min = crime_station.groupBy("DR_NO") \
        .agg(spark_min("distance").alias("min_distance"))

    crime_nearest = crime_min.join(
        crime_station,
        (crime_min.DR_NO == crime_station.DR_NO) &
        (crime_min.min_distance == crime_station.distance),
        "inner"
    ).select("division", "distance")

    # Aggregate statistics by division
    division_stats = crime_nearest.groupBy("division") \
        .agg(
            spark_round(avg("distance"), 3).alias("average_distance"),
            count("*").alias("#")
        ).select("division", "average_distance", "#") \
         .orderBy(col("#").desc())

    return division_stats

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    start_time = time.time()
    df1 = run_query_4()

    df1.show(50, truncate=False)
    df1.explain(mode="extended")

    end_time = time.time()
    print(f"Execution time for run_query_4: {end_time - start_time:.2f} seconds")
