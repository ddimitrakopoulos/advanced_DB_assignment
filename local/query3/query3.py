# la_crime_mo_analysis.py

import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, trim, count, broadcast

# ========================
# CONFIGURABLE PARAMETERS
# ========================

SPARK_CONFIG = {
    "spark.executor.instances": "4",
    "spark.executor.cores": "1",
    "spark.executor.memory": "2g",
    "spark.driver.memory": "2g"
}

CRIME_FILES = [
    "../LA_Data/LA_Crime_Data/LA_Crime_Data_2010_2019.csv",
    "../LA_Data/LA_Crime_Data/LA_Crime_Data_2020_2025.csv"
]

MO_FILE = "../LA_Data/MO_codes.txt"

PAGE_SIZE = 50

# ========================
# UTILITY FUNCTIONS
# ========================

def load_crime_data(spark):
    dfs = [spark.read.option("header", True).csv(f) for f in CRIME_FILES]
    return dfs[0].unionByName(dfs[1])

def load_mo_codes(spark):
    return (spark.read.text(MO_FILE)
            .withColumn("split_cols", split(col("value"), " ", 2))
            .withColumn("MO_Code", col("split_cols").getItem(0))
            .withColumn("MO_Desc", col("split_cols").getItem(1))
            .drop("value", "split_cols"))

def explode_mocodes(df):
    return df.withColumn("MO_Code", explode(split(col("Mocodes"), " "))) \
             .withColumn("MO_Code", trim(col("MO_Code")))

def scrollable_print(results):
    total_rows = len(results)
    print(f"{'MO Code':<10} | {'Description':<50} | {'Frequency':<10}")
    print("-" * 80)
    for start_idx in range(0, total_rows, PAGE_SIZE):
        chunk = results[start_idx:start_idx + PAGE_SIZE]
        for row in chunk:
            if isinstance(row, tuple):
                # RDD-style row: ((code, desc), freq)
                print(f"{row[0][0]:<10} | {row[0][1]:<50} | {row[1]:<10}")
            else:
                # DataFrame row
                print(f"{row['MO_Code']:<10} | {row['MO_Desc']:<50} | {row['Frequency']:<10}")
        if start_idx + PAGE_SIZE < total_rows:
            print(f"-- Showing rows {start_idx+1}-{min(start_idx+PAGE_SIZE, total_rows)} of {total_rows} --\n")

# ========================
# JOIN STRATEGIES
# ========================

def default_join(spark):
    crime_df = explode_mocodes(load_crime_data(spark))
    mo_df = load_mo_codes(spark)
    joined_df = crime_df.join(mo_df, on="MO_Code", how="left")

    summary = (joined_df
               .groupBy("MO_Code", "MO_Desc")
               .agg(count("*").alias("Frequency"))
               .filter((col("MO_Code").isNotNull()) & (trim(col("MO_Code")) != "") &
                       (col("MO_Desc").isNotNull()) & (trim(col("MO_Desc")) != ""))
               .orderBy(col("Frequency").desc()))

    print("=== Default Join ===")
    summary.explain(mode="extended")
    scrollable_print(summary.collect())

def broadcast_join(spark):
    crime_df = explode_mocodes(load_crime_data(spark))
    mo_df = load_mo_codes(spark)
    joined_df = crime_df.join(broadcast(mo_df), on="MO_Code", how="left")

    summary = (joined_df
               .groupBy("MO_Code", "MO_Desc")
               .agg(count("*").alias("Frequency"))
               .filter((col("MO_Code").isNotNull()) & (trim(col("MO_Code")) != "") &
                       (col("MO_Desc").isNotNull()) & (trim(col("MO_Desc")) != ""))
               .orderBy(col("Frequency").desc()))

    print("=== Broadcast Join ===")
    summary.explain(mode="extended")
    scrollable_print(summary.collect())

def hint_join(spark, hint_type="MERGE"):
    crime_df = explode_mocodes(load_crime_data(spark))
    mo_df = load_mo_codes(spark)
    joined_df = crime_df.hint(hint_type).join(mo_df.hint(hint_type), on="MO_Code", how="left")

    summary = (joined_df
               .groupBy("MO_Code", "MO_Desc")
               .agg(count("*").alias("Frequency"))
               .filter((col("MO_Code").isNotNull()) & (trim(col("MO_Code")) != "") &
                       (col("MO_Desc").isNotNull()) & (trim(col("MO_Desc")) != ""))
               .orderBy(col("Frequency").desc()))

    print(f"=== {hint_type} Join Hint ===")
    summary.explain(mode="extended")
    scrollable_print(summary.collect())

# ========================
# RDD-BASED JOIN
# ========================

def rdd_join(spark, method="broadcast"):
    sc = spark.sparkContext
    crime_rdd = load_crime_data(spark).rdd.map(lambda row: row.asDict())
    mo_rdd = (sc.textFile(MO_FILE)
              .map(lambda line: line.strip().split(" ", 1))
              .filter(lambda parts: len(parts) == 2)
              .map(lambda parts: (parts[0].strip(), parts[1].strip())))
    # Explode Mocodes
    exploded = crime_rdd.flatMap(lambda row: [(m.strip(), row) for m in (row.get("Mocodes") or "").split(" ") if m.strip() != ""])

    if method == "broadcast":
        mo_dict = dict(mo_rdd.collect())
        bc_mo = sc.broadcast(mo_dict)
        joined = exploded.map(lambda x: (x[0], bc_mo.value.get(x[0]))).filter(lambda x: x[1] is not None)
    elif method == "sortmerge":
        joined = exploded.sortBy(lambda x: x[0]).join(mo_rdd.sortBy(lambda x: x[0]))
    elif method == "hash":
        joined = exploded.partitionBy(200).join(mo_rdd.partitionBy(200))
    elif method == "replicate":
        mo_list = mo_rdd.collect()
        joined = exploded.flatMap(lambda row: [(code, desc) for (code, desc) in mo_list if code in (row[1].get("Mocodes") or "")])
        joined = joined.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: -x[1])
        scrollable_print(joined.collect())
        return

    # Count frequencies
    summary = joined.map(lambda x: ((x[0], x[1][1] if isinstance(x[1], tuple) else x[1]), 1)) \
                    .reduceByKey(lambda a, b: a + b) \
                    .sortBy(lambda x: -x[1])
    print(f"=== RDD Join ({method}) ===")
    scrollable_print(summary.collect())

# ========================
# MAIN EXECUTION
# ========================

if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("LA MO Analysis")
             .config("spark.executor.instances", SPARK_CONFIG["spark.executor.instances"])
             .config("spark.executor.cores", SPARK_CONFIG["spark.executor.cores"])
             .config("spark.executor.memory", SPARK_CONFIG["spark.executor.memory"])
             .config("spark.driver.memory", SPARK_CONFIG["spark.driver.memory"])
             .getOrCreate())

    spark.catalog.clearCache()
    start_time = time.time()

    # Uncomment any of the functions below to run a specific join strategy
    default_join(spark)
    broadcast_join(spark)
    hint_join(spark, "MERGE")
    hint_join(spark, "SHUFFLE_HASH")
    hint_join(spark, "SHUFFLE_REPLICATE_NL")
    rdd_join(spark, "broadcast")
    rdd_join(spark, "sortmerge")
    rdd_join(spark, "hash")
    rdd_join(spark, "replicate")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} sec")
