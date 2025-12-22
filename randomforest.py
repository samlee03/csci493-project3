import os
from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest

def main():

    os.environ["PYSPARK_PYTHON"] = r"C:\Users\13477\miniconda3\envs\spark-env\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\13477\miniconda3\envs\spark-env\python.exe"

    spark = SparkSession.builder \
        .appName("RandomForest_KFold") \
        .master("local[2]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel("WARN")

    def parse_line(line):
        parts = line.split(",")
        label = 1.0 if parts[1] == "M" else 0.0
        features = [float(x) for x in parts[2:]]
        return LabeledPoint(label, features)

    raw = sc.textFile("data/project3_data.csv")
    header = raw.first()

    data = raw.filter(lambda x: x != header).map(parse_line).cache()

    print("Total samples:", data.count())
    print("Number of features:", len(data.first().features))

    k = 5
    folded_data = data.zipWithIndex().map(lambda x: (x[1] % k, x[0])).cache()

    accuracies = []
    global_tp = global_fp = global_fn = global_tn = 0

    for fold in range(k):
        print(f"\nRunning fold {fold + 1}/{k}")

        trainRDD = folded_data.filter(lambda x: x[0] != fold).map(lambda x: x[1])
        testRDD  = folded_data.filter(lambda x: x[0] == fold).map(lambda x: x[1])

        model = RandomForest.trainClassifier(
            trainRDD,
            numClasses=2,
            categoricalFeaturesInfo={},
            numTrees=20,
            featureSubsetStrategy="auto",
            impurity="gini",
            maxDepth=5,
            maxBins=32,
            seed=42
        )

        preds = model.predict(testRDD.map(lambda x: x.features))
        labels = testRDD.map(lambda x: x.label)
        labels_preds = labels.zip(preds)

        def conf_matrix_count(x):
            if x[0] == 1.0 and x[1] == 1.0: return (1, 0, 0, 0)
            if x[0] == 0.0 and x[1] == 1.0: return (0, 1, 0, 0)
            if x[0] == 1.0 and x[1] == 0.0: return (0, 0, 1, 0)
            return (0, 0, 0, 1)

        tp, fp, fn, tn = labels_preds.map(conf_matrix_count).reduce(
            lambda a, b: (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
        )

        accuracy = (tp + tn) / (tp + fp + fn + tn)
        accuracies.append(accuracy)

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

        print(f"Fold Accuracy: {accuracy:.4f}")

    avg_accuracy = sum(accuracies) / k
    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) else 0.0
    recall    = global_tp / (global_tp + global_fn) if (global_tp + global_fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1 Score:         {f1:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()
