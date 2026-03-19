// ============================================================
//  Music Genre Classifier — Spark MLlib (Scala / spark-shell)
//  Run with:  spark-shell --driver-memory 2g
//             then inside shell: :load model/train_model.scala
// ============================================================

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame

// spark is already available in spark-shell; setLogLevel reduces noise
spark.sparkContext.setLogLevel("WARN")

// ─────────────────────────────────────────────────────────────
//  STEP 1 — Load & Clean Dataset
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
//  STEP 9 — Extended to 8 genres
//  Switch to Merged_dataset.csv which includes the 7 Mendeley genres
//  (pop, country, blues, rock, jazz, reggae, hip hop) PLUS soul
//  from Student_dataset.csv.  Everything else in the pipeline is
//  unchanged — StringIndexer and LogisticRegression handle 8 classes
//  automatically.
// ─────────────────────────────────────────────────────────────
val dataPath = "dataset/Merged_dataset.csv"

// 1-a  Load raw CSV
// NOTE: lines end with '.' so spark-shell :load treats them as one expression
val rawDF: DataFrame = spark.read.
  option("header",    "true").
  option("inferSchema", "true").
  option("multiLine", "true").
  option("escape",    "\"").
  csv(dataPath)

// 1-b  Keep only the five required columns
val selectedDF: DataFrame = rawDF.select(
  "artist_name", "track_name", "release_date", "genre", "lyrics")

// 1-c  Drop rows where lyrics or genre are null / blank
val notNullDF: DataFrame = selectedDF.
  filter(col("lyrics").isNotNull && col("genre").isNotNull).
  filter(trim(col("lyrics")) =!= "").
  filter(trim(col("genre"))  =!= "")

// 1-d  Lowercase lyrics (normalises casing for TF-IDF)
val cleanDF: DataFrame = notNullDF.
  withColumn("lyrics", lower(trim(col("lyrics"))))

// ─── Inspection ─────────────────────────────────────────────

println("\n=== Schema ===")
cleanDF.printSchema()

println(s"\n=== Total records after cleaning: ${cleanDF.count()} ===\n")

println("=== Genre distribution ===")
cleanDF.
  groupBy("genre").count().
  orderBy(desc("count")).
  show(truncate = false)

println("=== Sample rows (lyrics truncated to 80 chars) ===")
cleanDF.
  select(
    col("genre"),
    col("artist_name"),
    col("track_name"),
    substring(col("lyrics"), 1, 80).alias("lyrics_snippet")).
  show(5, truncate = false)

// ─────────────────────────────────────────────────────────────
//  STEP 2 — Train / Test Split  (80 / 20)
// ─────────────────────────────────────────────────────────────

// Fixed seed guarantees the same split every run (reproducibility)
val seed = 42L
val Array(trainDF, testDF) = cleanDF.randomSplit(Array(0.8, 0.2), seed)

// Cache both sets — they are reused multiple times by the pipeline
trainDF.cache()
testDF.cache()

println("\n=== Train / Test split ===")
println(s"Training rows : ${trainDF.count()}")
println(s"Testing  rows : ${testDF.count()}")
println(s"Total    rows : ${cleanDF.count()}")

println("\n=== Genre distribution — TRAIN ===")
trainDF.
  groupBy("genre").count().
  orderBy(desc("count")).
  show(truncate = false)

println("=== Genre distribution — TEST ===")
testDF.
  groupBy("genre").count().
  orderBy(desc("count")).
  show(truncate = false)

// ─────────────────────────────────────────────────────────────
//  STEP 3 — Build & Fit ML Pipeline
// ─────────────────────────────────────────────────────────────

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer}
import org.apache.spark.ml.classification.LogisticRegression

// Stage 1 — Tokenizer: splits lowercase lyrics string into an array of words
val tokenizer = new Tokenizer().
  setInputCol("lyrics").
  setOutputCol("words")

// Stage 2 — StopWordsRemover: drops common English words (the, a, is …)
val remover = new StopWordsRemover().
  setInputCol("words").
  setOutputCol("filtered_words")

// Stage 3 — HashingTF: maps filtered word arrays to fixed-length feature vectors
//   numFeatures = 65536 (2^16) — large enough for song lyrics vocabulary
val hashingTF = new HashingTF().
  setInputCol("filtered_words").
  setOutputCol("raw_features").
  setNumFeatures(65536)

// Stage 4 — IDF: re-weights TF features; down-weights words common across all docs
val idf = new IDF().
  setInputCol("raw_features").
  setOutputCol("features").
  setMinDocFreq(2)   // ignore terms that appear in fewer than 2 songs

// Stage 5 — StringIndexer: encodes genre strings as numeric labels
//   e.g. "pop"->0, "rock"->1, … (order determined by frequency, descending)
val labelIndexer = new StringIndexer().
  setInputCol("genre").
  setOutputCol("label").
  setHandleInvalid("keep")   // keeps unseen labels as extra index at predict time

// Stage 6 — Logistic Regression classifier
//   maxIter=100 gives good convergence; regParam=0.01 light L2 regularisation
val lr = new LogisticRegression().
  setFeaturesCol("features").
  setLabelCol("label").
  setMaxIter(100).
  setRegParam(0.01).
  setElasticNetParam(0.0)   // 0.0 = pure L2 (Ridge)

// Assemble the pipeline
val pipeline = new Pipeline().
  setStages(Array(tokenizer, remover, hashingTF, idf, labelIndexer, lr))

// ── Print stages for inspection ──
println("\n=== Pipeline stages ===")
pipeline.getStages.zipWithIndex.foreach { case (stage, i) =>
  println(s"  Stage $i : ${stage.getClass.getSimpleName}")
}

// ── Fit on training data ──
println("\nFitting pipeline on training data … (this may take a minute)")
val pipelineModel = pipeline.fit(trainDF)
println("Pipeline training complete.")

// ─────────────────────────────────────────────────────────────
//  STEP 4 — Evaluate Model
// ─────────────────────────────────────────────────────────────

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.col

// 4-a  Run predictions on the test set
val predictions = pipelineModel.transform(testDF)

// 4-b  Overall accuracy
val evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("label").
  setPredictionCol("prediction").
  setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println(s"\n=== Test Accuracy: ${math.round(accuracy * 10000) / 100.0} % ===")

// 4-c  Weighted F1 score
val f1Evaluator = new MulticlassClassificationEvaluator().
  setLabelCol("label").
  setPredictionCol("prediction").
  setMetricName("f1")

val f1 = f1Evaluator.evaluate(predictions)
println(s"=== Weighted F1 Score: ${math.round(f1 * 10000) / 100.0} % ===")

// 4-d  Confusion matrix via MLlib (needs RDD[(Double,Double)])
println("\n=== Confusion Matrix ===")
val predAndLabels = predictions.
  select(col("prediction"), col("label")).
  rdd.map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(predAndLabels)
println("(rows = actual, cols = predicted)\n")
println(metrics.confusionMatrix)

// 4-e  Recover the genre label mapping from StringIndexer model
//   PipelineModel stage index 4 is the StringIndexerModel
import org.apache.spark.ml.feature.StringIndexerModel
val indexerModel = pipelineModel.stages(4).asInstanceOf[StringIndexerModel]
val labels = indexerModel.labels   // Array[String] ordered by index

println("\n=== Label index mapping ===")
labels.zipWithIndex.foreach { case (genre, idx) =>
  println(f"  $idx%2d  ->  $genre")
}

// 4-f  Per-class precision / recall / F1
println("\n=== Per-class metrics ===")
println(f"  ${"Genre"}%-12s  ${"Precision"}%9s  ${"Recall"}%9s  ${"F1"}%9s")
labels.indices.foreach { i =>
  val p  = metrics.precision(i.toDouble)
  val r  = metrics.recall(i.toDouble)
  val f  = metrics.fMeasure(i.toDouble)
  println(f"  ${labels(i)}%-12s  ${p}%9.4f  ${r}%9.4f  ${f}%9.4f")
}

// 4-g  Sample predictions (10 rows)
println("\n=== Sample predictions ===")
predictions.
  select(
    substring(col("lyrics"), 1, 60).alias("lyrics_snippet"),
    col("genre").alias("actual"),
    col("prediction")).
  show(10, truncate = false)

// ─────────────────────────────────────────────────────────────
//  STEP 5 — Save Model
// ─────────────────────────────────────────────────────────────

// Save to the /model directory (relative to the project root where
// spark-shell was launched).  Spark writes a directory tree, not a
// single file — do NOT try to open it as a file.
val modelSavePath = "model/genre_classifier"

// overwrite = true: safe to re-run after retraining
pipelineModel.write.overwrite().save(modelSavePath)
println(s"\n=== Model saved to: $modelSavePath ===")

// ── Verify: reload and run one test prediction ──
import org.apache.spark.ml.PipelineModel

val reloadedModel = PipelineModel.load(modelSavePath)
println("=== Model reloaded successfully ===")

// Quick sanity check — transform a single row from the test set
val sampleRow = testDF.limit(1)
val samplePred = reloadedModel.transform(sampleRow)
println("\n=== Sanity-check prediction from reloaded model ===")
samplePred.
  select(
    col("genre").alias("actual"),
    col("prediction"),
    substring(col("lyrics"), 1, 60).alias("lyrics_snippet")).
  show(truncate = false)

// Also save the label array alongside the model as a plain text file
// so the Flask API can read it without a Spark context
import java.nio.file.{Files, Paths}
import java.nio.charset.StandardCharsets
val labelsFilePath = "model/labels.txt"
Files.write(
  Paths.get(labelsFilePath),
  labels.mkString("\n").getBytes(StandardCharsets.UTF_8))
println(s"=== Label list saved to: $labelsFilePath ===")

// ─────────────────────────────────────────────────────────────
//  STEP 9 — Verification: confirm 8 genres are present
// ─────────────────────────────────────────────────────────────
println("\n=== Step 9 — Genre classes in trained model ===")
labels.zipWithIndex.foreach { case (genre, idx) =>
  println(f"  $idx%2d  ->  $genre")
}
println(s"\nTotal classes: ${labels.length}")

// Confirm soul appears in the test predictions
println("\n=== Soul samples in test set ===")
testDF.
  filter(col("genre") === "soul").
  select(
    col("genre"),
    substring(col("lyrics"), 1, 60).alias("lyrics_snippet")).
  show(5, truncate = false)
