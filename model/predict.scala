// ============================================================
//  Music Genre Classifier — Prediction Function
//  Usage inside spark-shell (after train_model.scala has been run):
//    :load model/predict.scala
//    predictGenre("your lyrics here")
//
//  Or standalone in a fresh spark-shell session:
//    spark-shell --driver-memory 2g
//    :load model/predict.scala
// ============================================================

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructType, StructField, StringType}
import scala.io.Source
import java.io.File

spark.sparkContext.setLogLevel("WARN")

// ─────────────────────────────────────────────────────────────
//  STEP 6 — Reusable Prediction Function
// ─────────────────────────────────────────────────────────────

val modelPath  = "model/genre_classifier"
val labelsPath = "model/labels.txt"

// 6-a  Load model once at startup (expensive — not repeated per call)
println(s"Loading model from $modelPath ...")
val loadedModel = PipelineModel.load(modelPath)
println("Model loaded.")

// 6-b  Load label list saved by train_model.scala Step 5
val genreLabels: Array[String] = Source.fromFile(new File(labelsPath)).
  getLines().
  filter(_.nonEmpty).
  toArray
println(s"Labels loaded: ${genreLabels.mkString(", ")}")

// 6-c  Pre-build the single-column schema used by every predict call
//      (avoids rebuilding StructType on every invocation)
val lyricsSchema = StructType(
  StructField("lyrics", StringType, nullable = true) :: Nil)

// ─────────────────────────────────────────────────────────────
//  predictGenre : String => Map[String, Double]
//
//  Returns { genre -> probability } for all classes, e.g.:
//    Map("pop" -> 0.42, "rock" -> 0.31, ...)
// ─────────────────────────────────────────────────────────────
def predictGenre(lyrics: String): Map[String, Double] = {
  // Wrap raw string in a one-row DataFrame without relying on implicits
  val inputDF = spark.createDataFrame(
    spark.sparkContext.parallelize(Seq(Row(lyrics))),
    lyricsSchema)

  // Run the full Tokenizer -> TF-IDF -> LogisticRegression pipeline
  val result = loadedModel.transform(inputDF)

  // Extract probability vector (length = number of genre classes)
  val probVector = result.
    select("probability").
    first().
    getAs[DenseVector]("probability")

  // Zip label names with probability values and round to 4 decimal places
  genreLabels.zipWithIndex.map { case (genre, idx) =>
    genre -> (math.round(probVector(idx) * 10000.0) / 10000.0)
  }.toMap
}

// ─────────────────────────────────────────────────────────────
//  showPrediction : pretty-print results in the terminal
// ─────────────────────────────────────────────────────────────
def showPrediction(lyrics: String): Unit = {
  val probs    = predictGenre(lyrics)
  val sorted   = probs.toSeq.sortBy(-_._2)
  val topGenre = sorted.head._1

  println(s"\nLyrics snippet : ${lyrics.take(80)} ...")
  println("-" * 50)
  sorted.foreach { case (genre, prob) =>
    val bar   = "|" * (prob * 40).toInt
    val pct   = f"${prob * 100}%5.1f%%"
    val arrow = if (genre == topGenre) " <- predicted" else ""
    println(f"  ${genre}%-12s $pct  $bar$arrow")
  }
  println()
}

// ─────────────────────────────────────────────────────────────
//  Quick demo — runs immediately on :load
// ─────────────────────────────────────────────────────────────
val demoLyrics = "baby I love you every moment every day I need you close to me"
println("\n=== Demo prediction ===")
showPrediction(demoLyrics)

val demoResult: Map[String, Double] = predictGenre(demoLyrics)
println("Raw map output (for API use):")
println(demoResult)

