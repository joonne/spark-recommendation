import java.io.File

import org.apache.log4j.{Logger, Level}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd._
import org.apache.spark.mllib.recommendation.{ALS, Rating, MatrixFactorizationModel}

object MovieLensALS {
  def main(args: Array[String]) {
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.eclipse.jetty.server").setLevel(Level.OFF)

    val conf = new SparkConf()
      .setAppName("MovieLensALS")
    implicit val sc = new SparkContext(conf)

    // load personal ratings

    val personalRatings = sc.textFile("s3n://movielens-recommendation/personalRatings.txt")
      .map { line =>
        val fields = line.split(",")
        Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
      }.filter(_.rating > 0.0)

    // load ratings

    val ratings = sc.textFile("s3n://movielens-recommendation/ml-20m/ratings.csv")
      .filter(!isHeader("userId")(_))
      .map { line =>
        val fields = line.split(",")
        // format: (timestamp % 10, Rating(userId, movieId, rating))
        (fields(3).toLong % 10, Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble))
      }

    // load movies

    val movies = sc.textFile("s3n://movielens-recommendation/ml-20m/movies.csv")
      .filter(!isHeader("movieId")(_))
      .map { line =>
        val fields = line.split(",")
        // format: (movieId, movieName)
        (fields(0).toInt, fields(1))
      }.collect().toMap

    val numRatings = ratings.count
    val numUsers = ratings.map(_._2.user).distinct.count
    val numMovies = ratings.map(_._2.product).distinct.count

    println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

    val numPartitions = 4
    val training = ratings.filter(x => x._1 < 6)
      .values
      .union(personalRatings)
      .repartition(numPartitions)
      .cache()

    val validation = ratings.filter(x => x._1 >= 6 && x._1 < 8)
      .values
      .repartition(numPartitions)
      .cache()

    val test = ratings.filter(x => x._1 >= 8)
      .values
      .cache()

    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = test.count()

    println(s"Training: $numTraining, validation: $numValidation, test: $numTest")

    // training

    var bestModel: Option[MatrixFactorizationModel] = None
    val ranks = List(8, 12)
    val lambdas = List(0.01, 0.1, 1, 2, 3)
    val numIters = List(1, 3, 5, 15, 20)
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = -1.0
    var bestNumIter = -1
    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
      val model = ALS.train(training, rank, numIter, lambda)
      val validationRmse = computeRmse(model, validation, numValidation)
      println(s"RMSE (validation) = $validationRmse for the model trained with rank = $rank, lambda = $lambda, and numIter = $numIter.")
      if (validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }

    val testRmse = computeRmse(bestModel.get, test, numTest)

    println(s"The best model was trained with rank = $bestRank and lambda = $bestLambda and numIter = $bestNumIter and its RMSE on the test set is $testRmse .")

    val myRatedMovieIds = personalRatings.map(_.product).collect()
    val candidates = sc.parallelize(
      movies.keys
        .filter(!myRatedMovieIds.contains(_))
        .map((0, _))
        .toSeq
    )

    val recommendations = bestModel.get
      .predict(candidates)
      .collect()
      .sortBy(- _.rating)
      .take(10)

    var i = 1
    println("Movies recommended for you:")
    recommendations.foreach { r =>
      println("%2d".format(i) + ": " + movies(r.product))
      i += 1
    }

    // clean up
    sc.stop()
  }

  def isHeader(headerId: String)(line: String): Boolean = {
    line.contains(headerId)
  }

  // Compute RMSE (Root Mean Squared Error)
  def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
    val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
    val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
      .join(data.map(x => ((x.user, x.product), x.rating)))
      .values
    math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2) * (x._1 - x._2)).reduce(_ + _) / n)
  }
}