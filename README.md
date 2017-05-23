# spark-recommendation #

## Compile

```sbt package```

## Run

```~/Applications/spark-2.1.0-bin-hadoop2.7/bin/spark-submit --class "MovieLensALS" --master local[4] target/scala-2.11/movielens-recommendations_2.11-1.0.jar```