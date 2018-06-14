# spark-recommendation #

### compile

```sh
sbt package
```

### run

```sh
spark-submit --class "MovieLensALS" --master local[4] target/scala-2.11/movielens-recommendations_2.11-1.0.jar
```

## AWS EMR

> aws s3 cp target/scala-2.11/movielens-recommendations_2.11-1.0.jar s3://movielens-recommendation/movielens-recommendations_2.11-1.0.jar

> aws s3 cp s3://movielens-recommendation/movielens-recommendations_2.11-1.0.jar .

> spark-submit ./movielens-recommendations_2.11-1.0.jar