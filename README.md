# spark-recommendation

## compile

```sh
sbt package
```

## run locally

```sh
spark-submit --class "MovieLensALS" --master local[4] target/scala-2.11/movielens-recommendations_2.11-1.0.jar
```

## run in Amazon EMR

### upload to s3

```sh
aws s3 cp target/scala-2.11/movielens-recommendations_2.11-1.0.jar s3://movielens-recommendation/movielens-recommendations_2.11-1.0.jar
```

### download from s3

```sh
aws s3 cp s3://movielens-recommendation/movielens-recommendations_2.11-1.0.jar .
```

### run in EMR

```sh
spark-submit ./movielens-recommendations_2.11-1.0.jar
```
