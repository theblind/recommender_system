import math
import os
import sys

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel


def read_ratings(path):
    """Read Ratings to RDD and return it"""
    ratings_raw_data = sc.textFile(path)
    ratings_raw_data_header = ratings_raw_data.take(1)[0]
    ratings_data = ratings_raw_data.filter(
        lambda line: line != ratings_raw_data_header).map(
        lambda line: line.split(",")).map(
        lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()
    return ratings_data


def read_movies(path):
    """Read Movies to RDD and return it"""
    movies_raw_data = sc.textFile(path)
    movies_raw_data_header = movies_raw_data.take(1)[0]
    movies_data = movies_raw_data.filter(
        lambda line: line != movies_raw_data_header).map(
        lambda line: line.split(",")).map(
        lambda tokens: (tokens[0], tokens[1])).cache()
    return movies_data


def split_data(ratings_data):
    """Split ratings data into train/val/test"""
    training_RDD, validation_RDD, test_RDD = ratings_data.randomSplit([7, 1, 2])
    return training_RDD, validation_RDD, test_RDD


def validata_model(training_RDD, validation_RDD, iterations=10):
    ranks = [4, 8, 12]
    regularizations = [0.1]
    result = {}
    min_error = float("inf")
    best_rank = 0
    best_regularization = 0.1
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    for rank in ranks:
        for regularization in regularizations:
            model = ALS.train(training_RDD,
                              rank=rank,
                              iterations=iterations,
                              lambda_=regularization)
            predictions = model.predictAll(validation_for_predict_RDD).map(
                lambda r: ((r[0], r[1]), r[2]))
            rates_and_preds = validation_RDD.map(
                lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
            error = math.sqrt(rates_and_preds.map(
                lambda r: (r[1][0] - r[1][1])**2).mean())
            print("For rank {} the RMSE is {}".format(rank, error))
            result[(rank, regularization)] = error
            if error < min_error:
                min_error = error
                best_rank = rank
                best_regularization = regularization
    return best_rank, best_regularization


def train_model(training_RDD, test_RDD, rank=8, regularization=0.1, iterations=10):
    """Use ALS to train the model"""
    model = ALS.train(training_RDD,
                      rank=rank,
                      iterations=iterations,
                      lambda_=regularization)
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    predictions = model.predictAll(test_for_predict_RDD).map(
        lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(
        lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(
        lambda r: (r[1][0] - r[1][1])**2).mean())
    print(error)
    return model


def main():
    # Read dataset
    dataset_path = os.path.join("datasets", "ml-20m")
    ratings_file_path = os.path.join(dataset_path, "ratings.dat")
    ratings_data = read_ratings(ratings_file_path)

    # Split data
    training_RDD, validation_RDD, test_RDD = split_data(ratings_data)

    # Choose the best training parameter
    best_rank, best_regularization = validata_model(
        training_RDD, validation_RDD)

    # Train the model
    model = train_model(training_RDD, test_RDD,
                        rank=best_rank,
                        regularization=best_regularization)

    # Save the model
    model_path = os.path.join("models", "spark_ALS")
    model.save(sc, model_path)


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    main()
    sc.stop()
