import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


def read_ratings_data(path):
    """Read Movielens data"""
    ratings = pd.read_csv(path)
    return ratings.userId, ratings.movieId, ratings.rating


def build_the_model(n_users, n_movies, n_user_dimension=32, n_movie_dimension=32, dropout_rate=0.25):
    """Build the computational graph according to feature vectors"""
    # Set Input Placeholder
    user_input = tf.keras.layers.Input(shape=[1])
    movie_input = tf.keras.layers.Input(shape=[1])

    # Transfer user/movie index to vector
    user_vec = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_users, n_user_dimension)(user_input))
    movie_vec = tf.keras.layers.Flatten()(
        tf.keras.layers.Embedding(n_movies, n_movie_dimension)(movie_input))

    # Build Computational Graph
    nn = tf.keras.layers.concatenate([user_vec, movie_vec])
    nn = tf.keras.layers.Dense(
        (n_user_dimension+n_movie_dimension)*4, activation="relu")(nn)
    nn = tf.keras.layers.Dropout(dropout_rate)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Dense(
        (n_user_dimension+n_movie_dimension)*2, activation="relu")(nn)
    nn = tf.keras.layers.Dropout(dropout_rate)(nn)
    nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Dense(
        (n_user_dimension+n_movie_dimension), activation="relu")(nn)
    result = tf.keras.layers.Dense(1)(nn)

    # Compile the model
    model = tf.keras.models.Model(
        inputs=[user_input, movie_input], outputs=result)
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


def validate_the_model(n_users, n_movies, users_train, movies_train, ratings_train):
    """Choose the best parameter to build the model"""
    n_user_dimensions = [8, 16, 32, 64]
    n_movie_dimensions = [8, 16, 32, 64]

    min_loss = float("inf")
    best_parameter = (n_user_dimensions[0], n_movie_dimensions[0])
    for n_user_dimension in n_user_dimensions:
        for n_movie_dimension in n_movie_dimensions:
            model = build_the_model(
                n_users, n_movies, n_user_dimension, n_movie_dimension)
            history = model.fit([users_train, movies_train], ratings_train,
                                batch_size=10000,
                                validation_split=0.1)
            if min_loss > min(history.history["val_loss"]):
                min_loss = min(history.history["val_loss"])
                best_parameter = (n_user_dimension, n_movie_dimension)
                print("Find best parameters: n_user_dimension={}, n_movie_dimension={}".format(
                    n_user_dimension, n_movie_dimension))
    return best_parameter


def train_the_model(model, users_train, movies_train, ratings_train, epochs_to_run=100):
    """Train the model"""
    history = model.fit([users_train, movies_train], ratings_train,
                        batch_size=10000,
                        epochs=epochs_to_run)
    plt.plot(history.history["loss"])
    plt.show()
    return model


def evalute_the_model(model, users_test, movies_test, ratings_test):
    """Evaluate the model"""
    score = model.evaluate([users_test, movies_test], ratings_test)
    return score


def main():
    dataset_path = os.path.join("datasets", "ml-20m")
    ratings_path = os.path.join(dataset_path, "ratings.csv")
    user_ids, movie_ids, rating_scores = read_ratings_data(ratings_path)

    # Get the number of users and movies
    n_users = len(user_ids.unique())
    n_movies = len(movie_ids.unique())

    # Split the data into train and test sets
    users_train, users_test, movies_train, movies_test, ratings_train, ratings_test = train_test_split(
        user_ids, movie_ids, rating_scores)

    # Build the model
    n_user_dimension, n_movie_dimension = validate_the_model(
        n_users, n_movies, users_train, movies_train, ratings_train)
    model = build_the_model(
        n_users, n_movies, n_user_dimension, n_movie_dimension)
    model = train_the_model(model, user_ids, movie_ids, rating_scores)
    score = evalute_the_model(model, users_test, movies_test, ratings_test)
    print("\nCurrent model score is {}".format(score))


if __name__ == "__main__":
    main()
