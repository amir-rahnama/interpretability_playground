import pandas as pd
import matplotlib.pyplot as plt
from utils import make_label_column_numeric, df_to_examples, create_feature_spec, tfexamples_input_fn, create_feature_columns
import tensorflow as tf
import functools
import numpy as np
import witwidget
import tensorflow as tf

# Set the path to the CSV containing the dataset to train on.
csv_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'

# Set the column names for the columns in the CSV. If the CSV's first line is a header line containing
# the column names, then set this to None.
csv_columns = [
  "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital-Status",
  "Occupation", "Relationship", "Race", "Sex", "Capital-Gain", "Capital-Loss",
  "Hours-per-week", "Country", "Over-50K"]

# Read the dataset from the provided CSV and print out information about it.
df = pd.read_csv(csv_path, names=csv_columns, skipinitialspace=True)

# Set the column in the dataset you wish for the model to predict
label_column = 'Over-50K'

# Make the label column numeric (0 and 1), for use in our model.
# In this case, examples with a target value of '>50K' are considered to be in
# the '1' (positive) class and all other examples are considered to be in the
# '0' (negative) class.
make_label_column_numeric(df, label_column, lambda val: val == '>50K')

# Set list of all columns from the dataset we will use for model input.
input_features = [
  'Age', 'Workclass', 'Education', 'Marital-Status', 'Occupation',
  'Relationship', 'Race', 'Sex', 'Capital-Gain', 'Capital-Loss',
  'Hours-per-week', 'Country']

# Create a list containing all input features and the label column
features_and_labels = input_features + [label_column]

examples = df_to_examples(df)

num_steps = 2000  #@param {type: "number"}
tf.logging.set_verbosity(tf.logging.DEBUG)

# Create a feature spec for the classifier
feature_spec = create_feature_spec(df, features_and_labels)

# Define and train the classifier
train_inpf = functools.partial(tfexamples_input_fn, examples, feature_spec, label_column)
classifier = tf.estimator.LinearClassifier(
    feature_columns=create_feature_columns(input_features, feature_spec, df))
classifier.train(train_inpf, steps=num_steps)

#@title Create and train the DNN classifier {display-mode: "form"}
tf.logging.set_verbosity(tf.logging.DEBUG)
num_steps_2 = 2000  #@param {type: "number"}

classifier2 = tf.estimator.DNNClassifier(
    feature_columns=create_feature_columns(input_features, feature_spec, df),
    hidden_units=[128, 64, 32])
classifier2.train(train_inpf, steps=num_steps_2)

serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = classifier.export_savedmodel('models/linear', serving_input_receiver_fn)

serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
export_dir = classifier2.export_savedmodel('models/dnn', serving_input_receiver_fn)
