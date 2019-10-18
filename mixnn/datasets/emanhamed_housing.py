import os
import pandas as pd

from keras.utils.data_utils import get_file
from sklearn.model_selection import train_test_split

dataset_URL = "https://codeload.github.com/emanhamed/Houses-dataset/zip/master"


def load_data(path='emanhamed-housing.zip'):
    zip_path = get_file(path,
                    origin='https://codeload.github.com/emanhamed/Houses-dataset/zip/master',
                    extract=True)
    datasets_directory = os.path.dirname(zip_path)
    housing_directory = os.path.join(datasets_directory, "Houses-dataset-master", "Houses Dataset")
    houses_info_file = os.path.join(housing_directory, "HousesInfo.txt")

    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(houses_info_file, sep=" ", header=None, names=cols)

    # Drop houses
    df = remove_houses(df)
    # Drop missing data
    df = df[~pd.isna(df['price'])]

    df['bathroom_img'] = df.index.map(lambda house_index: get_image_path(house_index, housing_directory, "bathroom"))
    df['bedroom_img'] = df.index.map(lambda house_index: get_image_path(house_index, housing_directory, "bedroom"))
    df['kitchen_img'] = df.index.map(lambda house_index: get_image_path(house_index, housing_directory, "kitchen"))
    df['frontal_img'] = df.index.map(lambda house_index: get_image_path(house_index, housing_directory, "frontal"))

    features = [
        {"name": "bedrooms", "type": "numerical"},
        {"name": "bathrooms", "type": "numerical"},
        {"name": "area", "type": "numerical"},
        {"name": "zipcode", "type": "categorical"},
        {"name": "bathroom_img", "type": "image", "cnn": "medium"},
        {"name": "bedroom_img", "type": "image", "cnn": "medium"},
        {"name": "kitchen_img", "type": "image", "cnn": "medium"},
        {"name": "frontal_img", "type": "image", "cnn": "medium"},
    ]

    X = df[[f['name'] for f in features]].values
    y = df['price'].values

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, y_train), (X_validation, y_validation), features


def get_image_path(house_index, housing_directory, category):
    filename = "%s_%s.jpg" % (house_index + 1, category)
    return os.path.join(housing_directory, filename)


def remove_houses(df):
    # determine (1) the unique zip codes and (2) the number of data
    # points with each zip code
    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    # loop over each of the unique zip codes and their corresponding
    # count
    for (zipcode, count) in zip(zipcodes, counts):
        # the zip code counts for our housing dataset is *extremely*
        # unbalanced (some only having 1 or 2 houses per zip code)
        # so let's sanitize our data by removing any houses with less
        # than 25 houses per zip code
        if count < 25:
            idxs = df[df["zipcode"] == zipcode].index
            df = df.drop(idxs)
    return df
