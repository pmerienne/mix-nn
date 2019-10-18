import pandas as pd
import pkgutil
import io

from mixnn.datasets import mnist
from sklearn.model_selection import train_test_split


def load_mnist_digits_data():
    return mnist.load_data()


def load_titanic_data():
    titanic_data = pkgutil.get_data('mixnn', 'datasets/titanic.csv')
    titanic_data = io.BytesIO(titanic_data)
    titanic_df = pd.read_csv(titanic_data)

    target = 'Survived'
    features = [
        {"name": "PassengerId", "type": "categorical"},
        {"name": "Pclass", "type": "categorical"},
        {"name": "Name", "type": "categorical"},
        {"name": "Sex", "type": "categorical"},
        {"name": "SibSp", "type": "categorical"},
        {"name": "Parch", "type": "categorical"},
        {"name": "Ticket", "type": "categorical"},
        {"name": "Cabin", "type": "categorical"},
        {"name": "Embarked", "type": "categorical"},
        {"name": "Age", "type": "numerical"},
        {"name": "Fare", "type": "numerical"},
    ]

    # Fill Missing values
    for feature in features:
        column = feature['name']
        if feature['type'] == "categorical":
            titanic_df[column] = titanic_df[column].fillna('N/A')
        elif feature['type'] == "numerical":
            titanic_df[column] = titanic_df[column].fillna(titanic_df[column].mean())

    columns = [feature['name'] for feature in features]
    X = titanic_df[columns].values
    y = titanic_df[target].values
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    return features, (X_train, y_train), (X_validation, y_validation)
