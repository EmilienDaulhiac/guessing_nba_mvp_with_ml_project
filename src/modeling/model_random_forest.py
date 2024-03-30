# src/models/model1.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score


def train_random_forest(X_train, y_train, 
                        n_estimators=100, 
                        max_depth=None,
                        min_samples_split =  2,
                        min_samples_leaf =  1,
                        max_features = 1.0,
                        bootstrap = True):
    """
    Parameters:
    - X_train: Features of the training dataset.
    - y_train: Labels corresponding to the training features.
    - n_estimators: The number of trees in the forest (default=100).
    - max_depth: The maximum depth of each tree in the forest (default=None).
    - min_samples_split: The minimum number of samples required to split an internal node (default=2).
    - min_samples_leaf: The minimum number of samples required to be at a leaf node (default=1).
    - max_features: The number of features to consider when looking for the best split (default=1.0, i.e., all features).
    - bootstrap: Whether to bootstrap samples when building trees (default=True).

    Returns:
    - trained_model: Trained RandomForestClassifier
    """
    # Initialize the RandomForestClassifier
    model = RandomForestRegressor(n_estimators=n_estimators, 
                                  max_depth=max_depth, 
                                  min_samples_split = min_samples_split,
                                  min_samples_leaf = min_samples_leaf,
                                  max_features = max_features,
                                  bootstrap = bootstrap,
                                  random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)

    # Print training and validation accuracies
    print(f"Training Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train)}")
    print(f"Training Mean Squarred Error: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Training R2 score: {r2_score(y_train, y_pred_train)}")

    return model