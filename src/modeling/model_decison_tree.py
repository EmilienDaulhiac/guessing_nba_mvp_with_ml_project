# src/models/model1.py
from sklearn.tree import DecisionTreeRegressor  
from sklearn.metrics import  mean_absolute_error, mean_squared_error, r2_score

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_leaf=2, min_samples_split=1):
    """
    Train a DecisonTreeRegressor.

    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - max_depth: Maximum depth of the tree (default=None)
    - min_samples_split: Minimum number of samples required to split an internal node
    - min_samples_leaf: Minimum number of samples required to be at a leaf node

    Returns:
    - trained_model: Trained RandomForestClassifier
    """
    model = DecisionTreeRegressor(max_depth = max_depth, 
                                 min_samples_leaf = min_samples_leaf, 
                                 min_samples_split = min_samples_split, 
                                 random_state=42)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)

    print(f"Training Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train)}")
    print(f"Training Mean Squarred Error: {mean_squared_error(y_train, y_pred_train)}")
    print(f"Training R2 score: {r2_score(y_train, y_pred_train)}")

    return model
