import tensorflow as tf
from tensorflow.keras import layers, models

def train_cnn(X_train, y_train, 
              num_epochs=10,
              batch_size=32,
              learning_rate=0.001,
              conv_filters=[64],
              kernel_sizes=[3],
              pool_sizes=[2],
              verbose = 0):
    """
    Train a Convolutional Neural Network (CNN).

    Parameters:
    - X_train: Features of the training dataset.
    - y_train: Labels corresponding to the training features.
    - num_epochs: Number of training epochs (default=10).
    - batch_size: Batch size for training (default=32).
    - learning_rate: Learning rate for optimization (default=0.001).
    - dropout_rate: Dropout rate to prevent overfitting (default=0.5).
    - conv_filters: List of integers specifying the number of filters in each convolutional layer (default=[32, 64]).
    - kernel_sizes: List of integers specifying the kernel size in each convolutional layer (default=[3, 3]).
    - pool_sizes: List of integers specifying the pooling size in each pooling layer (default=[2, 2]).

    Returns:
    model (object)
    """
    model = models.Sequential()
    input_shape_of_Conv = X_train.shape[1:]
    for i, conv_filter in enumerate(conv_filters):
        model.add(layers.Conv2D(
            conv_filter,
            kernel_size=(3, input_shape_of_Conv[1]),
            activation='relu',
            input_shape=input_shape_of_Conv,
        ))
        model.add(layers.MaxPooling2D(pool_size=(pool_sizes[i], 1)))

    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=num_epochs, validation_split=0.2, verbose=verbose)

    return model
