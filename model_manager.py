import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score


class ModelManager:
    """
    Class for building, training, and evaluating the CNN model with residual blocks.
    """

    def __init__(self, num_emotions, emotion_columns):
        self.num_emotions = num_emotions
        self.emotion_columns = emotion_columns

    def compute_class_weights(self, y_train):
        """
        Computes class weights for balancing the loss function.

        Parameters:
            y_train (np.ndarray): Training labels.

        Returns:
            np.ndarray: Array of class weights.
        """
        num_samples = y_train.shape[0]
        num_classes = y_train.shape[1]
        class_weights = np.zeros(num_classes)

        for i in range(num_classes):
            count_pos = np.sum(y_train[:, i])
            if count_pos == 0:
                class_weights[i] = 1.0
            else:
                class_weights[i] = num_samples / (2.0 * count_pos)

        return class_weights

    def weighted_binary_crossentropy(self, class_weights):
        """
        Creates a weighted binary crossentropy loss function.

        Parameters:
            class_weights (array-like): Pre-computed class weights.

        Returns:
            function: Weighted binary crossentropy loss function.
        """
        class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

        def loss(y_true, y_pred):
            bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
            weights = y_true * class_weights_tensor + (1 - y_true)
            weighted_bce = weights * bce
            return tf.keras.backend.mean(weighted_bce)

        return loss

    def residual_block(self, x, filters, kernel_size=(3, 3), pool=True, dropout_rate=0.3):
        """
        Builds a residual block.

        Parameters:
            x: Input tensor.
            filters (int): Number of filters.
            kernel_size (tuple): Kernel size.
            pool (bool): Whether to apply pooling.
            dropout_rate (float): Dropout rate.

        Returns:
            Tensor: Output tensor.
        """
        shortcut = x
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if pool:
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")(x)
            shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=(2, 2), padding="same")(shortcut)

        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        return x

    def build_model(self, input_shape, class_weights, dropout_rate=0.5, learning_rate=0.0005):
        """
        Builds and compiles the CNN model.

        Parameters:
            input_shape (tuple): Shape of the input.
            class_weights (array-like): Class weights for loss function.
            dropout_rate (float): Dropout rate.
            learning_rate (float): Learning rate.

        Returns:
            tf.keras.Model: The compiled model.
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)

        x = self.residual_block(x, filters=32, kernel_size=(3, 3), pool=True, dropout_rate=dropout_rate / 2)
        x = self.residual_block(x, filters=64, kernel_size=(3, 3), pool=True, dropout_rate=dropout_rate)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        outputs = tf.keras.layers.Dense(self.num_emotions, activation="sigmoid")(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.weighted_binary_crossentropy(class_weights),
            metrics=[
                "binary_accuracy",
                tf.keras.metrics.AUC(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
            ],
        )

        return model

    def train_model(self, model, X_train, y_train, X_val, y_val, input_shape,
                    epochs=50, batch_size=32, patience=10):
        """
        Trains the CNN model.

        Parameters:
            model (tf.keras.Model): Compiled model.
            X_train, y_train: Training data and labels.
            X_val, y_val: Validation data and labels.
            input_shape (tuple): Input shape.
            epochs (int): Maximum epochs.
            batch_size (int): Batch size.
            patience (int): Early stopping patience.

        Returns:
            History: Training history.
        """
        # Use data augmentation if training set is small
        if X_train.shape[0] < 1000:
            print("Data augmentation in progress...")
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode="nearest",
            )

        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        )

        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        )

        # Model checkpoint
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
        )

        # Train model
        if X_train.shape[0] < 1000:
            # Use data augmentation
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=[lr_scheduler, early_stopping, model_checkpoint],
                verbose=1,
            )
        else:
            # Train without augmentation for larger datasets
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[lr_scheduler, early_stopping, model_checkpoint],
                verbose=1,
            )

        return history

    def find_best_thresholds(self, y_true, y_pred_prob, thresholds=np.arange(0.3, 0.71, 0.05)):
        """
        Finds optimal thresholds per emotion based on F1 score.

        Parameters:
            y_true (np.ndarray): True labels.
            y_pred_prob (np.ndarray): Predicted probabilities.
            thresholds (iterable): Threshold values to try.

        Returns:
            list: Best threshold for each emotion.
        """
        best_thresholds = []

        for i in range(y_true.shape[1]):
            best_f1 = 0.0
            best_threshold = 0.5

            for t in thresholds:
                y_pred = (y_pred_prob[:, i] >= t).astype(int)
                f1 = f1_score(y_true[:, i], y_pred)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            best_thresholds.append(best_threshold)

        return best_thresholds

    def evaluate_model(self, model, X_test, y_test, thresholds=None):
        """
        Evaluates the model on test data.

        Parameters:
            model (tf.keras.Model): Trained model.
            X_test (np.ndarray): Test data.
            y_test (np.ndarray): Test labels.
            thresholds (list, optional): Thresholds per emotion.

        Returns:
            tuple: (metrics, used_thresholds)
        """
        # Get predictions
        y_pred_prob = model.predict(X_test)

        if thresholds is None:
            used_thresholds = [0.5] * y_test.shape[1]
        else:
            used_thresholds = thresholds

        y_pred = np.zeros_like(y_pred_prob)
        for i, t in enumerate(used_thresholds):
            y_pred[:, i] = (y_pred_prob[:, i] >= t).astype(int)

        # Calculate metrics
        metrics = {
            "hamming_loss": hamming_loss(y_test, y_pred),
            "sample_f1": f1_score(y_test, y_pred, average="samples"),
            "micro_f1": f1_score(y_test, y_pred, average="micro"),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "precision_micro": precision_score(y_test, y_pred, average="micro"),
            "recall_micro": recall_score(y_test, y_pred, average="micro"),
        }

        # Per-emotion metrics
        for i, emotion in enumerate(self.emotion_columns):
            metrics[f"{emotion}_f1"] = f1_score(y_test[:, i], y_pred[:, i])
            metrics[f"{emotion}_precision"] = precision_score(y_test[:, i], y_pred[:, i], zero_division=0)
            metrics[f"{emotion}_recall"] = recall_score(y_test[:, i], y_pred[:, i])

        return metrics, used_thresholds
