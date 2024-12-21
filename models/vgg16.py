from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_absolute_error


class VGG16Model:
    def __init__(self, input_shape=(224, 224, 3), dropout=None, dense=None, learning_rate=1e-4):
        self.input_shape = input_shape
        self.dropout = dropout
        self.dense = dense
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = Flatten()(x)
        if self.dropout:
            x = Dropout(self.dropout)(x)
        x = Dense(4096, activation='relu')(x)
        if self.dense:
            x = Dropout(self.dropout)(x)
            x = Dense(self.dense, activation='relu')(x)
        predictions = Dense(units=1, activation='linear')(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='mean_squared_error',
                      metrics=['mae'])
        return model

    def train(self, x_train, y_train, x_val, y_val, batch_size=32, epochs=50):
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        early_stopping = EarlyStopping(
            monitor='val_mae',
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1,
            callbacks=[early_stopping]
        )

    def evaluate(self, x_test, y_test):
        predictions = self.model.predict(x_test)
        mae = mean_absolute_error(y_test, predictions.flatten())
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        return mae

    def save(self, filepath):
        self.model.save(filepath)
