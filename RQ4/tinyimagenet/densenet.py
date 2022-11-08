import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


def DenseNet_model():
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=200
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(200)(x)
    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == "__main__":
    model = DenseNet_model()
    model.summary()

    f = open('/mnt/irisgpfs/projects/acc-estima-aiapp/data/tiny_imagenet/tiny_imagenet_data.pkl', 'rb')
    X = pickle.load(f)
    f = open('/mnt/irisgpfs/projects/acc-estima-aiapp/data/tiny_imagenet/tiny_imagenet_label.pkl', 'rb')
    y = pickle.load(f)
    num_classes = 200
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    # image_shape = (64, 64, 3)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15)
    model.save('densenet.h5')