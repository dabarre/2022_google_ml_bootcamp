# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='auto')
    rl = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=15, factor=0.5, verbose=1)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train / 255.
    x_test = x_test / 255.

    print(x_train[0].shape)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])
    ### END CODE HERE

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # YOUR CODE HERE
    model.fit(x_train, y_train, epochs=10, batch_size=64, callbacks=[es, rl], validation_split=0.2)

    result = model.evaluate(x_test, y_test, batch_size=1)
    print("loss : ", result[0])
    print("acc : ", result[1])

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
