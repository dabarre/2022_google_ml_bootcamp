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
# Computer vision with CNNs
#
# Create and train a classifier for horses or humans using the provided data.
# Make sure your final layer is a 1 neuron, activated by sigmoid as shown.
#
# The test will use images that are 300x300 with 3 bytes color depth so be sure to
# design your neural network accordingly

import tensorflow as tf
import urllib
import zipfile


def solution_model():
    _TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
    _TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
    urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
    local_zip = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/horse-or-human/')
    zip_ref.close()
    urllib.request.urlretrieve(_TEST_URL, 'testdata.zip')
    local_zip = 'testdata.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/testdata/')
    zip_ref.close()

    # YOUR CODE
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)
    tf.keras.backend.clear_session()
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10,
                                                      monitor='val_loss')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     min_lr=1e-5,
                                                     patience=5,
                                                     mode='min')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_accuracy',
                                                          filepath='mymodel.h5',
                                                          save_best_only=True,
                                                          verbose=1)
    callbacks = [
        #myCallback(),
        #early_stopping,
        #reduce_lr,
        model_checkpoint
    ]

    img_shape = (300, 300)
    batch_size = 32

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        # Your code here. Should at least have a rescale. Other parameters can help with overfitting.
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #Your Code here
        rescale=1. / 255
    )

    train_dir = "tmp/horse-or-human/"
    valid_dir = "tmp/testdata/"
    train_generator = train_datagen.flow_from_directory(
        #Your Code Here
        directory=train_dir,
        batch_size=batch_size,
        class_mode='binary',
        target_size=img_shape
    )
    validation_generator = validation_datagen.flow_from_directory(
        #Your Code Here
        directory=valid_dir,
        batch_size=batch_size,
        class_mode='binary',
        target_size=img_shape
    )

    model = tf.keras.models.Sequential([
        # Note the input shape specified on your first layer must be (300,300,3)
        # Your Code here

        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
        tf.keras.layers.GlobalMaxPool2D(),

        # This is the last layer. You should not change this code.
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())
    model.compile(
        #Your Code Here
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        #Your Code Here
        train_generator,
        validation_data=validation_generator,
        batch_size=batch_size,
        shuffle=True,
        epochs=100,
        callbacks=callbacks
    )

    return model
    # NOTE: If training is taking a very long time, you should consider setting the batch size
    # appropriately on the generator, and the steps per epoch in the model.fit() function.

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    #model.save("mymodel.h5")