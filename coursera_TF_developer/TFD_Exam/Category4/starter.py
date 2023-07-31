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
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
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
        # early_stopping,
        #reduce_lr,
        #model_checkpoint
    ]

    with open('sarcasm.json', 'r') as file:
        data = json.load(file)
        for row in data:
            sentences.append(row['headline'])
            labels.append(row['is_sarcastic'])

    # Prepare data
    train_sentences = sentences[:training_size]
    valid_sentences = sentences[training_size:]

    train_labels = labels[:training_size]
    valid_labels = labels[training_size:]

    # Prepare input
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size,
                                                      oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    train_seqs = tokenizer.texts_to_sequences(train_sentences)
    train_pad_seqs = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,
                                                                   maxlen=max_length,
                                                                   padding=padding_type,
                                                                   truncating=trunc_type)
    valid_seqs = tokenizer.texts_to_sequences(valid_sentences)
    valid_pad_seqs = tf.keras.preprocessing.sequence.pad_sequences(valid_seqs,
                                                                   maxlen=max_length,
                                                                   padding=padding_type,
                                                                   truncating=trunc_type)

    train_labels = np.array(train_labels)
    valid_labels = np.array(valid_labels)

    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        tf.keras.layers.Embedding(input_dim=vocab_size + 1,
                                  output_dim=embedding_dim,
                                  input_length=max_length,
                                  # weights=[embeddings_matrix],
                                  trainable=True),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=['accuracy'])

    model.fit(x=train_pad_seqs,
              y=train_labels,
              validation_data=(valid_pad_seqs, valid_labels),
              batch_size=256,
              shuffle=True,
              epochs=50,
              callbacks=callbacks)

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
