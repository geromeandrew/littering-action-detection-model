from keras import layers
from keras.models import Sequential


def model_1(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST):
    model = Sequential()

    model.add(layers.ConvLSTM2D(filters=4, kernel_size=(3, 3), activation='tanh', data_format="channels_last", 
                                recurrent_dropout=0.2, return_sequences=True, 
                                input_shape=(SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=8, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=14, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=16, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                recurrent_dropout=0.2, return_sequences=True))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2),  padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(len(CLASSES_LIST), activation="softmax"))

    model.summary()
    return model




def model_2(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST):
    model = Sequential()

    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                                     input_shape=(SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()
    return model




def model_3(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST):
    model = Sequential()

    model.add(layers.TimeDistributed(layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
                                     input_shape=(SEQUENCE_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Conv2D(
        32, (3, 3), padding='same', activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((4, 4))))
    model.add(layers.TimeDistributed(layers.Dropout(0.25)))

    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.Dense(len(CLASSES_LIST), activation='softmax'))

    model.summary()
    return model
