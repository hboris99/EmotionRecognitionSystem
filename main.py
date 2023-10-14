
import numpy as np

import cv2

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.saving import model_from_json
from keras.optimizers import Adam

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization, Activation, MaxPooling2D

base_train_path = "dataset/train"
base_validation_path = "dataset/validation"
train_data = ImageDataGenerator()
test_data = ImageDataGenerator()


def create_train_set():
    batch_size = 128
    train_set = train_data.flow_from_directory(base_train_path,
                                               target_size=(48, 48),
                                               color_mode="grayscale",
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                               shuffle=True)

    return train_set


def create_validation_set():
    batch_size = 128
    test_set = test_data.flow_from_directory(base_validation_path,
                                             target_size=(48, 48),
                                             color_mode="grayscale",
                                             batch_size=batch_size,
                                             class_mode='categorical',
                                             shuffle=False)
    return test_set


def create_nn():
    model = Sequential()
    # model.add(
    #     Conv2D(input_shape=(48, 48, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    #
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(BatchNormalization())
    #
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd CNN layer
    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd CNN layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th CNN layer
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # Fully connected 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(7, activation='softmax'))

    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def train_nn(model, train, test):
    checkpoint = ModelCheckpoint("cnn.h5", monitor='accuracy', verbose=1, save_best_only=True, mode='max')
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=3,
                                  verbose=1,
                                  min_delta=0.0001)
    callbacks = [ checkpoint, reduce_lr]
    epochs = 160
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

    result = model.fit_generator(generator=train, steps_per_epoch=train.n // train.batch_size, epochs=epochs,
                                 validation_data=test,
                                 validation_steps=test.n // test.batch_size, callbacks=callbacks)
    return result, model


def save_model(model):
    serialized = model.to_json();
    with open("model.json", "w") as json_file:
        json_file.write(serialized)
        model.save_weights("model.h5")


def loadmodel():
    try:
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('model.h5')
        return loaded_model
    except FileNotFoundError:
        return None

def display_graphs(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    train = create_train_set()
    test = create_validation_set()
    model = loadmodel()
    classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    if model is None:
        nn = create_nn()
        res, model = train_nn(nn, train, test)
        save_model(model)
        display_graphs(res)
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
q
                prediction = model.predict(roi)[0]
                print(prediction)
                label = 'haoppy'
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
