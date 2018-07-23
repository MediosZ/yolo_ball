import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Lambda
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from input import input_data
from model import yolo_loss, data_generator


"""
# Generate dummy data
x_train = np.random.random((1000, 20))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

model = Sequential()
model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
"""

def constructModel(grid_shape=(20, 15)):
    y_true = Input(shape=(grid_shape[0], grid_shape[1], 5))
    inputs = Input(shape=(None, None, 3), name='input', dtype='float32')
    # layer one(320, 240, 3)
    x = Conv2D(4, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    # layer two
    x = Conv2D(8, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    # layer three
    x = Conv2D(16, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    # layer four
    x = Conv2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    # layer five
    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPool2D()(x)
    # layer six
    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    # layer seven
    x = Conv2D(128, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    # layer eight
    outputs = Conv2D(5, (3,3), padding="same")(x)
    print(outputs.shape)
    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss", arguments={
        "batch_size": 16, "grid_shape":(20,15)
    })([outputs, y_true])
    return Model(inputs=[inputs, y_true], outputs=model_loss)

#images, labels = input_data('train.txt')

def main():
    new_yolo = constructModel()
    log_dir = 'log'
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    new_yolo.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    num_train = 460
    new_yolo.fit_generator(data_generator('./train.txt', 16), steps_per_epoch=max(1, num_train//16),
        validation_data=data_generator('./vel.txt', 16), validation_steps=1,
        callbacks=[logging, checkpoint, reduce_lr, early_stopping], epochs=60)

    #score = new_yolo.evaluate(x_test, y_test, batch_size=128)
#model = constructModel()

main()