from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

from data import polyvore_dataset, DataGenerator
from utils import Config

import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
import numpy as np 
import matplotlib.pyplot as plt



if __name__=='__main__':

    # data generators
    dataset = polyvore_dataset()
    transforms = dataset.get_data_transforms()
    X_train, X_test, y_train, y_test, n_classes = dataset.create_dataset()

    if Config['debug']:
        train_set = (X_train[:100], y_train[:100], transforms['train'])
        test_set = (X_test[:100], y_test[:100], transforms['test'])
        dataset_size = {'train': 100, 'test': 100}
    else:
        train_set = (X_train, y_train, transforms['train'])
        test_set = (X_test, y_test, transforms['test'])
        dataset_size = {'train': len(y_train), 'test': len(y_test)}

    params = {'batch_size': Config['batch_size'],
              'n_classes': n_classes,
              'shuffle': True
              }

    train_generator =  DataGenerator(train_set, dataset_size, params)
    test_generator = DataGenerator(test_set, dataset_size, params)


    # # Use GPU
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    #     base_model = ResNet50(weights='imagenet', include_top=False)
    #     x = base_model.output
    #     x = GlobalAveragePooling2D()(x)
    #     x = Dense(512, activation='relu')(x)

    #     predictions = Dense(n_classes, activation = 'softmax')(x)
    #     model = Model(inputs=base_model.input, outputs=predictions)

    #     for layer in base_model.layers:
    #         layer.trainable = False

    #     # define optimizers
    #     model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # model.summary()

    # # training
    # model.fit_generator(generator=train_generator,
    #                     validation_data=test_generator,
    #                     use_multiprocessing=True,
    #                     workers=Config['num_workers'],
    #                     )


    # Use GPU
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        base_model = MobileNet(weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(n_classes, activation = 'softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in base_model.layers:
            layer.trainable = False

    # define optimizers
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    # training
    results = model.fit(train_generator,validation_data=test_generator, epochs=Config['num_epochs'])
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    acc = results.history['accuracy']
    val_acc = results.history['val_accuracy']

    epochs = np.arange(len(loss))

    plt.figure()
    plt.plot(epochs, acc, label='acc')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig('learning_acc.png', dpi=256)