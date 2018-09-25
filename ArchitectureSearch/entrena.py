"""
Funciones utilizadas para entrenar una Red
"""
from cleaning import clean_data
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import numpy as np

# si después de 5 epoch no mejora se detiene el entrenamiento de ese modelo
early_stopper = EarlyStopping(patience=5)

def clean_data(df):
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)



def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = network['nb_neurons']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def entrena_red(red, datos_listos):
    """Entrena el modelo, regresa su evaluacion.

    Args:
        red (dict): los parámetros de una red
        outputs(int): número de outputs que queremos que nuestra red tenga
        datos_listos(tuple): tupla con los datos que van a utilizarse:
            X_train(np.array): valores de variables de entrenamiento
            X_test(np.array): valores variables de prueba
            y_train(np.array): valores de variable a predecir de entrenamiento
            y_test(np.array): valores de variable a predecir de prueba
    """
    X_train, X_test, y_train, y_test = datos_listos
    inputs = X_train[0].size #número de inputs de la red a partir de # de datos
    outputs = 1
    model = compile_model(red, outputs, inputs)

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # usamos EarlyStopping así que no es el límite real
              verbose=0,
              validation_data=(X_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(X_test, y_test, verbose=0)
    res_accuracy = score[1] # 1 es accuracy. 0 es loss.

    return res_accuracy
