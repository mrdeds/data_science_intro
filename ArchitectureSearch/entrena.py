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

# si después de 5 epochs no mejora se detiene el entrenamiento de ese modelo
early_stopper = EarlyStopping(patience=5)

def compila_modelo(red, outputs, inputs):
    """Compila un modelo secuencial.

    Args:
        red (dict): the parameters of the network


    Returns:
        model(Keras.model): Una red compilada.

    """
    # Obtenemos los parámetros de nuestra red.
    num_capas = red['num_capas']
    num_neurons = red['num_neurons']
    activacion = red['activacion']
    optimizador = red['optimizador']

    model = Sequential()

    # Se añade cada capa.
    for i in range(num_capas):
        # Necesitamos el número de inputs para la primer capa.
        if i == 0:
            model.add(Dense(num_neurons, activation=activacion, input_shape=inputs))
        else:
            model.add(Dense(num_neurons, activation=activacion))

        model.add(Dropout(0.2))
        # le añadimos una capa de Dropout antes de la última para mejor desempeño

    # Capa de salida.
    model.add(Dense(outputs, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=optimizador,
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
    model = compila_modelo(red, outputs, inputs)

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # usamos EarlyStopping así que no es el límite real
              verbose=0,
              validation_data=(X_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(X_test, y_test, verbose=0)
    res_accuracy = score[1] # 1 es accuracy. 0 es loss.

    return res_accuracy
