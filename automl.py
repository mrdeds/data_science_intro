#!/usr/bin/python3
# coding: utf-8
"""
Ejemplo de como se puede utilizar el optimizador de búsqueda de arquitecturas
"""
import logging
import sys
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from ArchitectureSearch.optimizador import Optimizador
from Extract.extract import db_extraction
from transform.transform import augment_data
from DataCleaning.cleaning import clean_data
sys.path.append('../')

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)


def separa_datos(d_frame, response):
    """
    Separa los datos de entrenamiento y validación que se van a utilizar
    Args:
        DF(DataFrame): Dataframe con todos los datos
        response(str): nombre de la columna donde se encuentran los valores de
            la variables a predecir
        test_size(float): porcentaje del tamaño a tomar del set de prueba
    Returns:
        X_train(np.array): valores de variables de entrenamiento
        X_test(np.array): valores variables de prueba
        y_train(np.array): valores de variable a predecir de entrenamiento
        y_test(np.array): valores de variable a predecir de prueba
    """
    data_frame = d_frame.copy()
    x_elements = np.array(data_frame.drop(response, axis=1))
    y_elements = list(data_frame[response])

    x_train, x_test, y_train, y_test = train_test_split(x_elements, y_elements)

    return x_train, x_test, y_train, y_test


def entrena_redes(poblacion, data_frame, response):
    """Train each network.

    Args:
        poblacion (list): Lista con la poblacion de redes
        data_frame (DataFrame): Dataframe con datos de entrenamiento
        response (str): String que indica la variable objetivo
    """
    # para ver el avance del entrenamiento de las poblaciones
    pbar = tqdm(total=len(poblacion))
    datos_listos = separa_datos(data_frame, response)

    for red in poblacion:
        red.entrena(datos_listos)
        pbar.update(1)
    pbar.close()


def get_average_accuracy(redes):
    """Obtiene el promedio de accuracy de una lista de redes

    Args:
        redes (list): lista de redes

    Returns:
        res (float): Promedio de accuracy de una lista de redes

    """
    total_accuracy = 0
    for red in redes:
        total_accuracy += red.accuracy

    res = total_accuracy / len(redes)

    return res


def genera_red(generaciones, tam_poblacion, nn_param_candidatos, data_frame, response):
    """
    Genera red nueva con algoritmo genético.

    Args:
        generaciones (int): numero de veces que se va a evolucionar una población
        tam_poblacion (int): número de redes en una población
        nn_param_candidatos (dict): parámetros que puede incluir la red.
        data_frame (DataFrame): Dataframe con
    """
    optimizador = Optimizador(nn_param_candidatos)
    redes = optimizador.crea_poblacion(tam_poblacion)

    # Evoluciona la generación
    for i in range(generaciones):
        print_count = i+1
        logging.info("***Haciendo generación %d de %d***", (print_count, generaciones))

        # Entrena y obtiene accuracy de cada red.
        logging.info("*** Parámetros creados a partir de la selección de variables ***")
        entrena_redes(redes, data_frame, response)

        # obtiene el accuracy promedio de esta generación.
        prom_accuracy = get_average_accuracy(redes)

        # imprimimos el promedio de accuracy por generación
        print_prom = prom_accuracy * 100
        logging.info("Promedio de generación: %.2f%%", (print_prom))
        logging.info('-' * 80)

        # Seguimos si aún no hemos acabado de optimizar.
        if i != generaciones - 1:
            redes = optimizador.evoluciona(redes)

    # Se ordena nuestra última iteración de redes por accuracy
    redes = sorted(redes, key=lambda x: x.accuracy, reverse=True)

    # Se impirmen el top 5 de las redes finales.
    imprime_redes(redes[:5])


def imprime_redes(redes):
    """Imprime una lista de redes.

    Args:
        redes (list): la población de redes

    """
    logging.info('-' * 80)
    for red in redes:
        red.imprime_red()


def main():
    """
    Ejemplo de evolucion de una red.
    """
    # Obtenemos datos para entrenar
    # Hacemos una conexión a la base con sus credenciales
    query = '''
        WITH sl as (select clientecerrado__c, createddate, id, email, phone,
               recomendado__c, mediocontactoagrupado__c,
               mediocontactodepurado__c
               from salesforce_lead
               where leadsource = 'RTD' and
               createddate >= DATEADD(month,-3, getdate())
               order by createddate desc),
        m as (select email_address, location_dstoff,
                location_gmtoff, location_latitude,
                location_longitude, member_rating,
                stats_avg_click_rate, stats_avg_open_rate,
                email_client, location_country_code,
                status from mailchimp_members)

        select * from sl left join m on sl.email = m.email_address

            limit 5000'''

    data_frame = db_extraction(query)

    generaciones = 2  # Número de veces a evolucionar una población
    tam_poblacion = 20  # número de redes en una población.
    response = 'clientecerrado__c'

    # limpiamos los datos antes de entrenar
    data_frame = clean_data(data_frame, max_unique=1000, response=response)
    data_frame, var_nuevas = augment_data(data_frame, response, categories=True)

    nn_param_candidatos = {
        'num_neurons': [64, 128, 256, 512, 768, 1024],
        'num_capas': [1, 2, 3, 4],
        'activacion': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizador': ['rmsprop', 'adam', 'sgd', 'adagrad',
                        'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolucionando %d generaciones, con población de  %d ***",
                 (generaciones, tam_poblacion))

    genera_red(generaciones, tam_poblacion, nn_param_candidatos, data_frame, response)


if __name__ == '__main__':
    main()
