"""Ejemplo de como se puede utilizar el optimizador de búsqueda de arquitecturas"""
import logging
import json
from tqdm import tqdm
from optimizador import Optimizador
import sys
sys.path.append('../')
from Extract.extract import db_connection, download_data
from DataCleaning.cleaning import clean_data
from transform.transform import augment_data
import numpy as np
from sklearn.model_selection import train_test_split
from transform.transform import nan_to_avg
# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG,
    #filename='log.txt'
)


def separa_datos(DF, response, test_size=0.33):
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
    df = DF.copy()
    x = np.array(df.drop(response, axis=1))
    y = list(df[response])

    x_train, x_test, y_train, y_test = train_test_split(x, y)

    return x_train, x_test, y_train, y_test


def entrena_redes(poblacion, df, response):
    """Train each network.

    Args:
        poblacion (list): Lista con la poblacion de redes
        df (DataFrame): Dataframe con datos de entrenamiento
        response (str): String que indica la variable objetivo
    """
    # para ver el avance del entrenamiento de las poblaciones
    pbar = tqdm(total=len(poblacion))
    datos_listos = separa_datos(df, response, test_size=0.33)

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


def genera_red(generaciones, tam_poblacion, nn_param_candidatos, df, response):
    """
    Genera red nueva con algoritmo genético.

    Args:
        generaciones (int): numero de veces que se va a evolucionar una población
        tam_poblacion (int): número de redes en una población
        nn_param_candidatos (dict): parámetros que puede incluir la red.
        df (DataFrame): Dataframe con
    """
    optimizador = Optimizador(nn_param_candidatos)
    redes = optimizador.crea_poblacion(tam_poblacion)

    # Evoluciona la generación
    for i in range(generaciones):
        logging.info("***Haciendo generación %d de %d***" %
                     (i + 1, generaciones))

        # Entrena y obtiene accuracy de cada red.
        entrena_redes(redes, df, response)

        # obtiene el accuracy promedio de esta generación.
        prom_accuracy = get_average_accuracy(redes)

        # imprimimos el promedio de accuracy por generación
        logging.info("Promedio de generación: %.2f%%" % (prom_accuracy * 100))
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
    # Obtenemos las claves de conexión
    with open('creds.txt', encoding='utf-8') as data_file:
        creds = json.loads(data_file.read())
    # Obtenemos datos para entrenar
    # Hacemos una conexión a la base con sus credenciales
    connect = db_connection(creds)
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

    df = download_data(connect, query)

    generaciones = 2  # Número de veces a evolucionar una población
    tam_poblacion = 20  # número de redes en una población.
    response = 'clientecerrado__c'

    # limpiamos los datos antes de entrenar
    df = clean_data(df, max_unique=1000, response=response)
    df, var_nuevas = augment_data(df, response, treshold=0.1, categories=True)

    nn_param_candidatos = {
        'num_neurons': [64, 128, 256, 512, 768, 1024],
        'num_capas': [1, 2, 3, 4],
        'activacion': ['relu', 'elu', 'tanh', 'sigmoid'],
        'optimizador': ['rmsprop', 'adam', 'sgd', 'adagrad',
                      'adadelta', 'adamax', 'nadam'],
    }

    logging.info("***Evolucionando %d generaciones, con población de  %d***" %
                 (generaciones, tam_poblacion))

    genera_red(generaciones, tam_poblacion, nn_param_candidatos, df, response)


if __name__ == '__main__':
    main()
