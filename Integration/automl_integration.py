import pandas as pd
import logging
import sys
sys.path.append('../')
from Extract.extract import db_extraction
from DataCleaning.cleaning import clean_data
from transform.transform import augment_data
from VariableSelection.selection import importance_corr
from ModelCreation.models import train_test, simple_model
from HyperparameterTuning.random_search import model_precision, best_nn

def automl_random_search(query,
                         response,
                         corr=0.1,
                         train_size=0.75,
                         tpot=False,
                         it=10,
                         epcs=10,
                         metric='auc'):
    """
    Crea un modelo simple, tpot y red neuronal con random
    search dada una base de datos y una variable objetivo

    Args:
        query (str): Query de SQL
        reponse (str): Variable objetivo
        corr (float): Correlación mínima con variable objetivo
        train_size (float): Porcentaje de datos con los que entrenar el modelo
        tpot (bool): Si queremos que se entrene TPOT
        it (int): Número de iteraciones o redes a entrenar y comparar
        epcs (int): Número de epochs
        metric (str): Métrica de precisión para seleccionar mejor modelo
                      ['acc', 'prec', 'rec', 'f1', 'mcc', 'auc']
    Returns:
        simple_mod (modelo): Modelo simple
        best_neuralnet (modelo): Mejor modelo de red neuronal
        tpot_mod (modelo): Modelo creado con TPOT

    """
    logging.info('Extracción')
    dfextract = db_extraction(query)
    logging.info('Limpieza')
    dfclean = clean_data(dfextract,
                         max_unique=50,
                         response=response,
                         mp=0.4,
                         safezone=None,
                         printdrops=False)
    logging.info('Transformación')
    dftransform, newvals = augment_data(dfclean,
                                        response,
                                        treshold=0.075,
                                        categories=False)
    logging.info('Selección de variables')
    best_vars = importance_corr(dftransform,
                                response,
                                corr=corr,
                                fif=0.01,
                                vif=False)
    best_vars = list(best_vars['feature'].values)
    best_vars.append(response)
    df = dftransform[best_vars]
    X_train, X_test, y_train, y_test = train_test(df,
                                                  response,
                                                  train_size=train_size,
                                                  time_series=False)
    logging.info('Entrenamiento de Modelos')
    simple_mod, tpot_mod = simple_model(X_train, y_train, tpot=tpot)
    best_neuralnet = best_nn(X_train,
                             y_train,
                             X_test,
                             y_test,
                             it=it,
                             epcs=epcs,
                             metric=metric)

    logging.info('Evaluación de modelos:\n')
    logging.info('Modelo Simple:')
    simple_predictions = simple_mod.predict(X_test)
    model_precision(y_test, simple_predictions, 0.5, disp=True)
    logging.info('Modelo de Red Neuronal:')
    nn_predictions = best_neuralnet.predict(X_test)
    model_precision(y_test, nn_predictions, 0.5, disp=True)
    if tpot:
        logging.info('Modelo de TPOT:')
        tpot_predictions = tpot_mod.predict(X_test)
        model_precision(y_test, tpot_predictions, 0.5, disp=True)

    return simple_mod, best_neuralnet, tpot_mod
