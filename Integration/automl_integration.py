#-*- coding: utf-8 -*-
"""
Integración de Pipeline de AutoML con random random search
"""

import pandas as pd
import logging
import sys
sys.path.append('../')
from Extract.extract import db_extraction
from DataCleaning.cleaning import clean_data
from transform.transform import augment_data
from VariableSelection.selection import importance_corr
from ModelCreation.models import train_test, simple_model
from Evaluation.metrics import show_metrics
from HyperparameterTuning.random_search import best_nn

def automl_random_search(query,
                         response,
                         corr=0.1,
                         train_size=0.75,
                         scaling=None,
                         tpot=False,
                         it=10,
                         epcs=10,
                         metric='auc',
                         graphs=True,
                         threshold=0.5,
                         quantiles=10):
    """
    Crea un modelo simple, tpot y red neuronal con random
    search dada una base de datos y una variable objetivo

    Args:
        query (str): Query de SQL
        reponse (str): Variable objetivo
        corr (float): Correlación mínima con variable objetivo
        train_size (float): Porcentaje de datos con los que entrenar el modelo
        scaling (str): ['standard', 'minmax', 'maxabs', 'robust', 'quantile']
        tpot (bool): Si queremos que se entrene TPOT
        it (int): Número de iteraciones o redes a entrenar y comparar
        epcs (int): Número de epochs
        metric (str): Métrica de precisión para seleccionar mejor modelo
                      ['acc', 'prec', 'rec', 'f1', 'mcc', 'auc']
        graphs (bool): Si se requieren tablas y gráficas de las métricas de
                       precisión
        threshold (float): Score de corte
        quantiles (int): Número de cuantiles para tabla de precisión
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
                                        threshold=0.075,
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
                                                  time_series=False,
                                                  scaling=scaling)
    logging.info('Entrenamiento de Modelos')
    simple_mod, tpot_mod = simple_model(X_train, y_train, tpot=tpot)
    best_neuralnet = best_nn(X_train,
                             y_train,
                             X_test,
                             y_test,
                             it=it,
                             epcs=epcs,
                             metric=metric,
                             threshold=threshold)

    logging.info('Evaluación de modelos:\n')
    logging.info('Modelo Simple:')
    simple_predictions = simple_mod.predict(X_test)
    if graphs:
        show_metrics(y_test, simple_predictions, threshold, disp=graphs, n=quantiles)
    logging.info('Modelo de Red Neuronal:')
    nn_predictions = best_neuralnet.predict(X_test)
    if graphs:
        show_metrics(y_test, nn_predictions, threshold, disp=graphs, n=quantiles)
    if tpot:
        logging.info('Modelo de TPOT:')
        tpot_predictions = tpot_mod.predict(X_test)
        if graphs:
            show_metrics(y_test, tpot_predictions, threshold, disp=graphs, n=quantiles)


    return simple_mod, best_neuralnet, tpot_mod, best_vars
