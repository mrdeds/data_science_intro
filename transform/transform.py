#!/usr/bin/python3
# coding: utf-8
"""
Funciones que transforman y enriquecen los datos para entrenamiento
"""
import math
import logging
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from datetime import datetime as dt
from tqdm import tqdm
from transform.replacenans import replace_nan
sys.path.append('../')
from DataCleaning.cleaning import datatypes

logging.getLogger().setLevel(logging.DEBUG)

def check_correlation(df_pair, treshold=0.1):
    """
    Checa la correlación de un par de columnas de un DataFrame
    Args:
        - df_pair (Datframe): Dataframe con la primer columna a comparar por la segunda
    Returns:
        - result (list) : lista con el valor de la variable y su correlación
    """
    varname = df_pair.columns[0]
    response = df_pair.columns[1]
    correlation = df_pair.corr()[response][0]

    if abs(correlation) < abs(treshold):
        result = varname
    else:
        result = ''

    return result

def drop_correlation(DF, var_list, response, treshold=0.1):
    """
    De una lista de variables quita las que tengan menor correlación del DataFrame
    Args:
        DF (Dataframe): Dataframe con los datos aumentados
        var_list (list): lista de variables a verificar correlación
        response (string): nombre de la variable dependiente a predecir
    Response:
        DF (Datframe): Dataframe con variables que tienen mayor relación con
                        la variable a predecir
        dropped (list): lista de variables desechadas de la lista por baja correlación
    """
    logging.info("*** Checando correlación de variables nuevas contra {} \
                 (puede tardar algunos minutos)***".format(response))
    df = DF.copy()
    # Correlación con cada variable de la lista contra la variable dependiente
    pair_columns = [[element, response] for element in var_list]

    df_pairs = [df[pair] for pair in pair_columns]

    pool = mp.Pool(processes=4)

    correlations_drop = pool.map(check_correlation, df_pairs)
    pool.close()
    pool.join()

    correlations_drop = [varname for varname in correlations_drop if varname not in ('')]
    #eliminamos la columna que no cumple con correlación mínima

    df = df.drop(correlations_drop, 1)
    return df, correlations_drop


def augment_numeric(DF, response):
    """
    Crea una lista con transformación de variables numéricas del DataFrame
    Args:
        DF (Dataframe): Dataframe con los datos numéricos a aumentar
        response (list): nombre de las variables a predecir
    Response:
        DF (DataFrame): Dataframe con los datos numéricos aumentados
        new_vars (list): lista con nombre de variables aumentadas candidatas
    """
    df = DF.copy()

    numericas = list(df.select_dtypes(include=['int','float']).columns)

    df = replace_nan(df, numericas) #cambiar a función que predice valores

    numericas = list(filter(lambda x: x not in response, numericas))
    new_vars = []

    for i in numericas:
        if isinstance(i, int) or isinstance(i, float):
            try:
                new_vars.append(i)

                varname = i + '^' + str(2)
                # Variable al cuadrado
                df[varname] = df[i]**2
                new_vars.append(varname)

                varname = i + '^' + str(3)
                # Variable al cubo
                df[varname] = df[i]**3
                new_vars.append(varname)

                varname = 'sqrt(' + i + ')'
                # Raíz cuadrada de la variable
                logging.info('tratamos de encontrar el cuadrado de {}'.format(varname))

                df[varname] = np.sqrt(df[i])
                new_vars.append(varname)

                varname = '1/' + i
                # Inverso de la variable
                df[varname] = 1 / df[i]
                new_vars.append(varname)

                varname = 'log(' + i + ')'
                # Logaritmo de la variable
                df[varname] = df[i].apply(np.log)
                new_vars.append(varname)

                #si tenemos logaritmos que dan infinitos
                df = df.replace(-np.inf, -1000)
                df = df.replace(np.inf, 1000)
            except Exception as e:
                logger.error(e)

    return df, new_vars


def augment_date(DF, response):
    """
    Crea una lista con transformación de variables de fechas del DataFrame
    Args:
        DF (Datframe): Datframe con los datos de fecha a aumentar
        response (list): nombre de las variables dependiente a predecir
    Response:
        DF (Dataframe): Dataframe con los datos de fechas aumentados
        new_vars(list): lista con las variables aumentadas candidatas
    """
    df = DF.copy()
    fechas = list(df.select_dtypes(include=['datetime']).columns)
    fechas = list(filter(lambda x: x not in response, fechas))
    original_cols = list(df.columns)
    newvars = []
    unuseful = []
    acum_fechas = []
    new_vars = []
    logging.info('Variables de tipo fecha encontradas: {}'.format(fechas))
    for i in fechas:
        varname = 'hora_' + i
        # Hora de la fecha
        df[varname] = df[i].dt.hour
        new_vars.append(varname)

        varname = 'dia_' + i
        # Día de la fecha
        df[varname] = df[i].dt.day
        new_vars.append(varname)

        varname = 'mes_' + i
        # Mes de la fecha
        df[varname] = df[i].dt.month
        new_vars.append(varname)

        varname = 'dia_semana_' + i
        # Día de la semana
        for ejemplo in df[i]: # TODO: Dias de la semana tomados en cuenta
            lista_de_dias_semana = int(ejemplo.weekday())
        df[varname] = lista_de_dias_semana
        new_vars.append(varname)
        acum_fechas.append(i)
        for j in [x for x in fechas if x not in acum_fechas]:
            logging.info('Resta de fechas {} y {}'.format(i,j))
            # Por cada fecha vamos a tomar la diferencia entre esa fecha y las demás
            # Diferencia de fechas (en días)
            varname = i + '-' + j
            df.loc[(df[i].notnull()) & (df[j].notnull()), varname] = (df[i] - df[j]).dt.days
            new_vars.append(varname)

    df = pd.get_dummies(df, columns=new_vars)
    logging.info('Vamos a eliminar estas fechas después de enriquecerlas: {}'.format(fechas))
    df.drop(fechas, 1)
    new_vars = [i for i in df.columns if i not in original_cols]
    return df, new_vars


def augment_categories(DF, response):
    """
    Se hacen transformaciones con operaciones lógicas entre variables categóricas
    dentro de un Dataframe
    Args:
        DF (DataFrame): Dataframe de donde se quieren aumentar las categorías
        response(list): lista con nombres de las variables dependientes a predecir
    Response:
        df (Dataframe): DataFrame con las variables categóricas aumentadas
        new_vars(list): lista con las variables categóricas candidatas aumentadas
    """
    df = DF.copy()
    dummy_vars = []
    for i in df.columns:
        if set(df[i].unique()) == set([0, 1]):
            dummy_vars.append(i)

    dummy_vars = list(filter(lambda x: x not in response, dummy_vars))
    new_vars = []

    logging.info("*** Aumentando {} categorías ***".format(len(dummy_vars)))
    pbar = tqdm(total=len(dummy_vars))
    for i in dummy_vars:
        for j in [x for x in dummy_vars if x not in new_vars]:
            # Multiplicación de conectores lógicos (AND, OR, NAND, NOR, XOR & XNOR)
            varname = i + '*' + j
            df[varname] = df[i].astype(int) & df[j].astype(int)
            new_vars.append(varname)

            varname = i + '+' + j
            df[varname] = df[i].astype(int) | df[j].astype(int)
            new_vars.append(varname)

            varname = 'nand(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) & df[j].astype(int))

            varname = 'nor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) | df[j].astype(int))

            varname = 'xor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & ~(df[j].astype(int))) | (~(df[i].astype(int)) & df[j].astype(int))

            varname = 'xnor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & df[j].astype(int)) | (~(df[i].astype(int)) & ~(df[j].astype(int)))
        pbar.update(1)
    pbar.close()
    #Algunas operaciones se quedan en valores lógicos. Hay que pasarlas a binarias

    logging.info("*** Verificando datos de {} nuevas variables ***".format(len(new_vars)))
    pbar = tqdm(total=len(new_vars))
    for var in new_vars: # TODO: hacer este chequeo cada que se añade una variable
        df.loc[df[var] == True, var] = 1
        df.loc[df[var] == False, var] = 0
        pbar.update(1)
    pbar.close()

    return df, new_vars


def augment_data(DF, response, treshold=0.1, categories=False):
    """
    Prueba ciertas transformaciones numéricas, de fecha y categóricas.
    Verifica si la correlación es buena, a partir de un threshold,
    para agregarlas al dataframe resultante
    Args:
        DF (DataFrame): DataFrame de tus datos
        response (str): Variable dependiente (la debe contener tu base)
        treshold (float): Correlación mínima que se espera de una variable que
                        quieres que entre al modelo
    Returns:
        df (DataFrame): DataFrame con transformaciones útiles
        new_vals (list): Lista de variables transformadas nuevas en el dataframe
    """

    logging.info('***Haciendo Agregación de datos***')
    df = DF.copy()
    catego = []
    df, numeric = augment_numeric(df, response)
    df, fecha = augment_date(df, response)
    #suele tardarse mucho la transformación de categorías
    numericas, categoricas, fechas = datatypes(df)
    df = pd.get_dummies(df, columns=categoricas)
    if categories:
        df, catego = augment_categories(df, response)
    aug_vars = numeric + fecha + catego
    df, dropped = drop_correlation(df, aug_vars, response, treshold)
    df.drop(df.select_dtypes(['datetime64[ns]']), inplace=True, axis=1)

    return df, dropped
