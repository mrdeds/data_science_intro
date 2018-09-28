#-*- coding: utf-8 -*-
"""
Limpieza de datos
"""

import pandas as pd
import numpy as np

def clean_numeric(df, numericas, mp=0.4):
    """
    Limpieza de datos numéricos
    Args:
        DF (DataFrame): DataFrame con todos los datos
        numericas (list): Variables numéricas
        mp (float): Máximo porcentaje permitido de datos faltantes
    Returns:
        numericas_dropped (list): Lista de variables numéricas a eliminar
    """
    # limpiamos variables numéricas y elegimos las que tienen datos completos
    DF = df.copy()
    num = []
    numericas_dropped = []
    for i in numericas:
        # si son constantes
        if len(DF[i].unique()) == 1:
            numericas_dropped.append(i)
        else:
            # si faltan por lo menos el x% de los datos (default 40%)
            if len(DF[DF[i].isna()]) > mp * len(DF):
                numericas_dropped.append(i)
            else:
                num.append(i) # dejamos las demás

    return numericas_dropped

def clean_categoric(df, categoricas, mp=0.4, max_unique=1000):
    """
    Limpieza de datos categóricos
    Args:
        DF (DataFrame): DataFrame con todos los datos
        categoricas (str): Variable dependiente
        mp (float): Máximo porcentaje permitido de datos faltantes
        max_unique (int): Máximo número de valores únicos en una variable
                          categórica
    Returns:
        categoricas_dropped (list): Lista de variables categóricas a eliminar
    """
    DF = df.copy()
    cat = []
    categoricas_dropped = []
    # limpiamos variables categóricas y elegimos las que tienen datos completos
    for i in categoricas:
        # convertimios a string para no tener problema con los datos
        DF[i] = DF[i].astype(str)
        # Más de 1000 categorías (ids) o constante
        if len(DF[i].unique()) > max_unique or len(DF[i].unique()) == 1:
            categoricas_dropped.append(i)
        else:
            # si faltan por lo menos el x% de los datos (default 40%)
            if len(DF[(DF[i].isna()) | (DF[i] == 'nan')
                                     | (DF[i] == '')]) > mp * len(DF):
                categoricas_dropped.append(i)
            else:
                cat.append(i)

    return categoricas_dropped

def clean_dates(df, fechas, mp=0.4):
    """
    Limpieza de datos temporales (fechas)
    Args:
        DF (DataFrame): DataFrame con todos los datos
        fechas (list): Variables temporales
        mp (float): Máximo porcentaje permitido de datos faltantes
    Returns:
        fechas_dropped (list): Lista de fechas a eliminar
    """
    DF = df.copy()
    fechas_dropped = []
    # Limpiamos fechas y elegimos las que tienen datos completos
    for i in fechas:
        if len(DF[DF[i].isna()]) >= mp * len(DF):
            fechas_dropped.append(i)

    return fechas_dropped

def datatypes(df):
    """
    Genera lista de variables asignando el tipo de dato que les corresponde
    Args:
        df (DataFrame)
    Returns:
        numericas (list): Lista de variables numéricas
        categoricas (list): Lista de variables categóricas
        fechas (list): Lista de variables temporales
    """
    # variables numericas
    numericas = list(df.select_dtypes(include=['int','float']).columns)
    # variables categoricas
    categoricas = list(df.select_dtypes(include=['category', 'object']).columns)
    # variables temporales
    fechas = list(df.select_dtypes(include=['datetime']).columns)

    return numericas, categoricas, fechas

def clean_data(df, max_unique=1000, response=None, mp=0.4, safezone=None,
               printdrops=False):
    """
    Limpia datos dependiendo de cada tipo
    Args:
        DF (DataFrame): DataFrame con todos los datos
        response (str): Variable objetivo
        mp (float): Máximo porcentaje permitido de datos faltantes
        safezone (list): Variables que no querramos eliminar
        printdrops (boolean): Si queremos ver las variables que eliminamos
    Returns:
        DF (DataFrame): DataFrame con todos los datos limpios y útiles
    """
    DF = df.copy()
    DF0 = df.copy()
    if response != None:
        DF0 = DF0.drop(response,1)
    numericas, categoricas, fechas = datatypes(DF0)
    if safezone != None:
        numericas = [i for i in numericas if i not in safezone]
        categoricas = [i for i in categoricas if i not in safezone]
        fechas = [i for i in fechas if i not in safezone]
    numericas_dropped = clean_numeric(DF, numericas, mp=mp)
    categoricas_dropped = clean_categoric(DF, categoricas, mp=mp,
                                          max_unique=max_unique)
    fechas_dropped = clean_dates(DF, fechas, mp=mp)
    DF = DF.drop(numericas_dropped,1) # drop de numericas que no sirven
    DF = DF.drop(categoricas_dropped,1) # drop de categoricas que no sirven
    DF = DF.drop(fechas_dropped,1) # drop de fechas que no sirven

    if printdrops != False:
        print('Numéricas que eliminamos:')
        print(numericas_dropped)
        print('Categóricas que eliminamos:')
        print(categoricas_dropped)
        print('Fechas que eliminamos:')
        print(fechas_dropped)

    return DF
