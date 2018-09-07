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
        num_dropped (list): Lista de variables numéricas a eliminar
    """
    # limpiamos variables numéricas y elegimos las que tienen datos completos
    DF = df.copy()
    num = []
    num_dropped = []
    for i in numericas:
        # si son constantes
        if len(DF[i].unique()) == 1:
            num_dropped.append(i)
        else:
            # si faltan por lo menos el x% de los datos (default 40%)
            if len(DF[DF[i].isna()]) > mp * len(DF):
                num_dropped.append(i)
            else:
                num.append(i) # dejamos las demás

    return num_dropped

def clean_categoric(df, categoricas, mp=0.4):
    """
    Limpieza de datos categóricos
    Args:
        DF (DataFrame): DataFrame con todos los datos
        categoricas (str): Variable dependiente
        mp (float): Máximo porcentaje permitido de datos faltantes
    Returns:
        cat_dropped (list): Lista de variables categóricas a eliminar
    """
    DF = df.copy()
    cat = []
    cat_dropped = []
    # limpiamos variables categóricas y elegimos las que tienen datos completos
    for i in categoricas:
        # convertimios a string para no tener problema con los datos
        DF[i] = DF[i].astype(str)
        # Más de 1000 categorías (ids) o constante
        if len(DF[i].unique()) > 1000 or len(DF[i].unique()) == 1:
            cat_dropped.append(i)
        else:
            # si faltan por lo menos el x% de los datos (default 40%)
            if len(DF[(DF[i].isna()) | (DF[i] == 'nan')
                                     | (DF[i] == '')]) > mp * len(DF):
                cat_dropped.append(i)
            else:
                cat.append(i)

    return cat_dropped

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

def clean_data(df, response=None, mp=0.4, safezone=None, printdrops=False):
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
    numericas, categoricas, fechas = datatypes(df)
    if safezone != None:
        numericas = [i for i in numericas if i not in safezone]
        categoricas = [i for i in categoricas if i not in safezone]
        fechas = [i for i in fechas if i not in safezone]
    num_dropped = clean_numeric(DF, numericas, mp=mp)
    cat_dropped = clean_categoric(DF, categoricas, mp=mp)
    fech_dropped = clean_dates(DF, fechas, mp=mp)
    DF = DF.drop(num_dropped,1) # drop de numericas que no sirven
    DF = DF.drop(cat_dropped,1) # drop de categoricas que no sirven
    DF = DF.drop(fech_dropped,1) # drop de fechas que no sirven
    if printdrops != False:
        print('Numéricas que eliminamos:')
        display(num_dropped)
        print('Categóricas que eliminamos:')
        display(cat_dropped)
        print('Fechas que eliminamos:')
        display(fech_dropped)

    return DF
