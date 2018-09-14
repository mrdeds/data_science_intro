#-*- coding: utf-8 -*-
"""
Función que reemplaza datos faltantes con modelos simples
"""

import logging
import pandas as pd
import numpy as np
import sys
sys.path.append('../')
from ModelCreation.models import linreg, logreg

def nan_to_mean(df):
    """
    Todos los valores faltantes los cambia por el promedio
    Args:
        df (DataFrame): Datos
    Returns:
        data (DataFrame): Mismos datos con promedio en lugar de nan
    """
    data = df.copy()
    data = data.dropna(how='all', axis=1)
    numericas = list(data.select_dtypes(include=['int','float']).columns)
    for i in numericas:
        try:
            avg = data[i][data[i].notnull()].mean()
            data.loc[data[i].isnull(), i] = avg
        except Exception as e:
            pass
    return data

def replace_nan(data, numeric):
    """
    Rellena datos faltantes con modelos simples, utilizando mismas variables
    para todos los datos

    Args:
        data (DataFrame): Datos
        numeric (list): Lista de datos numéricos
    Returns:
        df (DataFrame): Datos con NaNs reemplazados
    """
    df = data[numeric].copy()
    df['intercept'] = 1
    df_rep = df.copy()
    # Primero cambiamos NaNs por promedio
    df_rep = nan_to_mean(df_rep)
    replaced = df.copy()
    for i in numeric:
        if len(df[df[i].isna()]) > 0:
            try:
                X = df_rep.drop(i, axis=1).values
                y = df_rep[i].values
                # Instancias que necesitan reemplazo
                X_rep = df_rep[df[i].isna()].drop(i, axis=1).values
                # Modelo simple (regresión lineal) para cada variable
                lr = linreg(X, y)
                df.loc[df[i].isna(), i] = lr.predict(X_rep)
                # Marcamos variables con instancias reemplazadas
                replaced[i+'_rep'] = 0
                replaced.loc[replaced[i].isna(), i + '_rep'] = 1
            except Exception as e:
                logging.info(i)
                logging.error(e)
    rep_columns = [i for i in replaced.columns if i.endswith('_rep')]
    for i in rep_columns:
        df[i] = replaced[i]

    return df
