#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones que toma pytest para evaluar la librería
"""
import os
import logging
import pytest
import sys
sys.path.append('../')
from transform.transform import drop_correlation, augment_numeric, augment_date
import pandas as pd
import random
import numpy as np
import psycopg2
import Extract
import pg_temp
from datetime import datetime
import random

logging.getLogger().setLevel(logging.DEBUG)

temp_db = pg_temp.TempDB(databases=['testdb'], verbosity=1)
os.environ['HOST'] = temp_db.pg_socket_dir
os.environ['PORT'] = '5432'
os.environ['USER'] = ''
os.environ['PASSWORD'] = ''
os.environ['DB'] = 'testdb'

def my_equal(df1, df2):
    """
    Función que permite comparar dos dataframes aunque no esten en el mismo orden
    sus columnas

    Args:
        df1 (Dataframe): Dataframe a ser comparado
        df2: (Dataframe): Segundo dataframe a ser comparado con df1

    Returns:
        Regresa un booleano [True, False]  si los dataframes df1 y df2 son iguales
    """
    from pandas.util.testing import assert_frame_equal
    try:
        assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True)
        return True
    except (AssertionError, ValueError, TypeError):
        return False

def test_db_connection():
    """
    Evalúa la función de conexión a la base de datos
    """
    connection = Extract.db_connection()
    connection.close()
    assert str(type(connection)) == "<class 'psycopg2.extensions.connection'>"

def test_download_data():
    """
    Evalúa la función extracción de datos
    """
    temp_db = pg_temp.TempDB(databases=['testdb'])
    select_query = "SELECT * FROM pg_attribute"

    conn = Extract.db_connection()
    df = Extract.download_data(conn, select_query)
    temp_db.cleanup()

    assert df.size>0

def test_drop_correlation():
    """
    Evalúa que de un dataframe obtenga correctamente la correlación de variables
    contra alguna variable objetivo dada
    """

    data = {'area' : pd.Series([100.0, 125.0, 150.0, 130.0, 145.0, 10.0, 1000.0, 20.0]),
            'zona_sur': pd.Series([1, 0, 0, 1, 0, 0, 1]),
            'zona_norte': pd.Series([0, 1, 1, 0, 1, 1, 0]),
            'precio': pd.Series([10.0, 12.50, 15.0, 13.0, 14.5, 1.0, 100.0, 2])}
    DF = pd.DataFrame(data)
    var_list = ['area', 'zona_sur', 'zona_norte']
    response = 'precio'
    df, dropped = drop_correlation(DF, var_list, response, threshold=0.01)

    assert len(df.columns) == 2 and len(dropped) == 2

def test_augment_numeric():
    """
    Evalúa que dado un dataframe se le aumenten las variables numéricas
    checa que se añadan las variables correctas al dataframe final.
    """
    data = {'area' : pd.Series([100.0, 125.0, 150.0, 130.0, 145.0, 10.0, 1000.0, 20.0]),
            'zona': pd.Series(['sur', 'norte', 'norte', 'sur', 'norte', 'sur', 'norte', 'sur']),
            'precio': pd.Series([10.0, 12.50, 15.0, 13.0, 14.5, 1.0, 100.0, 2.0])}
    DF = pd.DataFrame(data)
    response = 'precio'
    df, new_vars = augment_numeric(DF, response)
    print(new_vars)
    assert len(df.columns) == 8 and new_vars == ['area^2', 'area^3', 'sqrt(area)',
                                         '1/area', 'log(area)']

def test_augment_date():
    """
    Evalúa que dado un dataframe se le aumenten las variables de fecha
    """
    dates = []
    for i in range(8): #hacemos primero 8 fechas al azar. Año, mes, día, hora
        dates.append(datetime(random.randint(2000, 2018), random.randint(1, 12),
                             random.randint(1, 28), random.randint(0, 23)))
    data = {'fecha_compra' : pd.Series(dates),
            'area' : pd.Series([100.0, 125.0, 150.0, 130.0, 145.0, 10.0, 1000.0, 20.0]),
            'zona' : pd.Series(['sur', 'norte', 'norte', 'sur', 'norte', 'sur', 'norte', 'sur']),
            'precio' : pd.Series([10.0, 12.50, 15.0, 13.0, 14.5, 1.0, 100.0, 2.0])}
    DF = pd.DataFrame(data)
    response = 'precio'

    df, new_vars = augment_date(DF, response)

    meses = len(set([date.month for date in dates]))
    dias = len(set([date.day for date in dates]))
    horas = len(set([date.hour for date in dates]))
    dias_semana = len(set([date.weekday() for date in dates]))
    #las columnas que añadimos más las originales
    total = meses + dias + horas + dias_semana + 4

    check_binary = [column for column in df.columns if set(df[column].unique()) == set([0, 1])]

    assert len(df.columns) == total and new_vars == check_binary
