#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones de extracción de datos
"""
import logging
import psycopg2
import os
import pandas as pd
from Extract.memory import save_memory

def db_connection():
    """
    Método que hace la conexión a la base
    Returns:
        conn: conexión
    """

    HOST = os.environ['HOST']
    PORT = os.environ['PORT']
    USER = os.environ['USER']
    PASSWORD = os.environ['PASSWORD']
    DATABASE = os.environ['DB']

    conn = psycopg2.connect(
        host=HOST,
        port=PORT,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    return conn

def download_data(conn, query):
    """
    Descarga datos de la base de datos según la consulta insertada
    Args:
        conn (connection): objeto que contiene la sesión de una
                           conexión a la base de datos
        query (str): String donde se define el query a ejecutarse
    Returns:
        df (DataFrame): Tabla con los datos que elegimos
    """
    try:
        df = pd.read_sql(query, conn)
        conn.commit()
    finally:
        conn.close()

    return df

def db_extraction(query):
    """
    Descarga base en un DataFrame
    Args:
        query (str): String donde se define el query a ejecutarse
    Returns:
        df (DataFrame): Tabla con los datos que elegimos
    """
    conn = db_connection()
    df = download_data(conn, query)
    df = save_memory(df)

    return df
