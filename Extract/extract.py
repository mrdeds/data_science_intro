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

def db_connection(db):
    """
    Método que hace la conexión a la base
    Args:
        db (str): 'panoply', 'medicion', 'llamadas'
    Returns:
        conn: conexión
    """
    if db == 'panoply':
        HOST = os.environ['PANOPLY_HOST']
        PORT = os.environ['PANOPLY_PORT']
        USER = os.environ['PANOPLY_USER']
        PASSWORD = os.environ['PANOPLY_PASSWORD']
        DATABASE = os.environ['PANOPLY_DB']
    if db == 'medicion':
        HOST = os.environ['MEDICION_HOST']
        PORT = os.environ['MEDICION_PORT']
        USER = os.environ['MEDICION_USER']
        PASSWORD = os.environ['MEDICION_PASSWORD']
        DATABASE = os.environ['MEDICION_DB']

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

def db_extraction(db, query):
    """
    Descarga base en un DataFrame
    Args:
        db (str): Base de datos
        query (str): String donde se define el query a ejecutarse
    Returns:
        df (DataFrame): Tabla con los datos que elegimos
    """
    conn = db_connection(db)
    df = download_data(conn, query)
    df = save_memory(df)

    return df
