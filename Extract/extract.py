#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones de extracción de datos
"""
import logging
import psycopg2
import pandas as pd

def db_connection(conn_creds):
    """
    Método que hace la conexión a la base

    Args:
        conn_creds(dict): diccionario donde vienen las credenciales de
        la conexión a la base de datos
                         host(str): host que hospeda a la base de datos
                         port(str): puerto donde está disponible la BD
                         user(str): usuario con el que se hará la conexión
                         password(str): contraseña del usuario en la BD
                         database(str): nombre de la base de datos
    Returns:
        conn: objeto que contiene la sesión de una conexión a la BD
    """
    try:
        conn = psycopg2.connect(
            host=conn_creds['host'],
            port=conn_creds['port'],
            user= conn_creds['user'],
            password=conn_creds['password'],
            database=conn_creds['database'],
        )
        logging.info("Nueva conexión a base: %s", conn_creds['database'])
    except Exception as error:
        logging.error(error)

    return conn

def download_data(conn, query):
    """
    Baja datos elegidos y convierte categóricos en binarios

    Args:
        conn (connection): objeto que contiene la sesión de una
                           conexión a la BD
        query (str): String donde se define el query a ejecutarse
    Returns:
        df (DataFrame): Tabla con los datos que elegimos
    """
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        data_sql = cursor.fetchall()
        conn.commit()
    finally:
        conn.close()

    df = pd.DataFrame(data_sql) # Ponemos datos en DataFrame

    return df
