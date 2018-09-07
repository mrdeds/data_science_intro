#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones que toma pytest para evaluar la librería
"""
import logging
import pytest
import psycopg2
import Extract
import pg_temp

logging.getLogger().setLevel(logging.DEBUG)

def test_db_connection():
    """
    Evalúa la función de conexión a la base de datos
    """
    temp_db = pg_temp.TempDB(databases=['testdb'], verbosity=1)
    conn_creds = {'host':temp_db.pg_socket_dir, 'port': 5432, 'user':'','password':'','database':'testdb'}
    connection = Extract.db_connection(conn_creds)
    temp_db.cleanup()
    assert str(type(connection)) == "<class 'psycopg2.extensions.connection'>"

def test_download_data():
    """
    Evalúa la función extracción de datos
    """
    temp_db = pg_temp.TempDB(databases=['testdb'])
    conn_creds = {'host':temp_db.pg_socket_dir, 'port': 5432, 'user':'','password':'','database':'testdb'}
    select_query = "SELECT * FROM pg_attribute"

    conn = Extract.db_connection(conn_creds)
    df = Extract.download_data(conn, select_query)
    temp_db.cleanup()

    assert df.size>0
