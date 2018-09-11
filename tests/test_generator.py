#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones que toma pytest para evaluar la librería
"""
import logging
import pytest
from transform import divide_cats, nan_to_avg
import pandas as pd
import random
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

def my_equal(df1, df2):
    """
    Función que permite comparar dos dataframes aunque no esten en el mismo orden
    sus columnas
    """
    from pandas.util.testing import assert_frame_equal
    try:
        assert_frame_equal(df1.sort_index(axis=1), df2.sort_index(axis=1), check_names=True)
        return True
    except (AssertionError, ValueError, TypeError):
        return False

def test_divide_cats():
    """
    Evalúa la función divide_cats
    """
    data = {'color' : pd.Series(['rojo', 'verde', 'azul']),
            'sabor' : pd.Series(['frambuesa', 'limón', 'chicle']),
            'precio': pd.Series([10.0, 12.50, 15.0])}
    DF = pd.DataFrame(data)
    categorias = ['color', 'sabor']
    res = divide_cats(DF, categorias)
    data_assert = data
    data_assert.update({'color_azul' : pd.Series([0,0,1]),
                        'color_verde' : pd.Series([0,1,0]),
                        'color_rojo' : pd.Series([1,0,0]),
                        'sabor_limón' : pd.Series([0,1,0]),
                        'sabor_chicle' : pd.Series([0,0,1]),
                        'sabor_frambuesa': pd.Series([1,0,0])})
    DF_assert = pd.DataFrame(data_assert)

    assert my_equal(df1, df2) == True

def test_divide_cats():
    """
    Evalúa la función nan_to_avg
    """
    df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                       [3, 4, np.nan, 1],
                       [np.nan, np.nan, np.nan, 5],
                       [np.nan, 3, np.nan, 4]],
                       columns=list('ABCD'))

    df = nan_to_avg(df)
    print(df)
    assert  df.isnull().values.any() == False
