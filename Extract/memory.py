#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Funciones para mejora el uso de memoria
"""
import pandas as pd

def save_memory(df):
    """
    Si hay muchos datos tipo 'object' los cambiamos por 'category' para liberar
    espacio
    Args:
        df (DataFrame): Datos
    Returns:
        DF (DataFrame): Datos
    """
    DF = df.copy()
    try:
        df_obj = DF.select_dtypes(include=['object']).copy()
        converted_obj = pd.DataFrame()
        for col in df_obj.columns:
            num_unique_values = len(df_obj[col].unique())
            num_total_values = len(df_obj[col])
            if num_unique_values / num_total_values < 0.5:
                converted_obj.loc[:,col] = df_obj[col].astype('category')
            else:
                converted_obj.loc[:,col] = df_obj[col]
        DF[converted_obj.columns] = converted_obj
        del df_obj
        del converted_obj
    except:
        pass

    return DF
