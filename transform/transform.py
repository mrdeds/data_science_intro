#!/usr/bin/python3
# coding: utf-8
"""
Clase que hace deploy a ML Engine.
"""
import math

def divide_cats(DF, categorias):
    """
    Función que divide por categorías

    Args:
        DF (DataFrame): Dataframe con todos los datos
        categorias(list): lista con categorias del Data Frame

    Returns:
        DF (DataFrame): con columnas con categorías separadas
    """
    for cat in categorias:
        list_val_cats = set([value for value in DF[cat]]) #tomamos cada valor que tenga la categoría
        for value in list_val_cats:
            DF[cat+"_"+value] = 0
            DF.loc[DF[cat] == value, cat+"_"+value] = 1

    return DF


def nan_to_avg(DF):
    """
    Todos los valores faltantes los cambia por el promedio de la columna

    Args:
        DF (DataFrame): Dataframe con todos los datos
    Returns:
        DF (DataFrame): Dataframe con promedio en lugar de nan
    """
    nums = list(DF.select_dtypes(include=['int','float']).columns)
    for col_vals in nums:
        try:
            avg = DF[col_vals][DF[col_vals].notnull()].mean()
            if math.isnan(avg): avg = 0
            print(avg)
            DF.loc[DF[col_vals].isnull(), col_vals] = avg
        except Exception as e:
            print(e)
    return DF
