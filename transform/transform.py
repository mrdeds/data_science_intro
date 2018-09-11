#!/usr/bin/python3
# coding: utf-8
"""
Clase que hace deploy a ML Engine.
"""
import math
from datetime import datetime as dt
import logging

logging.getLogger().setLevel(logging.DEBUG)

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


def augment_data(DF, response, correl=0.1, datetrans=False, convertdummies=False, dummy_transform=False):
    """
    Prueba ciertas transformaciones numéricas y verifica si la correlación es buena
    para agregarlas al dataframe

    Args:
        DF (DataFrame): DataFrame de tus datos
        response (str): Variable dependiente (la debe contener tu base)
        correl (float): Correlación mínima que se espera de una variable que
                        quieres que entre al modelo
        convertdummies (boolean): Si queremos convertir categóricas a variables
                                  binarias
        dummy_transform (boolean): Si queremos encontrar transformaciones en las
                                   variables binarias
    Returns:
        df (DataFrame): DataFrame con transformaciones útiles

    """
    df = DF.copy()

    numericas = list(df.select_dtypes(include=['int','float']).columns)
    fechas = list(df.select_dtypes(include=['datetime']).columns)

    df = nan_to_avg(df)

    numericas = [x for x in numericas if x != response]
    fechas = [x for x in fechas if x != response]

    newvars = []
    unuseful = []

    if convertdummies != False:
        cat = list(df.select_dtypes(include=['category', 'object']).columns)
        df = pd.get_dummies(df, columns=cat)

    # En caso de querer transformaciones en nuestras variables binarias hay
    # un gran tiempo de espera
    if dummy_transform != False:
        dummy_vars = []
        for i in df.columns:
            if set(df[i].unique()) == set([0, 1]):
                dummy_vars.append(i)

        dummy_vars = [x for x in dummy_vars if x != response]
        fechas = [i for i in fechas if i not in dummy_vars]
        numericas = [i for i in numericas if i not in dummy_vars]

        acum = []
        for i in dummy_vars:
            acum.append(i)
            for j in [x for x in dummy_vars if x not in acum]:
                # Multiplicación de conectores lógicos (AND)
                varname = i + '*' + j
                df[varname] = df[i] * df[j]
                correlagg = df[[varname, response]].corr()[response][0]
                # Se agrega si supera la correlación mínima
                if abs(correlagg) > abs(correl):
                    newvars.append(varname)
                else:
                    unuseful.append(varname)

    if datetrans != False:
        acum_fechas = []
        for i in fechas:
            varname = 'hora_' + i
            # Hora de la fecha
            df[varname] = df[i].dt.hour
            correlhora = df[[varname, response]].corr()[response][0]

            # Se agrega si supera la correlación mínima
            if abs(correlhora) > abs(correl):

                newvars.append(varname)
            else:
                unuseful.append(varname)

            varname = 'dia_' + i
            # Día de la fecha
            df[varname] = df[i].dt.day
            correldia = df[[varname, response]].corr()[response][0]

            # Se agrega si supera la correlación mínima
            if abs(correldia) > abs(correl):

                newvars.append(varname)
            else:
                unuseful.append(varname)

            varname = 'mes_' + i
            # Mes de la fecha
            df[varname] = df[i].dt.month
            correlmes = df[[varname, response]].corr()[response][0]

            # Se agrega si supera la correlación mínima
            if abs(correlmes) > abs(correl):

                newvars.append(varname)
            else:
                unuseful.append(varname)

            acum_fechas.append(i)
            for j in [x for x in fechas if x not in acum_fechas]:
                # Diferencia de fechas (en días)
                varname = i + '-' + j
                df.loc[(df[i].notnull()) & (df[j].notnull()), varname] = (df[i] - df[j]).dt.days
                correldif = df[[varname, response]].corr()[response][0]
                # Se agrega si supera la correlación mínima
                if abs(correldif) > abs(correl):
                    newvars.append(varname)
                else:
                    unuseful.append(varname)

    for i in numericas:
        # Correlación sin transformación
        correl1 = df[[i, response]].corr()[response][0]
        varname = i + '^' + str(2)
        # Variable al cuadrado
        df[varname] = df[i]**2

        # Correlación con cada variable al cuadrado
        correl2 = df[[varname, response]].corr()[response][0]
        # Se agrega si supera la correlación mínima y la correlación sin transformación
        if abs(correl2) > abs(correl) and abs(correl2) > abs(correl1):

            newvars.append(varname)
        else:
            unuseful.append(varname)

        varname = i + '^' + str(3)
        # Variable al cubo
        df[varname] = df[i]**3

        # Correlación con cada variable al cubo
        correl3 = df[[varname, response]].corr()[response][0]

        # Se agrega si supera la correlación mínima y la correlación sin transformación
        if abs(correl3) > abs(correl) and abs(correl3) > abs(correl1):

            newvars.append(varname)
        else:
            unuseful.append(varname)

        varname = 'sqrt(' + i + ')'
        # Raíz cuadrada de la variable
        df[varname] = np.sqrt(df[i])

        # Correlación con la raíz cuadrada de cada variable
        correlsqrt = df[[varname, response]].corr()[response][0]

        # Se agrega si supera la correlación mínima y la correlación sin transformación
        if abs(correlsqrt) > abs(correl) and abs(correlsqrt) > abs(correl1):

            newvars.append(varname)
        else:
            unuseful.append(varname)

        varname = '1/' + i
        # Inverso de la variable
        df[varname] = 1 / df[i]

        # Correlación con el inverso de cada variable
        correlinv = df[[varname, response]].corr()[response][0]

        # Se agrega si supera la correlación mínima y la correlación sin transformación
        if abs(correlinv) > abs(correl) and abs(correlinv) > abs(correl1):

            newvars.append(varname)
        else:
            unuseful.append(varname)

        varname = 'log(' + i + ')'
        # Logaritmo de la variable
        df[varname] = df[i].apply(np.log)

        # Correlación con el logaritmo de cada variable
        correllog = df[[varname, response]].corr()[response][0]

        # Se agrega si supera la correlación mínima y la correlación sin transformación
        if abs(correllog) > abs(correl) and abs(correllog) > abs(correl1):

            newvars.append(varname)
        else:
            unuseful.append(varname)

    df = df.drop(unuseful, 1)
    print('Agregamos las siguientes transformaciones:')
    display(newvars)

    df = change_nan_to_avg(df)

    df = df.replace(-np.inf, -1000)
    df = df.replace(np.inf, 1000)

    num = list(df.select_dtypes(include=['int', 'float']).columns)

    return df[num]
