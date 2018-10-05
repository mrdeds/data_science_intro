#!/usr/bin/python3
# coding: utf-8
"""
Funciones que transforman y enriquecen los datos para entrenamiento
"""
import math
from datetime import datetime as dt
import logging
import numpy as np

logging.getLogger().setLevel(logging.DEBUG)

def divide_cats(DF, categorias, esp_cat=''):
    """
    Función que divide por categorías

    Args:
        DF (DataFrame): Dataframe con todos los datos
        categorias(list): lista con categorias del Data Frame

    Returns:
        DF (DataFrame): con columnas con categorías separada
        new_vars (list): lista con las variables añadidas
    """
    new_vars = []
    for categoria in categorias:
        try:
            list_val_cats = set([str(value) for value in DF[categoria]]) #tomamos cada valor que tenga la categoría
            logging.info("list_val_cats({}): {}".format(categoria, list_val_cats))
            for value in list_val_cats:
                nueva_col = categoria + "_" + str(value)
                logging.info("nueva_col: {} viene de categoría: {} y valor de la categoría:{}".format(nueva_col, categoria, value))
                DF[nueva_col] = 0
                new_vars.append(nueva_col)
                if esp_cat == 'hour':
                    DF.loc[DF[categoria] == int(value), nueva_col] = 1
                if esp_cat == 'day':
                    DF.loc[DF[categoria] == int(value), nueva_col] = 1
                if esp_cat == 'month':
                    DF.loc[DF[categoria] == int(value), nueva_col] = 1
                if esp_cat == 'weekday':
                    DF.loc[DF[categoria] == int(value), nueva_col] = 1
                if esp_cat == '':
                    DF.loc[DF[categoria] == int(value), nueva_col] = 1

        except Exception as e:
            logging.error('error: {} en {}'.format(e,cat))

    return DF, new_vars


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
            DF.loc[DF[col_vals].isnull(), col_vals] = avg
        except Exception as e:
            logging.error(e)
    return DF


def drop_correlation(DF, var_list, response, treshold=0.1):
    """
    De una lista de variables quita las que tengan menor correlación del DataFrame

    Args:
        DF (Dataframe): Dataframe con los datos aumentados
        var_list (list): lista de variables a verificar correlación
        response (string): nombre de la variable dependiente a predecir
    Response:
        DF (Datframe): Dataframe con variables que tienen mayor relación con
                        la variable a predecir
        dropped (list): lista de variables desechadas de la lista por baja correlación
    """
    df = DF.copy()
    dropped = []
    # Correlación con cada variable de la lista contra la variable dependiente
    logging.info("*** Se checa correlación entre variables {} vs {}: ****".format(varname, response))
    for varname in var_list:
        correl = df[[varname, response]].corr()[response][0]
        logging.info("Variable {} vs {}: {} (treshold: {})".format(varname, response, correl, treshold))
        if abs(correl) < abs(treshold):
            dropped.append([varname, correl]) #añadimos a la lista de las que eliminamos
            df.drop(varname, 1) #eliminamos la columna que no cumple con correlación mínima

    return df, dropped


def augment_numeric(DF, response):
    """
    Crea una lista con transformación de variables numéricas del DataFrame

    Args:
        DF (Dataframe): Dataframe con los datos numéricos a aumentar
        response (list): nombre de las variables a predecir
    Response:
        DF (DataFrame): Dataframe con los datos numéricos aumentados
        new_vars (list): lista con nombre de variables aumentadas candidatas
    """
    df = DF.copy()

    numericas = list(df.select_dtypes(include=['int','float']).columns)

    df = nan_to_avg(df) #cambiar a función que predice valores

    numericas = list(filter(lambda x: x not in response, numericas))
    new_vars = []

    for i in numericas:
        if isinstance(i, int) or isinstance(i, float):
            try:
                new_vars.append(i)

                varname = i + '^' + str(2)
                # Variable al cuadrado
                df[varname] = df[i]**2
                new_vars.append(varname)

                varname = i + '^' + str(3)
                # Variable al cubo
                df[varname] = df[i]**3
                new_vars.append(varname)

                varname = 'sqrt(' + i + ')'
                # Raíz cuadrada de la variable
                print('tratamos de encontrar el cuadrado de {}'.format(varname))

                df[varname] = np.sqrt(df[i])
                new_vars.append(varname)

                varname = '1/' + i
                # Inverso de la variable
                df[varname] = 1 / df[i]
                new_vars.append(varname)

                varname = 'log(' + i + ')'
                # Logaritmo de la variable
                df[varname] = df[i].apply(np.log)
                new_vars.append(varname)

                #si tenemos logaritmos que dan infinitos
                df = df.replace(-np.inf, -1000)
                df = df.replace(np.inf, 1000)
            except Exception as e:
                logger.error(e)

    return df, new_vars


def augment_date(DF, response):
    """
    Crea una lista con transformación de variables de fechas del DataFrame

    Args:
        DF (Datframe): Datframe con los datos de fecha a aumentar
        response (list): nombre de las variables dependiente a predecir
    Response:
        DF (Dataframe): Dataframe con los datos de fechas aumentados
        new_vars(list): lista con las variables aumentadas candidatas
    """
    df = DF.copy()
    fechas = list(df.select_dtypes(include=['datetime']).columns)

    df = nan_to_avg(df)
    fechas = list(filter(lambda x: x not in response, fechas))

    newvars = []
    unuseful = []
    acum_fechas = []
    new_vars = []

    for i in fechas:
        logging.info("new_vars = {}".format(new_vars))
        varname = 'hora_' + i
        new_vars.append(varname)
        logging.info("new_vars = {}".format(new_vars))
        # Hora de la fecha
        df[varname] = df[i].dt.hour
        df,added = divide_cats(df, new_vars, esp_cat='hour') #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'dia_' + i
        # Día de la fecha
        df[varname] = df[i].dt.day
        df,added = divide_cats(df, new_vars, esp_cat='day') #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'mes_' + i
        # Mes de la fecha
        df[varname] = df[i].dt.month
        df,added = divide_cats(df, new_vars, esp_cat='month') #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'dia_semana_' + i
        # Día de la semana
        for ejemplo in df[i]: lista_de_dias_semana = ejemplo.weekday()
        df[varname] = lista_de_dias_semana
        df,added = divide_cats(df, new_vars, esp_cat='weekday') #se convierte en una categoría
        new_vars = new_vars + added

        acum_fechas.append(i)

        for j in [x for x in fechas if x not in acum_fechas]:
            # Por cada fecha vamos a tomar la diferencia entre esa fecha y las demás
            # Diferencia de fechas (en días)
            varname = i + '-' + j
            df.loc[(df[i].notnull()) & (df[j].notnull()), varname] = (df[i] - df[j]).dt.days
            new_vars.append(varname)

    return df, new_vars


def augment_categories(DF, response, exclude_metadata=True):
    """
    Se hacen transformaciones con operaciones lógicas entre variables categóricas
    dentro de un Dataframe

    Args:
        DF (DataFrame): Dataframe de donde se quieren aumentar las categorías
        response(list): lista con nombres de las variables dependientes a predecir

    Response:
        df (Dataframe): DataFrame con las variables categóricas aumentadas
        new_vars(list): lista con las variables categóricas candidatas aumentadas
    """
    df = DF.copy()
    dummy_vars = []
    for i in df.columns:
        if set(df[i].unique()) == set([0, 1]):
            dummy_vars.append(i)

    if exclude_metadata: [var for var in dummy_vars if not var.startswith('__')]

    dummy_vars = list(filter(lambda x: x not in response, dummy_vars))
    new_vars = []
    logging.info("*** Lista de variables a trabajar:{}***".format(dummy_vars))

    for i in dummy_vars:
        logging.info("*** Haciendo Multiplicación de conectores lógicos\
         (AND, OR, NAND, NOR, XOR & XNOR) con la variable {}***".format(i))
        new_vars.append(i)
        for j in [x for x in dummy_vars if x not in new_vars]:
            # Multiplicación de conectores lógicos (AND, OR, NAND, NOR, XOR & XNOR)
            logging.info("""*** Multiplicación de conectores lógicos
             (AND, OR, NAND, NOR, XOR & XNOR) de {} con {}***""".format(i, j))
            varname = i + '*' + j
            df[varname] = df[i].astype(int) & df[j].astype(int)
            new_vars.append(varname)

            varname = i + '+' + j
            df[varname] = df[i].astype(int) | df[j].astype(int)
            new_vars.append(varname)

            varname = 'nand(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) & df[j].astype(int))

            varname = 'nor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = ~(df[i].astype(int) | df[j].astype(int))

            varname = 'xor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & ~(df[j].astype(int))) | (~(df[i].astype(int)) & df[j].astype(int))

            varname = 'xnor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i].astype(int) & df[j].astype(int)) | (~(df[i].astype(int)) & ~(df[j].astype(int)))

    #Algunas operaciones se quedan en valores lógicos. Hay que pasarlas a binarias
    for var in new_vars:
        df.loc[df[var] == True, var] = 1
        df.loc[df[var] == False, var] = 0

    return df, new_vars


def augment_data(DF, response, treshold=0.1, categories=False):
    """
    Prueba ciertas transformaciones numéricas, de fecha y categóricas.
    Verifica si la correlación es buena, a partir de un threshold,
    para agregarlas al dataframe resultante

    Args:
        DF (DataFrame): DataFrame de tus datos
        response (str): Variable dependiente (la debe contener tu base)
        treshold (float): Correlación mínima que se espera de una variable que
                        quieres que entre al modelo
    Returns:
        df (DataFrame): DataFrame con transformaciones útiles
        new_vals (list): Lista de variables transformadas nuevas en el dataframe
    """

    logging.info('***Haciendo agregación de datos***')
    df = DF.copy()
    catego = []
    df, numeric = augment_numeric(df, response)
    df, fecha = augment_date(df, response)
    #suele tardarse mucho la transformación de categorías
    if categories:
        catego = df.select_dtypes(['category'])
        df, catego = augment_categories(df,response, exclude_metadata=True)


    aug_vars = numeric + fecha + catego

    df, dropped = drop_correlation(df, aug_vars, response, treshold)

    new_vals = list(filter(lambda x: x not in dropped, aug_vars))

    num = list(df.select_dtypes(include=['int', 'float']).columns)

    return df, new_vals
