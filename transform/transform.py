#!/usr/bin/python3
# coding: utf-8
"""
Funciones que transforman y enriquecen los datos para entrenamiento
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
        DF (DataFrame): con columnas con categorías separada
        new_vars (list): lista con las variables añadidas
    """
    for cat in categorias:
        list_val_cats = set([value for value in DF[cat]]) #tomamos cada valor que tenga la categoría
        for value in list_val_cats:
            varname = cat + "_" + value
            DF[varname] = 0
            new_vars.append(varname)
            DF.loc[DF[cat] == value, varname] = 1

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
            print(avg)
            DF.loc[DF[col_vals].isnull(), col_vals] = avg
        except Exception as e:
            print(e)
    return DF


def check_correl(DF, var_list, response, treshold=0.1):
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
    for varname in var_list:
        correl = df[[varname, response]].corr()[response][0]
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
    fechas = list(df.select_dtypes(include=['datetime']).columns)

    df = nan_to_avg(df)
    fechas = list(filter(lambda x: x not in response, fechas))

    newvars = []
    unuseful = []
    acum_fechas = []
    new_vars = []

    for i in fechas:
        new_vars.append(varname)
        varname = 'hora_' + i
        # Hora de la fecha
        df[varname] = df[i].dt.hour
        df,added = divide_cats(df, varname) #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'dia_' + i
        # Día de la fecha
        df[varname] = df[i].dt.day
        df,added = divide_cats(df, varname) #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'mes_' + i
        # Mes de la fecha
        df[varname] = df[i].dt.month
        df,added = divide_cats(df, varname) #se convierte en una categoría
        new_vars = new_vars + added

        varname = 'dia_semana'
        # Día de la semana
        df[varname] = df[i].dt.weekday()
        df,added = divide_cats(df, varname) #se convierte en una categoría
        new_vars = new_vars + added

        acum_fechas.append(i)

        for j in [x for x in fechas if x not in acum_fechas]:
            # Por cada fecha vamos a tomar la diferencia entre esa fecha y las demás
            # Diferencia de fechas (en días)
            varname = i + '-' + j
            df.loc[(df[i].notnull()) & (df[j].notnull()), varname] = (df[i] - df[j]).dt.days
            new_vars.append(varname)

    return df, new_vars


def augment_categories(DF, response):
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
    dummy_vars = []
    for i in df.columns:
        if set(df[i].unique()) == set([0, 1]):
            dummy_vars.append(i)

    dummy_vars = list(filter(lambda x: x not in response, dummy_vars))
    new_vars = []
    for i in dummy_vars:
        new_vars.append(i)
        for j in [x for x in dummy_vars if x not in acum]:
            # Multiplicación de conectores lógicos (AND, OR, NAND, NOR, XOR & XNOR)
            varname = i + '*' + j
            df[varname] = df[i] and df[j]
            new_vars.append(varname)

            varname = i + '+' + j
            df[varname] = df[i] or df[j]
            new_vars.append(varname)

            varname = 'nand(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = not(df[i] and df[j])

            varname = 'nor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = not(df[i] or df[j])

            varname = 'xor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i] and not(df[j])) or (not(df[i]) and df[j])

            varname = 'xnor(' + i + ',' + j + ')'
            new_vars.append(varname)
            df[varname] = (df[i] and df[j]) or (not(df[i]) and not(df[j]))

    #Algunas operaciones se quedan en valores lógicos. Hay que pasarlas a binarias
    for var in new_vars:
        df.loc[df[var] == True, var] = 1
        df.loc[df[var] == False, var] = 0

    return df, new_vars


def augment_data(DF, response, treshold=0.1, categories=False):
    """
    Prueba ciertas transformaciones numéricas y verifica si la correlación es buena
    para agregarlas al dataframe

    Args:
        DF (DataFrame): DataFrame de tus datos
        response (str): Variable dependiente (la debe contener tu base)
        treshold (float): Correlación mínima que se espera de una variable que
                        quieres que entre al modelo
    Returns:
        df (DataFrame): DataFrame con transformaciones útiles
        new_vals (list): Lista de variables transformadas nuevas en el dataframe
    """
    df = DF.copy()
    catego = []
    df, numeric = augment_numeric(df, response)
    df, fecha = augment_date(df, response)
    #suele tardarse mucho la transformación de categorías
    if categories: df, catego = augment_categories(df, response)

    aug_vars = numeric + fecha + catego

    df, dropped = check_correl(df, aug_vars, response, treshold)

    new_vals = list(filter(lambda x: x not in dropped, aug_vars))

    num = list(df.select_dtypes(include=['int', 'float']).columns)

    return df[num], new_vals
