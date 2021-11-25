import numpy as np
import pandas as pd


def preprocess(df:pd.DataFrame):
    
    #Convertimos la presion a un tipo numérico, establecimos los valores inválidos como NaN.
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'],errors='coerce')    
    #Convertimos las fechas al tipo datetime.
    df['dia'] = pd.to_datetime(df['dia'])
    
    #Eliminamos todas las filas del dataframe donde haya valores faltantes
    df = df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente']) 
    
    #Eliminamos las dos columnas con mayor porcentaje de missings
    df = df.drop(columns = ['horas_de_sol', 'mm_de_agua_avaporados']) 
    
    #Llenamos las columnas con grandes porcentajes de missings con la mediana
    df['nubosidad_tarde'] = df['nubosidad_tarde'].fillna(df['nubosidad_tarde'].median())
        
   
    return df
  
    

