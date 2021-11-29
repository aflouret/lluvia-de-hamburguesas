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
    df = df.drop(columns = ['horas_de_sol', 'mm_evaporados_agua']) 
    
    df = df.drop(columns = ['id'])
    
    #Llenamos las columnas con grandes porcentajes de missings con la mean
    df['nubosidad_tarde'] = df['nubosidad_tarde'].fillna(df['nubosidad_tarde'].mean())
    df['nubosidad_temprano'] = df['nubosidad_temprano'].fillna(df['nubosidad_temprano'].mean())
    df['presion_atmosferica_temprano'] = df['presion_atmosferica_temprano'].fillna(df['presion_atmosferica_temprano'].mean())
    df['presion_atmosferica_tarde'] = df['presion_atmosferica_tarde'].fillna(df['presion_atmosferica_tarde'].mean())
    df['rafaga_viento_max_velocidad'] = df['rafaga_viento_max_velocidad'].fillna(df['rafaga_viento_max_velocidad'].mean())
    df['humedad_tarde'] = df['humedad_tarde'].fillna(df['humedad_tarde'].mean())
    df['temperatura_tarde'] = df['temperatura_tarde'].fillna(df['temperatura_tarde'].mean())
    df['mm_lluvia_dia'] = df['mm_lluvia_dia'].fillna(df['mm_lluvia_dia'].mean())
    df['velocidad_viendo_tarde'] = df['velocidad_viendo_tarde'].fillna(df['velocidad_viendo_tarde'].mean())
    df['humedad_temprano'] = df['humedad_temprano'].fillna(df['humedad_temprano'].mean())
    df['velocidad_viendo_tarde'] = df['velocidad_viendo_tarde'].fillna(df['velocidad_viendo_tarde'].mean())
    df['velocidad_viendo_temprano'] = df['velocidad_viendo_temprano'].fillna(df['velocidad_viendo_temprano'].mean())
    df['temperatura_temprano'] = df['temperatura_temprano'].fillna(df['temperatura_temprano'].mean())
    df['temp_min'] = df['temp_min'].fillna(df['temp_min'].mean())
    df['temp_max'] = df['temp_max'].fillna(df['temp_max'].mean())

    
        
   
    return df
  
    

