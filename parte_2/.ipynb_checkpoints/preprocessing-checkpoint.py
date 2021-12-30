import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


def basic_preprocessing(df: pd.DataFrame, prediction_dataset=False):
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'],errors='coerce')    
    df['dia'] = pd.to_datetime(df['dia'])
    
    df['mes'] = df['dia'].dt.month
    df = df.drop(columns = ['dia'])
    if prediction_dataset == False:
        df = df.dropna(subset=['llovieron_hamburguesas_hoy'])
        df = df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente'])
        df = df[df.isnull().mean(1) < 0.4]

    label_encoder = LabelEncoder()
    
    label_encoder.fit(df['llovieron_hamburguesas_hoy'])
    df['llovieron_hamburguesas_hoy'] = label_encoder.transform(df['llovieron_hamburguesas_hoy'])
    
    if prediction_dataset == False:
        label_encoder.fit(df['llovieron_hamburguesas_al_dia_siguiente'])
        df['llovieron_hamburguesas_al_dia_siguiente'] = label_encoder.transform(df['llovieron_hamburguesas_al_dia_siguiente'])
    
    return df
    
def split(df: pd.DataFrame):
    X = df.drop(columns=['llovieron_hamburguesas_al_dia_siguiente'])
    y = df[['id', 'llovieron_hamburguesas_al_dia_siguiente']]
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=117, test_size=0.1, stratify=y['llovieron_hamburguesas_al_dia_siguiente'].astype(str))
    
    df_train = X_train.merge(y_train, on='id').drop(columns = ['id'])
    df_holdout = X_holdout.merge(y_holdout, on='id').drop(columns = ['id'])
    
    return df_train, df_holdout
    
def fill_numerical_missings(X: pd.DataFrame, X_train: pd.DataFrame):
    numerical_features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano', 
                          'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde',
                          'mm_lluvia_dia', 'velocidad_viendo_tarde', 'humedad_temprano', 'velocidad_viendo_temprano',
                          'temperatura_temprano', 'temp_min', 'temp_max', 'mm_evaporados_agua']
    
    for feature in numerical_features:
        if feature == 'mm_lluvia_dia' or feature == 'mm_evaporados_agua':
            X[feature] = X[feature].fillna(X_train[feature].median())
        else:
            X[feature] = X[feature].fillna(X_train[feature].mean())
    
    return X

def preprocessing_knn_standard(df: pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
        
    return df

def preprocessing_knn_min_max(df: pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
            'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
            'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
        
    return df

def preprocessing_knn_normalizer(df: pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
       
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = Normalizer()
    df[features] = scaler.fit_transform(df[features])
        
    return df

def preprocessing_arboles_1(df:pd.DataFrame):

    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])

    return df

def preprocessing_arboles_2(df:pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'llovieron_hamburguesas_hoy', 'barrio']) 
    
    return df

def preprocessing_arboles_3(df:pd.DataFrame):  
    df = df.drop(columns = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano']) 
    
    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    
    return df
    
def preprocessing_arboles_4(df:pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'barrio', 'horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'temperatura_temprano', 'presion_atmosferica_temprano', 'velocidad_viendo_temprano', 'temperatura_temprano', 'temp_min', 'temp_max']) 
    
    return df


def preprocessing_redes_1(df:pd.DataFrame):
    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])

    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano', 'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia', 'velocidad_viendo_tarde', 'humedad_temprano','temp_min', 'temp_max', 'velocidad_viendo_temprano', 'temperatura_temprano', 'mm_evaporados_agua']
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
        
    return df

def preprocessing_redes_2(df: pd.DataFrame):
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano', 'mm_evaporados_agua'])  
     
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano', 'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia', 'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
        
    return df