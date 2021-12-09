import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split


def basic_preprocessing(df: pd.DataFrame, test_size=0.1):
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'],errors='coerce')    
    df['dia'] = pd.to_datetime(df['dia'])
    df = df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente']) 
    df = df.dropna(subset=['llovieron_hamburguesas_hoy'])
    df = df.drop(columns = ['id'])
    df['mes'] = df['dia'].dt.month
    df = df.drop(columns = ['dia']) 
    df = df[df.isnull().mean(1) < 0.4]
    
    label_encoder = LabelEncoder()
    
    label_encoder.fit(df['llovieron_hamburguesas_hoy'])
    df['llovieron_hamburguesas_hoy'] = label_encoder.transform(df['llovieron_hamburguesas_hoy'])

    label_encoder.fit(df['llovieron_hamburguesas_al_dia_siguiente'])
    df['llovieron_hamburguesas_al_dia_siguiente'] = label_encoder.transform(df['llovieron_hamburguesas_al_dia_siguiente'])
    
    
    X = df.drop(columns=['llovieron_hamburguesas_al_dia_siguiente'])
    y = df['llovieron_hamburguesas_al_dia_siguiente']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=117, test_size=test_size, stratify=y.astype(str))
    
    numerical_features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano', 
                          'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde',
                          'mm_lluvia_dia', 'velocidad_viendo_tarde', 'humedad_temprano', 'velocidad_viendo_temprano',
                          'temperatura_temprano', 'temp_min', 'temp_max', 'mm_evaporados_agua']
    
    for feature in numerical_features:
        if feature == 'mm_lluvia_dia' or feature == 'mm_evaporados_agua':
            X_train[feature] = X_train[feature].fillna(X_train[feature].median())
            X_test[feature] = X_test[feature].fillna(X_test[feature].median())
        else:
            X_train[feature] = X_train[feature].fillna(X_train[feature].mean())
            X_test[feature] = X_test[feature].fillna(X_test[feature].mean())
            
    
    return X_train, X_test, y_train, y_test
    

def preprocessing_knn_standard(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df, test_size=0.2)

    X_train = X_train.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    X_test = X_test.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                    'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                    'mm_evaporados_agua'])  
    
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = StandardScaler()

    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.fit_transform(X_test[features])
        
    return X_train, X_test, y_train, y_test

def preprocessing_knn_min_max(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df, test_size=0.2)

    X_train = X_train.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    X_test = X_test.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                    'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                    'mm_evaporados_agua'])  
    
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
      
    scaler = MinMaxScaler()

    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.fit_transform(X_test[features])
        
    return X_train, X_test, y_train, y_test

def preprocessing_knn_normalizer(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df, test_size=0.2)

    X_train = X_train.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    X_test = X_test.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                    'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                    'mm_evaporados_agua'])  
    
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = Normalizer()

    X_train[features] = scaler.fit_transform(X_train[features])
    X_test[features] = scaler.fit_transform(X_test[features])
        
    return X_train, X_test, y_train, y_test

def preprocessing_arboles_1(df:pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df)

    X_train = pd.get_dummies(X_train, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    X_test = pd.get_dummies(X_test, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])

    return X_train, X_test, y_train, y_test

def preprocessing_arboles_2(df:pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df)
    X_train = X_train.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'llovieron_hamburguesas_hoy', 'barrio']) 
    X_test = X_test.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'llovieron_hamburguesas_hoy', 'barrio']) 
    
    return X_train, X_test, y_train, y_test

def preprocessing_arboles_3(df:pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df)
  
    X_train = X_train.drop(columns = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano']) 
    X_test = X_test.drop(columns = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano'])
    
    X_train = pd.get_dummies(X_train, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    X_test = pd.get_dummies(X_test, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    
    return X_train, X_test, y_train, y_test
    
def preprocessing_arboles_4(df:pd.DataFrame):
    X_train, X_test, y_train, y_test = basic_preprocessing(df)

    X_train = X_train.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'barrio', 'horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'temperatura_temprano', 'presion_atmosferica_temprano', 'velocidad_viendo_temprano', 'temperatura_temprano', 'temp_min', 'temp_max']) 
    
    X_test = X_test.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'barrio', 'horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'temperatura_temprano', 'presion_atmosferica_temprano', 'velocidad_viendo_temprano', 'temperatura_temprano', 'temp_min', 'temp_max'])
    
    return X_train, X_test, y_train, y_test


def basic_prediction_preprocessing(df: pd.DataFrame, test_size=0.1):
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'],errors='coerce')    
    df['dia'] = pd.to_datetime(df['dia'])
    df = df.drop(columns = ['id'])
    df['mes'] = df['dia'].dt.month
    df = df.drop(columns = ['dia']) 
    
    
    df['llovieron_hamburguesas_hoy'] = df['llovieron_hamburguesas_hoy'].fillna('no')
    label_encoder = LabelEncoder()
    label_encoder.fit(df['llovieron_hamburguesas_hoy'])
    df['llovieron_hamburguesas_hoy'] = label_encoder.transform(df['llovieron_hamburguesas_hoy'])

    numerical_features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano', 
                          'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde',
                          'mm_lluvia_dia', 'velocidad_viendo_tarde', 'humedad_temprano', 'velocidad_viendo_temprano',
                          'temperatura_temprano', 'temp_min', 'temp_max', 'mm_evaporados_agua']
    
    for feature in numerical_features:
        if feature == 'mm_lluvia_dia' or feature == 'mm_evaporados_agua':
            df[feature] = df[feature].fillna(df[feature].median())
        else:
            df[feature] = df[feature].fillna(df[feature].mean())
            
    
    return df

def preprocessing_nb_pred(df: pd.DataFrame):
    df = basic_prediction_preprocessing(df)
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes', 'llovieron_hamburguesas_hoy', 'barrio']) 
    return df

def preprocessing_arboles_pred(df: pd.DataFrame):
    df = basic_prediction_preprocessing(df)
    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    return df

def preprocessing_knn_pred(df: pd.DataFrame):
    df = basic_prediction_preprocessing(df)
    df = df.drop(columns = ['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'mes',
                                      'barrio', 'llovieron_hamburguesas_hoy', 'velocidad_viendo_temprano', 'temperatura_temprano',
                                      'mm_evaporados_agua'])  
    
    features = ['horas_de_sol', 'nubosidad_tarde', 'nubosidad_temprano', 'presion_atmosferica_temprano',
                'presion_atmosferica_tarde', 'rafaga_viento_max_velocidad', 'humedad_tarde', 'temperatura_tarde', 'mm_lluvia_dia',
                'velocidad_viendo_tarde', 'humedad_temprano', 'temp_min', 'temp_max']
    
    scaler = StandardScaler()

    df[features] = scaler.fit_transform(df[features])
    return df

def preprocessing_random_forest_pred(df: pd.DataFrame):
    df = basic_prediction_preprocessing(df)
    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    return df

def preprocessing_boosting_pred(df: pd.DataFrame):
    df = basic_prediction_preprocessing(df)
    df = pd.get_dummies(df, drop_first=True, dummy_na=True, columns=['direccion_viento_temprano', 'rafaga_viento_max_direccion', 'direccion_viento_tarde', 'barrio'])
    return df