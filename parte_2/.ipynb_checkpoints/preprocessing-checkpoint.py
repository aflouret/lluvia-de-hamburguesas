import numpy as np
import pandas as pd


def preprocess(df):
    df['presion_atmosferica_tarde'] = pd.to_numeric(df['presion_atmosferica_tarde'],errors='coerce')
    df = df.dropna(subset=['llovieron_hamburguesas_al_dia_siguiente'])
    df['dia'] = pd.to_datetime(df['dia'])
    

