from src.steps.base_step_ import BaseStep
from src.utils import save_results
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import logging
from src.steps.my_icecream import Show

class QualityAssurance(BaseStep):
    def run(self):
        # DIAGNÓSTICO COMPLETO
        print("=== DIAGNÓSTICO DE DATOS ===")
        print(f"Tipo de self.data: {type(self.data)}")
        print(f"Tamaño de self.data: {len(self.data) if hasattr(self.data, '__len__') else 'N/A'}")
        
        if isinstance(self.data, dict):
            print(f"Claves en self.data: {list(self.data.keys())}")
            for key, value in self.data.items():
                print(f"  {key}: {type(value)}, tamaño: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                    print(f"    Primer elemento: {type(value[0])}")
        
        # INTENTAR OBTENER EL DATAFRAME
        df = None
        
        # Estrategia 1: Verificar si ya es un DataFrame
        if isinstance(self.data, pd.DataFrame):
            df = self.data.copy()
            print("✓ self.data ya es un DataFrame")
        
        # Estrategia 2: Buscar DataFrame en el diccionario
        elif isinstance(self.data, dict):
            # Buscar claves comunes que contengan DataFrames
            possible_keys = ['data', 'df', 'dataframe', 'results', 'output', 'processed_data']
            for key in possible_keys:
                if key in self.data and isinstance(self.data[key], pd.DataFrame):
                    df = self.data[key].copy()
                    print(f"✓ DataFrame encontrado en clave '{key}'")
                    break
            
            # Si no encontramos un DataFrame, buscar en todas las claves
            if df is None:
                for key, value in self.data.items():
                    if isinstance(value, pd.DataFrame):
                        df = value.copy()
                        print(f"✓ DataFrame encontrado en clave '{key}'")
                        break
            
            # Estrategia 3: Si el diccionario contiene listas/arrays del mismo tamaño
            if df is None:
                try:
                    # Verificar si todas las claves tienen la misma longitud
                    lengths = []
                    for key, value in self.data.items():
                        if isinstance(value, (list, tuple, np.ndarray)):
                            lengths.append(len(value))
                        else:
                            lengths.append(1)  # Scalar
                    
                    if len(set(lengths)) == 1 and lengths[0] > 1:
                        # Todas tienen la misma longitud, intentar crear DataFrame
                        df = pd.DataFrame(self.data)
                        print("✓ DataFrame creado desde diccionario con listas")
                    elif len(set(lengths)) == 1 and lengths[0] == 1:
                        # Todos son escalares, crear DataFrame con una sola fila
                        df = pd.DataFrame([self.data])
                        print("✓ DataFrame creado desde diccionario con escalares (1 fila)")
                    else:
                        print("✗ El diccionario tiene elementos de diferentes longitudes")
                        # Intentar crear DataFrame solo con elementos de la misma longitud
                        max_length = max(lengths)
                        filtered_data = {}
                        for key, value in self.data.items():
                            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == max_length:
                                filtered_data[key] = value
                        
                        if filtered_data:
                            df = pd.DataFrame(filtered_data)
                            print(f"✓ DataFrame creado con {len(filtered_data)} columnas de longitud {max_length}")
                        
                except Exception as e:
                    print(f"✗ Error al crear DataFrame: {e}")
        
        # VERIFICAR SI TENEMOS UN DATAFRAME VÁLIDO
        if df is None or df.empty:
            raise ValueError("No se pudo obtener un DataFrame válido o está vacío")
        
        print(f"\n=== DATAFRAME OBTENIDO ===")
        print(f"Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        print(f"Tipos de datos:\n{df.dtypes}")
        print(f"Primeras 3 filas:\n{df.head(3)}")
        
        # VERIFICAR COLUMNAS NUMÉRICAS
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        print(f"\nColumnas numéricas encontradas: {list(numeric_cols)}")
        
        if len(numeric_cols) == 0:
            # Intentar convertir columnas que parezcan numéricas
            potential_numeric = []
            for col in df.columns:
                if col.lower() in ['image_path', 'path', 'filename', 'name', 'id']:
                    continue  # Saltar columnas que claramente no son numéricas
                try:
                    pd.to_numeric(df[col], errors='raise')
                    potential_numeric.append(col)
                except:
                    pass
            
            if potential_numeric:
                print(f"Columnas potencialmente numéricas: {potential_numeric}")
                for col in potential_numeric:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
                print(f"Columnas numéricas después de conversión: {list(numeric_cols)}")
        
        if len(numeric_cols) == 0:
            # Si aún no hay columnas numéricas, crear una columna dummy para que el proceso continúe
            print("⚠️  No se encontraron columnas numéricas. Creando columna dummy.")
            df['dummy_numeric'] = range(len(df))
            numeric_cols = ['dummy_numeric']
        
        # CONTINUAR CON EL PROCESAMIENTO
        output_dir = Path(self.general_config['output_dir']) / 'quality_assurance'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        method = self.params.get('method', 'IsolationForest')
        
        # Obtener solo las características numéricas y eliminar NaN
        features = df[numeric_cols].dropna()
        
        if features.empty:
            print("⚠️  No quedan datos después de eliminar NaN. Usando datos originales.")
            features = df[numeric_cols].fillna(0)  # Llenar NaN con 0
        
        print(f"\nCaracterísticas para análisis: {features.shape}")
        print(f"Columnas usadas: {list(features.columns)}")
        
        ids = features.index
        X = StandardScaler().fit_transform(features)
        
        # Inicializar columna de anomalías
        df['is_anomaly'] = 0
        
        if method.lower() == 'isolationforest':
            model = IsolationForest(contamination='auto', random_state=42)
            is_outlier = model.fit_predict(X)
            df.loc[ids, 'is_anomaly'] = (is_outlier == -1).astype(int)
        elif method.lower() == 'localoutlierfactor':
            model = LocalOutlierFactor(n_neighbors=min(20, len(X)-1))  # Ajustar n_neighbors si hay pocos datos
            is_outlier = model.fit_predict(X)
            df.loc[ids, 'is_anomaly'] = (is_outlier == -1).astype(int)
        else:
            raise ValueError(f"Método de aseguramiento de calidad no soportado: {method}")
        
        print(f"\nAnomalías detectadas: {df['is_anomaly'].sum()}/{len(df)}")
        
        # Guardar resultados
        save_results(output_dir, 'anomaly_labels.csv', df[['is_anomaly']])
        
        return df