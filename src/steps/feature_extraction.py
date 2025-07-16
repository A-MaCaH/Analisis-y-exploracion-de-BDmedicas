from abc import ABC, abstractmethod
from src.steps.base_step_ import BaseStep
from src.utils import save_results
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from PIL import Image
import logging
import os
from tqdm import tqdm
from src.steps.my_icecream import Show

class FeatureStrategy(ABC):
    @abstractmethod
    def extract(self, row):
        pass
    
    @abstractmethod
    def get_feature_names(self):
        """Devuelve los nombres de las características que extrae esta estrategia"""
        pass

class IntensityFeatures(FeatureStrategy):
    def get_feature_names(self):
        return [
            'intensity_mean',
            'intensity_std', 
            'intensity_skew',
            'intensity_kurtosis'
        ]
    
    def extract(self, row):
        img_path = row['image_path']
        try:
            # Verificar que el archivo existe
            if not os.path.exists(img_path):
                #logging.warning(f"Archivo no encontrado: {img_path}")
                return self._get_nan_features()
            
            # Leer imagen
            img = imread(img_path, as_gray=True)
            
            # Verificar que la imagen no esté vacía
            if img.size == 0:
                logging.warning(f"Imagen vacía: {img_path}")
                return self._get_nan_features()
            
            flat = img.flatten()
            
            # Calcular estadísticas
            mean_val = np.mean(flat)
            std_val = np.std(flat)
            
            # Usar pandas para skew y kurtosis (más robusto)
            series = pd.Series(flat)
            skew_val = series.skew()
            kurt_val = series.kurtosis()
            
            return {
                'intensity_mean': float(mean_val) if not np.isnan(mean_val) else np.nan,
                'intensity_std': float(std_val) if not np.isnan(std_val) else np.nan,
                'intensity_skew': float(skew_val) if not np.isnan(skew_val) else np.nan,
                'intensity_kurtosis': float(kurt_val) if not np.isnan(kurt_val) else np.nan
            }
            
        except Exception as e:
            logging.error(f"Error procesando {img_path}: {e}")
            return self._get_nan_features()
    
    def _get_nan_features(self):
        return {name: np.nan for name in self.get_feature_names()}

class TextureFeatures(FeatureStrategy):
    def get_feature_names(self):
        return [
            'texture_contrast',
            'texture_dissimilarity',
            'texture_homogeneity',
            'texture_energy',
            'texture_correlation'
        ]
    
    def extract(self, row):
        img_path = row['image_path']
        try:
            # Verificar que el archivo existe
            if not os.path.exists(img_path):
                logging.warning(f"Archivo no encontrado: {img_path}")
                return self._get_nan_features()
            
            # Leer imagen
            img = imread(img_path, as_gray=True)
            
            # Verificar que la imagen no esté vacía
            if img.size == 0:
                logging.warning(f"Imagen vacía: {img_path}")
                return self._get_nan_features()
            
            # Convertir a uint8 para GLCM
            img = (img * 255).astype(np.uint8)
            
            # Calcular GLCM (Gray Level Co-occurrence Matrix)
            glcm = graycomatrix(
                img, 
                distances=[1], 
                angles=[0], 
                levels=256, 
                symmetric=True, 
                normed=True
            )
            
            # Extraer propiedades de textura
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            
            return {
                'texture_contrast': float(contrast) if not np.isnan(contrast) else np.nan,
                'texture_dissimilarity': float(dissimilarity) if not np.isnan(dissimilarity) else np.nan,
                'texture_homogeneity': float(homogeneity) if not np.isnan(homogeneity) else np.nan,
                'texture_energy': float(energy) if not np.isnan(energy) else np.nan,
                'texture_correlation': float(correlation) if not np.isnan(correlation) else np.nan
            }
            
        except Exception as e:
            logging.error(f"Error procesando {img_path}: {e}")
            return self._get_nan_features()
    
    def _get_nan_features(self):
        return {name: np.nan for name in self.get_feature_names()}

class ShapeFeatures(FeatureStrategy):
    def get_feature_names(self):
        return [
            'shape_area',
            'shape_perimeter',
            'shape_circularity',
            'shape_aspect_ratio',
            'shape_solidity'
        ]
    
    def extract(self, row):
        img_path = row['image_path']
        try:
            # Verificar que el archivo existe
            if not os.path.exists(img_path):
                logging.warning(f"Archivo no encontrado: {img_path}")
                return self._get_nan_features()
            
            # Leer imagen
            img = imread(img_path, as_gray=True)
            
            # Verificar que la imagen no esté vacía
            if img.size == 0:
                logging.warning(f"Imagen vacía: {img_path}")
                return self._get_nan_features()
            
            # Binarizar imagen (asumiendo que queremos formas en primer plano)
            from skimage.filters import threshold_otsu
            from skimage.measure import label, regionprops
            
            thresh = threshold_otsu(img)
            binary = img > thresh
            
            # Encontrar regiones
            labeled = label(binary)
            regions = regionprops(labeled)
            
            if not regions:
                logging.warning(f"No se encontraron regiones en {img_path}")
                return self._get_nan_features()
            
            # Tomar la región más grande
            largest_region = max(regions, key=lambda r: r.area)
            
            # Calcular características de forma
            area = largest_region.area
            perimeter = largest_region.perimeter
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            bbox = largest_region.bbox
            aspect_ratio = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]) if (bbox[3] - bbox[1]) > 0 else 0
            
            solidity = largest_region.solidity
            
            return {
                'shape_area': float(area),
                'shape_perimeter': float(perimeter),
                'shape_circularity': float(circularity),
                'shape_aspect_ratio': float(aspect_ratio),
                'shape_solidity': float(solidity)
            }
            
        except Exception as e:
            logging.error(f"Error procesando {img_path}: {e}")
            return self._get_nan_features()
    
    def _get_nan_features(self):
        return {name: np.nan for name in self.get_feature_names()}

class FeatureExtraction(BaseStep):
    def run(self):
        try:
            # Obtener y validar datos
            df = self._get_dataframe()
            
            if df is None or df.empty:
                logging.error("No se pudo obtener un DataFrame válido o está vacío")
                return None
            
            # Validar que existe la columna image_path
            if 'image_path' not in df.columns:
                logging.error("La columna 'image_path' no existe en el DataFrame")
                return None
            
            # Obtener estrategias
            strategies = self.params.get('strategies', [])
            if not strategies:
                logging.warning("No se especificaron estrategias de extracción")
                return df
            
            # Crear directorio de salida
            output_dir = Path(self.general_config['output_dir']) / 'features'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Mapeo de estrategias disponibles
            strategy_map = {
                'intensity': IntensityFeatures(),
                'texture': TextureFeatures(),
                'shape': ShapeFeatures()
            }
            
            # Validar estrategias
            valid_strategies = []
            for strat_name in strategies:
                if strat_name in strategy_map:
                    valid_strategies.append(strat_name)
                else:
                    logging.warning(f"Estrategia no reconocida: {strat_name}")
            
            if not valid_strategies:
                logging.error("No se encontraron estrategias válidas")
                return df
            
            logging.info(f"Iniciando extracción de características para {len(df)} imágenes")
            logging.info(f"Estrategias a usar: {valid_strategies}")
            print("!!!!!!!!!!!!!1")
            print(df.head)
            # Procesar cada estrategia
            for strat_name in valid_strategies:
                logging.info(f"Procesando estrategia: {strat_name}")
                strat = strategy_map[strat_name]
                
                # Extraer características con barra de progreso
                results_list = []
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extrayendo {strat_name}"):
                    features = strat.extract(row)
                    results_list.append(features)
                
                # Convertir a DataFrame
                results_df = pd.DataFrame(results_list, index=df.index)
                
                # Concatenar con el DataFrame original
                df = pd.concat([df, results_df], axis=1)
                
                # Guardar resultados intermedios
                intermediate_file = f'features_{strat_name}.csv'
                save_results(output_dir, intermediate_file, df)
                
                logging.info(f"Características de {strat_name} extraídas exitosamente")
            
            # Guardar resultado final
            save_results(output_dir, 'features_complete.csv', df)
            
            # Generar reporte de extracción
            self._generate_extraction_report(df, output_dir, valid_strategies)
            
            logging.info("Extracción de características completada exitosamente")
            return df
            
        except Exception as e:
            logging.error(f"Error en la extracción de características: {e}")
            return None
    
    def _get_dataframe(self):
        """Obtiene y valida el DataFrame desde self.data"""
        try:
            if isinstance(self.data, pd.DataFrame):
                return self.data
            
            elif isinstance(self.data, dict):
                # Buscar DataFrame en claves comunes
                possible_keys = ['data', 'df', 'main_data', 'dataset', 'dataframe']
                
                for key in possible_keys:
                    if key in self.data and isinstance(self.data[key], pd.DataFrame):
                        logging.info(f"DataFrame encontrado en la clave: {key}")
                        return self.data[key]
                
                # Si no encontramos DataFrame, verificar si todas las claves son DataFrames
                dataframes = {k: v for k, v in self.data.items() if isinstance(v, pd.DataFrame)}
                if dataframes:
                    largest_df = max(dataframes.values(), key=len)
                    logging.info(f"Usando el DataFrame más grande con {len(largest_df)} filas")
                    return largest_df
                
                # Intentar convertir diccionario a DataFrame
                try:
                    df = pd.DataFrame(self.data)
                    logging.info("Diccionario convertido a DataFrame exitosamente")
                    return df
                except Exception as e:
                    logging.error(f"No se pudo convertir el diccionario a DataFrame: {e}")
                    return None
            
            else:
                logging.error(f"Tipo de datos no soportado: {type(self.data)}")
                return None
                
        except Exception as e:
            logging.error(f"Error al obtener DataFrame: {e}")
            return None
    
    def _generate_extraction_report(self, df, output_dir, strategies):
        """Genera un reporte de la extracción de características"""
        try:
            report = {
                'total_images': len(df),
                'strategies_used': strategies,
                'features_extracted': []
            }
            
            strategy_map = {
                'intensity': IntensityFeatures(),
                'texture': TextureFeatures(),
                'shape': ShapeFeatures()
            }
            
            for strat_name in strategies:
                strat = strategy_map[strat_name]
                feature_names = strat.get_feature_names()
                
                # Contar valores válidos para cada característica
                valid_counts = {}
                for feature in feature_names:
                    if feature in df.columns:
                        valid_counts[feature] = df[feature].count()
                
                report['features_extracted'].append({
                    'strategy': strat_name,
                    'features': feature_names,
                    'valid_counts': valid_counts
                })
            
            # Guardar reporte
            report_df = pd.DataFrame([report])
            save_results(output_dir, 'extraction_report.csv', report_df)
            
            logging.info("Reporte de extracción generado")
            
        except Exception as e:
            logging.error(f"Error generando reporte: {e}")