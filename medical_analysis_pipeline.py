#!/usr/bin/env python3
"""
Pipeline completo de análisis de imágenes médicas.
Incluye configuración interactiva, ejecución y visualización de resultados.
"""

import yaml
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from importlib import import_module
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalAnalysisPipeline:
    def __init__(self):
        self.config = {
            'general': {},
            'pipeline': []
        }
        self.data = None
        self.results = {}
        
    def print_header(self):
        """Imprime el encabezado del pipeline"""
        print("=" * 70)
        print("  🏥 PIPELINE DE ANÁLISIS DE IMÁGENES MÉDICAS")
        print("=" * 70)
        print()
    
    def get_input(self, prompt: str, default: str = "", required: bool = True) -> str:
        """Obtiene entrada del usuario con validación"""
        while True:
            if default:
                user_input = input(f"{prompt} [{default}]: ").strip()
                if not user_input:
                    user_input = default
            else:
                user_input = input(f"{prompt}: ").strip()
            
            if not required or user_input:
                return user_input
            print("❌ Este campo es obligatorio. Por favor, ingresa un valor.")
    
    def get_yes_no(self, prompt: str, default: str = "y") -> bool:
        """Obtiene respuesta sí/no del usuario"""
        while True:
            response = self.get_input(prompt, default).lower()
            if response in ['y', 'yes', 'sí', 'si', '1']:
                return True
            elif response in ['n', 'no', '0']:
                return False
            print("❌ Por favor responde 'y' (sí) o 'n' (no).")
    
    def get_multiple_choice(self, prompt: str, options: List[str], default: int = 0) -> str:
        """Obtiene selección múltiple del usuario"""
        print(f"\n{prompt}")
        for i, option in enumerate(options):
            marker = "→" if i == default else " "
            print(f"  {marker} {i+1}. {option}")
        
        while True:
            try:
                choice = input(f"\nSelecciona una opción [1-{len(options)}] [{default+1}]: ").strip()
                if not choice:
                    choice = default + 1
                else:
                    choice = int(choice)
                
                if 1 <= choice <= len(options):
                    return options[choice - 1]
                else:
                    print(f"❌ Por favor selecciona un número entre 1 y {len(options)}.")
            except ValueError:
                print("❌ Por favor ingresa un número válido.")
    
    def get_multiple_selections(self, prompt: str, options: List[str]) -> List[str]:
        """Obtiene múltiples selecciones del usuario"""
        print(f"\n{prompt}")
        for i, option in enumerate(options):
            print(f"  {i+1}. {option}")
        
        while True:
            try:
                selections = input(f"\nSelecciona opciones (separadas por coma, ej: 1,3) [1-{len(options)}]: ").strip()
                if not selections:
                    return [options[0]]
                
                indices = [int(x.strip()) - 1 for x in selections.split(',')]
                if all(0 <= i < len(options) for i in indices):
                    return [options[i] for i in indices]
                else:
                    print(f"❌ Por favor selecciona números entre 1 y {len(options)}.")
            except ValueError:
                print("❌ Por favor ingresa números válidos separados por coma.")
    
    def configure_pipeline(self):
        """Configura todo el pipeline de forma interactiva"""
        print("🔧 CONFIGURACIÓN INTERACTIVA DEL PIPELINE")
        print("-" * 50)
        
        # Configuración general
        print("\n📁 CONFIGURACIÓN GENERAL")
        print("-" * 30)
        
        metadata_path = self.get_input(
            "Ruta al archivo CSV/Excel con metadatos",
            "data/mnist_medical_metadata.csv"
        )
        self.config['general']['input_metadata_path'] = metadata_path
        
        images_dir = self.get_input(
            "Directorio con las imágenes médicas",
            "data/mnist_medical_images/"
        )
        self.config['general']['input_images_dir'] = images_dir
        
        output_dir = self.get_input(
            "Directorio de salida para resultados",
            "outputs/medical_analysis"
        )
        self.config['general']['output_dir'] = output_dir
        
        # Data Loader
        print("\n📥 CONFIGURACIÓN: CARGA DE DATOS")
        print("-" * 30)
        
        strategy = self.get_multiple_choice(
            "Estrategia para valores faltantes:",
            ["drop", "impute"],
            1
        )
        
        impute_method = "mean"
        if strategy == "impute":
            impute_method = self.get_multiple_choice(
                "Método de imputación:",
                ["mean", "median", "mode"],
                1
            )
        
        self.config['pipeline'].append({
            'name': 'data_loader',
            'params': {
                'missing_values_strategy': strategy,
                'impute_method': impute_method
            }
        })
        
        # Descriptive Stats
        print("\n📊 CONFIGURACIÓN: ESTADÍSTICAS DESCRIPTIVAS")
        print("-" * 30)
        
        use_stats = self.get_yes_no("¿Incluir análisis estadístico descriptivo?", "y")
        
        if use_stats:
            group_by = self.get_input(
                "Variable para agrupar análisis (dejar vacío para omitir)",
                "diagnosis",
                required=False
            )
            
            params = {}
            if group_by:
                params['group_by_variable'] = group_by
            
            self.config['pipeline'].append({
                'name': 'descriptive_stats',
                'params': params
            })
        
        # Feature Extraction
        print("\n🔍 CONFIGURACIÓN: EXTRACCIÓN DE CARACTERÍSTICAS")
        print("-" * 30)
        
        use_features = self.get_yes_no("¿Incluir extracción de características?", "y")
        
        if use_features:
            strategies = self.get_multiple_selections(
                "Estrategias de extracción de características:",
                ["intensity", "texture", "shape"]
            )
            
            self.config['pipeline'].append({
                'name': 'feature_extraction',
                'params': {
                    'strategies': strategies
                }
            })
        
        # Projection
        print("\n📈 CONFIGURACIÓN: PROYECCIÓN Y VISUALIZACIÓN")
        print("-" * 30)
        
        use_projection = self.get_yes_no("¿Incluir análisis de proyección?", "y")
        
        if use_projection:
            methods = self.get_multiple_selections(
                "Métodos de proyección:",
                ["pca", "tsne", "isomap", "umap"]
            )
            
            color_by = self.get_input(
                "Variable para colorear visualizaciones",
                "diagnosis"
            )
            
            self.config['pipeline'].append({
                'name': 'projection',
                'params': {
                    'methods': methods,
                    'color_by_variable': color_by
                }
            })
        
        # Quality Assurance
        print("\n🔍 CONFIGURACIÓN: CONTROL DE CALIDAD")
        print("-" * 30)
        
        use_qa = self.get_yes_no("¿Incluir control de calidad?", "y")
        
        if use_qa:
            method = self.get_multiple_choice(
                "Método de detección de anomalías:",
                ["IsolationForest", "LocalOutlierFactor", "EllipticEnvelope"],
                0
            )
            
            self.config['pipeline'].append({
                'name': 'quality_assurance',
                'params': {
                    'method': method
                }
            })
    
    def show_config_summary(self):
        """Muestra un resumen de la configuración"""
        print("\n" + "=" * 70)
        print("  📋 RESUMEN DE CONFIGURACIÓN")
        print("=" * 70)
        
        print("\n📁 CONFIGURACIÓN GENERAL:")
        for key, value in self.config['general'].items():
            print(f"  • {key}: {value}")
        
        print(f"\n🔄 PASOS DEL PIPELINE ({len(self.config['pipeline'])}):")
        for i, step in enumerate(self.config['pipeline'], 1):
            print(f"  {i}. {step['name']}")
            if step['params']:
                for param, value in step['params'].items():
                    print(f"     • {param}: {value}")
        
        print(f"\n✅ Configuración completada con {len(self.config['pipeline'])} pasos.")
    
    def execute_pipeline(self):
        """Ejecuta el pipeline completo"""
        print("\n🚀 EJECUTANDO PIPELINE")
        print("-" * 50)
        
        # Crear directorio de salida
        output_dir = Path(self.config['general']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ejecutar cada paso
        for i, step_conf in enumerate(self.config['pipeline'], 1):
            print(f"\n🔄 Paso {i}/{len(self.config['pipeline'])}: {step_conf['name']}")
            print("-" * 40)
            
            try:
                # Importar y ejecutar el paso
                module = import_module(f"src.steps.{step_conf['name']}")
                step_class_name = ''.join(part.capitalize() for part in step_conf['name'].split('_'))
                StepClass = getattr(module, step_class_name)
                
                step_instance = StepClass(self.data, step_conf.get('params', {}), self.config['general'])
                self.data = step_instance.run()
                
                # Guardar resultados del paso
                if self.data is not None:
                    self.results[step_conf['name']] = self.data
                    print(f"✅ Paso {step_conf['name']} completado exitosamente")
                else:
                    print(f"⚠️ Paso {step_conf['name']} no retornó datos")
                    
            except Exception as e:
                print(f"❌ Error en paso {step_conf['name']}: {e}")
                logging.error(f"Error en paso {step_conf['name']}: {e}")
                continue
        
        # Guardar resultado final
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                final_output_path = output_dir / 'final_output.csv'
                self.data.to_csv(final_output_path, index=False)
                print(f"\n💾 Resultado final guardado en: {final_output_path}")
        
        print("\n✅ Pipeline completado!")
    
    def visualize_results(self):
        """Visualiza los resultados del pipeline"""
        print("\n📊 VISUALIZANDO RESULTADOS")
        print("-" * 50)
        
        if not self.results:
            print("❌ No hay resultados para visualizar")
            return
        
        # Configurar matplotlib para mostrar gráficos
        plt.ion()  # Modo interactivo
        
        # Visualizar estadísticas descriptivas
        if 'descriptive_stats' in self.results:
            self._show_descriptive_stats()
        
        # Visualizar proyecciones
        if 'projection' in self.results:
            self._show_projections()
        
        # Visualizar características extraídas
        if 'feature_extraction' in self.results:
            self._show_feature_analysis()
        
        # Visualizar control de calidad
        if 'quality_assurance' in self.results:
            self._show_quality_assurance()
        
        print("\n📈 Visualizaciones mostradas. Cierra las ventanas para continuar.")
        plt.show(block=True)
    
    def _show_descriptive_stats(self):
        """Muestra estadísticas descriptivas"""
        print("📊 Mostrando estadísticas descriptivas...")
        
        df = self.results['descriptive_stats']
        if not isinstance(df, pd.DataFrame):
            return
        
        # Mostrar información básica del dataset
        print(f"Dataset: {len(df)} filas, {len(df.columns)} columnas")
        print(f"Columnas: {list(df.columns)}")
        
        # Mostrar primeras filas
        print("\nPrimeras 5 filas:")
        print(df.head())
        
        # Mostrar estadísticas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\nEstadísticas de columnas numéricas ({len(numeric_cols)}):")
            print(df[numeric_cols].describe())
    
    def _show_projections(self):
        """Muestra las proyecciones generadas"""
        print("📈 Mostrando proyecciones...")
        
        output_dir = Path(self.config['general']['output_dir'])
        projection_dir = output_dir / 'projection'
        
        if not projection_dir.exists():
            print("❌ No se encontró directorio de proyecciones")
            return
        
        # Buscar archivos de proyección
        projection_files = list(projection_dir.glob('*.png'))
        
        if not projection_files:
            print("❌ No se encontraron archivos de proyección")
            return
        
        print(f"📁 Encontrados {len(projection_files)} archivos de proyección:")
        
        # Mostrar cada proyección
        for i, file_path in enumerate(projection_files):
            print(f"  {i+1}. {file_path.name}")
            
            # Cargar y mostrar imagen
            try:
           
                img = plt.imread(file_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.title(f"Proyección: {file_path.stem}")
                plt.axis('off')
                plt.tight_layout()
            except Exception as e:
                print(f"❌ Error al cargar {file_path}: {e}")
    
    def _show_feature_analysis(self):
        """Muestra análisis de características extraídas"""
        print("🔍 Mostrando análisis de características...")
        
        df = self.results['feature_extraction']
        if not isinstance(df, pd.DataFrame):
            return
        
        # Identificar columnas de características
        feature_cols = [col for col in df.columns if any(prefix in col for prefix in ['intensity_', 'texture_', 'shape_'])]
        
        if not feature_cols:
            print("❌ No se encontraron características extraídas")
            return
        
        print(f"📊 Características extraídas ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"  • {col}")
        
        # Mostrar estadísticas de características
        if len(feature_cols) > 0:
            print(f"\nEstadísticas de características:")
            print(df[feature_cols].describe())
            
            # Crear histogramas de características
            n_features = len(feature_cols)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            for i, col in enumerate(feature_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.hist(df[col].dropna(), bins=30, alpha=0.7)
                plt.title(f'Distribución de {col}')
                plt.xlabel(col)
                plt.ylabel('Frecuencia')
            plt.tight_layout()
    
    def _show_quality_assurance(self):
        """Muestra resultados del control de calidad"""
        print("🔍 Mostrando control de calidad...")
        
        output_dir = Path(self.config['general']['output_dir'])
        qa_dir = output_dir / 'quality_assurance'
        
        if not qa_dir.exists():
            print("❌ No se encontró directorio de control de calidad")
            return
        
        # Buscar archivos de anomalías
        anomaly_files = list(qa_dir.glob('*.png'))
        
        if not anomaly_files:
            print("❌ No se encontraron archivos de control de calidad")
            return
        
        print(f"📁 Encontrados {len(anomaly_files)} archivos de control de calidad:")
        
        # Mostrar cada visualización
        for i, file_path in enumerate(anomaly_files):
            print(f"  {i+1}. {file_path.name}")
            
            try:
                img = plt.imread(file_path)
                plt.figure(figsize=(10, 8))
                plt.imshow(img)
                plt.title(f"Control de Calidad: {file_path.stem}")
                plt.axis('off')
                plt.tight_layout()
            except Exception as e:
                print(f"❌ Error al cargar {file_path}: {e}")
    
    def save_config(self, filename: str = "pipeline_config.yaml"):
        """Guarda la configuración en un archivo YAML"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
            print(f"\n💾 Configuración guardada en: {filename}")
            return True
        except Exception as e:
            print(f"\n❌ Error al guardar configuración: {e}")
            return False
    
    def run(self):
        """Ejecuta el pipeline completo"""
        self.print_header()
        
        # Configurar pipeline
        self.configure_pipeline()
        
        # Mostrar resumen
        self.show_config_summary()
        
        # Confirmar ejecución
        if not self.get_yes_no("\n¿Ejecutar el pipeline con esta configuración?", "y"):
            print("❌ Pipeline cancelado")
            return
        
        # Ejecutar pipeline
        self.execute_pipeline()
        
        # Preguntar si visualizar resultados
        if self.get_yes_no("\n¿Visualizar resultados?", "y"):
            self.visualize_results()
        
        # Preguntar si guardar configuración
        if self.get_yes_no("\n¿Guardar configuración para uso futuro?", "y"):
            filename = self.get_input("Nombre del archivo", "pipeline_config.yaml")
            self.save_config(filename)
        
        print("\n🎉 ¡Análisis completado exitosamente!")

def main():
    """Función principal"""
    try:
        pipeline = MedicalAnalysisPipeline()
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\n❌ Pipeline cancelado por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        logging.error(f"Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 