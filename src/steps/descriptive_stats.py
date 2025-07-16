import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.steps.base_step_ import BaseStep
from src.utils import save_results
import warnings
import logging
from src.steps.my_icecream import Show
show = Show(condition=True)
class DescriptiveStats(BaseStep):
    def run(self):
        print(f"----...{type(self.data)}")
        try:
            # Obtener y validar los datos
            
            df = self._get_dataframe()
            
            if df is None or df.empty:
                logging.error("No se pudo obtener un DataFrame válido o está vacío")
                return None
            
            # Crear directorio de salida
            output_dir = Path(self.general_config['output_dir']) / 'descriptive_stats'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Procesando DataFrame con {len(df)} filas y {len(df.columns)} columnas")
            
            # Generar estadísticas descriptivas
            self._generate_summary_statistics(df, output_dir)
            show.show(self._generate_summary_statistics(df, output_dir))
            # Generar histogramas para variables numéricas
            self._generate_histograms(df, output_dir)
            show.show()
            # Generar gráficos de barras para variables categóricas
            self._generate_bar_plots(df, output_dir)
            
            # Generar boxplots agrupados si se especifica
            self._generate_grouped_boxplots(df, output_dir)
            
            # Generar matriz de correlación
            self._generate_correlation_matrix(df, output_dir)
            
            logging.info("Análisis descriptivo completado exitosamente")
            return df
            
        except Exception as e:
            logging.error(f"Error en el análisis descriptivo: {e}")
            return None
    
    def _get_dataframe(self):
        """Obtiene y valida el DataFrame desde self.data"""
        try:            
            if isinstance(self.data, pd.DataFrame):
                show.show("get dateframe devuelve dataframe")
                return self.data
            
            elif isinstance(self.data, dict):
                # Buscar DataFrame en claves comunes
                possible_keys = ['group_by_variable','data', 'df', 'main_data', 'dataset', 'dataframe']
                for key in possible_keys:
                    if key in self.data.keys():
                        show.show("DataFrame encontrado")
                        show.show(self.data)
                        return self.data[key]
                
                # Si no encontramos DataFrame, verificar si todas las claves son DataFrames
                dataframes = {k: v for k, v in self.data.items() if isinstance(v, pd.DataFrame)}
                if dataframes:
                    # Tomar el DataFrame más grande
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
            
            elif isinstance(self.data, list):
                # Intentar convertir lista a DataFrame
                try:
                    df = pd.DataFrame(self.data)
                    logging.info("Lista convertida a DataFrame exitosamente")
                    return df
                except Exception as e:
                    logging.error(f"No se pudo convertir la lista a DataFrame: {e}")
                    return None
            
            else:
                logging.error(f"Tipo de datos no soportado: {type(self.data)}")
                return None
                
        except Exception as e:
            logging.error(f"Error al obtener DataFrame: {e}")
            return None
    
    def _generate_summary_statistics(self, df, output_dir):
        """Genera estadísticas descriptivas básicas"""
        try:
            desc = df.describe(include='all')
            save_results(output_dir, 'summary_statistics.csv', desc)
            logging.info("Estadísticas descriptivas generadas")
        except Exception as e:
            logging.error(f"Error generando estadísticas descriptivas: {e}")
    
    def _generate_histograms(self, df, output_dir):
        """Genera histogramas para variables numéricas"""
        try:
            numeric_columns = df.select_dtypes(include='number').columns
            
            if len(numeric_columns) == 0:
                logging.warning("No se encontraron columnas numéricas para histogramas")
                return
            
            for col in numeric_columns:
                try:
                    # Verificar que la columna tenga datos válidos
                    valid_data = df[col].dropna()
                    if len(valid_data) == 0:
                        logging.warning(f"La columna {col} no tiene datos válidos")
                        continue
                    
                    plt.figure(figsize=(10, 6))
                    sns.histplot(valid_data, kde=True, bins=30)
                    plt.title(f"Histograma de {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frecuencia")
                    plt.grid(True, alpha=0.3)
                    
                    save_results(output_dir, f"hist_{col}.png", plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Error generando histograma para {col}: {e}")
                    plt.close()
            
            logging.info(f"Histogramas generados para {len(numeric_columns)} columnas numéricas")
            
        except Exception as e:
            logging.error(f"Error generando histogramas: {e}")
    
    def _generate_bar_plots(self, df, output_dir):
        """Genera gráficos de barras para variables categóricas"""
        try:
            categorical_columns = df.select_dtypes(include='object').columns
            
            if len(categorical_columns) == 0:
                logging.warning("No se encontraron columnas categóricas para gráficos de barras")
                return
            
            for col in categorical_columns:
                try:
                    # Verificar que la columna tenga datos válidos
                    valid_data = df[col].dropna()
                    if len(valid_data) == 0:
                        logging.warning(f"La columna {col} no tiene datos válidos")
                        continue
                    
                    # Limitar a las top 20 categorías para evitar gráficos muy densos
                    value_counts = valid_data.value_counts().head(20)
                    
                    plt.figure(figsize=(12, 6))
                    value_counts.plot(kind='bar')
                    plt.title(f"Distribución de {col}")
                    plt.xlabel(col)
                    plt.ylabel("Frecuencia")
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    save_results(output_dir, f"bar_{col}.png", plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Error generando gráfico de barras para {col}: {e}")
                    plt.close()
            
            logging.info(f"Gráficos de barras generados para {len(categorical_columns)} columnas categóricas")
            
        except Exception as e:
            logging.error(f"Error generando gráficos de barras: {e}")
    
    def _generate_grouped_boxplots(self, df, output_dir):
        """Genera boxplots agrupados si se especifica group_by_variable"""
        try:
            group_var = self.params.get('group_by_variable')
            
            if not group_var:
                logging.info("No se especificó group_by_variable, omitiendo boxplots agrupados")
                return
            
            if group_var not in df.columns:
                logging.warning(f"La variable de agrupación '{group_var}' no existe en el DataFrame")
                return
            
            numeric_columns = df.select_dtypes(include='number').columns
            
            if len(numeric_columns) == 0:
                logging.warning("No se encontraron columnas numéricas para boxplots")
                return
            
            for col in numeric_columns:
                try:
                    # Verificar que ambas columnas tengan datos válidos
                    subset = df[[group_var, col]].dropna()
                    if len(subset) == 0:
                        logging.warning(f"No hay datos válidos para {col} y {group_var}")
                        continue
                    
                    plt.figure(figsize=(12, 6))
                    sns.boxplot(data=subset, x=group_var, y=col)
                    plt.title(f"Boxplot de {col} por {group_var}")
                    plt.xticks(rotation=45, ha='right')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    save_results(output_dir, f"box_{col}_by_{group_var}.png", plt.gcf())
                    plt.close()
                    
                except Exception as e:
                    logging.error(f"Error generando boxplot para {col} por {group_var}: {e}")
                    plt.close()
            
            logging.info(f"Boxplots agrupados generados para {len(numeric_columns)} columnas numéricas")
            
        except Exception as e:
            logging.error(f"Error generando boxplots agrupados: {e}")
    
    def _generate_correlation_matrix(self, df, output_dir):
        """Genera matriz de correlación para variables numéricas"""
        try:
            numeric_columns = df.select_dtypes(include='number').columns
            
            if len(numeric_columns) < 2:
                logging.warning("Se necesitan al menos 2 columnas numéricas para matriz de correlación")
                return
            
            # Calcular matriz de correlación
            corr_matrix = df[numeric_columns].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f')
            plt.title("Matriz de Correlación")
            plt.tight_layout()
            
            save_results(output_dir, "correlation_matrix.png", plt.gcf())
            plt.close()
            
            # Guardar matriz de correlación como CSV
            save_results(output_dir, "correlation_matrix.csv", corr_matrix)
            
            logging.info("Matriz de correlación generada")
            
        except Exception as e:
            logging.error(f"Error generando matriz de correlación: {e}")
            plt.close()