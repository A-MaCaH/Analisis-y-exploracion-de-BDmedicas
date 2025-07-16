from src.steps.base_step_ import BaseStep
from src.utils import save_results
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import pairwise_distances
import logging
import warnings
from tqdm import tqdm
from src.steps.my_icecream import Show


class Projection(BaseStep):
    def run(self):
        try:
            # Obtener y validar datos
            df = self._get_dataframe()
            
            if df is None or df.empty:
                logging.error("No se pudo obtener un DataFrame válido o está vacío")
                return None
            
            # Crear una copia para trabajar
            df = df.copy()
            
            # Crear directorio de salida
            output_dir = Path(self.general_config['output_dir']) / 'projection'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Obtener parámetros
            methods = self.params.get('methods', ['pca'])
            color_var = self.params.get('color_by_variable', None)
            n_components = self.params.get('n_components', 2)
            
            logging.info(f"Iniciando proyección con métodos: {methods}")
            logging.info(f"DataFrame shape: {df.shape}")
            
            # Preparar datos para proyección
            features_df, feature_names = self._prepare_features(df)
            
            if features_df is None or features_df.empty:
                logging.error("No se encontraron características numéricas válidas")
                return df
            
            logging.info(f"Características seleccionadas: {len(feature_names)} features")
            logging.info(f"Muestras válidas: {len(features_df)}")
            
            # Estandarizar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)
            
            # Guardar información de escalado
            scaling_info = pd.DataFrame({
                'feature': feature_names,
                'mean': scaler.mean_,
                'std': scaler.scale_
            })
            save_results(output_dir, 'scaling_info.csv', scaling_info)
            
            # Aplicar métodos de proyección
            for method in methods:
                logging.info(f"Aplicando método: {method}")
                try:
                    projected_df = self._apply_projection_method(
                        method, X_scaled, features_df.index, n_components
                    )
                    
                    if projected_df is not None:
                        # Unir proyección con datos originales
                        df = df.join(projected_df, how='left')
                        
                        # Generar visualización
                        self._generate_projection_plot(
                            df, method, color_var, output_dir, n_components
                        )
                        
                        # Guardar proyección
                        save_results(output_dir, f'{method}_projection.csv', projected_df)
                        
                        logging.info(f"Proyección {method} completada exitosamente")
                    
                except Exception as e:
                    logging.error(f"Error aplicando método {method}: {e}")
                    continue
            
            # Generar matrices de distancia y correlación
            self._generate_distance_matrices(X_scaled, features_df.index, output_dir)
            
            # Generar matriz de correlación de características
            self._generate_feature_correlation(features_df, output_dir)
            
            # Generar reporte de proyección
            self._generate_projection_report(df, methods, feature_names, output_dir)
            
            logging.info("Proyección completada exitosamente")
            return df
            
        except Exception as e:
            logging.error(f"Error en la proyección: {e}")
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
    
    def _prepare_features(self, df):
        """Prepara las características numéricas para proyección"""
        try:
            # Seleccionar solo columnas numéricas
            numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
            
            if len(numeric_columns) == 0:
                logging.error("No se encontraron columnas numéricas")
                return None, []
            
            # Filtrar columnas con suficientes datos válidos
            min_valid_ratio = 0.5  # Al menos 50% de datos válidos
            valid_columns = []
            
            for col in numeric_columns:
                valid_ratio = df[col].count() / len(df)
                if valid_ratio >= min_valid_ratio:
                    valid_columns.append(col)
                else:
                    logging.warning(f"Columna {col} descartada: solo {valid_ratio:.2%} datos válidos")
            
            if len(valid_columns) == 0:
                logging.error("No se encontraron columnas con suficientes datos válidos")
                return None, []
            
            # Crear DataFrame solo con características válidas
            features_df = df[valid_columns].copy()
            
            # Eliminar filas con valores faltantes
            features_df = features_df.dropna()
            
            if len(features_df) == 0:
                logging.error("No quedan filas después de eliminar valores faltantes")
                return None, []
            
            # Verificar varianza (eliminar columnas constantes)
            constant_columns = []
            for col in features_df.columns:
                if features_df[col].std() == 0:
                    constant_columns.append(col)
            
            if constant_columns:
                logging.warning(f"Eliminando columnas constantes: {constant_columns}")
                features_df = features_df.drop(columns=constant_columns)
            
            if features_df.empty:
                logging.error("No quedan características después de filtrar columnas constantes")
                return None, []
            
            return features_df, list(features_df.columns)
            
        except Exception as e:
            logging.error(f"Error preparando características: {e}")
            return None, []
    
    def _apply_projection_method(self, method, X, index, n_components):
        """Aplica un método específico de proyección"""
        try:
            if method == 'pca':
                model = PCA(n_components=n_components, random_state=42)
            elif method == 'tsne':
                # t-SNE es más lento, usar menos iteraciones para datasets grandes
                n_iter = 1000 if len(X) > 1000 else 1000
                model = TSNE(
                    n_components=n_components, 
                    random_state=42,
                    n_iter=n_iter,
                    perplexity=min(30, len(X) // 4)  # Ajustar perplexity según el tamaño
                )
            elif method == 'isomap':
                # Ajustar n_neighbors según el tamaño del dataset
                n_neighbors = min(15, len(X) // 2)
                model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            elif method == 'umap':
                try:
                    from umap import UMAP
                    model = UMAP(n_components=n_components, random_state=42)
                except ImportError:
                    logging.warning("UMAP no está instalado, saltando...")
                    return None
            else:
                logging.error(f"Método desconocido: {method}")
                return None
            
            # Aplicar transformación
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                components = model.fit_transform(X)
            
            # Crear DataFrame con los componentes
            column_names = [f'{method}_{i+1}' for i in range(n_components)]
            projected_df = pd.DataFrame(
                components, 
                columns=column_names, 
                index=index
            )
            
            # Guardar información del modelo si es PCA
            if method == 'pca':
                self._save_pca_info(model, output_dir)
            
            return projected_df
            
        except Exception as e:
            logging.error(f"Error aplicando método {method}: {e}")
            return None
    
    def _save_pca_info(self, pca_model, output_dir):
        """Guarda información adicional de PCA"""
        try:
            # Varianza explicada
            variance_info = pd.DataFrame({
                'component': [f'PC{i+1}' for i in range(len(pca_model.explained_variance_ratio_))],
                'explained_variance_ratio': pca_model.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca_model.explained_variance_ratio_)
            })
            save_results(output_dir, 'pca_variance_explained.csv', variance_info)
            
            # Componentes principales
            components_info = pd.DataFrame(
                pca_model.components_.T,
                columns=[f'PC{i+1}' for i in range(pca_model.n_components_)]
            )
            save_results(output_dir, 'pca_components.csv', components_info)
            
        except Exception as e:
            logging.error(f"Error guardando información de PCA: {e}")
    
    def _generate_projection_plot(self, df, method, color_var, output_dir, n_components):
        """Genera visualización de la proyección"""
        try:
            if n_components == 2:
                self._plot_2d_projection(df, method, color_var, output_dir)
            elif n_components == 3:
                self._plot_3d_projection(df, method, color_var, output_dir)
            else:
                logging.warning(f"Visualización no soportada para {n_components} componentes")
                
        except Exception as e:
            logging.error(f"Error generando gráfico para {method}: {e}")
    
    def _plot_2d_projection(self, df, method, color_var, output_dir):
        """Genera gráfico 2D de la proyección"""
        try:
            x_col = f'{method}_1'
            y_col = f'{method}_2'
            
            if x_col not in df.columns or y_col not in df.columns:
                logging.warning(f"Columnas de proyección no encontradas para {method}")
                return
            
            plt.figure(figsize=(12, 8))
            
            # Filtrar datos válidos
            plot_data = df[[x_col, y_col]].dropna()
            
            if color_var and color_var in df.columns:
                # Filtrar también por variable de color
                color_data = df.loc[plot_data.index, color_var]
                plot_data = plot_data[color_data.notna()]
                color_data = color_data[color_data.notna()]
                
                if len(plot_data) > 0:
                    scatter = plt.scatter(
                        plot_data[x_col], 
                        plot_data[y_col], 
                        c=color_data, 
                        cmap='viridis', 
                        alpha=0.7
                    )
                    plt.colorbar(scatter, label=color_var)
            else:
                plt.scatter(
                    plot_data[x_col], 
                    plot_data[y_col], 
                    alpha=0.7
                )
            
            plt.xlabel(f'{method.upper()} Component 1')
            plt.ylabel(f'{method.upper()} Component 2')
            plt.title(f'{method.upper()} Projection')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            save_results(output_dir, f'{method}_scatter.png', plt.gcf())
            plt.close()
            
        except Exception as e:
            logging.error(f"Error generando gráfico 2D para {method}: {e}")
            plt.close()
    
    def _plot_3d_projection(self, df, method, color_var, output_dir):
        """Genera gráfico 3D de la proyección"""
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            x_col = f'{method}_1'
            y_col = f'{method}_2'
            z_col = f'{method}_3'
            
            if any(col not in df.columns for col in [x_col, y_col, z_col]):
                logging.warning(f"Columnas de proyección 3D no encontradas para {method}")
                return
            
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Filtrar datos válidos
            plot_data = df[[x_col, y_col, z_col]].dropna()
            
            if color_var and color_var in df.columns:
                color_data = df.loc[plot_data.index, color_var]
                plot_data = plot_data[color_data.notna()]
                color_data = color_data[color_data.notna()]
                
                if len(plot_data) > 0:
                    scatter = ax.scatter(
                        plot_data[x_col], 
                        plot_data[y_col], 
                        plot_data[z_col],
                        c=color_data, 
                        cmap='viridis', 
                        alpha=0.7
                    )
                    plt.colorbar(scatter, label=color_var)
            else:
                ax.scatter(
                    plot_data[x_col], 
                    plot_data[y_col], 
                    plot_data[z_col],
                    alpha=0.7
                )
            
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_zlabel(f'{method.upper()} Component 3')
            ax.set_title(f'{method.upper()} 3D Projection')
            
            save_results(output_dir, f'{method}_3d_scatter.png', plt.gcf())
            plt.close()
            
        except Exception as e:
            logging.error(f"Error generando gráfico 3D para {method}: {e}")
            plt.close()
    
    def _generate_distance_matrices(self, X, index, output_dir):
        """Genera matrices de distancia"""
        try:
            # Matriz de distancias euclidianas
            dist_matrix = pairwise_distances(X, metric='euclidean')
            dist_df = pd.DataFrame(dist_matrix, index=index, columns=index)
            save_results(output_dir, 'euclidean_distance_matrix.csv', dist_df)
            
            # Matriz de distancias coseno
            cosine_matrix = pairwise_distances(X, metric='cosine')
            cosine_df = pd.DataFrame(cosine_matrix, index=index, columns=index)
            save_results(output_dir, 'cosine_distance_matrix.csv', cosine_df)
            
            logging.info("Matrices de distancia generadas")
            
        except Exception as e:
            logging.error(f"Error generando matrices de distancia: {e}")
    
    def _generate_feature_correlation(self, features_df, output_dir):
        """Genera matriz de correlación de características"""
        try:
            corr_matrix = features_df.corr()
            save_results(output_dir, 'feature_correlation_matrix.csv', corr_matrix)
            
            # Visualizar matriz de correlación
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            
            save_results(output_dir, 'feature_correlation_heatmap.png', plt.gcf())
            plt.close()
            
            logging.info("Matriz de correlación de características generada")
            
        except Exception as e:
            logging.error(f"Error generando matriz de correlación: {e}")
            plt.close()
    
    def _generate_projection_report(self, df, methods, feature_names, output_dir):
        """Genera reporte de la proyección"""
        try:
            report = {
                'total_samples': len(df),
                'features_used': feature_names,
                'num_features': len(feature_names),
                'methods_applied': methods,
                'projection_columns': []
            }
            
            # Recopilar columnas de proyección generadas
            for method in methods:
                method_cols = [col for col in df.columns if col.startswith(f'{method}_')]
                if method_cols:
                    report['projection_columns'].extend(method_cols)
            
            # Estadísticas de proyección
            projection_stats = {}
            for col in report['projection_columns']:
                if col in df.columns:
                    projection_stats[col] = {
                        'mean': float(df[col].mean()) if not df[col].isna().all() else np.nan,
                        'std': float(df[col].std()) if not df[col].isna().all() else np.nan,
                        'min': float(df[col].min()) if not df[col].isna().all() else np.nan,
                        'max': float(df[col].max()) if not df[col].isna().all() else np.nan
                    }
            
            report['projection_statistics'] = projection_stats
            
            # Guardar reporte
            report_df = pd.DataFrame([report])
            save_results(output_dir, 'projection_report.csv', report_df)
            
            logging.info("Reporte de proyección generado")
            
        except Exception as e:
            logging.error(f"Error generando reporte: {e}")