from src.steps.base_step_ import BaseStep
from src.utils import save_results
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, kruskal
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import logging
from src.steps.my_icecream import Show

class StatisticalAnalysis(BaseStep):
    def _get_dataframe_from_data(self):
        """Extrae o crea un DataFrame desde self.data"""
        print("=== DIAGNÓSTICO StatisticalAnalysis ===")
        print(f"Tipo de self.data: {type(self.data)}")
        
        if isinstance(self.data, pd.DataFrame):
            print("✓ self.data ya es un DataFrame")
            return self.data.copy()
        
        elif isinstance(self.data, dict):
            print(f"Claves en self.data: {list(self.data.keys())}")
            
            # Buscar DataFrame en claves comunes
            possible_keys = ['data', 'df', 'dataframe', 'results', 'output', 'processed_data']
            for key in possible_keys:
                if key in self.data and isinstance(self.data[key], pd.DataFrame):
                    print(f"✓ DataFrame encontrado en clave '{key}'")
                    return self.data[key].copy()
            
            # Buscar cualquier DataFrame en el diccionario
            for key, value in self.data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"✓ DataFrame encontrado en clave '{key}'")
                    return value.copy()
            
            # Si no hay DataFrame, intentar crear uno
            try:
                # Verificar si todos los valores son listas/arrays de la misma longitud
                lengths = []
                for key, value in self.data.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        lengths.append(len(value))
                    else:
                        lengths.append(1)  # Scalar
                
                if len(set(lengths)) == 1 and lengths[0] > 1:
                    # Todas tienen la misma longitud > 1
                    df = pd.DataFrame(self.data)
                    print("✓ DataFrame creado desde diccionario con listas")
                    return df
                elif len(set(lengths)) == 1 and lengths[0] == 1:
                    # Todos son escalares
                    df = pd.DataFrame([self.data])
                    print("✓ DataFrame creado desde diccionario con escalares (1 fila)")
                    return df
                else:
                    # Longitudes diferentes, tomar la máxima
                    max_length = max(lengths)
                    filtered_data = {}
                    for key, value in self.data.items():
                        if isinstance(value, (list, tuple, np.ndarray)) and len(value) == max_length:
                            filtered_data[key] = value
                    
                    if filtered_data:
                        df = pd.DataFrame(filtered_data)
                        print(f"✓ DataFrame creado con {len(filtered_data)} columnas de longitud {max_length}")
                        return df
                    else:
                        raise ValueError("No se pudo crear DataFrame: elementos de diferentes longitudes")
            
            except Exception as e:
                print(f"✗ Error creando DataFrame: {e}")
                raise ValueError(f"No se pudo convertir el diccionario a DataFrame: {e}")
        
        else:
            raise ValueError(f"Tipo de datos no soportado: {type(self.data)}")

    def run(self):
        # Obtener DataFrame
        df = self._get_dataframe_from_data()
        
        if df is None or df.empty:
            raise ValueError("No se pudo obtener un DataFrame válido o está vacío")
        
        print(f"DataFrame obtenido - Shape: {df.shape}")
        print(f"Columnas: {list(df.columns)}")
        
        output_dir = Path(self.general_config['output_dir']) / 'statistical_analysis'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener parámetros
        group_var = self.params.get('group_by_variable')
        continuous_vars = self.params.get('continuous_variables', [])
        
        # Validar que las variables existen
        if group_var and group_var not in df.columns:
            print(f"⚠️  Variable de agrupación '{group_var}' no encontrada en columnas: {list(df.columns)}")
            # Si no está, buscar una columna categórica
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                group_var = categorical_cols[0]
                print(f"Usando '{group_var}' como variable de agrupación")
            else:
                raise ValueError(f"Variable de agrupación '{group_var}' no encontrada")
        
        # Validar variables continuas
        available_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if not continuous_vars:
            continuous_vars = available_numeric
            print(f"Variables continuas no especificadas. Usando: {continuous_vars}")
        else:
            # Filtrar solo las que existen
            continuous_vars = [var for var in continuous_vars if var in df.columns]
            if not continuous_vars:
                continuous_vars = available_numeric
                print(f"Variables continuas especificadas no encontradas. Usando: {continuous_vars}")
        
        if not continuous_vars:
            print("⚠️  No se encontraron variables continuas. Creando variable dummy.")
            df['dummy_continuous'] = np.random.normal(0, 1, len(df))
            continuous_vars = ['dummy_continuous']
        
        # Realizar análisis estadístico
        report_lines = [f"# Análisis Estadístico por Grupo: {group_var}\n"]
        
        for var in continuous_vars:
            if var not in df.columns:
                continue
                
            report_lines.append(f"\n## Variable: {var}\n")
            
            # Filtrar datos válidos
            sub_df = df[[group_var, var]].dropna()
            
            if sub_df.empty:
                report_lines.append("⚠️  No hay datos válidos para esta variable\n")
                continue
            
            # Agrupar datos
            groups = [g[var].values for name, g in sub_df.groupby(group_var)]
            k = len(groups)
            
            # Verificar que hay suficientes datos
            if k < 2:
                report_lines.append("⚠️  Se necesitan al menos 2 grupos para el análisis\n")
                continue
            
            # Verificar que cada grupo tiene suficientes datos
            min_group_size = min(len(g) for g in groups)
            if min_group_size < 3:
                report_lines.append("⚠️  Se necesitan al menos 3 observaciones por grupo\n")
                continue
            
            report_lines.append(f"- Número de grupos: {k}\n")
            
            # Shapiro-Wilk por grupo
            normal = True
            for i, g in enumerate(groups):
                if len(g) >= 3:  # Shapiro necesita al menos 3 observaciones
                    stat, p = shapiro(g)
                    report_lines.append(f"  - Grupo {i+1}: Shapiro-Wilk p = {p:.4f}")
                    if p < 0.05:
                        normal = False
                else:
                    report_lines.append(f"  - Grupo {i+1}: Pocos datos para Shapiro-Wilk")
                    normal = False
            
            # Test de Levene
            try:
                stat, p_levene = levene(*groups)
                homoscedastic = p_levene >= 0.05
                report_lines.append(f"- Levene p = {p_levene:.4f} → {'Homocedástico' if homoscedastic else 'Heterocedástico'}")
            except Exception as e:
                report_lines.append(f"- Error en test de Levene: {e}")
                homoscedastic = False
            
            # Prueba principal
            try:
                if k == 2:
                    if normal and homoscedastic:
                        stat, p = ttest_ind(groups[0], groups[1])
                        method = "T-test"
                    elif normal:
                        stat, p = ttest_ind(groups[0], groups[1], equal_var=False)
                        method = "Welch T-test"
                    else:
                        stat, p = mannwhitneyu(groups[0], groups[1])
                        method = "Mann-Whitney U"
                    report_lines.append(f"- {method} → p = {p:.4f}")
                else:
                    if normal and homoscedastic:
                        try:
                            formula = f"{var} ~ C({group_var})"
                            model = ols(formula, data=sub_df).fit()
                            aov_table = anova_lm(model, typ=2)
                            p = aov_table['PR(>F)'][0]
                            report_lines.append(f"- ANOVA → p = {p:.4f}")
                            
                            if p < 0.05:
                                tukey = pairwise_tukeyhsd(sub_df[var], sub_df[group_var])
                                report_lines.append("#### Tukey HSD:\n")
                                report_lines.append(str(tukey.summary()))
                        except Exception as e:
                            report_lines.append(f"- Error en ANOVA: {e}")
                    else:
                        stat, p = kruskal(*groups)
                        report_lines.append(f"- Kruskal-Wallis → p = {p:.4f}")
                        
                        if p < 0.05:
                            try:
                                dunn = sp.posthoc_dunn(sub_df, val_col=var, group_col=group_var, p_adjust='bonferroni')
                                report_lines.append("#### Dunn Post-hoc (Bonferroni):\n")
                                report_lines.append(dunn.to_string())
                            except Exception as e:
                                report_lines.append(f"- Error en Dunn post-hoc: {e}")
                                
            except Exception as e:
                report_lines.append(f"- Error en análisis estadístico: {e}")
        
        # Crear reporte final
        report = "\n".join(report_lines)
        
        # Guardar reporte como string, no como objeto
        try:
            with open(output_dir / 'statistical_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"✓ Reporte guardado en {output_dir / 'statistical_report.md'}")
        except Exception as e:
            print(f"Error guardando reporte: {e}")
        
        return df