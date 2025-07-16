import pandas as pd
import os
from pathlib import Path
from src.steps.base_step_ import BaseStep
import pydicom
import nibabel as nib
from PIL import Image
import icecream as ic
from src.steps.my_icecream import Show
show = Show(condition=False)
class DataLoader(BaseStep):
    def run(self):       
        metadata_path = self.general_config['input_metadata_path']
        images_dir = Path(self.general_config['input_images_dir'])
        strategy = self.params.get('missing_values_strategy', 'drop')
        impute_method = self.params.get('impute_method', 'mean')

        # Cargar metadatos
        if metadata_path.endswith('.csv'):
            df = pd.read_csv(metadata_path)
        elif metadata_path.endswith('.xlsx'):
            df = pd.read_excel(metadata_path)
        else:
            raise ValueError("Formato de archivo no soportado para metadatos")

        # Limpieza básica
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'impute':
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                if impute_method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif impute_method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif impute_method == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)

        # Asociar imágenes con cada fila de metadatos
        if 'filename' not in df.columns:
            raise ValueError("La columna 'filename' debe existir en el archivo de metadatos")

        df['image_path'] = df['filename'].apply(lambda x: str(x))
        df['image_exists'] = df['image_path'].apply(lambda p: os.path.isfile(p))
        # print(f" ...... \n {df.head()}")
        return df
