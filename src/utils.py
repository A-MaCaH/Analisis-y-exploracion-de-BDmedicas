import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle

def save_results(output_dir, filename, data):
    """
    Guarda diferentes tipos de datos en archivos
    
    Args:
        output_dir: Directorio donde guardar
        filename: Nombre del archivo
        data: Datos a guardar
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    
    try:
        if isinstance(data, pd.DataFrame):
            # DataFrame a CSV
            data.to_csv(filepath, index=False)
            print(f"✓ DataFrame guardado en {filepath}")
            
        elif isinstance(data, pd.Series):
            # Series a CSV
            data.to_csv(filepath, index=False)
            print(f"✓ Series guardado en {filepath}")
            
        elif isinstance(data, dict):
            # Diccionario a JSON
            if filename.endswith('.csv'):
                # Si el filename es CSV pero el data es dict, intentar convertir
                try:
                    df = pd.DataFrame(data)
                    df.to_csv(filepath, index=False)
                    print(f"✓ Diccionario convertido a DataFrame y guardado en {filepath}")
                except:
                    # Si falla, guardar como JSON
                    json_filepath = filepath.with_suffix('.json')
                    with open(json_filepath, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"✓ Diccionario guardado como JSON en {json_filepath}")
            else:
                # Guardar como JSON
                json_filepath = filepath.with_suffix('.json')
                with open(json_filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"✓ Diccionario guardado como JSON en {json_filepath}")
                
        elif isinstance(data, str):
            # String a archivo de texto
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data)
            print(f"✓ String guardado en {filepath}")
            
        elif isinstance(data, (list, tuple)):
            # Lista/tupla a JSON
            json_filepath = filepath.with_suffix('.json')
            with open(json_filepath, 'w', encoding='utf-8') as f:
                json.dump(list(data), f, indent=2, ensure_ascii=False)
            print(f"✓ Lista guardada como JSON en {json_filepath}")
            
        elif isinstance(data, np.ndarray):
            # Array de numpy
            if filename.endswith('.csv'):
                np.savetxt(filepath, data, delimiter=',')
                print(f"✓ Array guardado como CSV en {filepath}")
            else:
                np.save(filepath.with_suffix('.npy'), data)
                print(f"✓ Array guardado como NPY en {filepath.with_suffix('.npy')}")
                
        else:
            # Otros tipos - usar pickle como último recurso
            pickle_filepath = filepath.with_suffix('.pkl')
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Objeto guardado con pickle en {pickle_filepath}")
            
    except Exception as e:
        print(f"✗ Error guardando {filename}: {e}")
        # Intentar guardar como pickle en caso de error
        try:
            pickle_filepath = filepath.with_suffix('.pkl')
            with open(pickle_filepath, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Guardado como pickle en {pickle_filepath}")
        except Exception as e2:
            print(f"✗ Error crítico guardando {filename}: {e2}")
            raise e2