
from pathlib import Path
import pandas as pd

def cargar_datos():
    # ruta real del archivo .py
    src_dir = Path(__file__).resolve().parent
    
    # raíz del proyecto (un nivel arriba de src)
    project_root = src_dir.parent
    
    file_path = project_root / "Base_de_datos.xlsx"
    return pd.read_excel(file_path)

