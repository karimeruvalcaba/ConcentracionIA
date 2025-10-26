#Este no jala para el contrafactual, lo hice para ver si jalaba la lectura del primer archivo q nos dio el profe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import getpass
import os, sys, json, textwrap, time, math, random
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

# Initialize the client
client = OpenAI(api_key=api_key)

# Local folder where the CSV is located
base_path = r"C:\Users\abdel\Downloads\concentracionIA"

# Input and output
input_file = "youtube-senti-labelled-short(Sheet1).csv"
output_dir = "results_llms"

# Combine paths safely
DATA_PATH = os.path.join(base_path, input_file)
OUTPUT_DIR = os.path.join(base_path, output_dir)

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the CSV
df = pd.read_csv(DATA_PATH, encoding='latin1')

#= CONTROL DE COSTOS =======
N_SAMPLE = 30     # número de instancias de test para pruebas rápidas
MAX_CALLS = 50    # tope de llamadas a la API (ajuste según presupuesto)
API_CALLS_USED = 0

def check_quota():
    """Verifica y consume 1 crédito de llamada a la API."""
    global API_CALLS_USED
    if API_CALLS_USED >= MAX_CALLS:
        raise RuntimeError(
            f"Se alcanzó el límite de {MAX_CALLS} llamadas a la API. "
            f"Aumente MAX_CALLS o reduzca el dataset de prueba (N_SAMPLE)."
        )
    API_CALLS_USED += 1


# Candidatos de nombres de columna
TEXT_CANDIDATES = ["text","texto","review","content","sentence","tweet","document"]
LABEL_CANDIDATES = ["label","labels","etiqueta","etiquetas","sentiment","Sentiment","target","y"]

def load_dataset(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo en: {p}")

    if p.suffix.lower() in [".csv", ".tsv"]:
        sep = "," if p.suffix.lower()==".csv" else "\t"
        df = pd.read_csv(p, sep=sep)
    elif p.suffix.lower() in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif p.suffix.lower() in [".xls", ".xlsx"]:
        # Lee primera hoja por defecto; puede especificarse sheet_name si es necesario
        df = pd.read_excel(p, engine="openpyxl")
    else:
        # Intento por defecto: leer como CSV
        df = pd.read_csv(p)
    return df


df = load_dataset(DATA_PATH).copy()
print("Tamaño bruto:", len(df))


# Inferencia de columnas
TEXT_COL = next((c for c in TEXT_CANDIDATES if c in df.columns), None)
LABEL_COL = next((c for c in LABEL_CANDIDATES if c in df.columns), None)

if TEXT_COL is None or LABEL_COL is None:
    raise ValueError(
        f"No se detectaron columnas esperadas.\n"
        f"Candidatos texto: {TEXT_CANDIDATES}\n"
        f"Candidatos etiqueta: {LABEL_CANDIDATES}\n"
        f"Columnas disponibles: {list(df.columns)}"
    )

df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)
#display(df.head())