import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb


# ============================
# CONFIGURACIÓN DE RUTAS
# ============================

DATA_PROCESSED = Path("data/processed")
MODELS_DIR = DATA_PROCESSED / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ARTICLES_PATH = DATA_PROCESSED / "articulos_total.csv"

# Nueva ruta para almacenar ChromaDB
CHROMA_DIR = MODELS_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


# ============================
# Cargar artículos
# ============================

def load_articles():
    df = pd.read_csv(ARTICLES_PATH)

    df = df.dropna(subset=["texto"])
    df = df[df["texto"].str.strip() != ""]
    df["texto"] = df["texto"].astype(str)

    print(f"Artículos cargados: {len(df)}")
    return df


# ============================
# Construir ChromaDB con embeddings
# ============================

def build_chroma_index(df):
    print("Cargando modelo de embeddings…")
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    texts = df["texto"].tolist()
    embeddings = model.encode(texts, normalize_embeddings=True)

    # --- Nueva API de Chroma (correcta) ---
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Creamos o recuperamos colección
    collection = client.get_or_create_collection(
        name="normativa_transito",
        metadata={"hnsw:space": "cosine"}
    )

    ids = [str(i) for i in df.index]

    # Inserción segura
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=df.to_dict(orient="records"),
    )

    print(f"ChromaDB creada correctamente en: {CHROMA_DIR}")


# ============================
# MAIN
# ============================

def main():
    df = load_articles()
    build_chroma_index(df)
    print("\nProceso completado: ChromaDB lista.")


if __name__ == "__main__":
    main()
