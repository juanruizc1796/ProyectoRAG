import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb


# ============================
# 0. CONFIGURACIÓN GENERAL
# ============================

CHROMA_PATH = "data/processed/models/chroma_db"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"


# ============================
# 1. CARGA DE RECURSOS
# ============================

@st.cache_resource
def load_resources():

    # --- Encoder ---
    encoder = SentenceTransformer(EMBED_MODEL_NAME)

    # --- Conectar a ChromaDB (API moderna) ---
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # La colección que creaste se llama EXACTAMENTE "normativa"
    collection = client.get_collection("normativa")

    # --- Groq ---
    groq_api_key = os.getenv("GROQ_API_KEY", st.secrets.get("GROQ_API_KEY"))
    if groq_api_key is None:
        raise ValueError("No se encontró GROQ_API_KEY en secrets o variables de entorno.")

    groq_client = Groq(api_key=groq_api_key)

    return encoder, collection, groq_client


encoder, collection, groq_client = load_resources()


# ============================
# 2. RAG
# ============================

def buscar_articulos(query, top_k=4):

    query_emb = encoder.encode([query], normalize_embeddings=True).tolist()

    resultados = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        include=["metadatas", "documents"]
    )

    docs = resultados["documents"][0]
    metas = resultados["metadatas"][0]

    contexto = ""
    for meta, doc in zip(metas, docs):
        contexto += (
            f"ARTÍCULO {meta.get('id_articulo', 's/d')} - {meta.get('titulo', '')}\n"
            f"{doc}\n\n"
        )

    return contexto


def call_groq(prompt: str) -> str:

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un asistente experto en normativa de tránsito en Colombia. "
                    "Responde SOLO con el contexto proporcionado. "
                    "Si no hay información relevante, responde exactamente: "
                    "'No encontré información relevante en la normativa cargada.'"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return completion.choices[0].message.content


def rag_responder(query):
    contexto = buscar_articulos(query)

    prompt = f"""
PREGUNTA:
{query}

ARTÍCULOS RELEVANTES (fragmentos):
{contexto}

Instrucciones:
- Usa lenguaje claro.
- Explica qué artículos se usaron.
- Si no se encuentra información: dilo explícitamente.
"""

    return call_groq(prompt)


# ============================
# 3. INTERFAZ STREAMLIT
# ============================

st.set_page_config(page_title="Asistente Legal de Tránsito", page_icon="🚦")

st.title("🚦 Asistente Legal de Tránsito • Colombia")
st.write("Consulta normativa oficial usando un sistema RAG optimizado con **ChromaDB + Groq**.")

st.subheader("Preguntas sugeridas:")

col1, col2, col3 = st.columns(3)

q1 = "¿Qué debo hacer si me imponen un comparendo?"
q2 = "¿Cuánto tiempo tengo para pagar una multa?"
q3 = "¿En qué casos inmovilizan mi vehículo?"

if col1.button(q1):
    st.session_state["pregunta"] = q1
if col2.button(q2):
    st.session_state["pregunta"] = q2
if col3.button(q3):
    st.session_state["pregunta"] = q3

pregunta = st.text_input(
    "Pregunta sobre tránsito:",
    value=st.session_state.get("pregunta", "")
)

if st.button("Consultar"):
    if pregunta.strip() == "":
        st.warning("Por favor ingresa una pregunta.")
    else:
        with st.spinner("Generando respuesta…"):
            respuesta = rag_responder(pregunta)

        st.subheader("Respuesta")
        st.write(respuesta)
