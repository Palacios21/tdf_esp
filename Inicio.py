import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.title("🔍 Escribe un texto en Español")

# Documentos de ejemplo
default_docs = """El Susurro del Viento
Bajo el palio del cielo azul,
el viento danza con suavidad,
acariciando las hojas con luz,
en un abrazo de pura libertad.
Las flores despiertan con el sol,
bañadas en rocío y esplendor,
mientras el río en su caracol,
murmura versos de paz y amor."""

# Stemmer en español
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    # Minúsculas
    text = text.lower()
    # Solo letras españolas y espacios
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    # Tokenizar
    tokens = [t for t in text.split() if len(t) > 1]
    # Aplicar stemming
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# Layout en dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("❓ Escribe tu pregunta:", "¿Qué elemento de la naturaleza es el que "danza con suavidad" y "acaricia las hojas" en la primera estrofa del poema?")

with col2:
    st.markdown("### 💡 Preguntas sugeridas:")
    
    # NUEVAS preguntas optimizadas para mayor similitud
    if st.button("¿De qué color es el cielo??", use_container_width=True):
        st.session_state.question = "¿De qué color es el cielo?"
        st.rerun()
    
    if st.button("¿Qué sale a despertar a las flores?", use_container_width=True):
        st.session_state.question = "¿Qué sale a despertar a las flores?"
        st.rerun()
        
    if st.button("¿Qué es lo que danza suavemente?", use_container_width=True):
        st.session_state.question = "¿Qué es lo que danza suavemente?"
        st.rerun()
        
    if st.button("¿En qué lugar murmura el río?", use_container_width=True):
        st.session_state.question = "¿En qué lugar murmura el río?"
        st.rerun()
        
    if st.button("¿De qué están bañadas las flores al despertar?", use_container_width=True):
        st.session_state.question = "¿De qué están bañadas las flores al despertar?"
        st.rerun()

# Actualizar pregunta si se seleccionó una sugerida
if 'question' in st.session_state:
    question = st.session_state.question

if st.button("🔍 Analizar", type="primary"):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        # Crear vectorizador TF-IDF
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1  # Incluir todas las palabras
        )
        
        # Ajustar con documentos
        X = vectorizer.fit_transform(documents)
        
        # Mostrar matriz TF-IDF
        st.markdown("### 📊 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # Calcular similitud con la pregunta
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        
        # Encontrar mejor respuesta
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # Mostrar respuesta
        st.markdown("### 🎯 Respuesta")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:  # Umbral muy bajo
            st.success(f"**Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"**Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")
