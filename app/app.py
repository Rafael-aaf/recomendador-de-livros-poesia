import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.data_prep.user_data import load_user_csv
from src.run import SOURCE_LABELS, Recommender

st.set_page_config(page_title="Recomendador de livros de poesia", layout="wide")


@st.cache_resource(show_spinner="Carregando dataset e embeddings (só na primeira vez)...")
def get_recommender():
    return Recommender.load()


def format_score(value):
    return "-" if pd.isna(value) else f"{value:.3f}"


st.title("Recomendador de livros de poesia")

st.markdown(
    """
Envie sua biblioteca exportada do Goodreads e receba sugestões de livros de poesia! Este programa usa filtro colaborativo e busca semântica para as recomendações.
"""
)

st.header("1. Envie o seu CSV")
st.markdown(
    """
No site do Goodreads (versão para computador), abra **My Books**, clique em **Import and export** (menu lateral, seção Tools)
e depois em **Export Library**. O arquivo costuma se chamar `goodreads_library_export.csv`.
Envie esse arquivo abaixo. Ele é lido só durante a execução e não fica salvo.
"""
)
uploaded = st.file_uploader("Arquivo CSV do Goodreads", type="csv")

col_cf, col_sem = st.columns(2)
n_collaborative = col_cf.number_input("Livros do filtro colaborativo", min_value=0, max_value=50, value=10)
n_semantic = col_sem.number_input("Livros da busca semântica", min_value=0, max_value=50, value=5)

st.header("2. Recomendações")

if uploaded is None:
    st.info("Envie o CSV para gerar as recomendações.")
elif st.button("Gerar recomendações", type="primary"):
    recommender = get_recommender()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "library.csv"
        path.write_bytes(uploaded.getvalue())
        df_user = load_user_csv(path)

    try:
        result = recommender.recommend(df_user, int(n_collaborative), int(n_semantic))
    except ValueError as error:
        st.error(str(error))
    else:
        table = pd.DataFrame(
            {
                "Título": result["title"],
                "Autor": result["authors"],
                "Origem": result["source"].map(SOURCE_LABELS),
                "Score do filtro": result["cf_score"].map(format_score),
                "Score da busca": result["sem_score"].map(format_score),
            }
        )
        table.index = range(1, len(table) + 1)
        st.dataframe(table, height=(15 + 1) * 35 + 3)

        st.markdown(
            """
**Como ler a tabela**

- **Origem:** *filtro* são livros que leitores parecidos com você avaliaram bem; *busca* são livros com sinopse parecida com a dos livros que você avaliou bem.
- **Score do filtro:** soma da similaridade dos leitores parecidos que gostaram do livro.
- **Score da busca:** similaridade de cosseno entre a sinopse do livro e o seu perfil.
- Os dois scores não são comparáveis entre si.
"""
        )
