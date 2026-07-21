"""Minimal Streamlit UI: search, rerank, and recommended movies as cards."""

from concurrent.futures import ThreadPoolExecutor

import streamlit as st

from embeddings.embed import get_collection, get_model
from scripts.history import append_history
from scripts.posters import fetch_poster
from scripts.query import search_and_rerank

st.set_page_config(page_title="Movie RAG Search", page_icon="🎬")


@st.cache_resource
def load_model():
    return get_model()


@st.cache_resource
def load_collection():
    return get_collection()


st.title("🎬 Movie Search")

query = st.text_input(
    "What are you in the mood for?",
    placeholder="a heist movie with a twist ending",
)

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Enter a search query first.")
    else:
        with st.spinner("Searching and ranking..."):
            try:
                ranking = search_and_rerank(
                    query, model=load_model(), collection=load_collection()
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
            else:
                append_history(query, ranking.model_dump())

                titles = [movie.title for movie in ranking.rankings]
                poster_urls = []
                if titles:
                    with ThreadPoolExecutor(max_workers=len(titles)) as executor:
                        poster_urls = list(executor.map(fetch_poster, titles))

                for movie, poster_url in zip(ranking.rankings, poster_urls):
                    with st.container(border=True):
                        poster_col, info_col = st.columns([1, 3])
                        with poster_col:
                            if poster_url:
                                st.image(poster_url)
                            else:
                                st.markdown("🎞️\n\n*No poster*")
                        with info_col:
                            st.write(f"{movie.rank}. {movie.title}")
                            st.write(movie.justification)
