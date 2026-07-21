"""Minimal Streamlit UI: search, rerank, and recommended movies as cards."""

import logging

import streamlit as st

# Silence a harmless Streamlit warning: its file-watcher walks every loaded
# module's __path__ to decide what to watch, and transformers' lazy-loaded
# zoedepth submodule (unused here) raises when torchvision isn't installed.
# Must be set after streamlit's own logging setup runs, hence the ordering.
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

from embeddings.embed import get_collection, get_model  # noqa: E402
from scripts.history import append_history  # noqa: E402
from scripts.posters import fetch_poster  # noqa: E402
from scripts.query import search_and_rerank  # noqa: E402

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
                for movie in ranking.rankings:
                    with st.container(border=True):
                        poster_col, info_col = st.columns([1, 3])
                        with poster_col:
                            poster_url = fetch_poster(movie.title)
                            if poster_url:
                                st.image(poster_url)
                            else:
                                st.markdown("🎞️\n\n*No poster*")
                        with info_col:
                            st.write(f"{movie.rank}. {movie.title}")
                            st.write(movie.justification)
