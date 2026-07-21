# Rag movie search

A retrieval-augmented-generation movie search engine. It embeds a corpus of ~34K movie plots, retrieves candidates for a natural-language query via vector search, then uses an LLM to rerank the shortlist and justify each pick with a plot-grounded, one-line explanation.

## Approach

The pipeline is a classic retrieve-then-rerank RAG setup, split into four
independent stages:

1. **Ingest** — download the raw dataset from Hugging Face and clean it into a flat `title` / `plot` / `genre` / `release_year` / `director` / `origin` table.
2. **Embed** — build a `"{title}\n\n{plot}"` text chunk per movie, embed it with a sentence-transformer, and store the vector plus metadata in a local Chroma collection.
3. **Retrieve** — embed the user's query with the same model and pull the nearest ~30 candidates from Chroma by cosine distance.
4. **Rerank + augment** — send the candidate shortlist to an LLM with a structured-output schema. The model selects and orders the top 10 and writes a one-line justification per pick. 

Vector search alone is fast but semantically shallow (it matches surface wording); the LLM rerank step adds judgment — it can weigh plot details, tone, and narrative structure the way a person skimming a shortlist would.

## Data

- **Source**: [`vishnupriyavr/wiki-movie-plots-with-summaries`](https://huggingface.co/datasets/vishnupriyavr/wiki-movie-plots-with-summaries) on Hugging Face — ~34k movies scraped from Wikipedia.
- **Fields kept**: `title`, `plot`, `release_year`, `genre`, `director`, `origin`.
- **Cleaning**: rows with a missing title or plot are dropped, and duplicate titles are deduplicated, leaving ~32,432 movies.
- Raw and cleaned data are stored under `data/raw/` and `data/processed/` (gitignored — regenerate them locally, see Setup below).

## Models

| Stage | Model | Why |
|---|---|---|
| Embedding | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Small, fast, well-suited to short plot-summary text; runs locally with no API cost. |
| Reranking + justification | `gemini-3.1-flash-lite` | Structured-output JSON generation to rank and justify candidates grounded in their plots. |
| Vector store | [ChromaDB](https://www.trychroma.com/) | Simple embedded vector DB, no separate service to run. |


## Setup

```bash
uv sync
uv run --env-file .env python -m ingestion.download_dataset
uv run --env-file .env python -m ingestion.load_data
uv run --env-file .env python -m embeddings.embed
```

Reranking calls Gemini (`GEMINI_API_KEY=your-key-here`), so add a key to a local `.env`.
Poster lookups call [TMDB](https://www.themoviedb.org/settings/api) (`TMDB_API_KEY=your-key-here`, free) — optional, the UI just skips posters without it.


## Usage

```bash
# Search, rerank, and justify (writes to data/generated/query_history.json)
uv run --env-file .env python -m scripts.query "a heist movie with a twist ending"

# Skip the LLM step and show raw vector-search hits (no API key needed)
uv run python -m scripts.query "a heist movie with a twist ending" --no-rerank

# Tune result count / candidate pool size
uv run --env-file .env python -m scripts.query "a war drama" -n 5 -c 20
```

## Example output

From an actual run, logged in `data/generated/query_history.json`:

**Query:** `a heist movie with a twist ending`

| Rank | Title | Justification |
|---|---|---|
| 1 | Hero Wanted | The protagonist orchestrates an elaborate bank heist to play the hero, but the plan goes horribly wrong when the robbery turns violent and results in unexpected bloodshed. |
| 2 | Redirected | A casino heist in London goes awry due to betrayal and global events, leaving the perpetrators stranded in an unfamiliar country where they must navigate a surreal series of violent misunderstandings. |
| 3 | Rajathandhiram | After jobless youth attempt a high-stakes heist, they find themselves caught in a dangerous web of double-crosses by shady characters forcing them into a much larger operation. |
| 4 | Villain | While planning an ambitious raid on a plastics factory, an East End gangster must deal with complex internal dynamics and violent complications that deviate from his standard criminal methodology. |
| 5 | The Last Shot | An FBI agent stages a fake film production as an elaborate sting operation to entrap mobsters, only to find the line between reality and the heist-like deception blurring as he becomes obsessed with the movie business. |

**Query:** `an epic adventure movie with a sad ending`

| Rank | Title | Justification |
|---|---|---|
| 1 | The Disappeared | This epic struggle for survival in the vast North Atlantic follows six men whose isolation and deteriorating mental states reflect the grim reality of their stranded situation. |
| 2 | Halunda Tavaru | This tragedy drama details the harsh riches-to-rags transition of its protagonists, culminating in a sad ending where both lead characters die. |
| 3 | Anumati | The story depicts the emotional and financial struggle of an old man trying to save his wife, ending in his tragic death despite the help of a friend. |
| 4 | Tony de Peltrie | This short film captures the melancholy journey of a piano player reminiscing over his past glory days, ending on a deeply sad, fading note. |
| 5 | Bangarada Jinke | This film features an epic-scale adventure involving a treasure hunt for a golden deer, set against a backdrop of rebirth and revenge. |


## Future improvements

- **Incorporate user/critic reviews into the data.** The dataset currently only has plot summaries — no sense of reception, tone, or audience reaction. Pulling in reviews (e.g. from a ratings/reviews dataset) would let retrieval and reranking account for how a movie is actually experienced, not just what happens in it — enabling queries like "a heist movie critics called overly slow" or ranking by acclaim within a genre.
- Add automated evaluation (e.g. a small labeled query set with expected relevant titles) to measure retrieval and reranking quality over time.
- Cache embeddings for repeated ingestion runs instead of recomputing the full collection from scratch.
- Support filtering by metadata (genre, release year range, origin) alongside the semantic query.
- Add a lightweight web UI over `scripts/query.py` instead of the CLI only.
