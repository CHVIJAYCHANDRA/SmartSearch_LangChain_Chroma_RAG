# SmartSearch: Conversational RAG over Internal Documents

Single-turn RAG breaks the moment a user asks a follow-up. "How long is its
warranty?" is meaningless to a vector retriever — the pronoun carries all the
information, and it isn't in the query.

This repo demonstrates the fix and measures the cost of getting it wrong. A
dedicated LLM call rewrites each follow-up into a standalone question *before*
retrieval, so the vector search sees "how long is the ND-200 warranty?" instead
of a dangling reference.

It also fixes the mistake that quietly makes most Streamlit RAG demos expensive:
Streamlit re-runs the whole script on every interaction, so an uncached
`Chroma.from_documents` at module level re-embeds the entire corpus through the
paid embeddings API on **every single question**.

---

## Two scripts, one point each

| Script | Demonstrates |
|---|---|
| `rag_demo.py` | Baseline retrieval Q&A. CLI, handles `.txt` / `.md` / `.pdf`, persistent index, configurable `k`. |
| `historyaware_rag_demo.py` | Multi-turn chat with query rewriting before retrieval. Streamlit, cached index. |

The baseline exists as a control. It is what the conversational version is
measured *against* — not a separate feature.

---

## Pipeline

question ──► [rewrite w/ chat history] ──► retrieve top-k ──► answer w/ context
                     ▲                                              │
                     └──────────── chat history ◄───────────────────┘

Stage 1 (`create_history_aware_retriever`) is a separate LLM call whose prompt
explicitly forbids answering:

Resolve pronouns and implicit references to the entities discussed earlier.
If the question is already standalone, return it unchanged.
Do NOT answer the question. Return only the rewritten question.

Stage 2 answers from retrieved context only, and is instructed to refuse rather
than fall back on parametric knowledge:

Answer using only the context below. If the context does not contain the answer,
say you do not know - do not guess, and do not use outside knowledge.

Both stages run at `temperature=0`, so repeated runs are comparable — a
prerequisite for evaluating retrieval at all.

---

## Index caching

The corpus is embedded once per `(file content, chunk_size, chunk_overlap)`
combination. That tuple is hashed into the Chroma collection name:
python
def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"{path.stem}-{CHUNK_SIZE}-{CHUNK_OVERLAP}-{digest}"

A plain `persist_directory` cache is a trap: edit the corpus or change chunking,
and you keep querying a stale index while believing you tested the change.
Fingerprinting makes cache invalidation automatic, and makes chunking sweeps
safe because each configuration gets its own collection.

In Streamlit, `@st.cache_resource` holds the store and chain across reruns.
Both cached functions take only hashable primitives — the embedding and LLM
objects are constructed inside, since passing them as arguments would defeat
the cache.

Measured on `product-data.txt` (<fill in> chunks):

| | Setup | Per query |
|---|---|---|
| First build (cold) | <fill in>s | <fill in>s |
| Cached (warm) | <fill in>s | <fill in>s |

Reproduce: run any question twice and compare the sidebar status line.

---

## Setup
bash
git clone https://github.com/CHVIJAYCHANDRA/SmartSearch_LangChain_Chroma_RAG
cd SmartSearch_LangChain_Chroma_RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add your OPENAI_API_KEY

### Baseline, CLI
bash
python rag_demo.py                                       # interactive REPL
python rag_demo.py --question "Which gateway supports OPC-UA?" --show-context
python rag_demo.py --source paper.pdf --k 6              # any .txt/.md/.pdf
python rag_demo.py --rebuild                             # force reindex

### Conversational, Streamlit
bash
streamlit run historyaware_rag_demo.py

`product-data.txt` — a synthetic internal product knowledge base — is committed,
so both scripts run on a fresh clone with no extra files.

---

## Why this corpus

`product-data.txt` is written so retrieval actually has to discriminate rather
than keyword-match:

- **Two similar products** (ND-100 / ND-200) with overlapping attributes
- **Facts split across sections** — "which supports OPC-UA?" needs both the
  product and licensing sections
- **A stated absence** ("the ND-100 does not include onboard TPM") to test
  negative retrieval
- **Unambiguous numbers**, so ground truth is checkable
- **~3.4k characters** → roughly 5 chunks at 1000/200, so `k=4` is a real
  choice rather than "return the whole document"

### Behaviour worth checking

| Question | Expected | Tests |
|---|---|---|
| Which gateway supports OPC-UA? | ND-200 only | cross-section retrieval |
| What is the ND-100's throughput? | 800 msg/s | single fact |
| Does the ND-100 have a TPM? | No | negative fact |
| Who is the CEO? | *refuses* | grounding constraint |

Multi-turn, in order:

Tell me about the ND-200.
How long is its warranty?          # "its" resolves only via rewriting
Does the cheaper one have a TPM?   # "cheaper one" -> ND-100

Turn 2 is the reason this repo exists. Enable **Show retrieved chunks** to
confirm the ND-200 warranty chunk is what came back.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | required |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | generation and rewriting |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | embeddings |
| `CHROMA_DIR` | `./chroma_db` | persisted index location |
| `SOURCE_FILE` | `product-data.txt` | corpus for the Streamlit app |

Chunking (`1000` / `200`) and `k` (`4`) are exposed as constants and CLI/UI
controls. These values are LangChain defaults carried over deliberately as a
starting point; they have **not** been tuned against a labelled question set.

---

## Stack

LangChain 1.0 (`langchain-classic` for the retrieval chains) · Chroma ·
OpenAI embeddings + chat · Streamlit · pypdf · python-dotenv

Retrieval chains moved to `langchain-classic` in LangChain 1.0. Imports here
target that layout rather than the deprecated `langchain.chains` path.

---

## Limitations

- Chunk size, overlap and `k` are untuned — no ablation has been run, and no
  retrieval metric (recall@k, hit rate) is claimed.
- Answer quality is unmeasured. The behaviour table above is a manual smoke
  test, not an evaluation.
- Rewriting adds a second LLM call per turn, roughly doubling per-question token
  cost. That tradeoff is unquantified against the accuracy it buys.
- Single global session in the Streamlit app; concurrent users share history.
- Text-based PDFs only — no OCR.

## Planned

- Labelled question set over `product-data.txt` with retrieval hit-rate
- Ablation: naive vs. history-aware on follow-up questions specifically
- Chunk size / `k` sweep against retrieval accuracy
- Cost per answered question, rewriting on vs. off
