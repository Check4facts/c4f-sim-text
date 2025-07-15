from fastapi import FastAPI, Request
from TextFiltering import TextFiltering
from OllamaEmbeddings import OllamaEmbeddings
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import gc
import torch
from TextFiltering import TextFiltering
from pydantic import BaseModel

app = FastAPI()

# hf_model = SentenceTransformer("lighteternal/stsb-xlm-r-greek-transfer")
# ollama_handler = OllamaEmbeddings()


class SimTextRequest(BaseModel):
    text: str
    claim_embedding: list
    min_threshold: float = 0.3
    chunk_size: int = 1400


class EmbeddingRequest(BaseModel):
    text: str


@app.post("/sim_text_hf")
def sim_text_hf(request: SimTextRequest):
    handler = TextFiltering(hf=True)
    filtered_chunks = handler.get_sim_text_hf(
        text=request.text,
        claim_embedding=request.claim_embedding,
        min_threshold=request.min_threshold,
        chunk_size=request.chunk_size,
    )

    # Cleanup
    del handler
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {
        "filtered_chunks": [
            chunk.strip().replace("\n", " ") for chunk in filtered_chunks
        ]
    }


# @app.post("/sim_text_ollama")
# def sim_text_ollama(request: SimTextRequest):
#     handler = TextFiltering(hf=False)
#     filtered_chunks = handler.get_sim_text_ollama(
#         text=request.text,
#         claim_embedding=request.claim_embedding,
#         ollama_handler=ollama_handler,
#         min_threshold=request.min_threshold,
#         chunk_size=request.chunk_size,
#     )
#     return {
#         "filtered_chunks": [
#             chunk.strip().replace("\n", " ") for chunk in filtered_chunks
#         ]
#     }


@app.post("/claim_embedding_hf")
def claim_embedding_hf(request: EmbeddingRequest):
    handler = TextFiltering(hf=True)
    embedding = handler.get_embedding(request.text)
    return {"embedding": embedding}


# @app.post("/claim_embedding_ollama")
# def claim_embedding_ollama(request: EmbeddingRequest):
#     embedding = ollama_handler.compute_embedding(request.text)
#     return {"embedding": embedding.tolist()}
