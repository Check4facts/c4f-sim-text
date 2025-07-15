from OllamaEmbeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from sentence_transformers import util
import torch
import gc
import nltk

app = FastAPI()


class TextFiltering:

    def __init__(self, hf):
        self.hf = hf
        if hf:
            self.model = SentenceTransformer("lighteternal/stsb-xlm-r-greek-transfer")
            self.model = self.model.to("cuda")
            self.emb_dim = self.model.get_sentence_embedding_dimension()

    def chunk_text(self, text, chunk_size=1400, overlap_size=200):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                # print(current_chunk)
                # print("-----------------------------------")
                chunks.append(current_chunk)
                current_chunk = sentence[:overlap_size]
                current_chunk += sentence[overlap_size:]

        if current_chunk:
            # print(current_chunk)
            # print("-----------------------------------")
            chunks.append(current_chunk)

        return chunks

    def get_sim_text_ollama(
        self,
        text,
        claim_embedding,
        ollama_handler: OllamaEmbeddings,
        min_threshold=0.3,
        chunk_size=1400,
    ):
        if not text:
            return []

        filtered_results = []
        chunks = self.chunk_text(text, chunk_size)
        if not chunks:
            return []

        for chunk in chunks:
            try:
                chunk_embedding = ollama_handler.compute_embedding(chunk)
                similarity = ollama_handler.cosine_similarity(
                    claim_embedding, chunk_embedding
                )
            except Exception as e:
                print(f"Error computing embedding similarity for chunk: {e}")
                continue

            if similarity >= min_threshold:
                print(chunk)
                print()
                print(similarity)
                print("--------------------------------------------------")
                filtered_results.append(chunk)

        if len(filtered_results) == 0:
            return []

        return filtered_results

    def get_sim_text_hf(
        self,
        text,
        claim_embedding,
        min_threshold=0.3,
        chunk_size=1400,
        batch_size=16,
    ):
        if not text:
            return []

        device = self.model._target_device
        filtered_results = []
        chunks = self.chunk_text(text, chunk_size)
        if not chunks:
            return []

        if not torch.is_tensor(claim_embedding):
            claim_embedding = torch.tensor(claim_embedding, dtype=torch.float32)
        claim_embedding = claim_embedding.to(device)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            with torch.no_grad():
                chunk_embeddings = self.model.encode(
                    batch_chunks,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=device,
                )

                chunk_similarities = util.cos_sim(claim_embedding, chunk_embeddings)

                for chunk, similarity in zip(batch_chunks, chunk_similarities[0]):
                    if similarity >= min_threshold:
                        print(chunk)
                        print()
                        print(similarity.item())
                        print("--------------------------------------------------")
                        filtered_results.append(chunk)

            # Explicitly delete tensors and clear GPU cache after batch
            del chunk_embeddings
            del chunk_similarities
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Final cleanup after loop
        del claim_embedding
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        return filtered_results
