from OllamaEmbeddings import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import torch
import gc
import nltk


def log_gpu_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    print(
        f"{prefix} GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB"
    )


class TextFiltering:

    model = None

    def __init__(self, hf):
        self.hf = hf
        if hf:
            if TextFiltering.model is None:
                model = SentenceTransformer("lighteternal/stsb-xlm-r-greek-transfer")
                TextFiltering.model = model.to("cuda")
            self.model = TextFiltering.model
            self.emb_dim = self.model.get_sentence_embedding_dimension()

    def chunk_text(self, text, chunk_size=1400, overlap_size=200):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence[:overlap_size] + sentence[overlap_size:]
        if current_chunk:
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

        log_gpu_memory("Start")

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

            # Cleanup for each batch
            del chunk_embeddings, chunk_similarities
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            log_gpu_memory(f"After batch {i // batch_size + 1}")

        # Final cleanup
        del claim_embedding
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        log_gpu_memory("End")

        return filtered_results

    def get_embedding(self, text):
        result = self.model.encode(text)
        return result.tolist()
