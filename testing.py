import requests

# Claim text
claim_text = "The company aims to achieve net zero emissions by 2040."

# Test Hugging Face embedding generation
response_hf = requests.post(
    "http://127.0.0.1:8000/claim_embedding_hf", json={"text": claim_text}
)
hf_embedding = response_hf.json()["embedding"]

print("Hugging Face Claim Embedding:")
print(hf_embedding[:10])
print("--------------------------------------------")

# # Test Ollama embedding generation
# response_ollama = requests.post(
#     "http://127.0.0.1:8000/claim_embedding_ollama", json={"text": claim_text}
# )
# ollama_embedding = response_ollama.json()["embedding"]

# print("Ollama Claim Embedding:")
# print(ollama_embedding[:10])
# print("--------------------------------------------")

# Test HF similarity filtering
supporting_text = """
GreenFuture Corp has announced an ambitious goal to reach net zero emissions by the year 2040.
The company plans to invest heavily in renewable energy and carbon capture technologies.
"""

response_filter_hf = requests.post(
    "http://127.0.0.1:8000/sim_text_hf",
    json={
        "text": supporting_text,
        "claim_embedding": hf_embedding,
        "min_threshold": 0.3,
        "chunk_size": 1400,
    },
)

print("Filtered Chunks (HF):")
print(response_filter_hf.json())
print("--------------------------------------------")

# # Test Ollama similarity filtering
# response_filter_ollama = requests.post(
#     "http://127.0.0.1:8000/sim_text_ollama",
#     json={
#         "text": supporting_text,
#         "claim_embedding": ollama_embedding,
#         "min_threshold": 0.3,
#         "chunk_size": 1400,
#     },
# )

# print("Filtered Chunks (Ollama):")
# print(response_filter_ollama.json())
# print("--------------------------------------------")
