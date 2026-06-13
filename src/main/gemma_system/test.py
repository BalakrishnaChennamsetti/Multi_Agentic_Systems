from langchain_ollama import OllamaEmbeddings

print("Creating embeddings object...")

emb = OllamaEmbeddings(model="nomic-embed-text")

print("Calling embed_query...")

vector = emb.embed_query("hello world")

print("Done")
print(len(vector))
