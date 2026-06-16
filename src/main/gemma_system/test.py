from collections import Counter

from langchain_community.vectorstores import FAISS

from .vector_db.faiss_db import vector_db

_, embeddings = vector_db()
vector_store = FAISS.load_local(
    "src/main/gemma_system/vector_db/vector_store",
    embeddings,
    allow_dangerous_deserialization=True,
)

sources = []

for doc_id in vector_store.docstore._dict:
    doc = vector_store.docstore._dict[doc_id]
    sources.append(doc.metadata.get("source", "Unknown"))

for source in sorted(set(sources)):
    print(source)

for doc in vector_store.docstore._dict.values():
    print(doc.metadata)
    break

print(vector_store.index.ntotal)


"Find the next prime number after 2^{136,279,841}-1"
