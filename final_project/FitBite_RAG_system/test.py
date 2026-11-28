import chromadb
import os

CHROMA_PATH = "/Users/minahany/Desktop/Fit_bite/chroma_db"

print("Database exists:", os.path.exists(CHROMA_PATH))
print("Database files:", os.listdir(CHROMA_PATH))

client = chromadb.PersistentClient(path=CHROMA_PATH)

collections = client.list_collections()

print("\nAvailable Collections:")
print("=" * 40)

if not collections:
    print("DATABASE IS EMPTY!")
else:
    for col in collections:
        print(f"Name: {col.name}")
        print(f"Count: {col.count()}")
