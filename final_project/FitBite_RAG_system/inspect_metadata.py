
import chromadb
from collections import Counter

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="nutrition_knowledge"
)

data = collection.get(
    limit=10,
    include=["metadatas", "documents"]
)

print("Total chunks:", collection.count())

print("\nMETADATA EXAMPLES")
print("=" * 60)

for i, metadata in enumerate(data["metadatas"]):
    print(f"\nChunk {i + 1}")
    print(metadata)

print("\nMETADATA KEYS")
print("=" * 60)

key_counts = Counter()

all_data = collection.get(
    include=["metadatas"]
)

for metadata in all_data["metadatas"]:
    key_counts.update(metadata.keys())

for key, count in key_counts.items():
    print(f"{key}: {count}")
