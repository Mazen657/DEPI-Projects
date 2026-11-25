from app.retrieval import NutritionRetriever

retriever = NutritionRetriever()

results = retriever.collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

for i in range(len(results["documents"])):
    print(f"\n{'='*50}")
    print(f"CHUNK {i+1}")

    print("\nMETADATA:")
    print(results["metadatas"][i])

    print("\nCONTENT:")
    print(results["documents"][i][:500])