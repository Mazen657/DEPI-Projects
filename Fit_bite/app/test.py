from app.retrieval import NutritionRetriever

try:
    retriever = NutritionRetriever()

    print("Retriever loaded successfully!")
    print("Collection:", retriever.collection.name)
    print("Total chunks:", retriever.collection.count())

except Exception as e:
    print(f"Error: {e}")