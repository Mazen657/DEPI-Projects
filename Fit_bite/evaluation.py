import chromadb
import pandas as pd
import os

# ==============================
# Configuration
# ==============================

CHROMA_PATH = "/Users/minahany/Desktop/Fit_bite/chroma_db"
COLLECTION_NAME = "nutrition_knowledge_v2"

DATASET_PATH = "eval_dataset-2.csv"

K = 5

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv(DATASET_PATH)

# ==============================
# Connect to ChromaDB
# ==============================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print(f"Collection loaded: {collection.name}")
print(f"Total chunks: {collection.count()}")
print(f"Evaluation questions: {len(df)}")


# ==============================
# Helper Functions
# ==============================

def normalize_text(value):
    return str(value).strip().lower()


def is_relevant(metadata, row):

    retrieved_source = os.path.basename(
        str(metadata.get("source", ""))
    )

    expected_source = os.path.basename(
        str(row["Expected_PDF"])
    )

    if normalize_text(retrieved_source) != normalize_text(expected_source):
        return 0

    expected_page = normalize_text(row["Expected_Page"])

    # Some documents do not have page numbers
    if expected_page == "not_applicable":
        return 1

    retrieved_page = normalize_text(
        metadata.get("page", "")
    )

    try:
        return int(float(retrieved_page) == float(expected_page))
    except (ValueError, TypeError):
        return 0


# ==============================
# Evaluation
# ==============================

evaluation_results = []

for _, row in df.iterrows():

    query = row["Question"]

    results = collection.query(
        query_texts=[query],
        n_results=K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    relevance = []

    for metadata in metadatas:
        relevance.append(
            is_relevant(metadata, row)
        )

    # Precision@K
    precision = (
        sum(relevance) / len(relevance)
        if relevance else 0
    )

    # Recall@K
    # One expected source/page per question
    recall = 1 if sum(relevance) > 0 else 0

    # Reciprocal Rank
    reciprocal_rank = 0

    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            reciprocal_rank = 1 / rank
            break

    # Save results
    evaluation_results.append({

        "ID": row["ID"],

        "Question": query,

        "Question_Type": row["Question_Type"],

        "Precision@5": precision,

        "Recall@5": recall,

        "MRR@5": reciprocal_rank,

        "Hit": int(sum(relevance) > 0),

        "Retrieved_Sources": [
            {
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "distance": distance,
                "relevant": relevant
            }
            for metadata, distance, relevant
            in zip(metadatas, distances, relevance)
        ]
    })


# ==============================
# Calculate Final Metrics
# ==============================

results_df = pd.DataFrame(evaluation_results)

print("\n" + "=" * 50)
print("RETRIEVAL EVALUATION RESULTS")
print("=" * 50)

print(f"Precision@5: {results_df['Precision@5'].mean():.4f}")

print(f"Recall@5:    {results_df['Recall@5'].mean():.4f}")

print(f"MRR@5:       {results_df['MRR@5'].mean():.4f}")

print(f"Hit Rate@5:  {results_df['Hit'].mean():.4f}")


# ==============================
# Export Results
# ==============================

results_df.to_csv(
    "retrieval_evaluation_results.csv",
    index=False
)

print("\nResults saved to retrieval_evaluation_results.csv")