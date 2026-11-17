import chromadb
from sentence_transformers import SentenceTransformer
import ollama


chromadb_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "local_base"

collection = chromadb_client.get_or_create_collection(name=collection_name)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    (
        "doc_1",
        (
            "Architect project Opa: mobile app on React Native"
            "(Expo) with navigation"
            "validation and work with gallary"
            "React Hook Form"
        ),
        "doc_2",
        (
            "Architect project Nova: e-commerce web platform on Next.js "
            "with SSR, Tailwind CSS, "
            "Redux Toolkit state management "
            "and Stripe payment integration"
        ),

        "doc_3",
        (
            "Architect project Atlas: RESTful backend API on Node.js "
            "(Express) with TypeScript, PostgreSQL database, "
            "JWT authentication "
            "and Docker containerization"
        ),

        "doc_4",
        (
            "Architect project Pulse: dashboard analytics app on React "
            "with Chart.js visualization, Material-UI components, "
            "real-time WebSockets "
            "and internationalization support"
        ),

        "doc_5",
        (
            "Architect project Quantum: machine learning pipeline in Python "
            "with scikit-learn for predictive modeling, Pandas for data processing, "
            "FastAPI service "
            "and automated pytest testing"
        ),
    )
]

if collection.count() == 0:
    ids = [item[0] for item in documents]
    texts = [item[1] for item in documents]

    embeddings = embedding_model.encode(texts).tolist()

    collection.add(documents=texts, embeddings=embeddings, ids= ids)



user_query = "Which docs with backend?"

query_embendding = embedding_model.encode([user_query]).tolist()

result = collection.query(query_embeddings=query_embendding, n_results=2)

retieved_chunks = result["documents"][0]

for i, chunk in enumerate(retieved_chunks):
    print(f"[{i+1}] {chunk}")

context = "\n".join(retieved_chunks)

promt = f"""Give only the context provided below to answer the question.
    If the answer is not in the context say: "I do not have information in my base

    Context:
    {context}

    User query:
    {user_query}
"""

response = ollama.chat(
    model = "llama3",
    messages=[
        {
            "role":"system",
            "content":(
                "You - cleaver corparation assistant whose answeer on base docs"
            ),
            
        },
        {
            "role":"user", "content":promt
        }
    ]
)

print("="*40)
print("Answer: ")
print("="*40)
print(response["message"]["content"])
