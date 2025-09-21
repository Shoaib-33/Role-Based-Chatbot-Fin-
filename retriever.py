# retriever.py
import getpass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from users_db import users_db
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI


# -------------------------------
# Load Chroma DB
# -------------------------------
CHROMA_DIR = "chroma_db"
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name="company_docs"
)

# -------------------------------
# Login system
# -------------------------------
def login():
    username = input("👤 Username: ")
    password = getpass.getpass("Password: ")

    if username in users_db and users_db[username]["password"] == password:
        print(f"✅ Welcome {username}! Role: {users_db[username]['role']}")
        return users_db[username]["role"]
    else:
        print("Invalid credentials")
        return None

# -------------------------------
# Chat function with role filtering
# -------------------------------
def chat_with_bot(role):
    retriever = db.as_retriever(
        search_kwargs={"filter": {"role": role}, "k": 4}  # restrict to role
    )

    # Instead of:
# llm = ChatOpenAI(model="gpt-4o-mini")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",           # or another Gemini model you have access to
        temperature=0.7,              # optional: tune this
        google_api_key="" # or rely on env var
    )


    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are an assistant for {role} department.
        Use the following documents to answer:
        {context}

        Question: {query}
        """
        response = llm.invoke(prompt)
        print(f"Bot: {response.content}")


if __name__ == "__main__":
    role = login()
    if role:
        chat_with_bot(role)
