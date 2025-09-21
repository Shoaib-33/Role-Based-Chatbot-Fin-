# app.py
import streamlit as st
import pandas as pd
import sqlite3
from retriever import db, users_db
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

# -------------------------------
# Initialize SQLite DB (HR CSV)
# -------------------------------
@st.cache_resource
def init_db():
    df = pd.read_csv("resources/data/hr/hr_data.csv")
    conn = sqlite3.connect("employees.db", check_same_thread=False)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    return conn

conn = init_db()

# -------------------------------
# Initialize Reranker (Dense CrossEncoder)
# -------------------------------
@st.cache_resource
def init_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = init_reranker()

def rerank(query, docs, top_k=5):
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:top_k]]

# -------------------------------
# Initialize BM25 Retrievers per role
# -------------------------------
@st.cache_resource
def init_bm25_by_role():
    stored = db.get()  
    role_docs = {}

    for text, meta in zip(stored["documents"], stored["metadatas"]):
        role = meta.get("role", "general")
        if role not in role_docs:
            role_docs[role] = []
        role_docs[role].append(Document(page_content=text, metadata=meta))

    bm25_dict = {}
    for role, docs in role_docs.items():
        bm25_dict[role] = BM25Retriever.from_documents(docs)

    return bm25_dict

bm25_by_role = init_bm25_by_role()

# -------------------------------
# Hybrid Retrieval (Role-based BM25 + Dense)
# -------------------------------
def hybrid_retrieve(query, role, top_k=50):
    # Role-specific BM25
    bm25_docs = []
    if role in bm25_by_role:
        bm25_docs = bm25_by_role[role].get_relevant_documents(query)

    # Role-specific Dense
    dense_retriever = db.as_retriever(search_kwargs={"filter": {"role": role}, "k": top_k})
    dense_docs = dense_retriever.get_relevant_documents(query)

    # Merge results
    combined_docs = {d.page_content: d for d in bm25_docs + dense_docs}
    return list(combined_docs.values())

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("💬 Company Chatbot with Role-Based Access")

# Login
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if username in users_db and users_db[username]["password"] == password:
        role = users_db[username]["role"]
        st.session_state["role"] = role
        st.success(f"Logged in as {role}")
    else:
        st.error("Invalid login")

# Chat
if "role" in st.session_state:
    role = st.session_state["role"]
    query = st.text_input("Ask something...")

    if st.button("Send"):
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=""
        )

        # -------------------------------
        # Step 1: Detect structured queries
        # -------------------------------
        structured_keywords = ["greater than", "less than", "count", "list all", "average", "sum", "how many"]

        if any(kw in query.lower() for kw in structured_keywords):
            # ✅ Only HR can access SQL queries
            if role.lower() == "hr":
                code_prompt = f"""
                You are a SQL assistant. Translate the user query into an SQL statement 
                for the SQLite table `employees`. Only return SQL code, no explanations.
                Table columns: {pd.read_csv("resources/data/hr/hr_data.csv").columns.tolist()}
                User query: {query}
                """
                sql_resp = llm.invoke(code_prompt)
                sql_query = sql_resp.content.strip("```sql").strip("```").strip()

                try:
                    cursor = conn.cursor()
                    cursor.execute(sql_query)
                    rows = cursor.fetchall()
                    col_names = [desc[0] for desc in cursor.description]

                    result_df = pd.DataFrame(rows, columns=col_names)
                    st.dataframe(result_df)

                except Exception as e:
                    st.error(f"SQL Execution error: {e}")
                    st.text(f"Tried query: {sql_query}")
            else:
                st.warning("⚠️ You do not have permission to run structured queries. Please ask a general question instead.")
        
        else:
            # -------------------------------
            # Default RAG flow (non-structured)
            # -------------------------------
            retriever = db.as_retriever(search_kwargs={"filter": {"role": role}, "k": 4})
            docs = retriever.get_relevant_documents(query)

            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            You are an AI assistant at FinSolve Technologies. The user has the role: {role}.Dont answer outside of role.
            Use the context below to answer their question clearly.

            Instruction:
            1) If the context does not contain relevant information, respond with "I don't have that information."
            2) If the question is outside your role, respond with "I'm not authorized to answer that."
            3) Always keep answers concise and to the point.
            4) Dont make up answers and dont answer outside of your knowledge of RAG
            
            Context:
            {context}

            Question: {query}
            """
            response = llm.invoke(prompt)
            st.write("🤖 " + response.content)
