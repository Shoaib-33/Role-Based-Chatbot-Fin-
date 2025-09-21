# 🛡️ Role-Based RAG Chatbot  

A secure, role-aware chatbot powered by **LLMs + Hybrid Retrieval (BM25 + Dense Search + Reranking)** — with **Role-Based Access Control (RBAC)** for different departments.  

---

![Project Architecture](architecture.png)


## 🧩 Problem Background  

In large organizations, teams like **Finance, HR, Engineering, Marketing, and Executives** often struggle with fragmented document access. Employees waste time switching between tools, and sensitive information can leak across departments if access isn’t controlled.  

---

## 🧠 Solution Overview  

This chatbot solves that by combining:  

- **RAG (Retrieval-Augmented Generation)** for context-aware answers  
- **Role-Based Filtering** at both **BM25** and **Vector DB** levels  
- **Cross-Encoder Reranking** for more accurate responses  
- **Interactive Streamlit Chat UI** with login & role enforcement  

Every query is filtered by role, ensuring secure, role-relevant responses. If a user asks something outside their role, the chatbot simply says: **“I don’t know.”**  

---

## 🚀 Features  

- 🔐 **Role-Based Access Control (RBAC)**  
  - HR → employee salary, leaves, policies  
  - Finance → financial reports, reimbursements  
  - Engineering → system details, architecture  
  - Marketing → campaigns, customer insights  
  - Employees → general FAQs & policies  
  - Executives → unrestricted access  

- 🧠 **Hybrid Retrieval**  
  - **BM25** (keyword-based) + **Dense Vector Search**  
  - Results merged → **Cross-Encoder Reranker** → Final context  

- 💬 **Chat UI** (Streamlit)  
  - Login screen with roles  
  - Secure session persistence  
  - Interactive chat with AI assistant  

- 📊 **Structured Query Support**  
  - Detects SQL-type queries (e.g., “count employees”, “average salary”)  
  - Translates them into SQL queries against an **SQLite HR database**  

---

## 🛠 Tech Stack  

| Layer          | Tool/Library |
|----------------|-------------|
| Frontend       | Streamlit |
| Backend        | SQLite |
| Embeddings     | ChromaDB (Vector Store) |
| Sparse Search  | BM25Retriever |
| Dense Search   | Vector similarity search |
| Reranker       | CrossEncoder (MS MARCO MiniLM) |
| LLM            | Google Gemini (via LangChain) |
| Auth           | Simple username-password DB |

---

## 🧪 Sample Users  

```python
users_db = {
    "alice": {"password": "eng123", "role": "engineering"},
    "bob": {"password": "fin456", "role": "finance"},
    "carol": {"password": "hr789", "role": "hr"},
    "dave": {"password": "general123", "role": "general"},
}
}

```
## Project Structure
```
role-based-rag-chatbot/
├── app.py                  # Streamlit main app
├── retriever.py            # Vector DB setup + user DB
├── employees.db            # Auto-generated SQLite DB from HR CSV
├── resources/
│   └── data/
│       └── hr/hr_data.csv  # HR dataset (used for SQL queries)
├── requirements.txt
└── README.md
```


## Setup Instructions

Clone the repository
```
git clone https://github.com/your-username/role-based-rag-chatbot
cd role-based-rag-chatbot
```

Install dependencies
```
pip install -r requirements.txt
```

Run the app

```
streamlit run app.py
```

Login using one of the sample credentials above.



