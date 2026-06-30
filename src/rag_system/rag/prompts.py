from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a retrieval augmented assistant.

Answer ONLY from the provided context.

Rules:
1. Never use outside knowledge.
2. Never invent information.
3. If answer is unavailable say:
'I cannot find the answer in the provided documents.'
4. Cite only provided information.

Context:
{context}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ]
)

CONVERSATIONAL_SYSTEM_PROMPT = """You are a retrieval assistant.

Conversation History:
{history}

Retrieved Context:
{context}

Answer ONLY from the context.

If unavailable answer:
I cannot find the answer in the provided documents."""

CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONVERSATIONAL_SYSTEM_PROMPT),
        ("human", "{question}")
    ]
)
