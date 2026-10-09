import os
import time
import hashlib
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

load_dotenv()

CAREER_URLS = [
    "https://roadmap.sh/ai-engineer",
    "https://roadmap.sh/data-analyst",
    "https://www.python.org/about/gettingstarted/",
]


PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "careerpath-rag")
PINECONE_NAMESPACE = "career-guidance"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536 

def demo_chatbot():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found. Create a .env file with: "
            "OPENAI_API_KEY=your_key_here"
        )
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=api_key
    )

def get_embeddings():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_pinecone_index():
    """Connect to the Pinecone index, creating it on first run."""
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        raise ValueError(
            "PINECONE_API_KEY not found. Add it to your .env file."
        )

    pc = Pinecone(api_key=api_key)

    if not pc.has_index(PINECONE_INDEX_NAME):
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )

        while not pc.describe_index(
            PINECONE_INDEX_NAME
        ).status["ready"]:
            time.sleep(1)

    # IMPORTANT: Return the index whether it already existed or was created.
    return pc.Index(PINECONE_INDEX_NAME)
            


def namespace_vector_count(index):
    """Number of vectors already stored in our namespace (0 if none)."""
    stats = index.describe_index_stats()
    namespaces = stats.namespaces or {}
    ns = namespaces.get(PINECONE_NAMESPACE)
    return ns.vector_count if ns else 0

def make_chunk_ids(chunks):
    """
    Deterministic IDs (source URL + chunk position), so re-indexing
    overwrites existing records instead of creating duplicates.
    """
    ids, counters = [], {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        i = counters.get(source, 0)
        counters[source] = i + 1
        ids.append(hashlib.md5(f"{source}::{i}".encode()).hexdigest())
    return ids

def index_website(urls=CAREER_URLS, reset=True):
    """
    Scrape the URLs, chunk them, and upsert into Pinecone.
    reset=True clears the namespace first so removed/shortened
    pages don't leave stale chunks behind.
    Returns the number of chunks upserted.
    """
    docs = WebBaseLoader(urls).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(docs)

    index = get_pinecone_index()
    
    if reset:
            try:
                index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
            except Exception:
                pass  # namespace doesn't exist yet on the very first run
    
    vector_store = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=PINECONE_NAMESPACE,
        )
    vector_store.add_documents(chunks, ids=make_chunk_ids(chunks), batch_size=100)
    
    return len(chunks)

def build_retriever(urls=CAREER_URLS, k=4):
    """
    Connect to Pinecone. Only scrapes + embeds if the namespace is empty,
    so app restarts no longer re-index the whole website.
    """
    index = get_pinecone_index()

    if namespace_vector_count(index) == 0:
        index_website(urls, reset=False)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=get_embeddings(),
        namespace=PINECONE_NAMESPACE,
    )
    return vector_store.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)



examples = [
    {
        "input": "What does a data analyst do?",
        "output": "A data analyst collects, cleans, and analyzes data "
                  "to help organizations make informed decisions.",
    },
    {
        "input": "How can I start learning AI?",
        "output": "Start with Python fundamentals, basic mathematics, "
                  "machine learning concepts, and hands-on projects. "
                  "Then explore generative AI and RAG applications.",
    },
]


example_prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), ("ai", "{output}")]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are CareerPath AI, a helpful and friendly career guidance "
            "assistant. Answer using the career website context below. "
            "If the answer is not available in the context, say you are "
            "not sure and suggest consulting official learning or career "
            "resources. Do not invent facts. Give clear, beginner-friendly "
            "explanations.\n\n"
            "Career website context:\n{context}",
        ),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def demo_memory():
    return InMemoryChatMessageHistory()


def demo_conversation(input_text, memory, retriever):
    llm = demo_chatbot()

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | CHAT_PROMPT
        | llm
    )

    chat_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: memory,
        input_messages_key="input",
        history_messages_key="history",
    )

    try:
        result = chat_with_history.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": "bepec-session"}},
        )
        chat_reply = result.content
    except Exception as e:
        chat_reply = f"Sorry, something went wrong: {str(e)}"

    return chat_reply, memory


    
            