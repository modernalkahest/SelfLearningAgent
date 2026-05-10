from dotenv import load_dotenv
import os
import hashlib
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_tavily import (
    TavilySearch,
    TavilyExtract
)
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from langchain.chat_models import init_chat_model
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

load_dotenv()
MODEL = "gpt-5.4"
MAX_ITERATIONS = 6

FAISS_INDEX_PATH = "faiss_index"

# FAISS uses distance
# LOWER = better
SIMILARITY_THRESHOLD = 0.6
MIN_RELEVANT_DOCS = 2

# Embeddings
embeddings = OpenAIEmbeddings()

# Better chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# Tavily search
tavily_search = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False
)

# Tavily crawl/extract
tavily_extract = TavilyExtract(
    extract_depth="advanced",
    include_images=False
)

# Global cache for vector database
_vector_db_cache = None


def get_vector_db():
    """
    Get or load the vector database (singleton pattern).
    Ensures we always work with the same instance.
    """
    global _vector_db_cache
    
    if _vector_db_cache is None:
        if os.path.exists(FAISS_INDEX_PATH):
            _vector_db_cache = FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing FAISS index from disk")
    
    return _vector_db_cache


def save_vector_db(vector_db):
    """
    Save and cache the vector database.
    Ensures disk and memory stay in sync.
    """
    global _vector_db_cache
    
    vector_db.save_local(FAISS_INDEX_PATH)
    _vector_db_cache = vector_db
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")


def get_doc_id(text: str) -> str:
    """
    Stable hash for deduplication.
    """
    return hashlib.md5(
        text.encode()
    ).hexdigest()


def has_relevant_knowledge(
    vector_db,
    query: str
) -> bool:
    """
    Check if vector DB already has
    highly relevant information.
    """
    if not vector_db:
        return False

    results = (
        vector_db.similarity_search_with_score(
            query,
            k=5
        )
    )

    if not results:
        return False

    relevant_docs = [
        (doc, score)
        for doc, score in results
        if score < SIMILARITY_THRESHOLD
    ]

    if len(relevant_docs) >= MIN_RELEVANT_DOCS:

        avg_score = (
            sum(
                score
                for _, score in relevant_docs
            )
            / len(relevant_docs)
        )

        print(
            f"\nRelevant docs found: "
            f"{len(relevant_docs)}"
        )

        print(
            f"Average distance: "
            f"{avg_score:.4f}"
        )

        return True

    return False


@tool
def embed_search_results(query: str) -> str:
    """
    Search Tavily and embed
    missing information into FAISS.
    """
    vector_db = get_vector_db()

    # FIRST: Check vector DB
    if has_relevant_knowledge(
        vector_db,
        query
    ):
        return (
            "Relevant information already "
            "exists in vector database. "
            "Skipping Tavily search."
        )

    print("\nKnowledge missing.")
    print("Triggering Tavily search...")

    search_results = tavily_search.invoke(query)

    documents = []
    seen_urls = set()

    results = search_results.get(
        "results",
        []
    )

    # -----------------------------------
    # SEARCH RESULTS
    # -----------------------------------

    for result in results:

        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")

        if (
            not content
            or url in seen_urls
        ):
            continue

        seen_urls.add(url)

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "title": title,
                    "url": url,
                    "query": query,
                    "source_type": "search",
                    "timestamp": (
                        datetime.utcnow()
                        .isoformat()
                    ),
                    "doc_id": get_doc_id(
                        content
                    )
                }
            )
        )

    # -----------------------------------
    # CRAWL FALLBACK
    # -----------------------------------

    if len(documents) < 2:

        print(
            "\nWeak search results."
        )

        print(
            "Triggering Tavily crawl..."
        )

        for result in results:

            url = result.get("url")

            if (
                not url
                or url in seen_urls
            ):
                continue

            try:

                extracted = (
                    tavily_extract.invoke(
                        {
                            "urls": [url]
                        }
                    )
                )

                for item in extracted.get(
                    "results",
                    []
                ):

                    raw_content = item.get(
                        "raw_content",
                        ""
                    )

                    if not raw_content:
                        continue

                    documents.append(
                        Document(
                            page_content=raw_content,
                            metadata={
                                "title": item.get(
                                    "title",
                                    ""
                                ),
                                "url": url,
                                "query": query,
                                "source_type": "crawl",
                                "timestamp": (
                                    datetime.utcnow()
                                    .isoformat()
                                ),
                                "doc_id": (
                                    get_doc_id(
                                        raw_content
                                    )
                                )
                            }
                        )
                    )

            except Exception as e:

                print(
                    f"Crawl failed for "
                    f"{url}: {e}"
                )

    # -----------------------------------
    # DEDUPLICATION
    # -----------------------------------

    unique_docs = {}

    for doc in documents:

        unique_docs[
            doc.metadata["doc_id"]
        ] = doc

    documents = list(
        unique_docs.values()
    )

    if not documents:

        return (
            "No useful information found."
        )

    # -----------------------------------
    # SPLIT DOCUMENTS
    # -----------------------------------

    split_docs = (
        splitter.split_documents(
            documents
        )
    )

    # -----------------------------------
    # CREATE / UPDATE VECTOR DB
    # -----------------------------------

    if vector_db:
        # Add to existing index
        vector_db.add_documents(
            split_docs
        )
        print(f"Added {len(split_docs)} chunks to existing FAISS index")
    else:
        # Create new index
        vector_db = (
            FAISS.from_documents(
                split_docs,
                embeddings
            )
        )
        print(f"Created new FAISS index with {len(split_docs)} chunks")

    # Save the updated index
    save_vector_db(vector_db)

    return (
        f"Stored "
        f"{len(split_docs)} chunks "
        f"from "
        f"{len(documents)} documents."
    )


@tool
def search_vector_db(
    query: str
) -> str:
    """
    Search vector database using
    semantic retrieval.
    """
    vector_db = get_vector_db()

    if not vector_db:

        return (
            "Vector database empty. "
            "Run embed_search_results first."
        )

    results = (
        vector_db
        .similarity_search_with_score(
            query,
            k=5
        )
    )

    filtered_results = [

        (doc, score)

        for doc, score in results

        if score < SIMILARITY_THRESHOLD
    ]

    if not filtered_results:

        return (
            "No highly relevant "
            "documents found."
        )

    formatted_results = []

    for i, (
        doc,
        score
    ) in enumerate(
        filtered_results,
        start=1
    ):

        formatted_results.append(
            f"""
# Result {i}

### Similarity Distance
{score:.4f}

### Title
{doc.metadata.get("title", "N/A")}

### URL
{doc.metadata.get("url", "N/A")}

### Source Type
{doc.metadata.get("source_type", "N/A")}

### Content
{doc.page_content[:1000]}
"""
        )

    return "\n".join(
        formatted_results
    )


def agent(query: str) -> str:

    tools_list = [
        search_vector_db,
        embed_search_results
    ]

    tools_dict = {
        tool.name: tool
        for tool in tools_list
    }

    chat_model = init_chat_model(
        MODEL,
        model_provider="openai"
    )

    llm_with_tools = (
        chat_model.bind_tools(
            tools_list
        )
    )

    messages = [

        SystemMessage(
            content="""
You are an advanced RAG assistant.

WORKFLOW:

1. FIRST search_vector_db
2. If retrieval weak:
   use embed_search_results
3. Then retry search_vector_db
4. Then answer the user

RULES:

- Minimize Tavily usage
- Avoid repeated web searches
- if you using web search make sure that you update the vector db with the new information 
and try to use the vector db again before searching again
- Prefer vector DB memory
- Always cite source URLs
- Return clean markdown
- Only use highly relevant information
"""
        ),

        HumanMessage(
            content=query
        )
    ]

    for iteration in range(
        MAX_ITERATIONS
    ):

        response = (
            llm_with_tools.invoke(
                messages
            )
        )

        messages.append(response)

        if not response.tool_calls:

            return response.content

        for tool_call in (
            response.tool_calls
        ):

            tool_name = (
                tool_call["name"]
            )

            if (
                tool_name
                not in tools_dict
            ):

                messages.append(
                    ToolMessage(
                        content=(
                            f"Tool "
                            f"{tool_name} "
                            f"does not exist."
                        ),
                        tool_call_id=(
                            tool_call["id"]
                        )
                    )
                )

                continue

            print(
                f"\nIteration "
                f"{iteration + 1}"
            )

            print(
                f"Calling: "
                f"{tool_name}"
            )

            tool = tools_dict[
                tool_name
            ]

            try:

                result = tool.invoke(
                    tool_call["args"]
                )

            except Exception as e:

                result = (
                    f"Tool failed: {e}"
                )

            #print(result)

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=(
                        tool_call["id"]
                    )
                )
            )

    return "Max iterations reached."


if __name__ == "__main__":

    user_query = input(
        "\nWhat would you like "
        "to search for today: "
    )

    answer = agent(user_query)

    print("\n")
    print(answer)

    with open(
        "output.md",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(answer)

    print(
        "\nAnswer saved to output.md"
    )