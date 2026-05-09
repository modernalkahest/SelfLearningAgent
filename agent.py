from dotenv import load_dotenv
import os
import hashlib

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage
)
from langchain.chat_models import init_chat_model
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

MODEL = "gpt-5.4"
MAX_ITERATIONS = 4
FAISS_INDEX_PATH = "faiss_index"

# Embeddings
embeddings = OpenAIEmbeddings()

# Better chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# Tavily optimized for free tier
tavily_tool = TavilySearch(
    max_results=2,
    topic="general",
    search_depth="basic",
    include_answer=True,
    include_raw_content=False
)


def load_vector_db():
    """
    Load or create FAISS DB.
    """

    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    return None


def get_doc_id(text: str) -> str:
    """
    Create stable hash for deduplication.
    """

    return hashlib.md5(text.encode()).hexdigest()


@tool
def embed_search_results(query: str) -> str:
    """
    Search Tavily and embed results into FAISS.
    """

    vector_db = load_vector_db()

    # FIRST: Check if info already exists
    if vector_db:
        existing = vector_db.similarity_search(query, k=2)

        if existing:
            return (
                "Relevant information already exists "
                "inside vector database. Skipping Tavily search."
            )

    # Tavily search
    results = tavily_tool.invoke(query)

    documents = []
    seen_urls = set()

    for result in results.get("results", []):

        url = result.get("url", "")
        title = result.get("title", "")
        content = result.get("content", "")

        # Skip duplicates
        if not content or url in seen_urls:
            continue

        seen_urls.add(url)

        doc = Document(
            page_content=content,
            metadata={
                "title": title,
                "url": url,
                "query": query,
                "doc_id": get_doc_id(content)
            }
        )

        documents.append(doc)

    if not documents:
        return "No useful search results found."

    # Chunk documents
    split_docs = splitter.split_documents(documents)

    # Create/update vector DB
    if vector_db:
        vector_db.add_documents(split_docs)
    else:
        vector_db = FAISS.from_documents(
            split_docs,
            embeddings
        )

    vector_db.save_local(FAISS_INDEX_PATH)

    return (
        f"Stored {len(split_docs)} chunks "
        f"from {len(documents)} Tavily results."
    )


@tool
def search_vector_db(query: str) -> str:
    """
    Search vector DB using semantic retrieval.
    """

    vector_db = load_vector_db()

    if not vector_db:
        return (
            "Vector database empty. "
            "Run embed_search_results first."
        )

    # MMR = better retrieval diversity
    results = vector_db.max_marginal_relevance_search(
        query,
        k=4,
        fetch_k=10
    )

    if not results:
        return "No matching documents found."

    formatted_results = []

    for i, doc in enumerate(results, start=1):

        formatted_results.append(
            f"""
Result {i}

Title:
{doc.metadata.get("title", "N/A")}

URL:
{doc.metadata.get("url", "N/A")}

Content:
{doc.page_content[:700]}
"""
        )

    return "\n".join(formatted_results)


def agent(query: str) -> str:

    tools_list = [
        embed_search_results,
        search_vector_db
    ]

    tools_dict = {
        t.name: t
        for t in tools_list
    }

    chat_model = init_chat_model(
        MODEL,
        model_provider="openai"
    )

    llm_with_tools = chat_model.bind_tools(
        tools_list
    )

    messages = [
        SystemMessage(
            content="""
                You are a RAG assistant.

                Workflow:
                1. First use embed_search_results
                2. Then use search_vector_db
                3. Then answer the user

                RULES:
                - Minimize Tavily usage
                - Never repeatedly search Tavily
                - Prefer vector DB whenever possible
                - Always cite URLs
                - Return markdown
                """
        ),
        HumanMessage(content=query)
    ]

    for iteration in range(MAX_ITERATIONS):

        response = llm_with_tools.invoke(messages)

        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            if tool_name not in tools_dict:

                messages.append(
                    ToolMessage(
                        content=f"Tool '{tool_name}' does not exist.",
                        tool_call_id=tool_call["id"]
                    )
                )

                continue

            print(
                f"\nIteration {iteration + 1}"
            )

            print(
                f"Calling: {tool_name}"
            )

            tool = tools_dict[tool_name]

            result = tool.invoke(
                tool_call["args"]
            )

            print(result)

            messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                )
            )

    return "Max iterations reached."


if __name__ == "__main__":

    user_query = (
        "What is the median salary of an "
        "AI platform engineer in India "
        "with 6+ years of experience?"
    )

    answer = agent(user_query)

    print(answer)

    with open("output.md", "w") as f:
        f.write(answer)