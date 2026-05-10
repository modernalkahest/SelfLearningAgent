import streamlit as st
import threading
import time
from agent import agent, get_vector_db

# =============================
# PAGE SETUP
# =============================

st.set_page_config(
    page_title="Research Assistant",
    page_icon="📡",
    layout="centered"
)

# =============================
# STATE
# =============================

if "result" not in st.session_state:
    st.session_state.result = None
if "query" not in st.session_state:
    st.session_state.query = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# =============================
# SIDEBAR - STATS
# =============================

with st.sidebar:
    st.header("📊 Vector Database")
    
    vector_db = get_vector_db()
    
    if vector_db:
        try:
            index_size = vector_db.index.ntotal
            st.success("✅ Knowledge Base loaded")
            st.metric("Total Embeddings", f"{index_size:,}")
        except:
            st.info("Knowledge Base exists")
    else:
        st.warning("⚠️ No Knowledge Base yet")
        st.caption("Will be created after first search")
    
    st.markdown("---")
    
    # Query history
    if st.session_state.query_history:
        st.subheader("📜 Recent Queries")
        for i, past_query in enumerate(reversed(st.session_state.query_history[-5:])):
            if st.button(
                f"🔄 {past_query[:35]}...",
                key=f"history_{i}",
                use_container_width=True
            ):
                st.session_state.query = past_query
                st.session_state.result = None
                st.rerun()
        if vector_db:
            try:
                new_index_size = vector_db.index.ntotal
                st.metric("Additional Embeddings", f"{new_index_size - index_size:,}")
            except:
                st.info("Knowledge Base exists")
        else:
            st.warning("⚠️ No Knowledge Base yet")
            st.caption("Will be created after first search")

# =============================
# HERO
# =============================

st.title("Research Agent")
st.caption("Search the open web · Get structured intelligence")

# =============================
# INPUT
# =============================

query = st.text_area(
    label="query",
    placeholder="What do you want to research?",
    height=80,
    label_visibility="collapsed",
    value=st.session_state.query if st.session_state.query else ""
)

col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    run_clicked = st.button("Search →", use_container_width=True)

# =============================
# ACTION
# =============================

if run_clicked:
    if not query.strip():
        st.warning("Please enter a research query.")
    else:
        try:
            status_messages = [
                "Searching vector database...",
                "Retrieving sources...",
                "Embedding new information...",
                "Summarizing relevant sources...",
            ]

            result_holder = {}
            status_placeholder = st.empty()

            def call_agent():
                try:
                    answer = agent(query)
                    result_holder["response"] = answer
                except Exception as e:
                    result_holder["error"] = str(e)

            thread = threading.Thread(target=call_agent)
            thread.start()

            idx = 0
            while thread.is_alive():
                msg = status_messages[idx % len(status_messages)]
                status_placeholder.markdown(f"⟳ {msg}")
                time.sleep(2)
                idx += 1

            thread.join()
            status_placeholder.empty()

            if "error" in result_holder:
                raise Exception(result_holder["error"])

            answer = result_holder["response"]
            
            # Store results
            st.session_state.result = answer
            st.session_state.query = query
            
            # Add to history
            if query not in st.session_state.query_history:
                st.session_state.query_history.append(query)
            
            # Clear the query input for next search
            st.rerun()

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            import traceback
            with st.expander("🐛 Debug Info"):
                st.code(traceback.format_exc())

# =============================
# OUTPUT
# =============================

if st.session_state.result:
    st.markdown("---")
    st.subheader(f"📝 Results for: {st.session_state.query}")
    st.markdown(st.session_state.result)

    col_dl, col_clear = st.columns([1, 1])
    with col_dl:
        st.download_button(
            label="↓ Download .md",
            data=st.session_state.result,
            file_name=f"research_{st.session_state.query[:30].replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_clear:
        if st.button("🗑️ Clear Results", use_container_width=True):
            st.session_state.result = None
            st.session_state.query = None
            st.rerun()

# =============================
# FOOTER
# =============================

st.caption("Powered by LangChain · FAISS Vector DB · Tavily Search")