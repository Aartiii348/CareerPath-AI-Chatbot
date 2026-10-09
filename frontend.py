
import streamlit as st
import backend as demo

st.set_page_config(
    page_title="CareerPath AI Chatbot",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 CareerPath AI Chatbot")
st.caption(
    "Explore AI, Data Analytics, Python, and career roadmaps "
    "using RAG + LangChain + Pinecone"
)


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_retriever():
    return demo.build_retriever()


retriever = get_retriever()

with st.sidebar:
    st.header("Controls")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.memory = demo.demo_memory()
        st.session_state.chat_history = []
        st.rerun()

    if st.button("Re-index Career Resources", use_container_width=True):
        with st.spinner("Loading and indexing career resources..."):
            n_chunks = demo.index_website(reset=True)

        st.success(f"Indexed {n_chunks} chunks into Pinecone.")
        get_retriever.clear()
        st.rerun()

    st.divider()

    st.markdown(
        "**About this project**\n\n"
        "- Career-focused Retrieval-Augmented Generation\n"
        f"- Vector DB: Pinecone (`{demo.PINECONE_INDEX_NAME}`)\n"
        "- Few-shot prompting and chat memory\n"
        "- LLM: `gpt-4o-mini`"
    )


if "memory" not in st.session_state:
    st.session_state.memory = demo.demo_memory()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])


input_text = st.chat_input(
    "Ask about AI careers, data analyst skills, or Python..."
)

if input_text:
    with st.chat_message("user"):
        st.markdown(input_text)

    st.session_state.chat_history.append(
        {"role": "user", "text": input_text}
    )

    with st.chat_message("assistant"):
        with st.spinner("Finding relevant career information..."):
            response, updated_memory = demo.demo_conversation(
                input_text=input_text,
                memory=st.session_state.memory,
                retriever=retriever,
            )

        st.markdown(response)

    st.session_state.memory = updated_memory

    st.session_state.chat_history.append(
        {"role": "assistant", "text": response}
    )
