import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ML Tutor", page_icon="🤖")
st.title("🤖 ML Expert Chatbot")
st.caption("Ask me anything about Machine Learning!")

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if not st.session_state.session_id:
    name = st.text_input("Enter your name to start chatting:")
    if st.button("Start Chat") and name.strip():
        st.session_state.session_id = name.strip()
        st.rerun()
    st.stop() 

session_id = st.session_state.session_id

if "store" not in st.session_state:
    st.session_state.store = {}

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []


@st.cache_resource
def get_chain():
    model = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.8,
        reasoning_format="parsed",
        timeout=None,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a ML/AI expert assistant. Your ONLY job is to answer questions about:
                - Machine Learning
                - Deep Learning  
                - Artificial Intelligence
                - Data Science
                - Statistics (only when related to ML)
                
                STRICT RULES:
                - If the question is NOT related to the above topics, respond ONLY with:
                  "I'm only trained to answer ML/AI questions. Please ask me something related to Machine Learning or AI."
                - Do NOT make exceptions. Do NOT answer even partially.
                - Do NOT engage with off-topic questions about movies, anime, sports, coding (non-ML), etc.
                - If unsure whether a topic is ML-related, refuse it.
                
                Explain all ML/AI concepts so a complete beginner can understand easily.""",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    return prompt | model | StrOutputParser()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = InMemoryChatMessageHistory()
    return st.session_state.store[session_id]


chain = get_chain()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

with st.sidebar:
    st.header("Settings")
    st.info(f"Logged in as: **{session_id}**")
    if st.button("Clear Memory"):
        st.session_state.store.pop(session_id, None)
        st.session_state.chat_display = []
        st.success("Memory cleared!")
    if st.button("Switch User"):
        st.session_state.session_id = None
        st.session_state.chat_display = []
        st.rerun()
    st.markdown("---")
    msg_count = len(get_session_history(session_id).messages)
    st.metric("Messages in memory", msg_count)

for msg in st.session_state.chat_display:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input(f"Ask a ML question, {session_id}..."):
    st.session_state.chat_display.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = chain_with_history.invoke(
                {"question": question},
                config={"configurable": {"session_id": session_id}},
            )
        st.markdown(result)

    st.session_state.chat_display.append({"role": "assistant", "content": result})
    st.rerun()
