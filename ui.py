import os
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # ← NEW
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #fdf0f8 50%, #f5f0ff 100%);
    }
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(270deg, #e94560, #533483, #0f3460, #e94560);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        padding: 10px;
    }
    .subtitle {
        text-align: center;
        color: #555577;
        font-size: 1rem;
        margin-bottom: 10px;
    }
    .typing-dots {
        display: inline-flex;
        gap: 4px;
        padding: 8px 12px;
    }
    .typing-dots span {
        width: 8px;
        height: 8px;
        background: #e94560;
        border-radius: 50%;
        animation: bounce 1.2s infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid rgba(233,69,96,0.2) !important;
        border-radius: 16px !important;
        margin: 6px 0 !important;
        animation: slideUp 0.4s ease-out;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06) !important;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: #1a1a2e !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: rgba(233,69,96,0.07) !important;
        border: 1px solid rgba(233,69,96,0.3) !important;
    }
    [data-testid="stChatInput"] {
        border: 2px solid #e94560 !important;
        border-radius: 25px !important;
        background: #ffffff !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #1a1a2e !important;
        background: #ffffff !important;
        caret-color: #e94560 !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    textarea, input {
        color: #1a1a2e !important;
        -webkit-text-fill-color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff, #f0f4ff) !important;
        border-right: 1px solid rgba(233,69,96,0.2) !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #1a1a2e !important;
    }
    .stButton button {
        background: #ffffff !important;
        color: #e94560 !important;
        border: 1.5px solid #e94560 !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    .stButton button:hover {
        background: #e94560 !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
    }
    .stat-card {
        background: #ffffff;
        border: 1px solid rgba(233,69,96,0.25);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-card:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(233,69,96,0.2);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #e94560;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #555577;
    }
    .pulse-dot {
        width: 10px;
        height: 10px;
        background: #00cc66;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s infinite;
        margin-right: 6px;
    }
    .stMarkdown p { color: #1a1a2e !important; }
    hr { border-color: rgba(233,69,96,0.15) !important; }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #f0f4ff; }
    ::-webkit-scrollbar-thumb { background: #e94560; border-radius: 4px; }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,204,102,0.6); }
        70% { box-shadow: 0 0 0 8px rgba(0,204,102,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,204,102,0); }
    }
</style>
""", unsafe_allow_html=True)

# ---- State ----
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---- Memory + Graph (one time) ---- #
@st.cache_resource
def build_graph():
    memory = MemorySaver()  # ← Memory yahan add hui
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    def chatbot_node(state: State):
        return {"messages": [llm.invoke(state["messages"])]}
    
    gb = StateGraph(State)
    gb.add_node("chatbot", chatbot_node)
    gb.add_edge(START, "chatbot")
    gb.add_edge("chatbot", END)
    
    return gb.compile(checkpointer=memory)  # ← Checkpointer add kiya

graph = build_graph()

# Thread ID — har session ka alag memory
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user-session-1"

# Config — memory ke liye zaroori
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ---- Session init ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quick" not in st.session_state:
    st.session_state.quick = None

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    model_choice = st.selectbox(
        "🧠 Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "gemma2-9b-it"],
        index=0
    )

    temperature = st.slider("🌡️ Creativity", 0.0, 1.0, 0.7, 0.1)

    st.divider()
    st.markdown("## 📊 Stats")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.messages)}</div>
            <div class="stat-label">Messages</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{user_msgs}</div>
            <div class="stat-label">Questions</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = "user-session-1"
        st.rerun()

    st.markdown("<div style='text-align:center; color:#555577; font-size:0.75rem; margin-top:20px'>Powered by LangGraph + Groq</div>", unsafe_allow_html=True)

# ---- MAIN ----
st.markdown('<div class="main-title">🤖 My AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle"><span class="pulse-dot"></span>Online — Ask me anything!</div>', unsafe_allow_html=True)

# Quick buttons
st.markdown("**⚡ Quick Actions:**")
c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("💡 Explain AI"):
        st.session_state.quick = "Explain Artificial Intelligence in simple words"
with c2:
    if st.button("🐍 Python Tips"):
        st.session_state.quick = "Give me 3 Python tips for beginners"
with c3:
    if st.button("📝 Write Poem"):
        st.session_state.quick = "Write a short funny poem about coding"
with c4:
    if st.button("🌍 Fun Fact"):
        st.session_state.quick = "Tell me an interesting fun fact"

st.divider()

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---- LLM helper with memory ----
def get_response(prompt):
    for event in graph.stream(
        {"messages": [("user", prompt)]},
        config=config,  # ← Memory config pass kiya
        stream_mode="values"
    ):
        latest = event["messages"][-1]
    return latest.content

# Quick action handler
if st.session_state.quick:
    prompt = st.session_state.quick
    st.session_state.quick = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
if prompt := st.chat_input("💬 Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(prompt)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})