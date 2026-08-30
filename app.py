import streamlit as st 
import tempfile
import os 
from agent import create_rag_agent
from main import ingest, query 
from ingestion.config import graph 
from ingestion.vector_store import delete_paper_from_vector_store
from langchain_core.messages import AIMessage

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
import warnings
warnings.filterwarnings("ignore", message=".*zoedepth.*")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="ResearchComp",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Restore state from backend on fresh load
if "initialized" not in st.session_state:
    papers_dir = "./data/papers/"
    if os.path.exists(papers_dir):
        existing_files = set(f for f in os.listdir(papers_dir) if f.endswith(".pdf"))
    else:
        existing_files = set()

    if existing_files:
        st.session_state.ingested_files = existing_files
        st.session_state.pipeline_ready = True
    else:
        st.session_state.ingested_files = set()
        st.session_state.pipeline_ready = False

    st.session_state.initialized = True


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    *, *::before, *::after { box-sizing: border-box; }

    /* ─── Base ─────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(145deg, #EEF6FF 0%, #F0EBFF 50%, #E8F4FD 100%) !important;
        background-attachment: fixed !important;
        color: #1A1040 !important;
        font-family: 'Inter', sans-serif !important;
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hide streamlit chrome */
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] { display: none !important; }

    [data-testid="stMainBlockContainer"],
    .block-container {
        padding-top: 0 !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* ─── App Bar ───────────────────────────── */
    .app-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 4px 14px;
        margin-bottom: 20px;
        border-bottom: 1px solid rgba(139, 92, 246, 0.12);
    }
    .app-bar-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .app-logo {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #38BDF8 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
    }
    .app-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 60%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.4px;
        margin: 0;
        line-height: 1;
    }
    .app-subtitle {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 400;
        margin: 2px 0 0;
    }
    .app-badge {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(56,189,248,0.1));
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 14px;
        font-weight: 600;
        color: #6366F1;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }


    /* ─── Action Row ────────────────────────────────────── */
    /* Outer container — no card, just a flex row */
    .st-key-action_row {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 18px !important;
    }
    .st-key-action_row [data-testid="stVerticalBlock"],
    .st-key-action_row [data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
    }
    .st-key-action_row [data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
        gap: 12px !important;
    }

    /* Left column — white upload card */
    .st-key-action_row [data-testid="stColumn"]:nth-child(1) > [data-testid="stVerticalBlock"] {
        background: white !important;
        border-radius: 16px !important;
        padding: 8px 20px !important;
        border: 1.5px solid rgba(99, 102, 241, 0.15) !important;
        box-shadow: 0 2px 16px rgba(99, 102, 241, 0.08), 0 1px 4px rgba(0,0,0,0.04) !important;
        height: 100% !important;
    }

    /* Right column — indigo-tinted ingest card */
    .st-key-action_row [data-testid="stColumn"]:nth-child(2) > [data-testid="stVerticalBlock"] {
        background: linear-gradient(135deg, rgba(99,102,241,0.07) 0%, rgba(139,92,246,0.07) 100%) !important;
        border-radius: 16px !important;
        padding: 8px 16px !important;
        border: 1.5px solid rgba(99, 102, 241, 0.22) !important;
        box-shadow: 0 2px 16px rgba(99, 102, 241, 0.06), 0 1px 4px rgba(0,0,0,0.04) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
    }

    /* Transparent inner wrappers */
    .st-key-action_row [data-testid="stColumn"] [data-testid="stElementContainer"] {
        background: transparent !important;
    }

    /* Uploader inside left card */
    .st-key-action_row [data-testid="stFileUploader"] {
        margin-bottom: 0 !important;
    }
    .st-key-action_row [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }
    .st-key-action_row [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.08)) !important;
        color: #6366F1 !important;
        border: 1.5px solid rgba(99,102,241,0.3) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        height: auto !important;
        box-shadow: none !important;
    }
    .st-key-action_row [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15)) !important;
        border-color: #6366F1 !important;
        box-shadow: 0 2px 8px rgba(99,102,241,0.2) !important;
    }
    .st-key-action_row [data-testid="stFileUploaderDropzone"] small,
    .st-key-action_row [data-testid="stFileUploaderDropzone"] span {
        color: #94A3B8 !important;
        font-size: 13px !important;
    }
    .st-key-action_row [data-testid="stUploadedFile"] {
        background: rgba(99,102,241,0.06) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(99,102,241,0.15) !important;
        color: #4338CA !important;
    }


    /* ─── Ingest Button ─────────────────────── */
    .st-key-ingest_btn button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        height: 52px !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.40) !important;
    }

    .st-key-ingest_btn button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.55) !important;
    }

    .st-key-ingest_btn button:disabled {
        background: linear-gradient(135deg, #C7D2FE, #DDD6FE) !important;
        color: rgba(99,102,241,0.5) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }

    /* ─── Loading Overlay ───────────────────── */
    .loading-overlay {
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        background: rgba(238, 246, 255, 0.92);
        backdrop-filter: blur(8px);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 9999;
    }
    .loading-card {
        background: white;
        border-radius: 24px;
        padding: 48px 56px;
        box-shadow: 0 20px 60px rgba(99,102,241,0.15), 0 4px 16px rgba(0,0,0,0.08);
        text-align: center;
        border: 1px solid rgba(99,102,241,0.12);
    }
    .loading-spinner {
        width: 52px;
        height: 52px;
        border: 3px solid rgba(99,102,241,0.15);
        border-top: 3px solid #6366F1;
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
        margin: 0 auto 20px;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    .loading-title {
        color: #1A1040;
        font-size: 18px;
        font-weight: 700;
        font-family: 'Inter', sans-serif;
        margin-bottom: 6px;
    }
    .loading-text {
        color: #94A3B8;
        font-size: 14px;
        font-family: 'Inter', sans-serif;
    }

    /* ─── File Panel ────────────────────────── */
    .file-panel {
        background: white;
        border-radius: 18px;
        padding: 22px 20px;
        box-shadow: 0 2px 16px rgba(99,102,241,0.07), 0 1px 4px rgba(0,0,0,0.04);
        border: 1.5px solid rgba(99,102,241,0.1);
        height: 580px;
        overflow-y: auto;
    }
    .file-panel::-webkit-scrollbar { width: 4px; }
    .file-panel::-webkit-scrollbar-track { background: transparent; }
    .file-panel::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 4px; }

    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(99,102,241,0.08);
    }
    .panel-header-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(56,189,248,0.1));
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 15px;
    }
    .panel-header-text {
        font-size: 14px;
        font-weight: 700;
        color: #1A1040;
    }
    .file-item {
        background: linear-gradient(135deg, #F8F9FF 0%, #F5F3FF 100%);
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .file-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-ingested { background: #34D399; box-shadow: 0 0 6px rgba(52,211,153,0.5); }
    .status-pending  { background: #FBBF24; box-shadow: 0 0 6px rgba(251,191,36,0.5); }
    .file-name {
        color: #374151;
        font-size: 12.5px;
        font-weight: 500;
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .file-tag {
        font-size: 10px;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 20px;
        letter-spacing: 0.3px;
        text-transform: uppercase;
        flex-shrink: 0;
    }
    .tag-ingested { background: rgba(52,211,153,0.12); color: #059669; border: 1px solid rgba(52,211,153,0.25); }
    .tag-pending  { background: rgba(251,191,36,0.12);  color: #D97706; border: 1px solid rgba(251,191,36,0.25); }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70%;
        text-align: center;
        gap: 12px;
        padding: 20px;
    }
    .empty-state-icon { font-size: 48px; opacity: 0.5; }
    .empty-state-title { font-size: 14px; font-weight: 600; color: #94A3B8; }
    .empty-state-text  { font-size: 12px; color: #CBD5E1; }

    /* ─── Chat Panel ────────────────────────── */
    .st-key-main_content [data-testid="stColumn"]:nth-child(2) > [data-testid="stVerticalBlock"] {
        background: white;
        border-radius: 18px;
        padding: 22px 20px 16px;
        height: 580px;
        box-shadow: 0 2px 16px rgba(99,102,241,0.07), 0 1px 4px rgba(0,0,0,0.04);
        border: 1.5px solid rgba(99,102,241,0.1);
    }

    [data-testid="stChatMessage"] * {
        color: #1A1040 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px !important;
    }

    [data-testid="stChatMessage"] {
        background-color: white !important;
        border: 1px solid #EDE9FE !important;
        border-radius: 4px 12px 12px 12px !important;
        padding: 12px 16px !important;
        margin-bottom: 14px !important;
    }


    # [data-testid="stChatMessage"] p {
    #     color:#7C3AED !important; 
    # }

    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
        margin-top: 12px !important;
        margin-bottom: 6px !important;
    }

    /* Reduce list spacing */
    [data-testid="stChatMessage"] li {
        margin-bottom: 2px !important;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        display: none !important;
    }
    /* "You" label for user messages */
    [data-testid="stChatMessage"][data-testid-type="user"]::before,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])::before {
        content: "You";
        position: absolute;
        top: -18px;
        left: 4px;
        font-size: 10px;
        font-weight: 700;
        color: #7C3AED;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* "Research Comp" label for assistant messages */
    [data-testid="stChatMessage"][data-testid-type="assistant"]::before,
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])::before {
        content: "Research Comp";
        position: absolute;
        top: -18px;
        left: 4px;
        font-size: 10px;
        font-weight: 700;
        color: #6366F1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* User messages — tinted background */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: #F5F3FF !important;
        border: none !important;
    }

    /* Assistant messages — white with border */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: white !important;
        border: 1px solid #EDE9FE !important;
    }

    /* Chat input textbox */
    [data-testid="stChatInput"] {
        background-color: #F5F3FF;
        border: 1px solid rgba(124, 58, 237, 0.2) !important;
        border-radius: 0.5rem;
        box-sizing: border-box;
    }

    [data-testid="stChatInput"] > div {
        background-color: #F5F3FF;
    }

    [data-testid="stChatInput"] textarea {
        background: #F5F3FF !important;
        border: none !important;
        color: #1E1B4B !important;
        font-size: 14px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #9CA3AF !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
    }

    /* Send button */
    [data-testid="stChatInput"] button {
        background: #7C3AED !important;
        color: white !important;
        border-radius: 8px !important;
    }

    /* ─── Stats Row ─────────────────────────── */
    .stats-row {
        display: flex;
        gap: 10px;
        margin-bottom: 18px;
    }
    .stat-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        background: white;
        border: 1px solid rgba(99,102,241,0.12);
        border-radius: 10px;
        padding: 6px 12px;
        font-size: 12px;
        font-weight: 600;
        color: #6366F1;
        box-shadow: 0 1px 4px rgba(99,102,241,0.06);
    }

    /* ─── Chat Empty State ──────────────────── */
    .chat-empty {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        padding: 40px 20px;
        text-align: center;
    }
    .chat-empty-bubble {
        width: 72px;
        height: 72px;
        background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(56,189,248,0.08));
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        margin-bottom: 20px;
        border: 1.5px solid rgba(99,102,241,0.12);
    }
    .chat-empty-title { font-size: 18px; font-weight: 700; color: #374151; margin-bottom: 8px; }
    .chat-empty-text  { font-size: 13px; color: #94A3B8; max-width: 260px; line-height: 1.6; }
    .chat-empty-steps {
        margin-top: 24px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        width: 100%;
        max-width: 300px;
    }
    .chat-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: linear-gradient(135deg, #F8F9FF, #F5F3FF);
        border-radius: 10px;
        border: 1px solid rgba(99,102,241,0.1);
    }
    .chat-step-num {
        width: 22px;
        height: 22px;
        background: linear-gradient(135deg, #6366F1, #8B5CF6);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: white;
        flex-shrink: 0;
    }
    .chat-step-text { font-size: 12.5px; color: #64748B; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("is_error", False):
    st.error("Something went wrong. Please try again later.")


# ─── App Bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-bar">
    <div class="app-bar-left">
        <div class="app-logo">🔬</div>
        <div>
            <div class="app-title">ResearchComp</div>
            <div class="app-subtitle">Agentic Graph RAG · Research Assistant</div>
        </div>
    </div>
    <div class="app-badge">Graph RAG</div>
</div>
""", unsafe_allow_html=True)

def save_uploaded_files(uploaded_files):
    save_dir = "./data/papers/"
    os.makedirs(save_dir, exist_ok=True)
    for file in uploaded_files:
        path = os.path.join(save_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
    return save_dir

def delete_papers(filename):
    graph.query(
        "MATCH (n {source_file: $filename}) DETACH DELETE n",
        {"filename": filename},
    )
    delete_paper_from_vector_store(filename)

def extract_answer(result):
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            
            # Content is a list of blocks — extract only text
            if isinstance(msg.content, list):
                text_parts = [
                    block["text"] for block in msg.content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if text_parts:
                    return "\n\n".join(text_parts)
    
    return "I couldn't generate a response. Please try again."


# ─── Action Row: uploader + ingest button side by side ──────────────────────
with st.container(key="action_row"):
    upload_col, ingest_col = st.columns([7, 2])

    with upload_col:
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDFs)",
            type=['pdf'],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    st.session_state.uploaded_files = uploaded_files if uploaded_files else []
    current_files = set(f.name for f in st.session_state.uploaded_files)
    ingested_files = st.session_state.get("ingested_files", set())
    new_files = current_files - ingested_files
    removed_files = ingested_files - current_files
    for filename in removed_files:
        delete_papers(filename)
    if removed_files:
        st.session_state.ingested_files = ingested_files - removed_files
        if "agent" in st.session_state:
            del st.session_state.agent
        if not current_files:
            st.session_state.pipeline_ready = False
        st.rerun()

    with ingest_col:
        if new_files:
            st.session_state.is_error = False
            if st.button("⚡  Ingest Papers", type="primary", use_container_width=True, key="ingest_btn"):
                overlay = st.empty()
                overlay.markdown("""
                <div class="loading-overlay">
                    <div class="loading-card">
                        <div class="loading-spinner"></div>
                        <div class="loading-title">Processing Papers</div>
                        <div class="loading-text">Building knowledge graph & embeddings…</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                new_uploaded = [f for f in st.session_state.uploaded_files if f.name in new_files]
                ingestion = ingest(new_uploaded)
                overlay.empty()
                if ingestion["success"]:
                    st.session_state.pipeline_ready = True
                    st.session_state.ingested_files = ingested_files | new_files
                    st.rerun()
                else:
                    st.session_state.is_error = True 
                    st.rerun()

        else:
            st.button("⚡  Ingest Papers", type="primary", disabled=True, use_container_width=True, key="ingest_btn")



# ─── Stats Row ─────────────────────────────────────────────────────────────────
total_ingested = len(st.session_state.get("ingested_files", set()))
total_pending  = len(set(f.name for f in st.session_state.get("uploaded_files", [])) - st.session_state.get("ingested_files", set()))

st.markdown(f"""<div class="stats-row"><div class="stat-chip">📚 {total_ingested} paper{"s" if total_ingested != 1 else ""} ingested</div>{"<div class='stat-chip' style='color:#D97706;border-color:rgba(251,191,36,0.25);'>⏳ " + str(total_pending) + " pending</div>" if total_pending else ""}{"<div class='stat-chip' style='color:#059669;border-color:rgba(52,211,153,0.25);'>✓ Ready to chat</div>" if st.session_state.get("pipeline_ready") else ""}</div>""", unsafe_allow_html=True)


# ─── Main Split Layout ─────────────────────────────────────────────────────────
with st.container(key="main_content"):
    col1, col2 = st.columns([3, 7])

    # ── Left: File List ────────────────────────────────────────────────────────
    with col1:
        ingested = st.session_state.get('ingested_files', set())
        current_uploaded = st.session_state.get('uploaded_files', [])
        display_files = {}
        for filename in ingested:
            display_files[filename] = "ingested"
        for file in current_uploaded:
            if file.name not in ingested:
                display_files[file.name] = "pending"

        file_list_html = '<div class="file-panel">'
        file_list_html += """<div class="panel-header">
                                 <div class="panel-header-icon">📁</div>
                                 <div class="panel-header-text">Research Papers</div>
                               </div>
                           """

        if display_files:
            for name, status in display_files.items():
                dot_class = "status-ingested" if status == "ingested" else "status-pending"
                tag_class  = "tag-ingested"  if status == "ingested" else "tag-pending"
                tag_label  = "Ready"         if status == "ingested" else "Pending"
                file_list_html += f"""<div class="file-item">
                                             <div class="file-status-dot {dot_class}"></div>
                                             <div class="file-name" title="{name}">{name}</div>
                                             <span class="file-tag {tag_class}">{tag_label}</span>
                                          </div>"""
        else:
            file_list_html += """<div class="empty-state">
                                    <div class="empty-state-icon">📄</div>
                                    <div class="empty-state-title">No papers yet</div>
                                    <div class="empty-state-text">Upload PDF research papers above to get started</div>
                                </div>"""

        file_list_html += '</div>'
        st.markdown(file_list_html, unsafe_allow_html=True)

    # ── Right: Chat ────────────────────────────────────────────────────────────
    with col2:
        if st.session_state.uploaded_files:
            if st.session_state.get("pipeline_ready"):
                if "agent" not in st.session_state:
                    st.session_state.agent = create_rag_agent(st.session_state.ingested_files)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                chat_container = st.container(height=450)
                with chat_container:
                    for message in st.session_state.messages:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                if prompt := st.chat_input("Ask anything about your research papers…"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking…"):
                                result = st.session_state.agent.invoke(
                                    {"messages": [{"role": "user", "content": prompt}]}
                                )
                                answer = extract_answer(result)
                                st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
            else:
                st.markdown("""
                <div class="chat-empty">
                    <div class="chat-empty-bubble">⚡</div>
                    <div class="chat-empty-title">Papers ready to ingest</div>
                    <div class="chat-empty-text">
                        Click <strong>Ingest Papers</strong> above to process your files and unlock the chat.
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-empty">
                <div class="chat-empty-bubble">💬</div>
                <div class="chat-empty-title">Start a conversation</div>
                <div class="chat-empty-text">
                    Upload research papers and ingest them to unlock your AI research companion.
                </div>
                <div class="chat-empty-steps">
                    <div class="chat-step">
                        <div class="chat-step-num">1</div>
                        <div class="chat-step-text">Upload PDF research papers above</div>
                    </div>
                    <div class="chat-step">
                        <div class="chat-step-num">2</div>
                        <div class="chat-step-text">Click <strong>Ingest Papers</strong> to process</div>
                    </div>
                    <div class="chat-step">
                        <div class="chat-step-num">3</div>
                        <div class="chat-step-text">Ask questions about your papers</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
   