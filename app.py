import streamlit as st 
import tempfile
import os 
from agent import create_rag_agent
from main import ingest, query 
from ingestion.config import graph 
from ingestion.vector_store import delete_paper_from_vector_store

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
    # Check what papers are already saved
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
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,100..900;1,100..900&display=swap');
            *, *::before, *::after { box-sizing: border-box; }

            html, body, [data-testid="stAppViewContainer"] {
                background-color: #F5F3FF !important;
                color: #1E1B4B !important;
                font-family: 'Noto Sans', sans-serif !important;
            }

            [data-testid="stHeader"] {
                background-color: #F5F3FF !important;
            }

            /* Hide streamlit chrome */
            #MainMenu, footer, header,
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] { display: none !important; }

            [data-testid="stMainBlockContainer"],
            .block-container {
                padding-top: 1.5rem !important;
            }

            .main-wrap {
                max-width: 100%;
                margin: 0 auto;
                padding: 0 20px;
            }

            .hero {
                margin-bottom: 20px;
                padding: 0px 0 10px;
            }
            .hero-title {
                font-family: 'Noto Sans', sans-serif;
                font-size: 32px;
                font-weight: 700;
                line-height: 1.1;
                color: #7C3AED;
                margin: 0 0 8px;
            }
            .hero-sub {
                font-family: 'Noto Sans', sans-serif;
                font-size: 16px;
                color: #6366F1;
                margin: 0;
            }

            .upload-section {
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }

            .file-list {
                background: white;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
                height: 600px;
                overflow-y: auto;
            }

            .main-window {
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
                height: 600px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #6B7280;
                font-size: 18px;
            }

            .file-item {
                background: #F5F3FF;
                border: 1px solid #E5E7EB;
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .file-icon {
                color: #F472B6;
                font-size: 16px;
            }

            .file-name {
                color: #374151;
                font-size: 14px;
                font-weight: 500;
                flex: 1;
            }

            .file-size {
                color: #6B7280;
                font-size: 12px;
            }

            .section-title {
                color: #7C3AED;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 16px;
            }

            /* Streamlit component styling */


            /* Upload row — dark bar */
            .st-key-upload_row {
                background: linear-gradient(135deg, #1E1B4B 0%, #252060 100%) !important;
                border-radius: 16px !important;
                padding: 10px 16px !important;
            }

            .st-key-upload_row [data-testid="stVerticalBlock"],
            .st-key-upload_row [data-testid="stHorizontalBlock"],
            .st-key-upload_row [data-testid="stColumn"],
            .st-key-upload_row [data-testid="stVerticalBlockBorderWrapper"],
            .st-key-upload_row [data-testid="stElementContainer"] {
                background: transparent !important;
            }

            .st-key-upload_row [data-testid="stHorizontalBlock"] {
                align-items: center !important;
            }

            .st-key-upload_row [data-testid="stFileUploader"] {
                margin-bottom: 0 !important;
            }

            .st-key-upload_row [data-testid="stFileUploaderDropzone"] {
                background: transparent !important;
                border: none !important;
            }

            /* Browse/Upload button — outlined purple */
            .st-key-upload_row [data-testid="stFileUploaderDropzone"] button {
                background: transparent !important;
                color: #A78BFA !important;
                border: 2px solid #7C3AED !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                font-size: 14px !important;
                height: auto !important;
                box-shadow: none !important;
            }

            .st-key-upload_row [data-testid="stFileUploaderDropzone"] button:hover {
                background: rgba(124, 58, 237, 0.15) !important;
                color: #C4B5FD !important;
            }

            /* Dropzone helper text */
            .st-key-upload_row [data-testid="stFileUploaderDropzone"] small,
            .st-key-upload_row [data-testid="stFileUploaderDropzone"] span {
                color: #94A3B8 !important;
            }

            /* Uploaded file chips */
            .st-key-upload_row [data-testid="stUploadedFile"] {
                background: rgba(255, 255, 255, 0.95) !important;
                border-radius: 8px !important;
                border: 1px solid #E5E7EB !important;
                color: #1E1B4B !important;
            }

            /* Ingest button — solid purple */
            .st-key-ingest_btn button {
                background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important;
                color: white !important;
                border: none !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
                font-size: 16px !important;
                height: 56px !important;
                box-shadow: 0 4px 12px rgba(124, 58, 237, 0.35) !important;
            }

            .st-key-ingest_btn button:hover {
                background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
                box-shadow: 0 6px 16px rgba(124, 58, 237, 0.45) !important;
            }

            .st-key-ingest_btn button:disabled {
                background: linear-gradient(135deg, #8B5CF6, #7C3AED) !important;
                color: rgba(255, 255, 255, 0.5) !important;
                box-shadow: none !important;
                opacity: 0.6 !important;
            }

            /* Loader */
            .loading-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(30, 27, 75, 0.85);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                z-index: 9999;
            }

            .loading-spinner {
                width: 48px;
                height: 48px;
                border: 4px solid rgba(255, 255, 255, 0.2);
                border-top: 4px solid #A78BFA;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-bottom: 24px;
            }

            @keyframes spin {
                to { transform: rotate(360deg); }
            }

            .loading-text {
                color: white;
                font-size: 18px;
                font-weight: 500;
                font-family: 'Noto Sans', sans-serif;
            }
 


            /* Make col2 itself the white box */
            .st-key-main_content [data-testid="stColumn"]:nth-child(2) > [data-testid="stVerticalBlock"] {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                height:600px;
            }

            [data-testid="stChatMessage"] * {
                color: #1E1B4B !important;
            }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-wrap">
    <div class="hero">
        <h1 class="hero-title">Research Comp</h1>
        <p class="hero-sub">Your research companion</p>
    </div>
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
    # filepath = os.path.join("./data/papers", filename)
    # if os.path.exists(filepath):
    #     os.remove(filepath)
    graph.query(
        "MATCH (n {source_file: $filename}) DETACH DELETE n",
        {"filename": filename},
    )
    delete_paper_from_vector_store(filename)


# Initialize session state for uploaded files
with st.container(key="upload_row"):
    upload_col, button_col = st.columns([8, 2])

    with upload_col:
        # File upload section
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
    
    with button_col:
        # st.markdown("<br>", unsafe_allow_html=True)
        if new_files:
            if st.button("▷  Ingest", type="primary", use_container_width=True, key="ingest_btn"):
                overlay = st.empty()
                overlay.markdown("""
                <div class="loading-overlay">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">Ingesting papers...</div>
                </div>
                """, unsafe_allow_html=True)
                new_uploaded = [f for f in st.session_state.uploaded_files if f.name in new_files]
                # saved_paths = save_uploaded_files(new_uploaded)
                ingestion = ingest(new_uploaded)
                if ingestion:
                    st.session_state.pipeline_ready = True
                    st.session_state.ingested_files = ingested_files | new_files
                    st.rerun()
        else:
            st.button("▷  Ingest", type="primary", disabled=True, use_container_width=True, key="ingest_btn")


# Main content area with 30-70 split

with st.container(key="main_content"):
    col1, col2 = st.columns([3, 7])

    # Left column (30%) - File list
    with col1:
        file_list_content = '<div class="file-list"><h3 class="section-title">📁 Uploaded Files</h3>'
        
        # papers_dir = "./data/papers"
        # saved_files = os.listdir(papers_dir) if os.path.exists(papers_dir) else []
        # pdf_files = [f for f in saved_files if f.endswith(".pdf")]
        ingested = st.session_state.get('ingested_files', set())
        current_uploaded = st.session_state.get('uploaded_files', [])
        display_files = {}
        
        
        for filename in ingested:
            display_files[filename] = "ingested"
        for file in current_uploaded:
            if file.name not in ingested:
                display_files[file.name] = "pending"
        if display_files:
            for name, status in display_files.items():
                icon = "✅" if status == "ingested" else "⏳"
                # file_size = os.path.getsize(os.path.join(papers_dir, filename)) / (1024 * 1024)  # Convert to MB
                file_list_content += f"""<div class="file-item"><span class="file-icon">{icon}</span><div class="file-name">{name}</div></div>"""
        else:
            file_list_content += """
            <div style="text-align: center; color: #6B7280; padding: 40px 0;">
                <div style="font-size: 48px; margin-bottom: 16px;">📄</div>
                <div>No files uploaded yet</div>
                <div style="font-size: 14px; margin-top: 8px;">Upload PDFs to get started</div>
            </div>
            """

        file_list_content += '</div>'
        st.markdown(file_list_content, unsafe_allow_html=True)

    # Right column (70%) - Main window (placeholder for future chatbot)

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
                if prompt := st.chat_input("Ask about your research papers"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                result = st.session_state.agent.invoke(
                                    {"messages": [{"role": "user", "content": prompt}]}
                                )
                                answer = result["messages"][-1].content 
                                print("LLM ANSWER", answer)
                                st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()
        else:
            print(st.session_state.uploaded_files)
            st.markdown("""
            <div style="margin-bottom:15px;">
                <div style="text-align: center;">
                    <div style="font-size: 64px; margin-bottom: 24px; color: #6B7280;">💬</div>
                    <div style="font-size: 24px; font-weight: 600; color: #374151; margin-bottom: 12px;">
                        Upload files to start
                    </div>
                    <div style="color: #6B7280;">
                        The chatbot will appear here once you upload research papers
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
   