"""Streamlit application for RAG PDF ingestion and querying."""

import asyncio
import html
import json
import logging
import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import inngest
import nest_asyncio
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rag.core.config import get_settings
from rag.models.schemas import LLMModel

# Fix for asyncio event loop in Streamlit
nest_asyncio.apply()

logger = logging.getLogger(__name__)
settings = get_settings()

# =============================================================================
# Constants
# =============================================================================

UPLOADS_DIR = Path("uploads")
ALL_DOCUMENTS_LABEL = "All Documents"
DEFAULT_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 0.5
MAX_CHAT_HISTORY = 50
API_BASE_URL = "http://localhost:8000"  # Direct backend URL for streaming

MODEL_OPTIONS: dict[str, str] = {
    "Llama 3.3 70B (Powerful)": LLMModel.LLAMA_3_3_70B.value,
    "Llama 3.1 8B (Fast)": LLMModel.LLAMA_3_1_8B.value,
    "Mixtral 8x7B (Balanced)": LLMModel.MIXTRAL_8X7B.value,
    "DeepSeek R1 70B (Reasoning)": LLMModel.DEEPSEEK_R1_70B.value,
    "Qwen QWQ 32B (Reasoning)": LLMModel.QWEN_QWQ_32B.value,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ChatMessage:
    """Represents a single chat message in history."""

    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)
    contexts: list[str] = field(default_factory=list)
    model: str = ""


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = """
<style>
.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.chat-message {
    padding: 1rem;
    border-radius: 0.75rem;
    margin-bottom: 0.75rem;
    animation: fadeIn 0.3s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.user-message {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    margin-left: 2rem;
}

.assistant-message {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    border: 1px solid #4a5568;
    color: #e2e8f0;
    margin-right: 2rem;
}

.source-context {
    background: #2d3748;
    border-left: 3px solid #667eea;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border-radius: 0 0.5rem 0.5rem 0;
    font-size: 0.9rem;
    color: #a0aec0;
}

.citation-badge {
    background-color: rgba(102, 126, 234, 0.4);
    color: #fff;
    padding: 0.1rem 0.4rem;
    border-radius: 0.3rem;
    font-weight: bold;
    font-size: 0.9em;
    margin: 0 0.2rem;
    border: 1px solid #667eea;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
}

.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    transition: transform 0.2s, box-shadow 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
}

.doc-card {
    background: #2d3748;
    border-radius: 0.5rem;
    padding: 0.75rem;
    margin: 0.5rem 0;
    border: 1px solid #4a5568;
}
</style>
"""

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================


def initialize_session_state() -> None:
    """Initialize all session state variables."""
    defaults = {
        "documents": [],
        "documents_loaded": False,
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_chat_history(message: ChatMessage) -> None:
    """Add a message to chat history."""
    st.session_state.chat_history.append(message)
    if len(st.session_state.chat_history) > MAX_CHAT_HISTORY:
        st.session_state.chat_history = st.session_state.chat_history[
            -MAX_CHAT_HISTORY:
        ]


def clear_chat_history() -> None:
    """Clear all chat history."""
    st.session_state.chat_history = []


def export_chat_to_markdown() -> str:
    """Export chat history to Markdown format."""
    lines = ["# RAG Chat Export", f"*Exported: {datetime.now().isoformat()}*", ""]

    for msg in st.session_state.chat_history:
        if msg.role == "user":
            lines.append("## Question")
            lines.append(f"> {msg.content}")
        else:
            lines.append(f"## Answer ({msg.model})")
            lines.append(msg.content)
            if msg.sources:
                lines.append("")
                lines.append("**Sources:**")
                for src in set(msg.sources):
                    lines.append(f"- {src}")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Inngest Client
# =============================================================================


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """Get cached Inngest client instance."""
    return inngest.Inngest(
        app_id=settings.inngest_app_id,
        is_production=False,
    )


# =============================================================================
# File Operations
# =============================================================================


def save_uploaded_file(file: UploadedFile) -> Path:
    """Save uploaded file to uploads directory."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOADS_DIR / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    logger.info("Saved uploaded file: %s", file_path.name)
    return file_path


# =============================================================================
# Backend Communication (Streaming & Inngest)
# =============================================================================


async def send_rag_ingest_event(file_path: Path) -> None:
    """Send document ingestion event to Inngest."""
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "file_path": str(file_path.resolve()),
                "source_id": file_path.name,
            },
        )
    )


def stream_chat_response(
    question: str,
    top_k: int,
    model: str,
    source_filter: str | None,
) -> Generator[str | dict[str, Any], None, None]:
    """Stream chat response from backend API."""
    url = f"{API_BASE_URL}/api/chat"
    payload = {
        "question": question,
        "model": model,
        "top_k": top_k,
        "source_filter": source_filter,
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=60) as response:
            if not response.ok:
                yield f"Error: {response.text}"
                return

            # First line contains metadata (sources/contexts)
            metadata_read = False
            for line in response.iter_lines():
                if not line:
                    continue
                
                decoded_line = line.decode("utf-8")
                
                if not metadata_read:
                    try:
                        metadata = json.loads(decoded_line)
                        yield metadata  # Yield metadata dictionary first
                        metadata_read = True
                        continue
                    except json.JSONDecodeError:
                        pass # Should be JSON, but if not, treat as text

                yield decoded_line

    except Exception as exc:
        yield f"Connection Error: {exc}"


async def send_list_documents_event() -> str:
    """Send list documents event to Inngest."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(name="rag/list_documents", data={})
    )
    return result[0]


async def send_delete_document_event(source_id: str) -> str:
    """Send delete document event to Inngest."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(name="rag/delete_document", data={"source_id": source_id})
    )
    return result[0]


# =============================================================================
# Helper Functions
# =============================================================================


def fetch_runs(event_id: str) -> list[dict[str, Any]]:
    """Fetch run information for a given event ID."""
    url = f"{settings.inngest_api_base}/events/{event_id}/runs"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Wait for Inngest run to complete and return output."""
    start = time.time()
    last_status: str | None = None

    while True:
        try:
            runs = fetch_runs(event_id)
        except requests.RequestException as exc:
            logger.warning("Failed to fetch run status: %s", exc)
            runs = []

        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                error_msg = run.get("output", {}).get("error", f"Run {status}")
                raise RuntimeError(error_msg)

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s (last status: {last_status})"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def get_available_documents() -> list[str]:
    """Fetch list of available documents."""
    try:
        event_id = asyncio.run(send_list_documents_event())
        output = wait_for_run_output(event_id, timeout_s=30.0)
        return output.get("documents", [])
    except Exception as exc:
        logger.warning("Failed to fetch documents: %s", exc)
        return []


def load_documents_if_needed() -> list[str]:
    """Load documents into session state if not already loaded."""
    initialize_session_state()

    if not st.session_state.documents_loaded:
        with st.spinner("Loading documents..."):
            st.session_state.documents = get_available_documents()
            st.session_state.documents_loaded = True

    return st.session_state.documents


def refresh_documents() -> None:
    """Force refresh of document list."""
    st.session_state.documents = get_available_documents()
    st.session_state.documents_loaded = True


def format_citations_to_badges(text: str) -> str:
    """Convert [1], [2] to HTML badges."""
    escaped = html.escape(text).replace("\n", "<br>")
    # Use Regex to replace [digits] with span class
    badged = re.sub(
        r"\[(\d+)\]",
        r'<span class="citation-badge">Reference [\1]</span>',
        escaped
    )
    return badged


# =============================================================================
# UI Components
# =============================================================================


def render_chat_message(msg: ChatMessage) -> None:
    """Render a single chat message."""
    css_class = "user-message" if msg.role == "user" else "assistant-message"
    role_label = "You" if msg.role == "user" else "Assistant"
    
    content_html = html.escape(msg.content).replace("\n", "<br>")
    
    # Apply badges for assistant messages
    if msg.role == "assistant":
        content_html = format_citations_to_badges(msg.content)

    st.markdown(
        f'<div class="chat-message {css_class}">'
        f"<strong>{role_label}</strong><br>{content_html}</div>",
        unsafe_allow_html=True,
    )

    if msg.role == "assistant" and msg.contexts:
        with st.expander(f"References ({len(msg.contexts)})"):
            for i, ctx in enumerate(msg.contexts, 1):
                st.markdown(
                    f'<div class="source-context">'
                    f"<strong>Reference [{i}]:</strong><br>{ctx[:500]}..."
                    f"</div>",
                    unsafe_allow_html=True,
                )


def render_chat_history() -> None:
    """Render the complete chat history."""
    if not st.session_state.chat_history:
        st.info("Start a conversation by asking a question below.")
        return

    for msg in st.session_state.chat_history:
        render_chat_message(msg)


def render_document_card(doc_name: str) -> None:
    """Render a document card with delete button."""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            f'<div class="doc-card">{doc_name}</div>',
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Delete", key=f"del_{doc_name}"):
            with st.spinner(f"Deleting {doc_name}..."):
                try:
                    asyncio.run(send_delete_document_event(doc_name))
                    st.success(f"Deleted {doc_name}")
                    refresh_documents()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to delete: {exc}")


# =============================================================================
# Main Application
# =============================================================================

initialize_session_state()

# Sidebar
with st.sidebar:
    st.title("RAG Assistant")
    st.caption("Intelligent Document Q&A")

    st.divider()

    st.header("Documents")
    available_docs = load_documents_if_needed()
    st.metric("Loaded", len(available_docs))

    if st.button("Refresh", use_container_width=True):
        refresh_documents()
        st.rerun()

    if available_docs:
        with st.expander("Manage Documents", expanded=False):
            for doc in available_docs:
                render_document_card(doc)

    st.divider()

    st.header("Chat History")
    st.metric("Messages", len(st.session_state.chat_history))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear", use_container_width=True):
            clear_chat_history()
            st.rerun()
    with col2:
        if st.session_state.chat_history:
            md_export = export_chat_to_markdown()
            st.download_button(
                "Export",
                data=md_export,
                file_name=f"chat_export_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown",
                use_container_width=True,
            )

# Main Content
tab_chat, tab_upload = st.tabs(["Chat", "Upload Documents"])

with tab_chat:
    st.title("Ask Questions About Your Documents")

    chat_container = st.container()
    with chat_container:
        render_chat_history()

    st.divider()

    with st.form("query_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            placeholder="What would you like to know about your documents?",
            height=80,
        )

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            document_options = [ALL_DOCUMENTS_LABEL, *available_docs]
            selected_document = st.selectbox(
                "Document Filter",
                options=document_options,
            )
        with col2:
            selected_model_label = st.selectbox(
                "Model",
                options=list(MODEL_OPTIONS.keys()),
            )
        with col3:
            top_k = st.number_input("Chunks", min_value=1, max_value=20, value=5)

        submitted = st.form_submit_button("Submit", use_container_width=True)

    if submitted and question.strip():
        selected_model = MODEL_OPTIONS[selected_model_label]
        source_filter = (
            None if selected_document == ALL_DOCUMENTS_LABEL else selected_document
        )

        user_msg = ChatMessage(role="user", content=question.strip())
        add_to_chat_history(user_msg)
        
        # Display user message immediately (manual add to container so we don't need rerun yet)
        with chat_container:
            render_chat_message(user_msg)

        # Container for streaming response
        with chat_container:
            assistant_placeholder = st.empty()
            
            full_response = ""
            sources = []
            contexts = []
            
            # Start streaming
            stream = stream_chat_response(
                question.strip(),
                int(top_k),
                selected_model,
                source_filter,
            )
            
            # First item should be metadata
            try:
                first_chunk = next(stream)
                if isinstance(first_chunk, dict):
                    sources = first_chunk.get("sources", [])
                    contexts = first_chunk.get("contexts", [])
                else:
                    full_response += str(first_chunk)
                    # Styling for first chunk if streaming failure
                    content = format_citations_to_badges(full_response)
                    html_content = (
                        f'<div class="chat-message assistant-message">'
                        f'<strong>Assistant</strong><br>{content}▌</div>'
                    )
                    assistant_placeholder.markdown(html_content, unsafe_allow_html=True)
            except StopIteration:
                pass

            # Loop through remainder
            import html
            
            for chunk in stream:
                if isinstance(chunk, str):
                    full_response += chunk
                    # Wrap in HTML to match styling immediately
                    escaped_text = html.escape(full_response).replace("\n", "<br>")
                    html_content = (
                        f'<div class="chat-message assistant-message">'
                        f'<strong>Assistant</strong><br>{escaped_text}▌</div>'
                    )
                    assistant_placeholder.markdown(html_content, unsafe_allow_html=True)
            
            # Final render without cursor
            escaped_final = html.escape(full_response).replace("\n", "<br>")
            final_html = (
                f'<div class="chat-message assistant-message">'
                f'<strong>Assistant</strong><br>{escaped_final}</div>'
            )
            assistant_placeholder.markdown(final_html, unsafe_allow_html=True)
            
            # Add sources expander manually after stream
            if contexts:
                 with st.expander(f"References ({len(contexts)})"):
                    for i, ctx in enumerate(contexts, 1):
                        st.markdown(
                            f'<div class="source-context">'
                            f"<strong>Reference [{i}]:</strong><br>{ctx[:500]}..."
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Save to history
            assistant_msg = ChatMessage(
                role="assistant",
                content=full_response,
                sources=sources,
                contexts=contexts,
                model=selected_model_label,
            )
            add_to_chat_history(assistant_msg)


with tab_upload:
    st.title("Upload Documents for Ingestion")
    st.caption("Supported formats: PDF, DOCX, TXT, MD")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload files to add to your knowledge base",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")

        if st.button("Start Ingestion", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")
                progress = (i + 1) / len(uploaded_files)
                progress_bar.progress(progress)

                try:
                    path = save_uploaded_file(file)
                    asyncio.run(send_rag_ingest_event(path))
                    time.sleep(0.3)
                except Exception as exc:
                    st.error(f"Failed to process {file.name}: {exc}")
                    logger.exception("Upload failed for %s", file.name)

            progress_bar.progress(1.0)
            status_text.text("All files processed!")
            st.success(f"Ingested {len(uploaded_files)} file(s)")
            st.caption("Click 'Refresh' in the sidebar to see new documents.")

            refresh_documents()
