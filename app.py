"""
Document Intelligence Assistant (v4)
Entry point — Chainlit lifecycle hooks.
"""
import hashlib
import logging
import os

import chainlit as cl
from chainlit.input_widget import Select
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from src.cache import ResponseCache
from src.citation_parser import CitationParser
from src.config import AppConfig
from src.follow_up_detector import FollowUpDetector
from src.ingestor import DocumentIngestor
from src.prompt_builder import OperationalMode, PromptBuilder
from src.rag_engine import RAGEngine
from src.rate_limiter import QuotaExhaustedError, RateLimiter

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
try:
    config = AppConfig.from_env()
except ValueError as e:
    logger.error("Configuration error: %s", e)
    raise

if config.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = config.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = config.langsmith_project
    logger.info("LangSmith tracing enabled for project: %s", config.langsmith_project)
else:
    logger.warning("WARNING: LangSmith credentials not configured. Observability disabled.")

# ---------------------------------------------------------------------------
# Shared module-level instances
# ---------------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model=config.model_name,
    google_api_key=config.google_api_key,
    disable_streaming=False,
    temperature=0.1,
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

prompt_builder = PromptBuilder()
citation_parser = CitationParser()

MODE_LABELS = {
    "Single Document": OperationalMode.SINGLE_DOC,
    "Compare Documents": OperationalMode.COMPARE,
}

MODE_DISPLAY = {
    OperationalMode.SINGLE_DOC: "📄 Single Document Mode",
    OperationalMode.COMPARE: "🔀 Compare Documents Mode",
}


# ---------------------------------------------------------------------------
# on_chat_start
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", OperationalMode.SINGLE_DOC)
    cl.user_session.set("vector_store", None)
    cl.user_session.set("uploaded_docs", [])
    cl.user_session.set("doc_filenames", {})       # doc_id → filename
    cl.user_session.set("doc_bytes", {})            # filename → file_bytes (for rebuild)
    cl.user_session.set("pdf_paths", {})            # filename → file_path
    cl.user_session.set("active_pdf", None)
    cl.user_session.set("cache", ResponseCache())
    cl.user_session.set("rate_limiter", RateLimiter())
    cl.user_session.set("previous_turn", None)

    await cl.ChatSettings([
        Select(
            id="mode",
            label="Analysis Mode",
            values=list(MODE_LABELS.keys()),
            initial_value="Single Document",
        )
    ]).send()

    await cl.Message(content=(
        "# 📚 Document Intelligence Assistant\n\n"
        "Welcome! I'm your enterprise-grade PDF analysis assistant.\n\n"
        "**To get started:**\n"
        "1. **Select your mode** via ⚙️ Settings (top-right) or type a command:\n"
        "   - `/single` — Single Document mode (summary, key insights, Q&A)\n"
        "   - `/compare` — Compare Documents mode (synthesis, comparison table, discrepancies)\n"
        "2. Upload PDF file(s) using the 📎 attachment button\n"
        "3. Ask me anything about your documents!\n\n"
        "**Conversation tips:**\n"
        "- Each query is answered fresh by default\n"
        "- Use follow-up phrases like *\"tell me more\"*, *\"expand on that\"* to continue from the previous answer\n"
        "- Prefix with `new query:` or `fresh:` to force a standalone query\n\n"
        "*All responses are grounded exclusively in your uploaded documents.*"
    )).send()


# ---------------------------------------------------------------------------
# on_settings_update — mode toggle
# ---------------------------------------------------------------------------
@cl.on_settings_update
async def on_settings_update(settings: dict):
    mode_label = settings.get("mode", "Single Document")
    new_mode = MODE_LABELS.get(mode_label, OperationalMode.SINGLE_DOC)
    old_mode = cl.user_session.get("mode")

    if new_mode == old_mode:
        return

    cl.user_session.set("mode", new_mode)

    # Clear all document state on mode switch
    cache: ResponseCache = cl.user_session.get("cache")
    cache.invalidate_all()
    cl.user_session.set("vector_store", None)
    cl.user_session.set("uploaded_docs", [])
    cl.user_session.set("doc_filenames", {})
    cl.user_session.set("doc_bytes", {})
    cl.user_session.set("pdf_paths", {})
    cl.user_session.set("active_pdf", None)
    cl.user_session.set("previous_turn", None)

    if new_mode == OperationalMode.COMPARE:
        msg = (
            f"✅ Switched to **{MODE_DISPLAY[new_mode]}**.\n\n"
            "Please upload **2 or more PDF files** to compare. "
            "Once uploaded, ask any question and I'll provide:\n"
            "- A unified **Synthesis** across all documents\n"
            "- A **Comparison Table** of key points\n"
            "- **Discrepancy Analysis** highlighting conflicts"
        )
    else:
        msg = (
            f"✅ Switched to **{MODE_DISPLAY[new_mode]}**.\n"
            "Please upload a PDF to begin analysis."
        )

    await cl.Message(content=msg).send()


# ---------------------------------------------------------------------------
# on_message — file uploads + text queries
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):

    # --- File uploads ---
    if message.elements:
        for element in message.elements:
            if not isinstance(element, cl.File):
                continue

            filename = element.name
            file_path = element.path

            if not filename.lower().endswith(".pdf"):
                await cl.Message(
                    content=f"❌ **{filename}**: Only PDF files are accepted."
                ).send()
                continue

            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            except Exception as e:
                await cl.Message(content=f"❌ Could not read **{filename}**: {e}").send()
                continue

            # --- Update session state BEFORE indexing ---
            doc_id = hashlib.sha256(filename.encode("utf-8")).hexdigest()
            mode: OperationalMode = cl.user_session.get("mode")

            # Add to doc_bytes store
            doc_bytes: dict = cl.user_session.get("doc_bytes") or {}
            doc_bytes[filename] = file_bytes
            cl.user_session.set("doc_bytes", doc_bytes)

            # Add to doc_filenames
            doc_filenames: dict = cl.user_session.get("doc_filenames") or {}
            doc_filenames[doc_id] = filename
            cl.user_session.set("doc_filenames", doc_filenames)

            # Add to uploaded_docs
            uploaded_docs: list = cl.user_session.get("uploaded_docs") or []
            if doc_id not in uploaded_docs:
                uploaded_docs.append(doc_id)
            cl.user_session.set("uploaded_docs", uploaded_docs)

            # Add to pdf_paths
            pdf_paths: dict = cl.user_session.get("pdf_paths") or {}
            pdf_paths[filename] = file_path
            cl.user_session.set("pdf_paths", pdf_paths)
            cl.user_session.set("active_pdf", filename)
            cl.user_session.set("previous_turn", None)

            # --- Rebuild vector store from ALL documents ---
            all_files = list(doc_bytes.items())  # [(filename, bytes), ...]
            logger.info(
                "Rebuilding vector store from %d document(s): %s",
                len(all_files), [f for f, _ in all_files]
            )

            async with cl.Step(name=f"📄 Indexing {len(all_files)} document(s)") as step:
                step.output = f"Building index from {len(all_files)} document(s)…"
                ingestor = DocumentIngestor(
                    embeddings=embeddings,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                try:
                    await cl.sleep(0)
                    vector_store = ingestor.ingest(all_files)
                    cl.user_session.set("vector_store", vector_store)

                    # Verify — scan ALL docs in the store
                    all_stored_docs = list(vector_store.docstore._dict.values())
                    sources_in_store = set(
                        d.metadata.get("source", "?") for d in all_stored_docs
                    )
                    logger.info(
                        "✅ Vector store rebuilt. Sources: %s | Total chunks: %d",
                        sources_in_store, len(all_stored_docs)
                    )
                    step.output = f"✅ Indexed {len(all_files)} document(s)"

                except ValueError as e:
                    await cl.Message(content=f"❌ {e}").send()
                    # Rollback the doc_bytes entry for this failed file
                    doc_bytes.pop(filename, None)
                    cl.user_session.set("doc_bytes", doc_bytes)
                    continue

            num_docs = len(doc_filenames)

            # Do NOT auto-switch mode — the user controls mode explicitly
            # via /single, /compare, or the ⚙️ Settings panel.
            current_mode = cl.user_session.get("mode")
            if current_mode == OperationalMode.COMPARE:
                loaded_names = ", ".join(doc_filenames.values())
                status = (
                    f"✅ **{filename}** indexed.\n"
                    f"**{num_docs} document(s) loaded:** {loaded_names}\n\n"
                    "🔀 **Compare Documents mode** is active.\n"
                    "Ask a question to get synthesis, comparison table, and discrepancy analysis."
                )
            else:
                status = f"✅ **{filename}** uploaded and indexed."
                if num_docs > 1:
                    loaded_names = ", ".join(doc_filenames.values())
                    status += (
                        f"\n\n**{num_docs} document(s) loaded:** {loaded_names}\n"
                        "� Type `/compare` to switch to Compare Documents mode, "
                        "or keep asking questions in Single Document mode."
                    )
                else:
                    status += " Ask me anything about it!"

            await cl.Message(
                content=status,
                elements=[cl.Pdf(name=filename, display="side", path=file_path)],
            ).send()

        if not message.content or not message.content.strip():
            return

    # --- Text query ---
    query = (message.content or "").strip()
    if not query:
        return

    # --- Inline mode-switch commands ---
    query_lower = query.lower().strip()
    if query_lower in ("/single", "/single document", "switch to single", "single mode"):
        current_mode = cl.user_session.get("mode")
        if current_mode != OperationalMode.SINGLE_DOC:
            cl.user_session.set("mode", OperationalMode.SINGLE_DOC)
            cache: ResponseCache = cl.user_session.get("cache")
            cache.invalidate_all()
            cl.user_session.set("vector_store", None)
            cl.user_session.set("uploaded_docs", [])
            cl.user_session.set("doc_filenames", {})
            cl.user_session.set("doc_bytes", {})
            cl.user_session.set("pdf_paths", {})
            cl.user_session.set("active_pdf", None)
            cl.user_session.set("previous_turn", None)
            await cl.Message(
                content=f"✅ Switched to **{MODE_DISPLAY[OperationalMode.SINGLE_DOC]}**.\n"
                        "Please upload a PDF to begin analysis."
            ).send()
        else:
            await cl.Message(
                content=f"ℹ️ Already in **{MODE_DISPLAY[OperationalMode.SINGLE_DOC]}**."
            ).send()
        return

    if query_lower in ("/compare", "/compare documents", "switch to compare", "compare mode"):
        current_mode = cl.user_session.get("mode")
        if current_mode != OperationalMode.COMPARE:
            cl.user_session.set("mode", OperationalMode.COMPARE)
            cache: ResponseCache = cl.user_session.get("cache")
            cache.invalidate_all()
            cl.user_session.set("vector_store", None)
            cl.user_session.set("uploaded_docs", [])
            cl.user_session.set("doc_filenames", {})
            cl.user_session.set("doc_bytes", {})
            cl.user_session.set("pdf_paths", {})
            cl.user_session.set("active_pdf", None)
            cl.user_session.set("previous_turn", None)
            await cl.Message(
                content=f"✅ Switched to **{MODE_DISPLAY[OperationalMode.COMPARE]}**.\n\n"
                        "Please upload **2 or more PDF files** to compare. "
                        "Once uploaded, ask any question and I'll provide:\n"
                        "- A unified **Synthesis** across all documents\n"
                        "- A **Comparison Table** of key points\n"
                        "- **Discrepancy Analysis** highlighting conflicts"
            ).send()
        else:
            await cl.Message(
                content=f"ℹ️ Already in **{MODE_DISPLAY[OperationalMode.COMPARE]}**."
            ).send()
        return

    if query_lower in ("/mode", "/status", "what mode", "current mode"):
        current_mode = cl.user_session.get("mode")
        doc_filenames: dict = cl.user_session.get("doc_filenames", {})
        docs_info = f" | **{len(doc_filenames)} doc(s) loaded**" if doc_filenames else " | No documents loaded"
        await cl.Message(
            content=f"ℹ️ Current mode: **{MODE_DISPLAY[current_mode]}**{docs_info}\n\n"
                    "Switch with `/single` or `/compare`."
        ).send()
        return

    # --- Conversational context detection ---
    previous_turn = cl.user_session.get("previous_turn")
    _detector = FollowUpDetector()
    _detection = _detector.detect(query, previous_turn)
    query = _detection.cleaned_query  # use cleaned query (fresh prefix stripped if present)

    if _detection.is_followup:
        await cl.Message(content="🔗 *Treating as follow-up to previous response…*").send()

    # Pass previous_turn to RAGEngine only when it's a follow-up
    turn_context = previous_turn if _detection.is_followup else None

    vector_store = cl.user_session.get("vector_store")
    if vector_store is None:
        await cl.Message(content="⚠️ Please upload a PDF document first.").send()
        return

    mode: OperationalMode = cl.user_session.get("mode")
    doc_filenames: dict = cl.user_session.get("doc_filenames", {})
    uploaded_docs: list = cl.user_session.get("uploaded_docs", [])

    # Enforce minimum 2 docs for Compare mode
    if mode == OperationalMode.COMPARE and len(doc_filenames) < 2:
        await cl.Message(
            content=(
                "⚠️ **Compare mode requires at least 2 documents.**\n"
                f"You currently have **{len(doc_filenames)}** document(s) loaded. "
                "Please upload another PDF to enable comparison."
            )
        ).send()
        return

    cache: ResponseCache = cl.user_session.get("cache")
    rate_limiter: RateLimiter = cl.user_session.get("rate_limiter")
    pdf_paths: dict = cl.user_session.get("pdf_paths", {})

    rag_engine = RAGEngine(
        vector_store=vector_store,
        llm=llm,
        embeddings=embeddings,
        prompt_builder=prompt_builder,
        cache=cache,
        rate_limiter=rate_limiter,
        top_k=config.top_k,
        max_retries=config.max_retries,
        initial_retry_delay=config.initial_retry_delay,
    )

    # Show what we're doing
    if mode == OperationalMode.COMPARE:
        doc_list = ", ".join(f"**{v}**" for v in doc_filenames.values())
        thinking_msg = await cl.Message(
            content=f"🔍 Comparing {len(doc_filenames)} documents: {doc_list}\n\n*Retrieving context from all documents…*"
        ).send()
    else:
        thinking_msg = await cl.Message(
            content="🔍 *Retrieving context and generating response…*"
        ).send()

    try:
        callback_handler = None
        if config.langsmith_api_key:
            try:
                from langchain.callbacks.tracers.langchain import LangChainTracer
                from langsmith import Client
                callback_handler = LangChainTracer(
                    project_name=config.langsmith_project,
                    client=Client(api_key=config.langsmith_api_key),
                )
            except Exception as tracer_err:
                logger.warning("LangSmith tracer init failed: %s", tracer_err)

        rag_response = await rag_engine.query(
            query=query,
            mode=mode,
            doc_ids=uploaded_docs,
            callback_handler=callback_handler,
            doc_filenames=doc_filenames,
            previous_turn=turn_context,
        )

    except QuotaExhaustedError as e:
        await cl.Message(content=f"🚫 **Daily quota reached**: {e}").send()
        return
    except Exception as e:
        error_msg = str(e).lower()
        if "quota" in error_msg or "429" in error_msg or "rate" in error_msg:
            await cl.Message(
                content="⏳ **API quota reached.** Please wait a few minutes and try again."
            ).send()
        else:
            logger.error("RAG query failed: %s", e)
            await cl.Message(content=f"❌ An error occurred: {e}").send()
        return

    # Store this turn for potential follow-up queries (success path only)
    cl.user_session.set("previous_turn", {"query": query, "answer": rag_response.answer})

    # --- Render response ---
    answer = rag_response.answer
    quality_report = rag_response.quality_report

    import re
    # Defensive cleanup: strip any JSON quality block the LLM might still emit
    clean_answer = re.sub(
        r'''```(?:json)?\s*\{[^}]*"confidence_score"[^}]*\}\s*```''',
        "",
        answer,
        flags=re.DOTALL,
    ).strip()

    quality_footer = (
        f"\n\n---\n"
        f"📊 **Confidence:** `{quality_report.confidence_score:.0f}/100` | "
        f"**Keyword Match:** `{quality_report.keyword_match_accuracy:.0f}/100` | "
        f"**Faithfulness:** `{quality_report.answer_faithfulness_score:.0f}/100` | "
        f"**Retrieval Quality:** `{quality_report.retrieval_quality:.0f}/100` | "
        f"**Citation Accuracy:** `{quality_report.citation_accuracy:.0f}/100`"
    )

    final_content = clean_answer + quality_footer

    active_pdf = cl.user_session.get("active_pdf")
    elements = []
    if active_pdf and active_pdf in pdf_paths:
        elements = [cl.Pdf(name=active_pdf, display="side", path=pdf_paths[active_pdf])]

    await cl.Message(content=final_content, elements=elements).send()


# ---------------------------------------------------------------------------
# action_callback — PDF navigation
# ---------------------------------------------------------------------------
@cl.action_callback("navigate_pdf")
async def on_navigate_pdf(action: cl.Action):
    try:
        parts = action.value.rsplit(":", 1)
        if len(parts) != 2:
            return
        filename, page = parts[0], int(parts[1])
        pdf_paths: dict = cl.user_session.get("pdf_paths", {})
        if filename not in pdf_paths:
            await cl.Message(content=f"⚠️ PDF '{filename}' is not loaded.").send()
            return
        cl.user_session.set("active_pdf", filename)
        await cl.Message(
            content=f"📄 Navigating to **{filename}**, page **{page}**…",
            elements=[cl.Pdf(name=filename, display="side", path=pdf_paths[filename], page=page)],
        ).send()
    except Exception as e:
        logger.error("Citation navigation failed: %s", e)
        await cl.Message(content=f"❌ Could not navigate: {e}").send()
