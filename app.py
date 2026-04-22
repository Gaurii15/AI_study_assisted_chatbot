import os
from io import BytesIO
from typing import List, Dict

import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted


# -----------------------------
# App configuration + UI styles
# -----------------------------
st.set_page_config(
    page_title="StudyBuddy AI — AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

# Simple, modern color palette (dark-ish card UI on light background)
st.markdown(
    """
    <style>
      :root {
        --bg: #0b1220;
        --card: rgba(255, 255, 255, 0.06);
        --card2: rgba(255, 255, 255, 0.08);
        --text: rgba(255, 255, 255, 0.92);
        --muted: rgba(255, 255, 255, 0.70);
        --accent: #7c3aed; /* purple */
        --accent2: #06b6d4; /* cyan */
        --border: rgba(255, 255, 255, 0.14);
      }

      /* Background */
      .stApp {
        background: radial-gradient(1200px 600px at 15% 10%, rgba(124,58,237,0.35), transparent 60%),
                    radial-gradient(900px 500px at 80% 25%, rgba(6,182,212,0.25), transparent 60%),
                    linear-gradient(180deg, #070b14 0%, #0b1220 100%);
        color: var(--text);
      }

      /* Headings */
      h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.02em;
      }

      /* Sidebar */
      section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.04);
        border-right: 1px solid var(--border);
      }

      /* Containers look like cards */
      div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
      }

      /* Buttons */
      .stButton > button {
        border-radius: 12px;
        border: 1px solid var(--border);
        background: linear-gradient(135deg, rgba(124,58,237,0.55), rgba(6,182,212,0.45));
        color: white;
        font-weight: 650;
        padding: 0.6rem 0.95rem;
      }
      .stButton > button:hover {
        border-color: rgba(255,255,255,0.22);
        filter: brightness(1.05);
      }

      /* Inputs */
      .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        background: rgba(255,255,255,0.06) !important;
        color: var(--text) !important;
      }

      /* Chat bubbles */
      div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 10px 12px;
      }
      .stCaption, .stMarkdown p, .stMarkdown li {
        color: var(--muted);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def init_session_state() -> None:
    """Initialize Streamlit session state keys used by the app."""
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = ""
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = ""
    if "chat_history" not in st.session_state and "messages" not in st.session_state:
        # Single source of truth for the chat UI + model context.
        st.session_state.chat_history: List[Dict[str, str]] = []
        # Backward-compat alias: other app features already use `messages`.
        st.session_state.messages = st.session_state.chat_history
    elif "chat_history" not in st.session_state and "messages" in st.session_state:
        # Migrate existing sessions to the new key without copying.
        st.session_state.chat_history = st.session_state.messages
    elif "messages" not in st.session_state and "chat_history" in st.session_state:
        st.session_state.messages = st.session_state.chat_history
    if "mode" not in st.session_state:
        st.session_state.mode = "Normal"
    if "generated_notes" not in st.session_state:
        st.session_state.generated_notes = ""
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "user_score" not in st.session_state:
        st.session_state.user_score = 0
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
    if "quiz_feedback" not in st.session_state:
        st.session_state.quiz_feedback = ""
    if "quiz_verdict" not in st.session_state:
        st.session_state.quiz_verdict = ""
    if "quiz_answer" not in st.session_state:
        st.session_state.quiz_answer = ""


def extract_pdf_text(file) -> str:
    """
    Extract text from a PDF using PyPDF2.
    Returns a best-effort plaintext string (may be empty for scanned PDFs).
    """
    reader = PdfReader(file)
    chunks: List[str] = []
    for page in reader.pages:
        # page.extract_text() can return None
        text = page.extract_text() or ""
        if text.strip():
            chunks.append(text)
    return "\n\n".join(chunks).strip()


def get_api_key() -> str:
    """
    Read Groq API key from Streamlit secrets or environment variables.
    Preferred: st.secrets["GROQ_API_KEY"].
    """
    return st.secrets.get("GROQ_API_KEY", "") or os.getenv("GROQ_API_KEY", "")


def build_system_prompt(mode: str, pdf_text: str) -> str:
    """
    Create a system prompt that:
    - Sets the assistant behavior based on the selected mode
    - Constrains answers to the uploaded PDF content (when present)
    """
    mode_instruction = {
        "Normal": (
            "Answer clearly and accurately. Use the PDF content as the primary source. "
            "If the PDF doesn't contain the answer, say so and offer what you can infer."
        ),
        "Explain Simply": (
            "Explain like I'm a beginner. Keep it short, use simple language, and include a tiny example if helpful. "
            "Base the answer on the PDF content. If the PDF doesn't contain it, say so."
        ),
        "Exam Answer": (
            "Write an exam-style answer: structured, concise, and point-wise when appropriate. "
            "Use key terms from the PDF. If the PDF doesn't contain it, say so."
        ),
    }.get(mode, "Answer based on the PDF content.")

    # We include PDF content directly in the system prompt (simple RAG-in-prompt approach).
    # To reduce "context length exceeded" errors, we cap the included text.
    pdf_block = pdf_text.strip()
    max_chars = 25_000  # pragmatic cap for a single-file demo app
    truncated_note = ""
    if len(pdf_block) > max_chars:
        pdf_block = pdf_block[:max_chars]
        truncated_note = (
            "\n\nNote: The PDF is long; only the first portion was provided. "
            "If the answer is missing, ask for the relevant page/section."
        )
    if pdf_block:
        return (
            "You are StudyBuddy AI, an AI Study Assistant.\n\n"
            f"Mode: {mode}\n"
            f"Instructions: {mode_instruction}\n\n"
            "Use ONLY the following PDF content as your source of truth:\n"
            "----- BEGIN PDF -----\n"
            f"{pdf_block}\n"
            "----- END PDF -----\n"
            f"{truncated_note}\n"
        )

    return (
        "You are StudyBuddy AI, an AI Study Assistant.\n\n"
        f"Mode: {mode}\n"
        f"Instructions: {mode_instruction}\n\n"
        "No PDF content is available yet. Ask the user to upload a PDF, "
        "or answer as a general study helper if they explicitly request non-PDF help."
    )


def groq_chat(messages: List[Dict[str, str]]) -> str:
    """
    Call Groq Chat Completions via the Groq Python SDK.
    Uses model: llama3-8b-8192
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing Groq API key. Set GROQ_API_KEY in your environment or in Streamlit secrets."
        )

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
    )
    return (resp.choices[0].message.content or "").strip()


def build_study_pdf_bytes(
    title: str, sections: List[Dict[str, str]]
) -> bytes:
    """
    Build a PDF in-memory using reportlab.

    sections: [{"heading": "...", "content": "..."}]
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=title,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 14),
    ]

    for sec in sections:
        heading = (sec.get("heading") or "").strip()
        content = (sec.get("content") or "").strip()
        if not content:
            continue
        if heading:
            story.append(Paragraph(heading, styles["Heading2"]))
            story.append(Spacer(1, 8))
        story.append(Preformatted(content, styles["Code"]))
        story.append(Spacer(1, 14))

    doc.build(story)
    return buf.getvalue()


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """
    Convert chat history into a readable plain-text transcript.
    Example:
      User: Hello
      Bot: Hi, how can I help you?
    """
    lines: List[str] = []
    for m in chat_history:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Bot"
        else:
            prefix = role.capitalize() if role else "Message"
        lines.append(f"{prefix}: {content}")
    return "\n\n".join(lines).strip() + ("\n" if lines else "")


# -----------------------------
# App
# -----------------------------
init_session_state()

st.markdown(
    """
    <div style="display:flex; align-items:flex-end; gap:14px; padding: 2px 2px 10px 2px;">
      <div style="font-size:34px; line-height:1;">AI Study Assistant</div>
      <div style="color: rgba(255,255,255,0.70); font-size:14px; padding-bottom:6px;">
        Powered by StudyBuddy AI
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Upload a PDF in the sidebar, then ask questions and study smarter.")


# Sidebar: upload + modes + actions
with st.sidebar:
    st.subheader("Study Tools")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    st.session_state.mode = st.selectbox(
        "Answer mode",
        options=["Normal", "Explain Simply", "Exam Answer"],
        index=["Normal", "Explain Simply", "Exam Answer"].index(st.session_state.mode)
        if st.session_state.mode in ["Normal", "Explain Simply", "Exam Answer"]
        else 0,
        help="Changes how StudyBuddy AI formats the answer.",
    )

    col_a, col_b = st.columns(2)
    summarize_clicked = col_a.button("Summarize PDF (5)")
    questions_clicked = col_b.button("5 Questions")

    generate_notes_clicked = st.button("Generate Notes")
    start_quiz_clicked = st.button("Start Quiz")

    st.markdown("---")
    st.caption(
        "Upload a PDF and start asking questions 📄🤖"
    )


# Handle PDF upload (store extracted text in session state)
if uploaded_pdf is not None:
    # Only re-extract if file changed
    if st.session_state.pdf_name != uploaded_pdf.name:
        try:
            with st.spinner("Reading PDF..."):
                text = extract_pdf_text(uploaded_pdf)
            st.session_state.pdf_text = text
            st.session_state.pdf_name = uploaded_pdf.name
            if not text:
                st.sidebar.warning(
                    "PDF loaded, but no extractable text was found. If it's a scanned PDF, "
                    "you may need OCR before it can be used."
                )
            else:
                st.sidebar.success(f"Loaded: {uploaded_pdf.name}")
        except Exception as e:
            st.session_state.pdf_text = ""
            st.session_state.pdf_name = ""
            st.sidebar.error(f"Failed to read PDF: {e}")


# Main content layout
left, right = st.columns([1.25, 0.75], gap="large")

with right:
    st.subheader("PDF Status")
    if st.session_state.pdf_name:
        st.markdown(f"**File:** `{st.session_state.pdf_name}`")
        st.markdown(
            f"**Extracted text:** {len(st.session_state.pdf_text):,} characters"
        )
    else:
        st.info("No PDF uploaded yet. Upload one in the sidebar to enable PDF-based answers.")

    st.subheader("Controls")
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.session_state.messages = st.session_state.chat_history
        st.rerun()

    transcript = format_chat_history(st.session_state.chat_history)
    st.download_button(
        "Download chat history",
        data=transcript or "No chat history yet.\n",
        file_name="chat_history.txt",
        mime="text/plain",
    )

    # Download all generated study content as a PDF (notes/summary/quiz questions)
    sections: List[Dict[str, str]] = []

    if st.session_state.generated_notes.strip():
        sections.append(
            {"heading": "Notes", "content": st.session_state.generated_notes.strip()}
        )

    # Summary (best-effort): detect the most recent assistant message with exactly 5 bullets.
    detected_summary = ""
    for m in reversed(st.session_state.messages):
        if m.get("role") != "assistant":
            continue
        txt = (m.get("content") or "").strip()
        if not txt:
            continue
        bullet_lines = [
            ln.strip()
            for ln in txt.splitlines()
            if ln.strip().startswith(("-", "•"))
        ]
        if len(bullet_lines) == 5:
            detected_summary = txt
            break
    if detected_summary:
        sections.append({"heading": "Summary (5 points)", "content": detected_summary})

    if st.session_state.quiz_questions:
        quiz_txt = "\n\n".join(st.session_state.quiz_questions)
        sections.append({"heading": "Quiz Questions", "content": quiz_txt})

    if sections:
        pdf_bytes = build_study_pdf_bytes("StudyBuddy AI Notes", sections)
        pdf_base = (
            os.path.splitext(st.session_state.pdf_name)[0].strip()
            if st.session_state.pdf_name
            else "studybuddy"
        )
        st.download_button(
            "Download Study सामग्री as PDF",
            data=pdf_bytes,
            file_name=f"{pdf_base}_studybuddy.pdf",
            mime="application/pdf",
        )

    st.subheader("Generated Notes")
    if st.session_state.generated_notes.strip():
        st.markdown(st.session_state.generated_notes)
        pdf_base = (
            os.path.splitext(st.session_state.pdf_name)[0].strip()
            if st.session_state.pdf_name
            else "studybuddy"
        )
        st.download_button(
            "Download Notes",
            data=st.session_state.generated_notes,
            file_name=f"{pdf_base}_notes.md",
            mime="text/markdown",
        )
    else:
        st.caption("Generate notes from the uploaded PDF to see them here.")

    st.subheader("Quiz Mode")
    if st.session_state.quiz_active and st.session_state.quiz_questions:
        total_q = len(st.session_state.quiz_questions)
        idx = int(st.session_state.current_question_index)
        idx = max(0, min(idx, total_q - 1))
        st.session_state.current_question_index = idx

        st.markdown(
            f"**Question {idx + 1} / {total_q}**  \n"
            f"**Score:** {int(st.session_state.user_score)}"
        )
        st.markdown(st.session_state.quiz_questions[idx])

        st.session_state.quiz_answer = st.text_area(
            "Your answer",
            value=st.session_state.quiz_answer,
            key="quiz_answer_input",
            height=120,
        )
        submit_quiz_answer = st.button("Submit Answer")

        if submit_quiz_answer:
            if not st.session_state.quiz_answer.strip():
                st.warning("Please enter an answer before submitting.")
            else:
                try:
                    with st.spinner("Checking your answer..."):
                        system = build_system_prompt(
                            st.session_state.mode, st.session_state.pdf_text
                        )
                        question = st.session_state.quiz_questions[idx]
                        user_answer = st.session_state.quiz_answer.strip()
                        eval_prompt = (
                            "You are grading a student's answer based ONLY on the PDF content.\n\n"
                            f"Question:\n{question}\n\n"
                            f"Student answer:\n{user_answer}\n\n"
                            "Respond in this exact format:\n"
                            "Verdict: Correct or Incorrect\n"
                            "Explanation: <2-5 sentences>\n"
                            "Ideal answer: <brief>\n"
                        )
                        feedback = groq_chat(
                            [
                                {"role": "system", "content": system},
                                {"role": "user", "content": eval_prompt},
                            ]
                        )

                    verdict_line = ""
                    for line in feedback.splitlines():
                        if line.strip().lower().startswith("verdict:"):
                            verdict_line = line.strip()
                            break
                    is_correct = "correct" in verdict_line.lower() and "incorrect" not in verdict_line.lower()

                    st.session_state.quiz_feedback = feedback
                    st.session_state.quiz_verdict = "Correct" if is_correct else "Incorrect"
                    if is_correct:
                        st.session_state.user_score = int(st.session_state.user_score) + 1

                    st.session_state.quiz_answer = ""

                    if idx + 1 < total_q:
                        st.session_state.current_question_index = idx + 1
                    else:
                        st.session_state.quiz_active = False

                    st.rerun()
                except Exception as e:
                    st.error(f"Could not evaluate answer: {e}")

        if st.session_state.quiz_feedback.strip():
            if st.session_state.quiz_verdict == "Correct":
                st.success("Correct")
            elif st.session_state.quiz_verdict == "Incorrect":
                st.error("Incorrect")
            st.markdown(st.session_state.quiz_feedback)

        if not st.session_state.quiz_active and (idx + 1) >= total_q:
            st.markdown(
                f"**Quiz complete!** Final score: **{int(st.session_state.user_score)} / {total_q}**"
            )
    else:
        st.caption("Click **Start Quiz** to generate 5 questions from your PDF.")


def require_pdf_or_show_error() -> bool:
    """Return True if PDF text is available; otherwise show UI error and return False."""
    if not st.session_state.pdf_name or not st.session_state.pdf_text.strip():
        st.error("Please upload a PDF with extractable text in the sidebar first.")
        return False
    return True


with left:
    st.subheader("💬 Chat History")

    # Show chat history
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Sidebar actions: summarize / generate questions
    if summarize_clicked:
        if require_pdf_or_show_error():
            try:
                with st.spinner("Creating a 5-point summary..."):
                    system = build_system_prompt(st.session_state.mode, st.session_state.pdf_text)
                    prompt = (
                        "Summarize the PDF in exactly 5 bullet points.\n"
                        "- Keep each bullet 1–2 sentences.\n"
                        "- Focus on the most exam-relevant ideas.\n"
                    )
                    assistant_text = groq_chat(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ]
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not summarize: {e}")

    if questions_clicked:
        if require_pdf_or_show_error():
            try:
                with st.spinner("Generating important questions..."):
                    system = build_system_prompt(st.session_state.mode, st.session_state.pdf_text)
                    prompt = (
                        "Generate 5 important study questions from the PDF.\n"
                        "- Number them 1 to 5.\n"
                        "- Make them high-yield for revision/exams.\n"
                    )
                    assistant_text = groq_chat(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ]
                    )
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_text}
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate questions: {e}")

    if generate_notes_clicked:
        if require_pdf_or_show_error():
            try:
                with st.spinner("Generating structured notes..."):
                    system = build_system_prompt(
                        st.session_state.mode, st.session_state.pdf_text
                    )
                    prompt = (
                        "Create structured study notes from the PDF.\n\n"
                        "Format requirements:\n"
                        "- Start with a clear Title\n"
                        "- Use headings and subheadings\n"
                        "- Use bullet points under each heading\n"
                        "- Include an 'Important keywords' section with a bullet list\n\n"
                        "Write concise, exam-focused notes and keep the structure clean."
                    )
                    notes_text = groq_chat(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ]
                    )

                st.session_state.generated_notes = notes_text
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"### Generated Notes\n\n{notes_text}",
                    }
                )
                st.rerun()
            except Exception as e:
                st.error(f"Could not generate notes: {e}")

    if start_quiz_clicked:
        if require_pdf_or_show_error():
            try:
                with st.spinner("Preparing your quiz..."):
                    system = build_system_prompt(
                        st.session_state.mode, st.session_state.pdf_text
                    )
                    prompt = (
                        "Generate exactly 5 quiz questions from the PDF.\n"
                        "- Number them 1 to 5.\n"
                        "- Make them answerable from the PDF.\n"
                        "- Do NOT include answers.\n"
                    )
                    raw = groq_chat(
                        [
                            {"role": "system", "content": system},
                            {"role": "user", "content": prompt},
                        ]
                    )

                questions: List[str] = []
                current: List[str] = []
                for line in raw.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if (
                        stripped[0].isdigit()
                        and (stripped[1:2] in [".", ")"] or stripped[:2].isdigit())
                    ):
                        if current:
                            questions.append("\n".join(current).strip())
                            current = []
                        current.append(stripped)
                    else:
                        if current:
                            current.append(stripped)
                if current:
                    questions.append("\n".join(current).strip())

                if len(questions) < 5:
                    questions = [q.strip() for q in raw.split("\n\n") if q.strip()]

                st.session_state.quiz_questions = questions[:5]
                st.session_state.current_question_index = 0
                st.session_state.user_score = 0
                st.session_state.quiz_feedback = ""
                st.session_state.quiz_verdict = ""
                st.session_state.quiz_answer = ""
                st.session_state.quiz_active = True if st.session_state.quiz_questions else False
                st.rerun()
            except Exception as e:
                st.error(f"Could not start quiz: {e}")

    # Chat input (question answering)
    user_query = st.chat_input("Ask a question about your PDF...")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})

        # If no PDF, handle gracefully (requirement)
        if not st.session_state.pdf_text.strip():
            with st.chat_message("assistant"):
                st.markdown(
                    "I don’t have a PDF to study from yet. Please upload a PDF in the sidebar, "
                    "then ask your question again."
                )
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Please upload a PDF in the sidebar, then ask your question again.",
                }
            )
            st.stop()

        try:
            with st.spinner("Thinking..."):
                system = build_system_prompt(st.session_state.mode, st.session_state.pdf_text)

                # Keep a short recent chat window to reduce tokens.
                recent_history = st.session_state.messages[-10:]
                model_messages = [{"role": "system", "content": system}] + recent_history

                assistant_text = groq_chat(model_messages)

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_text}
            )
            st.rerun()
        except Exception as e:
            st.error(f"Groq request failed: {e}")


# -----------------------------
# Footer (UI only)
# -----------------------------
st.markdown(
    """
    <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.14); margin: 18px 0 12px 0;" />
    <div style="text-align:center; padding-bottom: 6px;">
      <div style="font-weight: 650; color: rgba(255,255,255,0.92);">
        StudyBuddyAI — Learn smarter, not harder.
      </div>
      <div style="margin-top: 6px; color: rgba(255,255,255,0.70);">
        Made with ❤️ by Gauri Borse
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
