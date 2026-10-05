import os
import ast
import operator
import tempfile

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GenAI Study & Career Assistant",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# GEMINI API SETUP
# =========================================================

def get_api_key():
    """Get Gemini API key from Streamlit secrets or environment."""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "")


API_KEY = get_api_key()

if not API_KEY:
    st.error(
        "Gemini API key is not configured. "
        "Add GEMINI_API_KEY in Streamlit Secrets."
    )
    st.stop()

client = genai.Client(api_key=API_KEY)

MODEL_NAME = "gemini-2.5-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "vectors" not in st.session_state:
    st.session_state.vectors = None


# =========================================================
# SAFE CALCULATOR TOOL
# =========================================================

ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_calculate(expression):
    """Safely calculate basic mathematical expressions."""

    def evaluate(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid number")

        if isinstance(node, ast.BinOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")

            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Power too large")

            return operation(left, right)

        if isinstance(node, ast.UnaryOp):
            operation = ALLOWED_OPERATORS.get(type(node.op))
            if operation is None:
                raise ValueError("Operator not allowed")

            return operation(evaluate(node.operand))

        raise ValueError("Invalid mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return evaluate(tree.body)
    except Exception:
        return None


# =========================================================
# DOCUMENT PROCESSING
# =========================================================

def extract_pdf_text(uploaded_file):
    """Extract text from an uploaded PDF."""

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def create_chunks(text, chunk_size=800, overlap=150):
    """Split document into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_document_index(text):
    """Create TF-IDF index for document retrieval."""

    chunks = create_chunks(text)

    if not chunks:
        return [], None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    vectors = vectorizer.fit_transform(chunks)

    return chunks, vectorizer, vectors


def retrieve_chunks(question, top_k=3):
    """Retrieve the most relevant document chunks."""

    chunks = st.session_state.document_chunks
    vectorizer = st.session_state.vectorizer
    vectors = st.session_state.vectors

    if not chunks or vectorizer is None or vectors is None:
        return []

    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        vectors
    ).flatten()

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indices:
        if similarities[index] > 0:
            results.append(chunks[index])

    return results


# =========================================================
# AGENT ROUTER
# =========================================================

def detect_calculation(question):
    """Detect simple calculation questions."""

    keywords = [
        "calculate",
        "solve",
        "what is",
        "compute"
    ]

    math_characters = set("0123456789+-*/%().")

    question_lower = question.lower()

    if any(word in question_lower for word in keywords):
        math_part = ""

        for char in question:
            if char in math_characters:
                math_part += char
            elif math_part:
                math_part += " "

        math_part = math_part.strip()

        if any(char.isdigit() for char in math_part):
            return math_part

    return None


def agent_route(question):
    """
    Simple agent router.

    Routes the question to:
    1. Calculator
    2. RAG
    3. Normal Gemini assistant
    """

    calculation = detect_calculation(question)

    if calculation:
        return "calculator", calculation

    if st.session_state.document_chunks:
        retrieved = retrieve_chunks(question)

        if retrieved:
            return "rag", retrieved

    return "llm", None


# =========================================================
# GEMINI RESPONSE
# =========================================================

def generate_response(question, mode, context=""):
    """Generate response using Gemini."""

    if mode == "study":
        system_prompt = """
You are an expert AI Study Assistant.

Explain concepts clearly and simply.
Use examples where useful.
Break difficult topics into easy steps.
Avoid unnecessary complexity.
"""

    elif mode == "career":
        system_prompt = """
You are an AI Career Assistant.

Give practical career guidance.
Provide structured roadmaps.
Mention useful skills, projects, tools and learning steps.
Keep recommendations realistic for a student.
"""

    elif mode == "coding":
        system_prompt = """
You are an AI Coding Assistant.

Explain programming concepts clearly.
Provide correct and beginner-friendly code.
Explain important parts of the code.
Prefer simple solutions before advanced ones.
"""

    else:
        system_prompt = """
You are a helpful Generative AI Assistant.
Answer accurately, clearly and professionally.
"""

    if context:
        user_prompt = f"""
{system_prompt}

Use the following retrieved document information when answering.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using the document context when relevant.
If the answer cannot be found in the document, clearly say so.
"""

    else:
        user_prompt = f"""
{system_prompt}

USER QUESTION:
{question}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt
    )

    return response.text


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🤖 GenAI Assistant")

mode = st.sidebar.selectbox(
    "Choose Assistant Mode",
    [
        "study",
        "career",
        "coding"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("📚 Document RAG")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# =========================================================
# PDF UPLOAD
# =========================================================

if uploaded_file is not None:

    if st.session_state.get("uploaded_filename") != uploaded_file.name:

        with st.spinner("Processing document..."):

            try:
                text = extract_pdf_text(uploaded_file)

                if not text.strip():
                    st.sidebar.error(
                        "No readable text was found in this PDF."
                    )

                else:
                    chunks, vectorizer, vectors = build_document_index(text)

                    st.session_state.document_text = text
                    st.session_state.document_chunks = chunks
                    st.session_state.vectorizer = vectorizer
                    st.session_state.vectors = vectors
                    st.session_state.uploaded_filename = uploaded_file.name

                    st.sidebar.success(
                        f"Document ready: {len(chunks)} chunks"
                    )

            except Exception as error:
                st.sidebar.error(
                    f"Could not process PDF: {error}"
                )


# =========================================================
# MAIN UI
# =========================================================

st.title("🤖 GenAI Study & Career Assistant")

st.write(
    "A Generative AI application combining "
    "**LLM, Prompt Engineering, RAG, Tools and Agent Routing.**"
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🧠 LLM\n\nGemini")

with col2:
    st.info("📚 RAG\n\nDocument Q&A")

with col3:
    st.info("🛠️ Tools\n\nCalculator")


# =========================================================
# QUESTION INPUT
# =========================================================

st.subheader("💬 Ask Your Question")

question = st.text_area(
    "Enter your question",
    placeholder="Example: Create a roadmap to become an AI Engineer.",
    height=120
)

ask_button = st.button(
    "🚀 Ask Assistant",
    type="primary"
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if ask_button:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Thinking..."):

        route, data = agent_route(question)

        # -------------------------------
        # CALCULATOR
        # -------------------------------

        if route == "calculator":

            result = safe_calculate(data)

            if result is not None:

                st.success("🛠️ Calculator Tool Used")

                st.subheader("Answer")

                st.write(
                    f"**{data} = {result}**"
                )

            else:

                response = generate_response(
                    question,
                    mode
                )

                st.subheader("🤖 Answer")

                st.write(response)

        # -------------------------------
        # RAG
        # -------------------------------

        elif route == "rag":

            context = "\n\n".join(data)

            response = generate_response(
                question,
                mode,
                context
            )

            st.success("📚 RAG Document Search Used")

            st.subheader("🤖 Answer")

            st.write(response)

            with st.expander("View Retrieved Context"):
                st.write(context)

        # -------------------------------
        # NORMAL LLM
        # -------------------------------

        else:

            response = generate_response(
                question,
                mode
            )

            st.success("🧠 Gemini LLM Used")

            st.subheader("🤖 Answer")

            st.write(response)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "GenAI Study & Career Assistant | "
    "Built with Gemini, RAG, Tools and Agent Routing"
)