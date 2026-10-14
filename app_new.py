import re
import math
import time
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import streamlit as st

try:
    import requests
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    SKLEARN_AVAILABLE = False

st.set_page_config(
    page_title="Adaptive Personalised Learning Prototype",
    page_icon="🎓",
    layout="wide"
)

STOPWORDS = set("""
a an the and or but if then than this that these those is are was were be been being in on at for from to of as by with without into about over under between within across using use used can could should would may might will shall do does did done it its their his her our your you we they he she i me my mine ours yours them what when where why how which who also such based according through during each every most more less very not no yes
""".split())

ACADEMIC_NOISE_WORDS = set("""
slide slides chapter lecture lectures week weeks unit units page pages topic topics introduction overview learning material materials note notes pdf ppt powerpoint ct107 txsa document documents ediscovery dimensionality student students module modules course code assessment question questions answer answers explain describe using include important concept concepts
""".split())

STOPWORDS = STOPWORDS | ACADEMIC_NOISE_WORDS

DEFAULT_NOTES = """
Supervised learning is a machine learning approach where a model learns from labelled data. Labelled data contains input features and the correct output or target value. The model is trained to identify patterns between the input and output so that it can make predictions on unseen data. Common supervised learning tasks include classification and regression. Classification predicts categories, such as spam or not spam. Regression predicts continuous values, such as house prices.

Unsupervised learning is a machine learning approach where the model learns from unlabelled data. The system tries to discover hidden patterns, groups, or structures without being given the correct answers. Common unsupervised learning tasks include clustering and dimensionality reduction. Clustering groups similar data points together, while dimensionality reduction reduces the number of features while preserving important information.

Retrieval-Augmented Generation, also known as RAG, combines information retrieval with text generation. In a learning system, RAG retrieves relevant chunks from uploaded learning materials before generating questions, explanations, or feedback. This helps the AI response stay grounded in the learner's own notes instead of relying only on general model knowledge.

A learner model stores information about a student's learning behaviour and knowledge state. It may include concept mastery, engagement, consistency, attempts, weak concepts, and learning progression. Knowledge state inference estimates how well the learner understands a topic based on their responses, activity history, and interaction patterns.

Adaptive learning personalises the learning experience according to the learner's current needs. If a learner has low mastery, the system can recommend revision notes and basic practice. If a learner has medium mastery, the system can recommend application questions. If a learner has high mastery, the system can recommend advanced tasks or allow the learner to proceed to the next topic.
"""


def init_state():
    defaults = {
        "raw_text": DEFAULT_NOTES,
        "chunks": [],
        "topics": [],
        "selected_topic": None,
        "current_question": None,
        "expected_answer": None,
        "topic_context": "",
        "history": [],
        "learner_profile": defaultdict(lambda: {"attempts": 0, "scores": [], "last_level": "Not attempted", "weak_concepts": [], "activity_types": []}),
        "use_llm": False,
        "llm_model": "llama3.2",
        "selected_llm_provider": "Ollama - Llama",
        "selected_llms_for_comparison": ["Ollama - Llama", "Ollama - Qwen", "Gemini"],
        "ollama_llama_model": "llama3.2",
        "ollama_qwen_model": "qwen2.5",
        "gemini_model": "gemini-1.5-flash",
        "gemini_api_key": "",
        "llm_comparison_results": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def call_ollama(prompt: str, model: str = "llama3.2", temperature: float = 0.4, max_words: int = 180):
    """
    Calls a local Ollama LLM server.
    To use:
    1. Install Ollama
    2. Run: ollama pull llama3.2
    3. Run this Streamlit app
    """
    if not REQUESTS_AVAILABLE:
        return None

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=45
        )
        if response.status_code == 200:
            output = response.json().get("response", "").strip()
            words = output.split()
            if len(words) > max_words:
                output = " ".join(words[:max_words]) + "..."
            return output
        return None
    except Exception:
        return None




def trim_words(output: str, max_words: int = 180):
    output = (output or "").strip()
    words = output.split()
    if len(words) > max_words:
        output = " ".join(words[:max_words]) + "..."
    return output


def call_gemini(prompt: str, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.4, max_words: int = 180):
    if not REQUESTS_AVAILABLE:
        return None, "Requests library is not available."

    if not api_key:
        return None, "Gemini API key is empty."

    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature}
        }

        response = requests.post(url, json=payload, timeout=45)

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text}"

        data = response.json()
        output = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        return trim_words(output, max_words=max_words), ""

    except Exception as e:
        return None, str(e)


def get_llm_config(provider: str):
    configs = {
        "Ollama - Llama": {
            "provider": "ollama",
            "display_name": "Llama via Ollama",
            "model": st.session_state.get("ollama_llama_model", "llama3.2")
        },
        "Ollama - Qwen": {
            "provider": "ollama",
            "display_name": "Qwen via Ollama",
            "model": st.session_state.get("ollama_qwen_model", "qwen2.5")
        },
        "Gemini": {
            "provider": "gemini",
            "display_name": "Google Gemini API",
            "model": st.session_state.get("gemini_model", "gemini-1.5-flash")
        }
    }
    return configs.get(provider, configs["Ollama - Llama"])


def call_selected_llm(prompt: str, provider: str = None, temperature: float = 0.4, max_words: int = 180):
    provider = provider or st.session_state.get("selected_llm_provider", "Ollama - Llama")
    config = get_llm_config(provider)

    start = time.time()
    output = None
    error = ""

    if config["provider"] == "ollama":
        output = call_ollama(prompt, model=config["model"], temperature=temperature, max_words=max_words)
        if not output:
            error = "No response. Check whether Ollama is running and the model is pulled."
    elif config["provider"] == "gemini":
        output, gemini_error = call_gemini(
            prompt,
            api_key=st.session_state.get("gemini_api_key", ""),
            model=config["model"],
            temperature=temperature,
            max_words=max_words
        )
        if not output:
            error = gemini_error

    elapsed = round(time.time() - start, 2)
    return {
        "LLM": provider,
        "Model": config["model"],
        "Success": bool(output),
        "Response Time (s)": elapsed,
        "Output": output or "",
        "Error": error
    }


def run_llm_feasibility_test(prompt: str, providers, temperature: float = 0.4, max_words: int = 180):
    results = []
    for provider in providers:
        results.append(call_selected_llm(prompt, provider=provider, temperature=temperature, max_words=max_words))
    return results

def llm_enabled():
    return bool(st.session_state.get("use_llm", False))


def build_llm_prompt_for_question(topic, level, context):
    return f"""
You are an adaptive learning tutor for university students.

Based only on the retrieved learning material below, generate ONE {level}-level question for the topic: {topic}.

Rules:
- Beginner: ask a simple definition/main idea question.
- Intermediate: ask an application/example question.
- Advanced: ask a comparison, scenario, or reasoning question.
- Do not mention slide numbers, course codes, or lecture metadata.
- Do not include duplicated keywords.
- Return only the question.

Retrieved learning material:
{context[:1800]}
"""


def build_llm_prompt_for_feedback(topic, question, learner_answer,
                                  reference_answer, level,
                                  mastery, missing):

    missing_text = ", ".join(missing[:5]) if missing else "None"

    return f"""
You are an adaptive learning tutor.

Return ONLY the format below.

Topic: {topic}
Question: {question}
Knowledge Level: {level}
Mastery Score: {mastery:.1f}%

Learner Answer:
{learner_answer}

Reference Answer:
{reference_answer[:1000]}

Weak Concepts:
{missing_text}

Format:

### What you did well
- bullet point
- bullet point

### What needs improvement
- bullet point
- bullet point

### Guided explanation
- bullet point
- bullet point

### Suggested next action
- bullet point

Rules:
- Maximum 2 bullet points per section.
- No long paragraphs.
- No course codes.
- No slide numbers.
- No lecture metadata.
- Keep under 150 words.
"""


def build_llm_prompt_for_learning_path(topic, level,
                                       mastery, missing,
                                       activity_count,
                                       topic_mastery):

    missing_text = ", ".join(missing[:5]) if missing else "None"

    return f"""
Generate a short adaptive learning path.

Topic: {topic}
Knowledge Level: {level}
Mastery Score: {mastery:.1f}%
Weak Concepts: {missing_text}

Return EXACTLY:

1. Review: ...
2. Practice: ...
3. Reassess: ...

Rules:
- Maximum 3 steps.
- Each step under 15 words.
- No explanations.
- No sub-bullets.
"""



def split_sentences(text: str):
    text = clean_text(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 20]


def chunk_text(text: str, chunk_size: int = 550, overlap: int = 80):
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        st.error("pypdf is not installed. Run: pip install pypdf")
        return ""
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_keywords(text: str, top_n: int = 10):
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text.lower())
    words = [w for w in words if w not in STOPWORDS and not re.search(r"\d", w) and len(w) > 3]
    counts = Counter(words)
    return [w for w, _ in counts.most_common(top_n)]


def extract_topics(chunks, top_n=8):
    if not chunks:
        return []
    if SKLEARN_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=80)
            X = vectorizer.fit_transform(chunks)
            terms = vectorizer.get_feature_names_out()
            scores = np.asarray(X.sum(axis=0)).ravel()
            ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)
            topics = []
            for term, score in ranked:
                pretty = term.title()
                if len(pretty) > 3 and pretty not in topics:
                    topics.append(pretty)
                if len(topics) >= top_n:
                    break
            return topics
        except Exception:
            pass
    return [w.title() for w in extract_keywords(" ".join(chunks), top_n)]


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformer("all-MiniLM-L6-v2")
    return None


def embed_texts(texts):
    model = load_embedding_model()
    if model is not None:
        return model.encode(texts, normalize_embeddings=True)
    if SKLEARN_AVAILABLE:
        vectorizer = TfidfVectorizer(stop_words="english")
        return vectorizer.fit_transform(texts), vectorizer
    return None


def semantic_similarity(a: str, b: str):
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0.0
    model = load_embedding_model()
    if model is not None:
        emb = model.encode([a, b], normalize_embeddings=True)
        return float(np.dot(emb[0], emb[1]))
    if SKLEARN_AVAILABLE:
        vec = TfidfVectorizer(stop_words="english").fit_transform([a, b])
        return float(cosine_similarity(vec[0], vec[1])[0][0])
    aw, bw = set(extract_keywords(a, 50)), set(extract_keywords(b, 50))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def retrieve_relevant_chunks(query: str, chunks, top_k=3):
    if not chunks:
        return []
    model = load_embedding_model()
    if model is not None:
        q = model.encode([query], normalize_embeddings=True)[0]
        c = model.encode(chunks, normalize_embeddings=True)
        sims = np.dot(c, q)
        idx = np.argsort(sims)[::-1][:top_k]
        return [(chunks[i], float(sims[i])) for i in idx]
    if SKLEARN_AVAILABLE:
        docs = chunks + [query]
        X = TfidfVectorizer(stop_words="english").fit_transform(docs)
        sims = cosine_similarity(X[-1], X[:-1]).ravel()
        idx = np.argsort(sims)[::-1][:top_k]
        return [(chunks[i], float(sims[i])) for i in idx]
    qwords = set(extract_keywords(query, 20))
    scored = []
    for ch in chunks:
        cwords = set(extract_keywords(ch, 30))
        score = len(qwords & cwords) / max(1, len(qwords | cwords))
        scored.append((ch, score))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


def clean_topic_name(topic):
    topic = re.sub(r"\b(slide|slides|chapter|lecture|week|unit|document|ct\d+[-\w]*|txsa)\b", "", topic, flags=re.I)
    words = topic.split()
    cleaned = []
    for w in words:
        if w.lower() not in [x.lower() for x in cleaned]:
            cleaned.append(w)
    return " ".join(cleaned).strip() or "Current Topic"


def get_topic_level(topic):
    profile = st.session_state.learner_profile[topic]
    if profile["attempts"] == 0 or not profile["scores"]:
        return "Beginner"
    avg_score = np.mean(profile["scores"][-3:])
    if avg_score < 40:
        return "Beginner"
    if avg_score < 70:
        return "Intermediate"
    return "Advanced"


def get_next_activity_type(level, topic):

    profile = st.session_state.learner_profile[topic]
    completed = profile.get("activity_types", [])

    sequence = [
        "Definition",
        "Application",
        "Example",
        "Scenario",
        "Comparison",
        "Challenge"
    ]

    for activity in sequence:
        if activity not in completed:
            return activity

    return "Challenge"


def generate_reference_answer(context: str):
    sentences = split_sentences(context)
    useful = []
    for s in sentences:
        low = s.lower()
        if any(noise in low for noise in ["slide", "ct107", "txsa", "chapter", "lecture"]):
            continue
        useful.append(s)
    return " ".join(useful[:4]) if useful else context[:700]


def generate_question(topic: str, context: str):
    topic = clean_topic_name(topic)
    level = get_topic_level(topic)
    activity_type = get_next_activity_type(level, topic)
    expected = generate_reference_answer(context)

    if llm_enabled():
        prompt = build_llm_prompt_for_question(topic, level, context)
        llm_result = call_selected_llm(prompt, max_words=60)
        llm_q = llm_result.get("Output")
        if llm_q:
            return llm_q, expected, level, activity_type, True

    if activity_type == "Definition":
        q = f"What is {topic}? Explain it in your own words."
    elif activity_type == "Application":
        q = f"How is {topic} used in a real-world situation? Give one example."
    elif activity_type == "Example":
        q = f"Give an example of {topic} and explain why it is useful."
    elif activity_type == "Scenario":
        q = f"A student or organisation wants to apply {topic}. Describe how it could be used in this scenario."
    elif activity_type == "Comparison":
        q = f"Compare {topic} with another related concept from the notes. What is the difference?"
    else:
        q = f"Create a short explanation of {topic} and apply it to a practical case."

    return q, expected, level, activity_type, False

def concept_coverage(answer: str, expected: str, top_n=8):
    concepts = extract_keywords(expected, top_n)
    if not concepts:
        return 0.0, []
    ans = answer.lower()
    covered = [c for c in concepts if c.lower() in ans]
    missing = [c for c in concepts if c not in covered]
    return len(covered) / len(concepts), missing


def answer_depth(answer: str):
    words = re.findall(r"\w+", answer)
    word_count = len(words)
    connectors = len(re.findall(r"\b(because|therefore|however|for example|such as|so that|while|whereas|although|means|includes)\b", answer.lower()))
    sentence_count = len(split_sentences(answer)) or (1 if word_count > 0 else 0)
    length_score = min(word_count / 80, 1.0)
    explanation_score = min((connectors + sentence_count) / 5, 1.0)
    return 0.65 * length_score + 0.35 * explanation_score, word_count, connectors


def infer_level(score):
    if score < 40:
        return "Beginner"
    if score < 70:
        return "Intermediate"
    return "Advanced"


def calculate_topic_mastery(topic):

    profile = st.session_state.learner_profile[topic]

    scores = profile["scores"]

    if not scores:
        return 0, "Not Started"

    avg_mastery = np.mean(scores)

    completed_activities = len(set(profile["activity_types"]))

    if avg_mastery >= 70 and completed_activities >= 5:
        status = "Mastered"

    elif avg_mastery >= 50:
        status = "In Progress"

    else:
        status = "Needs Improvement"

    return round(avg_mastery, 1), status


def personalised_feedback(level, sim, cov, depth, missing, topic):
    strengths = []
    improvements = []
    if sim >= 0.70:
        strengths.append("Your answer is semantically close to the expected explanation.")
    elif sim >= 0.45:
        strengths.append("Your answer shows partial understanding of the topic.")
    else:
        improvements.append("Your explanation does not yet match the key meaning from the notes.")

    if cov >= 0.70:
        strengths.append("You included most of the important concepts.")
    else:
        improvements.append("Some important concepts are missing: " + ", ".join(missing[:5]) + ".")

    if depth >= 0.65:
        strengths.append("Your response has enough explanation depth.")
    else:
        improvements.append("Add more explanation, examples, or reasoning to make your answer deeper.")

    if level == "Beginner":
        summary = f"You are currently at Beginner level for {topic}. Focus on understanding the basic meaning first."
    elif level == "Intermediate":
        summary = f"You are currently at Intermediate level for {topic}. You understand part of the topic, but need stronger concept coverage and explanation."
    else:
        summary = f"You are currently at Advanced level for {topic}. You can proceed to more challenging practice."
    return summary, strengths, improvements

def reference_explanation(expected_answer, missing_concepts):
    explanation = expected_answer.strip()

    if missing_concepts:
        focus = ", ".join(missing_concepts[:5])
        return f"""
### Reference Explanation
{explanation}

### Key concepts to focus on
{focus}

Try to include these concepts when you re-answer the question.
"""
    return f"""
### Reference Explanation
{explanation}

Your answer covered most of the key concepts. You can now try a harder application question.
"""

def learning_path(level, topic, missing):
    missing_text = ", ".join(missing[:3]) if missing else "the current topic"
    if level == "Beginner":
        return [
            f"Review the retrieved notes for {topic}.",
            f"Focus on weak concepts: {missing_text}.",
            "Try one basic definition question.",
            "Re-answer the same question after revision.",
            "Move to application practice only when mastery is above 40%."
        ]
    if level == "Intermediate":
        return [
            f"Revise the missing concepts: {missing_text}.",
            "Try an application-based question using an example.",
            "Compare your answer with the expected explanation.",
            "Attempt a second question to improve consistency.",
            "Move to the next topic when mastery is above 70%."
        ]
    return [
        f"Proceed to advanced practice for {topic}.",
        "Try a scenario-based or case-study question.",
        "Connect this topic with another related topic.",
        "Summarise the topic in your own words.",
        "Move to the next recommended topic."
    ]


def recommend_next_topic(current_topic):
    topics = st.session_state.topics
    if not topics:
        return "Upload notes to generate topics."
    weak = []
    unattempted = []
    for t in topics:
        prof = st.session_state.learner_profile[t]
        if prof["attempts"] == 0:
            unattempted.append(t)
        elif prof["scores"] and np.mean(prof["scores"][-3:]) < 70:
            weak.append(t)
    if weak:
        return weak[0]
    if unattempted:
        return unattempted[0]
    return "All extracted topics attempted. Continue advanced practice."


init_state()

st.title("🎓 Adaptive Personalised Learning System using LLM/RAG/NLP")
st.caption("Supervisor Demo Prototype aligned with FYP scope: NLP processing, RAG retrieval, knowledge state inference, adaptive feedback, learning path, and analytics dashboard.")

with st.sidebar:
    st.header("Prototype Modules")
    page = st.radio("Navigate", [
        "1. Upload & NLP Processing",
        "2. RAG Learning Activity",
        "3. Knowledge State Inference",
        "4. Personalised Learning Path",
        "5. Learner Analytics Dashboard",
        "6. LLM Feasibility Comparison",
        "7. System Architecture & Model Justification"
    ])
    st.divider()
    st.subheader("LLM Settings")
    st.session_state.use_llm = st.checkbox("Use LLM generation", value=st.session_state.get("use_llm", False))

    available_llms = ["Ollama - Llama", "Ollama - Qwen", "Gemini"]
    st.session_state.selected_llms_for_comparison = st.multiselect(
        "LLMs for feasibility comparison",
        available_llms,
        default=st.session_state.get("selected_llms_for_comparison", available_llms)
    )

    active_options = st.session_state.selected_llms_for_comparison or available_llms
    previous_provider = st.session_state.get("selected_llm_provider", "Ollama - Llama")
    default_index = active_options.index(previous_provider) if previous_provider in active_options else 0
    st.session_state.selected_llm_provider = st.selectbox(
        "Active LLM used by the learning system",
        active_options,
        index=default_index
    )

    with st.expander("Model settings"):
        st.session_state.ollama_llama_model = st.text_input(
            "Llama model name in Ollama",
            value=st.session_state.get("ollama_llama_model", "llama3.2")
        )
        st.session_state.ollama_qwen_model = st.text_input(
            "Qwen model name in Ollama",
            value=st.session_state.get("ollama_qwen_model", "qwen2.5")
        )
        st.session_state.gemini_model = st.text_input(
            "Gemini model name",
            value=st.session_state.get("gemini_model", "gemini-1.5-flash")
        )
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API key",
            value=st.session_state.get("gemini_api_key", ""),
            type="password"
        )
    st.caption("For Ollama: install Ollama and pull the selected models. For Gemini: paste your Google AI Studio API key.")
    st.divider()
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st.success("AI Embedding Model: all-MiniLM-L6-v2")
    elif SKLEARN_AVAILABLE:
        st.warning("Fallback mode: TF-IDF semantic similarity")
    else:
        st.error("Minimal fallback mode: keyword overlap")

if page == "1. Upload & NLP Processing":
    st.header("1. Upload Learning Material → NLP Processing")
    st.write("This module supports your scope: upload learning materials, preprocess text, extract topics, and chunk content for retrieval.")
    uploaded = st.file_uploader("Upload PDF or TXT notes", type=["pdf", "txt"])
    colA, colB = st.columns([1, 1])
    with colA:
        chunk_size = st.slider("Chunk size", 300, 1000, 550, 50)
    with colB:
        topic_count = st.slider("Number of topics to extract", 3, 12, 8)

    if uploaded is not None:
        if uploaded.name.lower().endswith(".pdf"):
            st.session_state.raw_text = extract_pdf_text(uploaded)
        else:
            st.session_state.raw_text = uploaded.read().decode("utf-8", errors="ignore")

    if st.button("Process Learning Material", type="primary") or not st.session_state.chunks:
        st.session_state.raw_text = clean_text(st.session_state.raw_text)
        st.session_state.chunks = chunk_text(st.session_state.raw_text, chunk_size=chunk_size)
        st.session_state.topics = extract_topics(st.session_state.chunks, top_n=topic_count)
        if st.session_state.topics and st.session_state.selected_topic is None:
            st.session_state.selected_topic = st.session_state.topics[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", len(st.session_state.raw_text))
    c2.metric("Chunks", len(st.session_state.chunks))
    c3.metric("Extracted Topics", len(st.session_state.topics))

    st.subheader("Extracted Topics")
    st.write(st.session_state.topics)

    st.subheader("Content Chunks for RAG Retrieval")
    for i, ch in enumerate(st.session_state.chunks[:6], start=1):
        with st.expander(f"Chunk {i}"):
            st.write(ch)

elif page == "2. RAG Learning Activity":
    st.header("2. RAG Retrieval → Dynamic Learning Activity Generation")
    if not st.session_state.chunks:
        st.warning("Please process learning material first.")
        st.stop()

    selected = st.selectbox("Choose a topic extracted from uploaded material", st.session_state.topics or ["General Topic"])
    st.session_state.selected_topic = selected

    retrieved = retrieve_relevant_chunks(selected, st.session_state.chunks, top_k=3)
    context = " ".join([r[0] for r in retrieved])
    st.session_state.topic_context = context

    if st.button("Generate Dynamic Question", type="primary") or st.session_state.current_question is None:
        q, exp, q_level, activity_type, used_llm = generate_question(selected, context)
        st.session_state.current_question = q
        st.session_state.expected_answer = exp
        st.session_state.current_question_level = q_level
        st.session_state.current_activity_type = activity_type
        st.session_state.question_used_llm = used_llm

    st.subheader("Retrieved Notes Used by RAG")
    for i, (ch, score) in enumerate(retrieved, start=1):
        with st.expander(f"Retrieved Chunk {i} | relevance score: {score:.3f}"):
            st.write(ch)

    st.subheader("Generated Learning Activity")
    st.info(st.session_state.current_question)
    st.caption(f"Generated for inferred level: {st.session_state.get('current_question_level', 'Beginner')} | Activity type: {st.session_state.get('current_activity_type', 'Definition')} | LLM used: {st.session_state.get('question_used_llm', False)}")

    st.subheader("Reference Answer / Ground Truth from Retrieved Notes")
    with st.expander("View reference answer used for semantic comparison"):
        st.write(st.session_state.expected_answer)

elif page == "3. Knowledge State Inference":
    st.header("3. Learner Response → Knowledge State Inference")
    if not st.session_state.current_question:
        st.warning("Please generate a learning activity first.")
        st.stop()

    st.subheader("Question")
    st.info(st.session_state.current_question)
    learner_answer = st.text_area("Learner Answer", height=160, placeholder="Type the learner's answer here...")

    def calculate_progression(topic, current_score):
        profile = st.session_state.learner_profile[topic]
        previous_scores = profile["scores"]

        if not previous_scores:
            return 0.0, "First attempt"

        progression = current_score - previous_scores[-1]

        if progression > 5:
            status = "Improving"
        elif progression < -5:
            status = "Needs attention"
        else:
            status = "Stable"

        return progression, status

    if st.button("Analyse Response & Infer Knowledge Level", type="primary"):
        topic = st.session_state.selected_topic or "Current Topic"
        expected = st.session_state.expected_answer or st.session_state.topic_context
        sim = semantic_similarity(learner_answer, expected)
        cov, missing = concept_coverage(learner_answer, expected, top_n=8)
        depth, word_count, connectors = answer_depth(learner_answer)
        progression, progression_status = calculate_progression(topic, (0.5 * sim + 0.3 * cov + 0.2 * depth) * 100)
        history_factor = 0.0
        if st.session_state.learner_profile[topic]["scores"]:
            recent_avg = np.mean(st.session_state.learner_profile[topic]["scores"][-3:]) / 100
            history_factor = 0.10 * recent_avg
        raw_score = (0.50 * sim + 0.30 * cov + 0.20 * depth + history_factor) / 1.10
        mastery = max(0, min(100, raw_score * 100))
        level = infer_level(mastery)

        summary, strengths, improvements = personalised_feedback(level, sim, cov, depth, missing, topic)
        path = learning_path(level, topic, missing)
        ref_exp = reference_explanation(expected, missing)

        llm_feedback = None
        llm_path_text = None
        if llm_enabled():
            feedback_prompt = build_llm_prompt_for_feedback(
                topic, st.session_state.current_question, learner_answer, expected, level, mastery, missing
            )
            feedback_result = call_selected_llm(
                feedback_prompt,
                temperature=0.35,
                max_words=220
            )
            llm_feedback = feedback_result.get("Output")

            current_scores = st.session_state.learner_profile[topic]["scores"]
            topic_mastery_preview = np.mean(current_scores + [mastery]) if current_scores else mastery
            path_prompt = build_llm_prompt_for_learning_path(
                topic, level, mastery, missing,
                st.session_state.learner_profile[topic]["attempts"] + 1,
                topic_mastery_preview
            )
            path_result = call_selected_llm(
                path_prompt,
                temperature=0.4,
                max_words=180
            )
            llm_path_text = path_result.get("Output")

        if llm_feedback:
            summary = llm_feedback
        if llm_path_text:
            path = [line.strip(" -") for line in llm_path_text.split("\n") if line.strip()]

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "question": st.session_state.current_question,
            "question_level": st.session_state.get("current_question_level", "Beginner"),
            "activity_type": st.session_state.get("current_activity_type", "Definition"),
            "question_used_llm": st.session_state.get("question_used_llm", False),
            "llm_feedback_used": bool(llm_feedback),
            "llm_learning_path_used": bool(llm_path_text),
            "answer": learner_answer,
            "semantic_similarity": round(sim, 3),
            "concept_coverage": round(cov, 3),
            "answer_depth": round(depth, 3),
            "word_count": word_count,
            "mastery_score": round(mastery, 1),
            "knowledge_level": level,
            "missing_concepts": missing,
            "progression": round(progression, 1),
            "progression_status": progression_status,
            "feedback_summary": summary,
            "reference_explanation": ref_exp,
            "learning_path": path
        }
        st.session_state.history.append(record)
        prof = st.session_state.learner_profile[topic]
        prof["attempts"] += 1
        prof["scores"].append(mastery)
        prof["last_level"] = level
        prof["weak_concepts"] = missing[:5]
        prof["activity_types"].append(st.session_state.get("current_activity_type", "Definition"))
        st.session_state.latest_record = record

    if "latest_record" in st.session_state:
        r = st.session_state.latest_record
        st.subheader("Inference Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Semantic Similarity", r["semantic_similarity"])
        c2.metric("Concept Coverage", r["concept_coverage"])
        c3.metric("Answer Depth", r["answer_depth"])
        c4.metric("Mastery Score", f"{r['mastery_score']}%")
        st.success(f"Knowledge Level: {r['knowledge_level']}")
        st.caption(f"Question level: {r.get('question_level', 'Beginner')} | Activity type: {r.get('activity_type', 'Definition')} | LLM question: {r.get('question_used_llm', False)} | LLM feedback: {r.get('llm_feedback_used', False)}")
        st.caption(f"Progression status: {r['progression_status']} ({r['progression']} from previous attempt)")

        st.subheader("Personalised Feedback")
        st.write(r["feedback_summary"])
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Strengths**")
            for s in personalised_feedback(r["knowledge_level"], r["semantic_similarity"], r["concept_coverage"], r["answer_depth"], r["missing_concepts"], r["topic"])[1]:
                st.write("✅", s)
        with col2:
            st.markdown("**Improvements**")
            for imp in personalised_feedback(r["knowledge_level"], r["semantic_similarity"], r["concept_coverage"], r["answer_depth"], r["missing_concepts"], r["topic"])[2]:
                st.write("⚠️", imp)

        st.subheader("Guided Reference Explanation")
        st.markdown(r["reference_explanation"])

        if r.get("llm_feedback_used"):
            st.info("This personalised feedback was generated by the LLM using the learner answer, reference answer, inferred level, and weak concepts.")
        else:
            st.warning("LLM fallback is active. Start Ollama and enable LLM mode in the sidebar to use LLM-generated feedback.")

elif page == "4. Personalised Learning Path":
    st.header("4. Adaptive Feedback → Personalised Learning Path")

    if "latest_record" not in st.session_state:
        st.warning("Analyse a learner answer first to generate a personalised learning path.")
        st.stop()

    r = st.session_state.latest_record

    st.metric("Current Mastery", f"{r['mastery_score']}%", r["knowledge_level"])

    st.subheader("Weak Concept Detection")
    if r["missing_concepts"]:
        st.write(", ".join(r["missing_concepts"][:8]))
    else:
        st.write("No major weak concepts detected.")

    st.subheader("Recommended Learning Path")
    for i, step in enumerate(r["learning_path"], start=1):
        st.write(f"**Step {i}:** {step}")

    topic_mastery, mastery_status = calculate_topic_mastery(r["topic"])

    st.subheader("Topic Mastery Status")
    col1, col2 = st.columns(2)
    col1.metric("Topic Mastery", f"{topic_mastery}%")
    col2.metric("Status", mastery_status)

    profile = st.session_state.learner_profile[r["topic"]]
    completed = profile["activity_types"]

    all_activities = [
        "Definition",
        "Application",
        "Example",
        "Scenario",
        "Comparison",
        "Challenge"
    ]

    st.subheader("Activity Progress")
    for activity in all_activities:
        if activity in completed:
            st.write(f"✅ {activity}")
        else:
            st.write(f"⬜ {activity}")

    next_topic = recommend_next_topic(r["topic"])

    st.subheader("Next Recommended Topic")
    st.info(next_topic)

    st.subheader("Recommended Activity Type")
    if r["knowledge_level"] == "Beginner":
        st.write("📘 Revision note + basic definition question")
    elif r["knowledge_level"] == "Intermediate":
        st.write("📝 Application question + targeted weak-concept practice")
    else:
        st.write("🚀 Challenge question + move to next topic")

elif page == "5. Learner Analytics Dashboard":
    st.header("5. Learner Analytics Dashboard")
    if not st.session_state.history:
        st.info("No attempts yet. Analyse at least one learner answer to populate the dashboard.")
        st.stop()
    df = pd.DataFrame(st.session_state.history)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Attempts", len(df))
    c2.metric("Average Mastery", f"{df['mastery_score'].mean():.1f}%")
    c3.metric("Topics Attempted", df["topic"].nunique())
    c4.metric("Latest Level", df.iloc[-1]["knowledge_level"])

    st.subheader("Mastery by Topic")
    topic_scores = df.groupby("topic")["mastery_score"].mean().sort_values(ascending=False)
    st.bar_chart(topic_scores)

    st.subheader("Learning Progression Over Attempts")
    progression_df = df[["timestamp", "topic", "mastery_score"]].copy()
    progression_df.index = range(1, len(progression_df) + 1)
    st.line_chart(progression_df["mastery_score"])

    st.subheader("Detailed Learning Log")
    display_cols = ["timestamp", "topic", "question_level", "activity_type", "semantic_similarity", "concept_coverage", "answer_depth", "mastery_score", "knowledge_level", "progression_status", "llm_feedback_used"]
    existing_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[existing_cols], use_container_width=True)

elif page == "6. LLM Feasibility Comparison":
    st.header("6. LLM Feasibility Comparison")
    st.write("Use this page to compare the same RAG-grounded prompt across Gemini, Llama, and Qwen for your FYP Objective 2.")

    if not st.session_state.chunks:
        st.warning("Please process learning material first so the models can be tested using the same RAG context.")
        st.stop()

    selected = st.selectbox("Topic for comparison", st.session_state.topics or ["General Topic"], key="compare_topic")
    retrieved = retrieve_relevant_chunks(selected, st.session_state.chunks, top_k=3)
    context = " ".join([r[0] for r in retrieved])
    level = get_topic_level(selected)

    comparison_task = st.radio(
        "Comparison task",
        ["Generate learning activity", "Generate personalised feedback", "Generate learning path"],
        horizontal=True
    )

    sample_answer = st.text_area(
        "Sample learner answer for feedback/path comparison",
        value="The topic is used to support learning by giving students suitable activities and feedback based on their understanding.",
        height=100
    )

    expected = generate_reference_answer(context)
    sim = semantic_similarity(sample_answer, expected)
    cov, missing = concept_coverage(sample_answer, expected, top_n=8)
    depth, _, _ = answer_depth(sample_answer)
    mastery = max(0, min(100, (0.5 * sim + 0.3 * cov + 0.2 * depth) * 100))
    inferred_level = infer_level(mastery)

    if comparison_task == "Generate learning activity":
        prompt = build_llm_prompt_for_question(selected, level, context)
        max_words = 80
    elif comparison_task == "Generate personalised feedback":
        prompt = build_llm_prompt_for_feedback(selected, "Explain the selected topic.", sample_answer, expected, inferred_level, mastery, missing)
        max_words = 220
    else:
        prompt = build_llm_prompt_for_learning_path(selected, inferred_level, mastery, missing, 1, mastery)
        max_words = 160

    with st.expander("View shared prompt used for all LLMs"):
        st.code(prompt, language="text")

    selected_llms = st.session_state.get("selected_llms_for_comparison", [])
    if not selected_llms:
        st.warning("Select at least one LLM in the sidebar.")
        st.stop()

    if st.button("Run Feasibility Test", type="primary"):
        st.session_state.llm_comparison_results = run_llm_feasibility_test(prompt, selected_llms, max_words=max_words)

    if st.session_state.get("llm_comparison_results"):
        results = st.session_state.llm_comparison_results
        summary_df = pd.DataFrame([
            {
                "LLM": r["LLM"],
                "Model": r["Model"],
                "Success": r["Success"],
                "Response Time (s)": r["Response Time (s)"],
                "Output Length": len(r["Output"].split()) if r["Output"] else 0,
                "Error": r["Error"]
            }
            for r in results
        ])
        st.subheader("Feasibility Summary")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Model Outputs")
        for r in results:
            with st.expander(f"{r['LLM']} | {r['Model']} | Success: {r['Success']}"):
                if r["Output"]:
                    st.write(r["Output"])
                else:
                    st.error(r["Error"])

        st.subheader("Suggested Evaluation Criteria for IR")
        st.table(pd.DataFrame([
            ["Output relevance", "How closely the response follows the uploaded notes and selected topic."],
            ["Personalisation quality", "How well the feedback/path addresses mastery level and weak concepts."],
            ["Instruction following", "Whether the model follows the required output format."],
            ["Response time", "Whether the model is practical for real-time adaptive learning."],
            ["Cost and accessibility", "Whether the model can be used within FYP budget and setup constraints."],
        ], columns=["Criterion", "Meaning"]))

elif page == "7. System Architecture & Model Justification":
    st.header("7. System Architecture & Model Justification")
    st.subheader("Dynamic Process Flow")
    st.code("""
Upload Notes/PDF
→ NLP Cleaning
→ Topic Extraction
→ Content Chunking
→ Embedding + RAG Retrieval
→ LLM Dynamic Question Generation
→ Learner Answer Input
→ Knowledge State Inference
   - Semantic Similarity
   - Concept Coverage
   - Answer Depth
   - Learning History
→ LLM Personalised Feedback
→ Weak Concept Detection
→ LLM Adaptive Learning Path
→ Analytics Dashboard
""", language="text")

    st.subheader("AI / NLP Models Used")
    model_rows = [
        ["Semantic Embedding", "sentence-transformers/all-MiniLM-L6-v2", "Converts notes and learner answers into embeddings for semantic similarity and RAG retrieval."],
        ["Fallback Retrieval", "TF-IDF", "Allows the prototype to run even without transformer model installation."],
        ["LLM Generation", "Gemini API, Llama via Ollama, Qwen via Ollama", "Compares multiple LLMs for dynamic questions, personalised feedback, and learning paths grounded by retrieved notes."],
        ["Knowledge State Inference", "Hybrid scoring model", "Combines semantic similarity, concept coverage, answer depth, and learner history."],
        ["Adaptive Engine", "LLM + rule-based fallback", "Maps mastery level to feedback, weak concept practice, and learning path."],
        ["Dashboard", "Streamlit + Pandas", "Displays mastery, progression, attempts, weak concepts, and learning logs."]
    ]
    st.table(pd.DataFrame(model_rows, columns=["Component", "Model/Technique", "Justification"]))

    st.subheader("How It Addresses the Three Problem Backgrounds")
    problem_rows = [
        ["Lack of continuous real-time personalisation", "The learner receives immediate analysis, feedback, and next-step recommendation after each answer."],
        ["Inaccurate knowledge state inference", "The prototype does not rely only on marks; it analyses meaning, concept coverage, explanation depth, and past attempts."],
        ["Limited support for SDL progression", "The system generates a personalised learning path and analytics dashboard to guide what the learner should do next."]
    ]
    st.table(pd.DataFrame(problem_rows, columns=["Problem", "Prototype Response"]))
