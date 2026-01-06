# =================== IMPORT LIBRARIES ===================
import streamlit as st
import nltk
import torch

from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

from rouge_score import rouge_scorer
from docx import Document

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# =================== NLTK DOWNLOAD ===================
nltk.download("punkt")

# =================== TEXT PREPROCESSING ===================
def preprocess_text(text, chunk_size=7000, chunk_overlap=2500):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# =================== EXTRACTIVE SUMMARIZATION ===================
def extractive_summarization(text, num_sentences=10):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# =================== LOAD MODEL SAFELY ===================
@st.cache_resource
def load_llm():
    model_name = "facebook/bart-large-cnn"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    model.to("cpu")

    hf_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)

llm = load_llm()

# =================== PROMPT ===================
prompt = PromptTemplate(
    input_variables=["text"],
    template="Provide a detailed and comprehensive summary of the following text:\n{text}"
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# =================== ABSTRATIVE SUMMARIZATION ===================
def abstractive_summarization_langchain(text, min_length=20, max_length=249):
    response = llm_chain.invoke(
        {"text": text},
        config={
            "min_length": min_length,
            "max_length": max_length
        }
    )
    return response["text"]

# =================== COMBINED PIPELINE ===================
def summarization_pipeline(text, lexrank_sentences=10, min_length=20):
    lexrank_summary = extractive_summarization(text, lexrank_sentences)
    return abstractive_summarization_langchain(
        lexrank_summary,
        min_length=min_length,
        max_length=249
    )

# =================== ROUGE SCORE ===================
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    return scorer.score(reference, generated)

# =================== DOCX EXTRACTION ===================
def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    return "\n".join(p.text for p in doc.paragraphs)

# =================== STREAMLIT STYLING ===================
st.markdown("""
<style>
.gradient-text {
    background: linear-gradient(-355deg, #65ff3b8c, #fec17bc9);
    color: #001a96;
    border-radius: 25px;
    font-size: 20px;
    font-weight: bold;
}
.gradient-header {
    background: linear-gradient(90deg, #11cb91, #fc53256b);
    color: #8b006a;
    font-size: 41px;
    font-weight: bold;
}
.gradient-subheader {
    background: linear-gradient(90deg, #00acffd1, #9575cda6);
    color: #283593f0;
    font-size: 70px;
    border-radius: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<h1 class='gradient-header' style='text-align:center;'>👉 Text Summarization Tool 👈</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h2 class='gradient-subheader' style='text-align:center;'>🤖 LangChain 🐦</h2>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 class='gradient-text' style='text-align:center;'>"
    "Modular pipeline (extractive + abstractive) summarization with optional ROUGE evaluation"
    "</h4>",
    unsafe_allow_html=True
)

# =================== INPUT METHOD ===================
input_method = st.radio(
    "Do you have Text or a DOCX file?",
    options=["Text 🔤", "DOCX 📄"]
)

user_text = ""

if input_method == "Text 🔤":
    user_text = st.text_area("Enter the text to summarize:", height=200)

elif input_method == "DOCX 📄":
    uploaded_file = st.file_uploader("Upload your DOCX file:", type=["docx"])
    if uploaded_file:
        with st.spinner("Extracting text from DOCX..."):
            user_text = extract_text_from_docx(uploaded_file)
            st.success("Text extracted successfully!")

# =================== SUMMARIZATION ===================
if user_text:
    has_reference = st.radio(
        "Do you have a reference summary?",
        options=["No", "Yes"],
        index=0
    )

    reference_summary = None
    if has_reference == "Yes":
        reference_summary = st.text_area(
            "Enter the reference summary:",
            height=100
        )

    if st.button("Summarize Text"):
        with st.spinner("Processing... Please wait ⏳"):
            chunks = preprocess_text(user_text)
            summaries = []

            for chunk in chunks:
                summaries.append(
                    summarization_pipeline(
                        chunk,
                        min_length=20 if reference_summary else 249
                    )
                )

            final_summary = " ".join(summaries)

        st.subheader("Generated Summary")
        st.write(final_summary)

        if reference_summary:
            with st.spinner("Calculating ROUGE scores..."):
                scores = calculate_rouge(reference_summary, final_summary)

            st.subheader("ROUGE Scores")
            for metric, score in scores.items():
                st.write(
                    f"**{metric.upper()}** — "
                    f"Precision: {score.precision:.4f}, "
                    f"Recall: {score.recall:.4f}, "
                    f"F1: {score.fmeasure:.4f}"
                )

# =================== SIDEBAR ===================
with st.sidebar:
    st.image(
    "https://i.graphicmama.com/uploads/2023/3/6414793b7befa-602a4f0ed6583-Flying%20Robot%20Cartoon%20Animated%20GIFs.gif",
    use_column_width=True
    )
    st.markdown(
        "<center>Made with ❤️ Kamran Mustafa<br>© 2024–2025</center>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<center>🎓 Birmingham City University 🎓</center>",
        unsafe_allow_html=True
    )
