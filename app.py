# ==============================================================================
# Smart Knowledge & Skills Hub - Prototype
# Author: Gemini (as a World-Class AI Engineer)
# Date: August 14, 2025
# ==============================================================================

# ==============================================================================
# 1. DEPENDENCIES & SETUP
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import openai
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import faiss  # For scalable similarity search

# ==============================================================================
# 2. CONFIGURATION & INITIALIZATION
# ==============================================================================

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Knowledge & Skills Hub",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Model & API Initialization ---
@st.cache_resource
def load_models():
    """Load the sentence transformer model once and cache it."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


embedding_model = load_models()

# --- OpenAI API Key ---
# Best practice: Use Streamlit secrets. For this prototype, we'll use the sidebar.
st.sidebar.title("Configuration")
st.sidebar.markdown(
    "Enter your OpenAI API key to enable summary and learning path generation."
)
api_key = st.sidebar.text_input(
    "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",
    type="password",
)

if api_key:
    openai.api_key = api_key
else:
    st.sidebar.warning(
        "OpenAI API key is missing. AI features will be disabled.", icon="⚠️"
    )


# ==============================================================================
# 3. DATA COLLECTION & PREPROCESSING
# ==============================================================================


@st.cache_data
def scrape_arxiv(query="machine learning", max_results=20):
    """
    Scrapes arXiv for papers related to a query to simulate live data collection.
    """
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "xml")
        entries = soup.find_all("entry")

        data = []
        for entry in entries:
            title = entry.title.text.strip()
            # Handle potential NoneType for author
            authors = (
                [author.find("name").text for author in entry.find_all("author")]
                if entry.find("author")
                else ["Unknown"]
            )
            url = entry.id.text.strip()
            summary = entry.summary.text.strip().replace("\n", " ")

            data.append(
                {
                    "title": title,
                    "author": ", ".join(authors),
                    "source": "arXiv",
                    "url": url,
                    "description": summary,
                    "skills_tags": ["research", "paper", query],  # Simple tagging
                    "content_type": "Paper",
                }
            )
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Failed to scrape arXiv: {e}")
        return pd.DataFrame()


@st.cache_data
def get_static_data():
    """
    Loads a static dataset to represent content from various sources.
    In a real application, this would be fed by multiple scrapers/APIs.
    """
    data = {
        "title": [
            "Machine Learning by Andrew Ng",
            "Introduction to Computer Science and Programming in Python",
            "Khan Academy: SQL - Querying and managing data",
            "Natural Language Processing with Deep Learning",
            "Crash Course: Statistics",
            "Linear Algebra",
            "Calculus 1",
        ],
        "author": [
            "Coursera",
            "MIT OpenCourseWare",
            "Khan Academy",
            "Stanford University",
            "Khan Academy",
            "MIT OpenCourseWare",
            "Khan Academy",
        ],
        "source": [
            "Coursera",
            "MIT OpenCourseWare",
            "Khan Academy",
            "Coursera",
            "Khan Academy",
            "MIT OpenCourseWare",
            "Khan Academy",
        ],
        "url": [
            "https://www.coursera.org/learn/machine-learning",
            "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/",
            "https://www.khanacademy.org/computing/computer-programming/sql",
            "https://www.coursera.org/learn/natural-language-processing-deep-learning",
            "https://www.khanacademy.org/math/statistics-probability",
            "https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/",
            "https://www.khanacademy.org/math/calculus-1",
        ],
        "description": [
            "A comprehensive introduction to machine learning, data mining, and statistical pattern recognition. Topics include supervised and unsupervised learning, and best practices in machine learning.",
            "An introduction to computation. It covers the Python programming language and fundamental concepts like data structures, algorithms, and object-oriented programming.",
            "Learn how to use SQL to store, query, and manipulate data. SQL is a powerful language used for relational databases.",
            "This course covers cutting-edge techniques in deep learning for NLP, including recurrent neural networks, transformers, and attention mechanisms.",
            "A full course on statistics and probability, covering everything from basic concepts to inference and modeling.",
            "A foundational course in linear algebra, covering vectors, matrices, determinants, and eigenvalues.",
            "An introduction to calculus, covering limits, derivatives, and integrals.",
        ],
        "skills_tags": [
            ["machine learning", "python", "algorithms"],
            ["python", "programming", "cs fundamentals"],
            ["sql", "database", "data management"],
            ["nlp", "deep learning", "python", "tensorflow"],
            ["statistics", "probability", "data analysis"],
            ["math", "algebra", "vectors"],
            ["math", "calculus", "derivatives"],
        ],
        "content_type": [
            "Course",
            "Course",
            "Tutorial",
            "Course",
            "Tutorial",
            "Course",
            "Tutorial",
        ],
    }
    return pd.DataFrame(data)


def preprocess_data(df):
    """Cleans and standardizes the data."""
    df.dropna(subset=["title", "description"], inplace=True)
    df.drop_duplicates(subset=["title", "url"], inplace=True)
    df["text_for_embedding"] = df["title"] + " " + df["description"]
    return df


# ==============================================================================
# 4. EMBEDDINGS & CLUSTERING
# ==============================================================================


@st.cache_resource
def generate_embeddings(texts):
    """Generates embeddings for a list of texts using the cached model."""
    return embedding_model.encode(texts, show_progress_bar=True)


@st.cache_data
def create_faiss_index(embeddings):
    """Creates a FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance
    index.add(embeddings)
    return index


@st.cache_data
def get_clusters(embeddings, num_clusters=5):
    """Clusters content using KMeans."""
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
    kmeans.fit(embeddings)
    return kmeans.labels_


# ==============================================================================
# 5. GPT-POWERED FEATURES
# ==============================================================================


def generate_summary(text):
    """Generates a short, engaging summary using GPT."""
    if not api_key:
        return "Summary generation requires an OpenAI API key."
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant who summarizes educational content concisely for learners.",
                },
                {
                    "role": "user",
                    "content": f"Summarize the following content description in one engaging sentence:\n\n---\n\n{text}",
                },
            ],
            max_tokens=60,
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate summary: {e}"


def generate_learning_path(query, search_results_df):
    """Generates a personalized learning path based on search results."""
    if not api_key:
        return "Learning path generation requires an OpenAI API key."

    content_list = ""
    for _, row in search_results_df.iterrows():
        content_list += f"- **{row['title']}** (Type: {row['content_type']}, Source: {row['source']}): {row['description']}\n"

    prompt = f"""
    A user is interested in learning about "{query}". Based on the following curated list of educational resources, create a logical, step-by-step learning path for a beginner.

    **Instructions:**
    1.  Start with the most foundational content.
    2.  Organize the resources in a numbered list, explaining *why* each step is important.
    3.  Suggest a simple, practical project at the end to help solidify the knowledge.
    4.  Keep the tone encouraging and clear.

    **Available Resources:**
    {content_list}

    **Your Output (as Markdown):**
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Using a more capable model for reasoning tasks
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert curriculum developer and academic advisor.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate learning path: {e}"


# ==============================================================================
# 6. STREAMLIT DASHBOARD
# ==============================================================================

st.title("🧠 Smart Knowledge & Skills Hub")
st.markdown(
    "Your AI-powered guide to personalized learning. Discover, search, and map your educational journey."
)


# --- Load, process, and cache all data ---
@st.cache_data
def load_and_prepare_data():
    """Main function to load all data sources and prepare them for the app."""
    static_df = get_static_data()
    arxiv_df = scrape_arxiv()
    combined_df = pd.concat([static_df, arxiv_df], ignore_index=True)
    processed_df = preprocess_data(combined_df)

    # Generate embeddings
    embeddings = generate_embeddings(processed_df["text_for_embedding"].tolist())

    # Create FAISS index
    faiss_index = create_faiss_index(embeddings)

    # Get clusters
    processed_df["cluster"] = get_clusters(embeddings)

    return processed_df, embeddings, faiss_index


df, embeddings, faiss_index = load_and_prepare_data()

# --- User Input & Search ---
st.header("🔍 Search for Content")
query = st.text_input(
    "What would you like to learn about today?", "e.g., Python for data science"
)

# --- Sidebar Filters ---
st.sidebar.header("Filters")
selected_source = st.sidebar.multiselect(
    "Filter by Source", options=df["source"].unique(), default=df["source"].unique()
)
selected_type = st.sidebar.multiselect(
    "Filter by Content Type",
    options=df["content_type"].unique(),
    default=df["content_type"].unique(),
)

# Filter dataframe based on sidebar selections
filtered_df = df[
    df["source"].isin(selected_source) & df["content_type"].isin(selected_type)
].copy()

if query:
    # --- Semantic Search Logic ---
    query_embedding = generate_embeddings([query])[0]

    # FAISS Search
    k = 20  # Number of nearest neighbors to retrieve
    distances, indices = faiss_index.search(np.array([query_embedding]), k)

    # Get results from the main dataframe
    search_results_indices = indices[0]
    results_df = df.iloc[search_results_indices]

    # Apply sidebar filters to the search results
    final_results = results_df[
        results_df["source"].isin(selected_source)
        & results_df["content_type"].isin(selected_type)
    ]

    st.header(f"Results for '{query}'")

    if final_results.empty:
        st.warning(
            "No results found for your query and filters. Try adjusting your search."
        )
    else:
        # --- Display Learning Path Recommendation ---
        st.subheader("🚀 Your Personalized Learning Path")
        if st.button("Generate Learning Path with AI"):
            if not api_key:
                st.error(
                    "Please enter your OpenAI API key in the sidebar to use this feature."
                )
            else:
                with st.spinner("🤖 GPT is thinking... Please wait."):
                    # Generate path using top 5 results
                    learning_path = generate_learning_path(query, final_results.head(5))
                    st.markdown(learning_path)

        st.markdown("---")  # Separator

        # --- Display Individual Results ---
        st.subheader("📚 Curated Content")
        for _, row in final_results.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"#### [{row['title']}]({row['url']})")
                    st.caption(
                        f"Source: **{row['source']}** | Type: **{row['content_type']}**"
                    )

                with col2:
                    st.markdown(f"**Cluster ID:** `{row['cluster']}`")

                with st.expander("Show Details & AI Summary"):
                    st.markdown(f"**Description:** {row['description']}")
                    st.markdown(f"**Skills:** `{'`, `'.join(row['skills_tags'])}`")
                    st.markdown("---")

                    if api_key:
                        summary = generate_summary(row["description"])
                        st.markdown(f"**✨ AI Summary:** {summary}")
                    else:
                        st.info(
                            "Enter an OpenAI API key in the sidebar to see AI-generated summaries."
                        )


# ==============================================================================
# 7. SCALABLE ARCHITECTURE NOTES
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.header("Scalability Notes")
st.sidebar.info(
    """
    **Prototype Workflow:**
    Scraping → Preprocessing → Embeddings → GPT → Dashboard (all in one script).

    **Production Architecture:**
    1.  **Async Scraping**: Use a task queue (e.g., Celery, RabbitMQ) to run scrapers as background jobs.
    2.  **Vector Database**: Store embeddings in a dedicated vector DB like Pinecone, Weaviate, or a managed FAISS instance for fast, large-scale semantic search.
    3.  **Backend API**: Create a separate backend service (e.g., using FastAPI) to handle API calls, search logic, and GPT interactions. The Streamlit app would be a pure frontend.
    4.  **Data Warehouse**: Store structured metadata in a robust database like PostgreSQL.
    5.  **Caching**: Use Redis for caching API responses and frequently accessed data to improve performance.
    """
)
