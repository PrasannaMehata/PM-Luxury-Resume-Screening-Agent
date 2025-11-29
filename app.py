

import os
import re
import io
import math
import base64
from typing import List, Tuple, Dict
import json
import time

import streamlit as st
import pandas as pd

# Text extraction
import PyPDF2
import docx2txt
import pdfplumber

# NLP & vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Vector Database - Updated ChromaDB
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    st.warning("ChromaDB not available. Install with: pip install chromadb")

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# ---------- ChromaDB Setup (Updated API) ----------

def initialize_chromadb():
    """Initialize ChromaDB client and collection with new API"""
    try:
        if not CHROMA_AVAILABLE:
            return None, None
            
        # New ChromaDB client initialization
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="resume_screening",
            metadata={"description": "Resume and job description embeddings"}
        )
        return client, collection
    except Exception as e:
        st.warning(f"ChromaDB initialization warning: {str(e)}")
        return None, None

def store_embeddings_in_chromadb(collection, documents, embeddings, metadatas, ids):
    """Store documents and embeddings in ChromaDB"""
    try:
        if collection is None:
            return False
            
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        return True
    except Exception as e:
        st.warning(f"Warning storing in ChromaDB: {str(e)}")
        return False

def query_chromadb(collection, query_embedding, n_results=10):
    """Query ChromaDB for similar documents"""
    try:
        if collection is None:
            return None
            
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results
    except Exception as e:
        st.warning(f"Warning querying ChromaDB: {str(e)}")
        return None

# ---------- Utility functions ----------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                ptext = page.extract_text()
                if ptext:
                    text.append(ptext)
    except Exception:
        pass

    if not text:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            for p in reader.pages:
                ptext = p.extract_text()
                if ptext:
                    text.append(ptext)
        except Exception:
            pass

    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        with open('/tmp/temp_resume.docx', 'wb') as f:
            f.write(file_bytes)
        text = docx2txt.process('/tmp/temp_resume.docx')
        return text or ''
    except Exception:
        return ''

def extract_text_from_uploaded(file) -> str:
    content = file.read()
    fname = file.name.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif fname.endswith('.docx') or fname.endswith('.doc'):
        return extract_text_from_docx(content)
    elif fname.endswith('.txt'):
        try:
            return content.decode('utf-8', errors='ignore')
        except Exception:
            return str(content)
    else:
        return extract_text_from_pdf(content)

def clean_text(t: str) -> str:
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

# Skills database (expanded)
COMMON_SKILLS = [
    'python','java','c++','c#','javascript','typescript','react','angular','vue','node','express',
    'aws','azure','gcp','docker','kubernetes','terraform','ansible','jenkins','git','ci/cd',
    'sql','nosql','postgres','mysql','mongodb','redis','django','flask','fastapi','spring',
    'pandas','numpy','scikit-learn','tensorflow','pytorch','keras','nlp','computer vision',
    'machine learning','deep learning','data analysis','tableau','power bi','excel',
    'agile','scrum','jira','confluence','rest','graphql','microservices','linux','bash',
    'salesforce','sap','oracle','html','css','sass','webpack','babel','jest','cypress'
]
COMMON_SKILLS = list(set(COMMON_SKILLS))

def extract_skills(text: str, skills_list: List[str]=COMMON_SKILLS) -> List[str]:
    text_low = text.lower()
    found = []
    for s in skills_list:
        pattern = r'\b' + re.escape(s.lower()) + r'\b'
        if re.search(pattern, text_low):
            found.append(s)
    return found

def estimate_years_experience(text: str) -> float:
    matches = re.findall(r'(\d{1,2})\s*(?:\+)?\s*(?:years|year|yrs|yr)\b', text.lower())
    nums = [int(m) for m in matches]
    if nums:
        return max(nums)
    ranges = re.findall(r'((?:19|20)\d{2})\s*[-–—]\s*((?:19|20)\d{2})', text)
    if ranges:
        try:
            start, end = ranges[-1]
            return max(0.0, int(end) - int(start))
        except Exception:
            pass
    return 0.0

def detect_education_level(text: str) -> str:
    t = text.lower()
    if 'phd' in t or 'doctor' in t:
        return 'PhD'
    if re.search(r'\b(master|msc|ms|m\.s\.|mba|mtech|m\.tech)\b', t):
        return 'Masters'
    if re.search(r'\b(bachelor|bsc|ba|b\.sc|b\.tech|btech|b\.e|b\.eng)\b', t):
        return 'Bachelors'
    return 'Unknown'

def extract_contact_info(text: str) -> Dict[str, str]:
    out = {'email': '', 'phone': '', 'linkedin': ''}
    email_m = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    if email_m:
        out['email'] = email_m.group(0)
    phone_m = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}', text)
    if phone_m:
        out['phone'] = phone_m.group(0)
    ln = re.search(r'(https?://)?(www\.)?linkedin\.com/[^\s,;]+', text.lower())
    if ln:
        out['linkedin'] = ln.group(0)
    return out

def summarize_text_tfidf(text: str, n_sentences: int = 3) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sents) <= n_sentences:
        return " ".join(sents)
    vect = TfidfVectorizer(stop_words='english', max_features=2000)
    try:
        X = vect.fit_transform(sents)
        scores = X.sum(axis=1).A1
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_sentences]
        ranked_idx.sort()
        summary = " ".join([sents[i] for i in ranked_idx])
        return summary
    except Exception:
        return " ".join(sents[:n_sentences])

def detect_seniority(text: str, years: float) -> str:
    t = text.lower()
    if re.search(r'\b(senior|sr\.|lead|principal|manager|director|head of)\b', t):
        return 'Senior'
    if re.search(r'\b(junior|jr\.|intern|entry-level|graduate|associate)\b', t):
        return 'Junior'
    if years >= 8:
        return 'Senior'
    if years >= 3:
        return 'Mid'
    return 'Junior'

# ---------- Embedding helpers ----------
import requests
import os
os.environ["GEMINI_API_KEY"] = "enter your api key"
def has_gemini_key() -> bool:
    return bool(os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY'))

def call_gemini_embedding(texts: List[str]) -> List[List[float]]:
    api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('No Gemini/OpenAI API key found in environment.')

    url = 'https://api.openai.com/v1/embeddings'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    model = 'text-embedding-3-small'
    payload = {'model': model, 'input': texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f'Embedding API failed: {resp.status_code} {resp.text}')
    data = resp.json()
    embeddings = [item['embedding'] for item in data['data']]
    return embeddings

# ---------- Enhanced Ranking logic with ChromaDB ----------

def rank_resumes_tfidf(jd_text: str, resumes: List[Tuple[str, str]], weights: Dict[str, float]) -> pd.DataFrame:
    docs = [jd_text] + [r[1] for r in resumes]
    vectorizer = TfidfVectorizer(stop_words='english', max_features=6000)
    X = vectorizer.fit_transform(docs)
    jd_vec = X[0]
    res_vecs = X[1:]
    sims = cosine_similarity(jd_vec, res_vecs)[0]

    rows = []
    for i, (fname, text) in enumerate(resumes):
        skills = extract_skills(text)
        exp = estimate_years_experience(text)
        edu = detect_education_level(text)
        skill_score = len(skills)
        sim_score = float(sims[i])
        combined = (weights['similarity'] * sim_score +
                    weights['skills'] * (skill_score / (len(COMMON_SKILLS) or 1)) +
                    weights['experience'] * min(exp / 10.0, 1.0))
        contact = extract_contact_info(text)
        seniority = detect_seniority(text, exp)
        summary = summarize_text_tfidf(text, n_sentences=2)
        rows.append({
            'filename': fname,
            'similarity': sim_score,
            'skill_match_count': skill_score,
            'skills_found': ','.join(skills),
            'years_experience': exp,
            'education': edu,
            'seniority': seniority,
            'contact_email': contact.get('email',''),
            'contact_phone': contact.get('phone',''),
            'linkedin': contact.get('linkedin',''),
            'summary': summary,
            'score': combined
        })

    df = pd.DataFrame(rows).sort_values(by='score', ascending=False).reset_index(drop=True)
    return df

def rank_resumes_embeddings(jd_text: str, resumes: List[Tuple[str, str]], weights: Dict[str, float]) -> pd.DataFrame:
    texts = [jd_text] + [r[1] for r in resumes]
    embeddings = call_gemini_embedding(texts)
    jd_emb = embeddings[0]
    res_embs = embeddings[1:]

    def cosine(a, b):
        num = sum(x*y for x,y in zip(a,b))
        den = math.sqrt(sum(x*x for x in a))*math.sqrt(sum(y*y for y in b))
        if den == 0:
            return 0.0
        return num/den

    rows = []
    for i, (fname, text) in enumerate(resumes):
        sim_score = cosine(jd_emb, res_embs[i])
        skills = extract_skills(text)
        exp = estimate_years_experience(text)
        edu = detect_education_level(text)
        skill_score = len(skills)
        combined = (weights['similarity'] * sim_score +
                    weights['skills'] * (skill_score / (len(COMMON_SKILLS) or 1)) +
                    weights['experience'] * min(exp / 10.0, 1.0))
        contact = extract_contact_info(text)
        seniority = detect_seniority(text, exp)
        summary = summarize_text_tfidf(text, n_sentences=2)
        rows.append({
            'filename': fname,
            'similarity': sim_score,
            'skill_match_count': skill_score,
            'skills_found': ','.join(skills),
            'years_experience': exp,
            'education': edu,
            'seniority': seniority,
            'contact_email': contact.get('email',''),
            'contact_phone': contact.get('phone',''),
            'linkedin': contact.get('linkedin',''),
            'summary': summary,
            'score': combined
        })

    df = pd.DataFrame(rows).sort_values(by='score', ascending=False).reset_index(drop=True)
    return df

def rank_resumes_chromadb(jd_text: str, resumes: List[Tuple[str, str]], weights: Dict[str, float], collection) -> pd.DataFrame:
    """Rank resumes using ChromaDB for similarity search"""
    try:
        if collection is None:
            return rank_resumes_tfidf(jd_text, resumes, weights)
            
        # Get JD embedding
        jd_embedding = call_gemini_embedding([jd_text])[0]
        
        # Store resumes in ChromaDB if not already there
        resume_texts = [r[1] for r in resumes]
        resume_names = [r[0] for r in resumes]
        
        # Get embeddings for all resumes
        resume_embeddings = call_gemini_embedding(resume_texts)
        
        # Store in ChromaDB
        metadatas = [{"filename": name, "type": "resume"} for name in resume_names]
        ids = [f"resume_{i}" for i in range(len(resumes))]
        
        store_embeddings_in_chromadb(collection, resume_texts, resume_embeddings, metadatas, ids)
        
        # Query ChromaDB for similar resumes
        results = query_chromadb(collection, jd_embedding, n_results=len(resumes))
        
        rows = []
        for i, (fname, text) in enumerate(resumes):
            # Find the similarity score from ChromaDB results
            sim_score = 0.0
            if results and 'documents' in results:
                for idx, doc in enumerate(results['documents'][0]):
                    if text in doc or fname in results['metadatas'][0][idx].get('filename', ''):
                        if 'distances' in results:
                            sim_score = 1 - results['distances'][0][idx]  # Convert distance to similarity
                        break
            
            skills = extract_skills(text)
            exp = estimate_years_experience(text)
            edu = detect_education_level(text)
            skill_score = len(skills)
            combined = (weights['similarity'] * sim_score +
                        weights['skills'] * (skill_score / (len(COMMON_SKILLS) or 1)) +
                        weights['experience'] * min(exp / 10.0, 1.0))
            contact = extract_contact_info(text)
            seniority = detect_seniority(text, exp)
            summary = summarize_text_tfidf(text, n_sentences=2)
            rows.append({
                'filename': fname,
                'similarity': sim_score,
                'skill_match_count': skill_score,
                'skills_found': ','.join(skills),
                'years_experience': exp,
                'education': edu,
                'seniority': seniority,
                'contact_email': contact.get('email',''),
                'contact_phone': contact.get('phone',''),
                'linkedin': contact.get('linkedin',''),
                'summary': summary,
                'score': combined
            })

        df = pd.DataFrame(rows).sort_values(by='score', ascending=False).reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"ChromaDB ranking failed, falling back to TF-IDF: {str(e)}")
        # Fallback to TF-IDF
        return rank_resumes_tfidf(jd_text, resumes, weights)

# ---------- Luxury UI/UX with Premium Color Scheme ----------

PAGE_CSS = """
<style>
/* Luxury color scheme with deep blues, gold, and elegant gradients */
.main {
    background-color: #0a0f2b;
}

.stApp {
    background: linear-gradient(135deg, #0a0f2b 0%, #1a1f3c 50%, #2d1b69 100%);
    min-height: 100vh;
}

/* Main content container with luxury glass morphism */
.main .block-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 24px;
    margin-top: 2rem;
    margin-bottom: 2rem;
    padding: 2.5rem;
    box-shadow: 0 25px 70px rgba(0,0,0,0.2);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.3);
    background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.95) 100%);
}

/* Premium text colors */
.stMarkdown, .stText, .stTitle, .stHeader {
    color: #1e293b !important;
}

.stMetric {
    color: #1e293b !important;
}

/* Luxury animated header */
.animated-header {
    background: linear-gradient(135deg, #1a1f3c 0%, #2d1b69 50%, #4a1e7a 100%);
    padding: 4rem 2rem;
    border-radius: 24px;
    margin-bottom: 3rem;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: floatHeader 8s ease-in-out infinite;
    border: 1px solid rgba(255,215,0,0.2);
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
}

@keyframes floatHeader {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-15px); }
}

.animated-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: linear-gradient(45deg, transparent, rgba(255,215,0,0.1), transparent);
    animation: shimmer 12s infinite;
}

.animated-header h1 {
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1.5rem;
    text-shadow: 0 6px 12px rgba(0,0,0,0.3);
    position: relative;
    color: #ffffff;
    background: linear-gradient(135deg, #ffffff 0%, #ffd700 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.animated-header p {
    font-size: 1.4rem;
    opacity: 0.9;
    margin-bottom: 0;
    font-weight: 300;
    position: relative;
    color: #e2e8f0;
}

/* Luxury floating elements */
.floating-element {
    animation: luxuryFloat 8s ease-in-out infinite;
}

@keyframes luxuryFloat {
    0%, 100% { transform: translateY(0px) rotate(0deg) scale(1); }
    33% { transform: translateY(-20px) rotate(2deg) scale(1.02); }
    66% { transform: translateY(-10px) rotate(-1deg) scale(1.01); }
}

/* Premium card styling */
.resume-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.resume-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,215,0,0.1), transparent);
    transition: left 0.7s;
}

.resume-card:hover::before {
    left: 100%;
}

.resume-card:hover {
    transform: translateY(-12px) scale(1.02);
    box-shadow: 0 25px 60px rgba(0,0,0,0.15);
    border-color: #ffd700;
}

/* Luxury progress bars */
.score-bar {
    height: 12px;
    background: rgba(26, 31, 60, 0.1);
    border-radius: 12px;
    overflow: hidden;
    margin: 1rem 0;
    position: relative;
}

.score-fill {
    height: 100%;
    border-radius: 12px;
    background: linear-gradient(90deg, #2d1b69, #4a1e7a, #6b21a8);
    transition: width 1.2s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.score-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 3s infinite;
}

/* Premium tags and badges */
.skill-tag {
    display: inline-block;
    background: linear-gradient(135deg, #2d1b69, #4a1e7a);
    color: white;
    padding: 0.5rem 1.2rem;
    border-radius: 25px;
    font-size: 0.85rem;
    margin: 0.3rem;
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.4s ease;
    animation: popIn 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(45, 27, 105, 0.2);
}

.skill-tag:hover {
    transform: scale(1.15) rotate(2deg);
    box-shadow: 0 6px 20px rgba(45, 27, 105, 0.4);
    background: linear-gradient(135deg, #4a1e7a, #6b21a8);
}

.status-badge {
    display: inline-block;
    padding: 0.6rem 1.2rem;
    border-radius: 25px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-left: 0.5rem;
    animation: slideIn 0.6s ease-out;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.badge-recommended {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.3);
}

.badge-hold {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    color: white;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
}

/* Luxury step indicators */
.step-indicator {
    display: flex;
    justify-content: space-between;
    margin: 4rem 0;
    position: relative;
}

.step-indicator::before {
    content: '';
    position: absolute;
    top: 30px;
    left: 10%;
    right: 10%;
    height: 6px;
    background: linear-gradient(90deg, #2d1b69, #4a1e7a, #6b21a8);
    z-index: 1;
    border-radius: 3px;
    box-shadow: 0 4px 15px rgba(45, 27, 105, 0.2);
}

.step {
    text-align: center;
    z-index: 2;
    position: relative;
    flex: 1;
    animation: fadeInUp 0.8s ease-out;
}

.step-number {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: white;
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-weight: 700;
    font-size: 1.4rem;
    border: 4px solid white;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.step.active .step-number {
    background: linear-gradient(135deg, #2d1b69, #4a1e7a);
    color: white;
    transform: scale(1.15);
    box-shadow: 0 12px 30px rgba(45, 27, 105, 0.4);
    border-color: #ffd700;
}

.step.completed .step-number {
    background: linear-gradient(135deg, #059669, #10b981);
    color: white;
    box-shadow: 0 12px 30px rgba(5, 150, 105, 0.4);
}

.step-label {
    font-size: 1rem;
    color: #64748b;
    font-weight: 600;
    transition: all 0.4s ease;
}

.step.active .step-label {
    color: #2d1b69;
    font-weight: 700;
    transform: scale(1.08);
}

/* Luxury section headers */
.section-header {
    font-size: 1.9rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 2.5rem 0 2rem 0;
    padding-bottom: 1rem;
    border-bottom: 4px solid;
    border-image: linear-gradient(90deg, #2d1b69, #4a1e7a, #6b21a8) 1;
    animation: slideInFromLeft 0.8s ease-out;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Premium buttons */
.stButton button {
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #2d1b69, #4a1e7a) !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 2rem !important;
    font-size: 1.1rem !important;
    box-shadow: 0 8px 25px rgba(45, 27, 105, 0.3) !important;
}

.stButton button:hover {
    transform: translateY(-4px) scale(1.05) !important;
    box-shadow: 0 15px 40px rgba(45, 27, 105, 0.4) !important;
    background: linear-gradient(135deg, #4a1e7a, #6b21a8) !important;
}

/* Luxury metric cards */
.metric-card {
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 12px 35px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.4);
    transition: all 0.4s ease;
    animation: fadeInUp 0.8s ease-out;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #2d1b69, #4a1e7a, #6b21a8);
}

.metric-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.15);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: 900;
    margin: 1rem 0;
    background: linear-gradient(135deg, #2d1b69, #4a1e7a);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 1rem;
    color: #64748b;
    font-weight: 600;
}

/* Luxury file uploader */
.uploadedFile {
    background: linear-gradient(135deg, #ffffff, #f8fafc) !important;
    border: 3px dashed #2d1b69 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin: 0.75rem 0 !important;
    transition: all 0.4s ease !important;
}

.uploadedFile:hover {
    border-color: #4a1e7a !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 30px rgba(45, 27, 105, 0.2) !important;
}

/* Premium Top Performers Section */
.top-performer-card {
    background: linear-gradient(135deg, #fffaf0, #ffe4b5);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 15px 40px rgba(0,0,0,0.12);
    border: 2px solid #ffd700;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.top-performer-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,215,0,0.2), transparent);
    transition: left 0.7s;
}

.top-performer-card:hover::before {
    left: 100%;
}

.top-performer-card:hover {
    transform: translateY(-10px) scale(1.03);
    box-shadow: 0 25px 60px rgba(255, 173, 51, 0.25);
    border-color: #ffa940;
}

.top-performer-badge {
    position: absolute;
    top: -12px;
    right: 25px;
    background: linear-gradient(135deg, #ffd700, #ffa940);
    color: #1e293b;
    padding: 0.6rem 1.5rem;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 800;
    box-shadow: 0 6px 20px rgba(255, 173, 51, 0.4);
    z-index: 2;
    border: 2px solid white;
}

/* Luxury Comparison View */
.comparison-view {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-radius: 20px;
    padding: 2rem;
    margin: 2rem 0;
    border: 2px solid #bae6fd;
    box-shadow: 0 12px 35px rgba(0,0,0,0.08);
}

/* Enhanced data frames */
.stDataFrame {
    color: #1e293b !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}

.stDataFrame th {
    color: white !important;
    background: linear-gradient(135deg, #2d1b69, #4a1e7a) !important;
    font-weight: 700 !important;
    padding: 1rem !important;
}

.stDataFrame td {
    color: #1e293b !important;
    padding: 0.75rem !important;
}

/* Horizontal flow container */
.horizontal-flow {
    display: flex;
    gap: 2rem;
    align-items: stretch;
    margin: 2rem 0;
}

.horizontal-card {
    flex: 1;
    background: linear-gradient(135deg, #ffffff, #f8fafc);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 12px 35px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.3);
    transition: all 0.4s ease;
    text-align: center;
}

.horizontal-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 50px rgba(0,0,0,0.15);
}

.horizontal-card h3 {
    color: #2d1b69;
    margin-bottom: 1rem;
    font-weight: 700;
}

.horizontal-card p {
    color: #64748b;
    line-height: 1.6;
}

/* ChromaDB status indicator */
.chroma-status {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 0.5rem;
}

.chroma-available {
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
}

.chroma-unavailable {
    background: linear-gradient(135deg, #f59e0b, #d97706);
    color: white;
}
</style>
"""

def render_step_indicator(current_step):
    steps = [
        {"label": "Upload Resumes", "key": "upload"},
        {"label": "Job Description", "key": "jd"},
        {"label": "Configure", "key": "config"},
        {"label": "Results", "key": "results"}
    ]
    
    st.markdown('<div class="step-indicator">', unsafe_allow_html=True)
    
    for i, step in enumerate(steps):
        is_active = current_step == step['key']
        is_completed = list(steps).index(step) < list(steps).index(next(s for s in steps if s['key'] == current_step))
        
        step_class = "step"
        if is_active:
            step_class += " active"
        elif is_completed:
            step_class += " completed"
            
        st.markdown(f"""
        <div class="{step_class}">
            <div class="step-number">{i+1}</div>
            <div class="step-label">{step['label']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def app():
    st.set_page_config(
        page_title='PM Luxury Resume Screening Agent', 
        layout='wide',
        page_icon="🎯",
        initial_sidebar_state="collapsed"
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Initialize ChromaDB
    if 'chroma_client' not in st.session_state:
        st.session_state.chroma_client, st.session_state.chroma_collection = initialize_chromadb()

    # Initialize session state
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'upload'
    if 'resumes' not in st.session_state:
        st.session_state.resumes = []
    if 'jd_text' not in st.session_state:
        st.session_state.jd_text = ''
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    if 'selected_candidates' not in st.session_state:
        st.session_state.selected_candidates = []

    # Luxury Animated Header
    st.markdown("""
    <div class="animated-header">
        <div class="floating-element">
            <h1>🎯 PM Luxury Resume Screening</h1>
            <p>AI-Powered Candidate Matching with Premium Experience(SCROLL DOWN)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step indicator
    render_step_indicator(st.session_state.current_step)

    # Main workflow
    if st.session_state.current_step == 'upload':
        render_upload_step()
    elif st.session_state.current_step == 'jd':
        render_jd_step()
    elif st.session_state.current_step == 'config':
        render_config_step()
    elif st.session_state.current_step == 'results':
        render_results_step()

def render_upload_step():
    st.markdown('<div class="section-header">📁 Upload Resumes</div>', unsafe_allow_html=True)
    
    # ChromaDB status
    chroma_status = "Available" if st.session_state.chroma_collection else "Unavailable"
    status_class = "chroma-available" if st.session_state.chroma_collection else "chroma-unavailable"
    st.markdown(f'<div style="text-align: right; margin-bottom: 1rem;">ChromaDB: <span class="chroma-status {status_class}">{chroma_status}</span></div>', unsafe_allow_html=True)
    
    # Horizontal flow layout
    st.markdown("""
    <div class="horizontal-flow">
        <div class="horizontal-card">
            <h3>🚀 Quick Upload</h3>
            <p>Upload multiple resumes in PDF, DOCX, or TXT format. Our AI will automatically extract and analyze the content.</p>
        </div>
        <div class="horizontal-card">
            <h3>🔍 Smart Processing</h3>
            <p>Advanced NLP extracts skills, experience, education, and contact information from each resume.</p>
        </div>
        <div class="horizontal-card">
            <h3>💾 Secure Storage</h3>
            <p>All data is processed securely with ChromaDB vector storage for fast and accurate matching.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
        "Upload resume files (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'doc', 'txt'],
        accept_multiple_files=True,
        help="You can upload multiple files at once"
    )
    
    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        parsed_resumes = []
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
            try:
                text = extract_text_from_uploaded(file)
                text = clean_text(text)
                parsed_resumes.append((file.name, text))
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        st.session_state.resumes = parsed_resumes
        
        # Success Message
        st.success(f"✅ Successfully processed {len(parsed_resumes)} resumes")
        
        # Preview Cards
        st.markdown('<div class="section-header">📋 Resume Preview</div>', unsafe_allow_html=True)
        preview_cols = st.columns(3)

        for idx, (name, text) in enumerate(parsed_resumes[:3]):
            with preview_cols[idx]:
                skills = extract_skills(text)
                exp = estimate_years_experience(text)

                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, #2b1055, #7597de);
                    padding: 1.5rem;
                    border-radius: 16px;
                    margin: 0.5rem 0;
                    border-left: 5px solid #ffffff;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                    color: white;
                '>
                    <h4 style='margin: 0 0 0.5rem 0; color: white;'>{name}</h4>
                    <p style='margin: 0; color: white; font-size: 0.9rem;'>
                        🛠️ {len(skills)} skills | 📅 {exp} years
                    </p>
                </div>
                """, unsafe_allow_html=True)

    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff, #f8fafc); padding: 2rem; border-radius: 20px; border: 1px solid rgba(45, 27, 105, 0.1); box-shadow: 0 12px 35px rgba(0,0,0,0.1);'>
            <h3 style='margin-top: 0; color: #1e293b;'>💡 Premium Features</h3>
            <ul style='color: #64748b; line-height: 1.7;'>
                <li><strong>Multi-format Support:</strong> PDF, DOCX, TXT</li>
                <li><strong>AI-Powered Analysis:</strong> Skills, experience, education</li>
                <li><strong>Vector Database:</strong> ChromaDB for fast retrieval</li>
                <li><strong>Smart Matching:</strong> Semantic similarity scoring</li>
                <li><strong>Batch Processing:</strong> Up to 100 resumes at once</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Load Sample Resumes", use_container_width=True):
            sample1 = """
            Prasanna Mehata
            Senior Software Engineer
            Experience: 8 years at Tech Company (2015-2023)
            Skills: Python, Django, AWS, Docker, Kubernetes, SQL, React
            Education: Masters in Computer Science
            Contact: prasannamehata6@email.com | +91 5565433332
            LinkedIn: linkedin.com/in/PrasannaMehata
            """
            sample2 = """
            Nuthan kb
            Data Scientist
            Experience: 5 years at Data Corp (2018-2023)
            Skills: Python, pandas, scikit-learn, tensorflow, SQL, Tableau
            Education: PhD in Data Science
            Contact: nuthankb@email.com | +91 3454345568
            """
            st.session_state.resumes = [
                ('prasanna_mehata.txt', sample1),
                ('nuthan_kb.txt', sample2)
            ]
            st.success("Sample resumes loaded!")
            st.rerun()
    
    # Navigation
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.session_state.resumes:
            if st.button("Next: Job Description →", type="primary", use_container_width=True):
                st.session_state.current_step = 'jd'
                st.rerun()

def render_jd_step():
    st.markdown('<div class="section-header">📝 Job Description</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        jd_source = st.radio(
            "Job Description Source",
            ["Paste text", "Upload file", "Use template"],
            horizontal=True
        )
        
        if jd_source == "Paste text":
            st.session_state.jd_text = st.text_area(
                "Paste job description",
                height=200,
                placeholder="Paste the complete job description here...",
                value=st.session_state.jd_text
            )
        elif jd_source == "Upload file":
            jd_file = st.file_uploader("Upload job description", type=['pdf', 'docx', 'txt'])
            if jd_file:
                st.session_state.jd_text = extract_text_from_uploaded(jd_file)
                st.session_state.jd_text = clean_text(st.session_state.jd_text)
        else:  # Use template
            template = st.selectbox(
                "Choose template",
                ["Senior Python Developer", "Data Scientist", "Frontend Developer", "DevOps Engineer"]
            )
            if template == "\033[90mSenior Python Developer.\033[0m":
                st.session_state.jd_text = """
                Senior Python Developer
                We are looking for an experienced Python developer with 5+ years of experience building scalable applications.
                
                Requirements:
                - 5+ years professional Python experience
                - Strong knowledge of Django or Flask frameworks
                - Experience with AWS cloud services
                - Database design and optimization (SQL, PostgreSQL)
                - Docker and Kubernetes experience
                - REST API design and development
                - Testing and code quality best practices
                
                Preferred Skills:
                - React or Vue.js knowledge
                - CI/CD pipeline experience
                - Microservices architecture
                - Machine learning basics
                """
            elif template == "Data Scientist":
                st.session_state.jd_text = """
                Data Scientist
                Seeking a data scientist to analyze complex datasets and build predictive models.
                
                Requirements:
                - 3+ years data science experience
                - Strong Python programming skills
                - Expertise in pandas, numpy, scikit-learn
                - Machine learning model development
                - SQL and database querying
                - Data visualization (Tableau, matplotlib)
                - Statistical analysis and A/B testing
                
                Preferred Skills:
                - TensorFlow or PyTorch
                - Big data technologies (Spark, Hadoop)
                - Cloud platforms (AWS, GCP, Azure)
                """
        
        if st.session_state.jd_text:
            with st.expander("🔍 Preview Job Description", expanded=True):
                st.write(st.session_state.jd_text[:500] + "..." if len(st.session_state.jd_text) > 500 else st.session_state.jd_text)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff, #f8fafc); padding: 2rem; border-radius: 20px; border: 1px solid rgba(45, 27, 105, 0.1); box-shadow: 0 12px 35px rgba(0,0,0,0.1);'>
            <h3 style='margin-top: 0; color: #1e293b;'>🎯 Best Practices</h3>
            <ul style='color: #64748b; line-height: 1.7;'>
                <li>Include specific skills and technologies</li>
                <li>Mention required years of experience</li>
                <li>List both required and preferred qualifications</li>
                <li>Be clear about role responsibilities</li>
                <li>Include company culture and values</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats if JD is provided
        if st.session_state.jd_text:
            jd_skills = extract_skills(st.session_state.jd_text)
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{}</div>
                <div class="metric-label">Skills Detected</div>
            </div>
            """.format(len(jd_skills)), unsafe_allow_html=True)
            
            if jd_skills:
                skills_html = " ".join([f'<span class="skill-tag">{skill}</span>' for skill in jd_skills[:6]])
                st.markdown(f"<div style='margin-top: 1rem;'><strong>Key Skills:</strong><br>{skills_html}</div>", unsafe_allow_html=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_step = 'upload'
            st.rerun()
    with col3:
        if st.session_state.jd_text and st.session_state.resumes:
            if st.button("Next: Configure →", type="primary", use_container_width=True):
                st.session_state.current_step = 'config'
                st.rerun()

def render_config_step():
    st.markdown('<div class="section-header">⚙️ Screening Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Scoring Weights")
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            sim_w = st.slider("Similarity Weight", 0.0, 1.0, 0.60, 0.05, help="How much to weight text similarity")
        with col1b:
            skills_w = st.slider("Skills Weight", 0.0, 1.0, 0.25, 0.05, help="How much to weight skills matching")
        with col1c:
            exp_w = st.slider("Experience Weight", 0.0, 1.0, 0.15, 0.05, help="How much to weight years of experience")
        
        # Normalize weights
        ssum = sim_w + skills_w + exp_w
        if ssum == 0:
            weights = {'similarity': 0.6, 'skills': 0.25, 'experience': 0.15}
        else:
            weights = {'similarity': sim_w/ssum, 'skills': skills_w/ssum, 'experience': exp_w/ssum}
        
        st.subheader("Screening Options")
        col2a, col2b = st.columns(2)
        with col2a:
            min_skill_count = st.slider("Minimum skills required", 0, 10, 2)
            interview_threshold = st.slider("Interview threshold", 0.0, 1.0, 0.025, 0.05)
        with col2b:
            # Show ChromaDB option only if available
            ranking_options = ["Auto (AI if available)", "TF-IDF (Local)"]
            if st.session_state.chroma_collection:
                ranking_options.append("ChromaDB (Vector)")
                
            ranking_mode = st.radio(
                "Ranking Method",
                ranking_options,
                help="AI method uses embeddings for better accuracy but requires API key"
            )
            show_details = st.checkbox("Show detailed analysis", value=False)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffffff, #f8fafc); padding: 2rem; border-radius: 20px; border: 1px solid rgba(45, 27, 105, 0.1); box-shadow: 0 12px 35px rgba(0,0,0,0.1);'>
            <h3 style='margin-top: 0; color: #1e293b;'>📊 Configuration Summary</h3>
            <div style='color: #64748b; line-height: 1.8;'>
                <p><strong>Resumes loaded:</strong> {resume_count}</p>
                <p><strong>JD loaded:</strong> {jd_status}</p>
                <p><strong>Method:</strong> {method}</p>
                <p><strong>Scoring:</strong></p>
                <ul style='margin: 0.5rem 0; padding-left: 1.2rem;'>
                    <li>Similarity: {sim_pct}%</li>
                    <li>Skills: {skills_pct}%</li>
                    <li>Experience: {exp_pct}%</li>
                </ul>
            </div>
        </div>
        """.format(
            resume_count=len(st.session_state.resumes),
            jd_status="✅" if st.session_state.jd_text else "❌",
            method=ranking_mode,
            sim_pct=int(weights['similarity'] * 100),
            skills_pct=int(weights['skills'] * 100),
            exp_pct=int(weights['experience'] * 100)
        ), unsafe_allow_html=True)
    
    # Navigation and Run button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.current_step = 'jd'
            st.rerun()
    with col3:
        if st.button("🚀 Run Screening", type="primary", use_container_width=True):
            with st.spinner("Screening resumes with AI..."):
                try:
                    # Choose backend
                    if ranking_mode == "Auto (AI if available)" and has_gemini_key():
                        backend = 'embeddings'
                    elif ranking_mode == "ChromaDB (Vector)" and st.session_state.chroma_collection:
                        backend = 'chromadb'
                    else:
                        backend = 'tfidf'
                    
                    if backend == 'embeddings':
                        df = rank_resumes_embeddings(st.session_state.jd_text, st.session_state.resumes, weights)
                    elif backend == 'chromadb':
                        df = rank_resumes_chromadb(st.session_state.jd_text, st.session_state.resumes, weights, st.session_state.chroma_collection)
                    else:
                        df = rank_resumes_tfidf(st.session_state.jd_text, st.session_state.resumes, weights)
                    
                    # Apply additional filters and flags
                    df['meets_skill_threshold'] = df['skill_match_count'] >= min_skill_count
                    df['interview_recommended'] = df['score'] >= interview_threshold
                    
                    st.session_state.results = df
                    st.session_state.current_step = 'results'
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error during screening: {str(e)}")

def render_results_step():
    st.markdown('<div class="section-header">📊 Screening Results</div>', unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.error("No results available. Please run the screening first.")
        if st.button("Back to Configuration"):
            st.session_state.current_step = 'config'
            st.rerun()
        return
    
    df = st.session_state.results
    
    # Summary metrics with luxury styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        recommended = df['interview_recommended'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{recommended}</div>
            <div class="metric-label">Recommended ({recommended/len(df)*100:.0f}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        avg_score = df['score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_score:.2f}</div>
            <div class="metric-label">Average Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        avg_exp = df['years_experience'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_exp:.1f}</div>
            <div class="metric-label">Avg Experience (yrs)</div>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Top Performers Section
    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Top Performers</div>', unsafe_allow_html=True)
    
    # Top 3 candidates with luxury styling
    top_performers = df.head(3)
    
    if len(top_performers) > 0:
        cols = st.columns(3)
        for idx, (col, (_, row)) in enumerate(zip(cols, top_performers.iterrows())):
            with col:
                rank_emoji = ["🥇", "🥈", "🥉"][idx]
                score_pct = max(0, min(100, row['score'] * 100))
                
                # Extract filename without extension for display
                display_name = row['filename'].split('.')[0]
                display_name = ' '.join([word.capitalize() for word in display_name.split('_')])
                
                st.markdown(f"""
                <div class="top-performer-card">
                    <div class="top-performer-badge">{rank_emoji} Rank {idx+1}</div>
                    <h3 style="color: #1e293b; margin-top: 1rem; font-size: 1.4rem;">{display_name}</h3>
                    <p style="color: #64748b; font-size: 1rem; margin-bottom: 1.5rem;">
                        {row['seniority']} • {row['education']} • {row['years_experience']} yrs
                    </p>
                    
                    
                </div>
                """, unsafe_allow_html=True)
                
                # Show skills for top performers
                skills = row['skills_found'].split(',') if row['skills_found'] else []
                if skills:
                    skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills[:6]])
                    st.markdown(f"<div style='margin-top: 1rem; text-align: center;'>{skills_html}</div>", unsafe_allow_html=True)

    # Candidate Comparison Feature
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Compare Candidates</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multi-select for comparison
        candidate_options = [f"{row['filename']} (Score: {row['score']:.3f})" for _, row in df.iterrows()]
        selected_comparison = st.multiselect(
            "Select candidates to compare:",
            options=candidate_options,
            default=candidate_options[:2] if len(candidate_options) >= 2 else candidate_options,
            help="Select 2 or more candidates to compare their attributes"
        )
    
    with col2:
        # Toggle for detailed comparison view
        show_comparison = st.checkbox("Show detailed comparison", value=st.session_state.show_comparison)
        st.session_state.show_comparison = show_comparison
    
    # Display comparison if candidates are selected
    if len(selected_comparison) >= 2 and st.session_state.show_comparison:
        #st.markdown('<div class="comparison-view">', unsafe_allow_html=True)
        st.subheader("📊 Candidate Comparison")
        
        # Extract selected candidate data
        selected_indices = [candidate_options.index(name) for name in selected_comparison]
        comparison_data = df.iloc[selected_indices]
        
        # Create comparison table
        comp_cols = st.columns(len(selected_comparison))
        
        for idx, (col, (_, candidate)) in enumerate(zip(comp_cols, comparison_data.iterrows())):
            with col:
                display_name = candidate['filename'].split('.')[0]
                display_name = ' '.join([word.capitalize() for word in display_name.split('_')])
                
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #ffffff, #f8fafc); border-radius: 16px; margin-bottom: 1rem; box-shadow: 0 8px 25px rgba(0,0,0,0.08);">
                    <h4 style="color: #1e293b; margin-bottom: 1rem;">{display_name}</h4>
                    <div style="font-size: 1.8rem; font-weight: bold; color: #2d1b69; margin-bottom: 1rem;">{candidate['score']:.3f}</div>
                    <div style="color: #64748b; font-size: 0.9rem; line-height: 1.8;">
                        <div>📊 {candidate['similarity']:.3f} similarity</div>
                        <div>🛠️ {candidate['skill_match_count']} skills</div>
                        <div>📅 {candidate['years_experience']} years</div>
                        <div>🎓 {candidate['education']}</div>
                        <div>💼 {candidate['seniority']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Skills comparison
        st.subheader("🛠️ Skills Comparison")
        all_skills = set()
        for _, candidate in comparison_data.iterrows():
            skills = candidate['skills_found'].split(',') if candidate['skills_found'] else []
            all_skills.update(skills)
        
        if all_skills:
            # Create a skills matrix
            skills_matrix = {}
            for skill in sorted(all_skills):
                skills_matrix[skill] = []
                for _, candidate in comparison_data.iterrows():
                    candidate_skills = candidate['skills_found'].split(',') if candidate['skills_found'] else []
                    skills_matrix[skill].append('✅' if skill in candidate_skills else '❌')
            
            # Display skills matrix
            display_names = [candidate['filename'].split('.')[0] for _, candidate in comparison_data.iterrows()]
            display_names = [' '.join([word.capitalize() for word in name.split('_')]) for name in display_names]
            skills_df = pd.DataFrame(skills_matrix, index=display_names)
            st.dataframe(skills_df, use_container_width=True)
        
        #st.markdown('</div>', unsafe_allow_html=True)

    # All candidates list
    #st.markdown("---")
    st.markdown('<div class="section-header">📋 All Candidates</div>', unsafe_allow_html=True)
    
    for idx, row in df.iterrows():
        with st.container():
            #st.markdown(f"<div class='resume-card'>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                # Candidate header
                badge_class = "badge-recommended" if row['interview_recommended'] else "badge-hold"
                badge_text = "RECOMMENDED" if row['interview_recommended'] else "HOLD"
                
                display_name = row['filename'].split('.')[0]
                display_name = ' '.join([word.capitalize() for word in display_name.split('_')])
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                    <h3 style="margin: 0; flex-grow: 1; color: #FFFFFF;">#{idx+1} {display_name}</h3>
                    <span class="status-badge {badge_class}">{badge_text}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Metadata
                st.markdown(f"""
                <div style="color: #FFFFFF; font-size: 1rem; margin-bottom: 1rem;">
                    {row['seniority']} • {row['education']} • {row['years_experience']} years experience
                </div>
                """, unsafe_allow_html=True)
                
                # Score bar
                score_pct = max(0, min(100, row['score'] * 100))
                st.markdown(f"""
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                        <span style="color: #FFFFFF; font-weight: 600;">Overall Match</span>
                        <span style="color: #FFFFFF; font-weight: 700; font-size: 1.1rem;">{row['score']:.3f} ({score_pct:.0f}%)</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" style="width: {score_pct}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Skills
                skills = row['skills_found'].split(',') if row['skills_found'] else []
                if skills:
                    skills_html = "".join([f'<span class="skill-tag">{skill}</span>' for skill in skills[:10]])
                    st.markdown(f"<div style='margin-bottom: 1.5rem;'><strong style='color: #FFFFFF;'>Skills:</strong><br>{skills_html}</div>", unsafe_allow_html=True)
                
                # Summary
                if row['summary']:
                    with st.expander("📄 View Summary"):
                        #st.write(row['summary'])
                        st.markdown(f"<span style='color:white;'>{row['summary']}</span>", unsafe_allow_html=True)

            
            with col2:
                # Detailed scores
                st.metric("Similarity", f"{row['similarity']:.3f}")
                st.metric("Skills Match", row['skill_match_count'])
                st.metric("Experience", f"{row['years_experience']} yrs")
                
                # Contact info
                if row['contact_email'] or row['contact_phone']:
                    with st.expander("📞 Contact"):
                        if row['contact_email']:
                            st.write(f"📧 {row['contact_email']}")
                        if row['contact_phone']:
                            st.write(f"📞 {row['contact_phone']}")
                        if row['linkedin']:
                            st.write(f"🔗 [LinkedIn]({row['linkedin']})")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Download options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="all_candidates.csv" style="display: inline-block; padding: 1rem 2rem; background: linear-gradient(135deg, #2d1b69, #4a1e7a); color: white; text-decoration: none; border-radius: 16px; font-weight: 700; transition: all 0.4s ease; box-shadow: 0 8px 25px rgba(45, 27, 105, 0.3);">📥 Download All Results (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if df['interview_recommended'].any():
            rec_df = df[df['interview_recommended']]
            csv_rec = rec_df.to_csv(index=False)
            b64_rec = base64.b64encode(csv_rec.encode()).decode()
            href_rec = f'<a href="data:file/csv;base64,{b64_rec}" download="recommended_candidates.csv" style="display: inline-block; padding: 1rem 2rem; background: linear-gradient(135deg, #059669, #10b981); color: white; text-decoration: none; border-radius: 16px; font-weight: 700; transition: all 0.4s ease; box-shadow: 0 8px 25px rgba(5, 150, 105, 0.3);">📥 Download Recommended Only (CSV)</a>'
            st.markdown(href_rec, unsafe_allow_html=True)
    
    with col3:
        if len(top_performers) > 0:
            top_csv = top_performers.to_csv(index=False)
            b64_top = base64.b64encode(top_csv.encode()).decode()
            href_top = f'<a href="data:file/csv;base64,{b64_top}" download="top_performers.csv" style="display: inline-block; padding: 1rem 2rem; background: linear-gradient(135deg, #f59e0b, #d97706); color: white; text-decoration: none; border-radius: 16px; font-weight: 700; transition: all 0.4s ease; box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);">📥 Download Top Performers (CSV)</a>'
            st.markdown(href_top, unsafe_allow_html=True)
    
    # New screening
    st.markdown("---")
    if st.button("🔄 Start New Screening", type="primary", use_container_width=True):
        st.session_state.current_step = 'upload'
        st.session_state.resumes = []
        st.session_state.jd_text = ''
        st.session_state.results = None
        st.session_state.show_comparison = False
        st.session_state.selected_candidates = []
        st.rerun()

if __name__ == '__main__':
    app()
