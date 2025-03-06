import os
import re
import nltk
import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Dictionary of Job Roles with Required Degrees and Skills
JOB_ROLES = {
    "Graphic Designer": {"degrees": ["bachelor", "diploma"], "skills": ["photoshop", "illustrator", "typography"]},
    "UI/UX Designer": {"degrees": ["bachelor", "diploma"], "skills": ["figma", "adobe xd", "wireframing"]},
    "Interior Designer": {"degrees": ["bachelor", "diploma"], "skills": ["autocad", "3ds max", "rendering"]},
    "Fashion Designer": {"degrees": ["bachelor", "diploma"], "skills": ["textile", "pattern making", "fashion illustration"]},
    "Business Analyst": {"degrees": ["bachelor", "mba"], "skills": ["sql", "excel", "data visualization"]},
    "Marketing Manager": {"degrees": ["bachelor", "mba"], "skills": ["seo", "social media", "branding"]},
    "Software Engineer": {"degrees": ["bachelor", "master"], "skills": ["python", "java", "c++"]},
    "Data Scientist": {"degrees": ["bachelor", "master", "phd"], "skills": ["machine learning", "deep learning", "nlp"]},
    "AI Engineer": {"degrees": ["bachelor", "master"], "skills": ["tensorflow", "pytorch", "computer vision"]},
    "Cloud Engineer": {"degrees": ["bachelor", "master"], "skills": ["aws", "azure", "gcp"]},
    "Musician": {"degrees": ["diploma", "bachelor"], "skills": ["composition", "instrument", "music production"]},
    "Film Director": {"degrees": ["bachelor", "master"], "skills": ["screenwriting", "cinematography", "storytelling"]},
    "Football Coach": {"degrees": ["diploma", "bachelor"], "skills": ["strategy", "fitness training", "team management"]},
    "Mathematics Teacher": {"degrees": ["bachelor", "master"], "skills": ["calculus", "algebra", "geometry"]},
    # Add more job roles as needed (total 100)...
}

# Extract text from PDFs

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        if not text.strip():
            print("No text found in PDF. Running OCR...")
            images = convert_from_path(pdf_path)
            for img in images:
                text += pytesseract.image_to_string(img) + "\n"
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text.strip()

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Extract experience
def extract_experience(text):
    match = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:of experience|experience)", text, re.IGNORECASE)
    return int(match.group(1)) if match else 0

# Extract qualifications
def extract_qualification(text, job_role):
    required_degrees = JOB_ROLES.get(job_role, {}).get("degrees", [])
    found_degrees = [degree for degree in required_degrees if degree in text.lower()]
    return len(found_degrees), found_degrees

# Extract key skills
def extract_skills(text, job_role):
    required_skills = JOB_ROLES.get(job_role, {}).get("skills", [])
    matched_skills = [skill for skill in required_skills if skill in text.lower()]
    return len(matched_skills), matched_skills

# Rank CVs
def rank_cvs(jd_text, cv_texts, job_role):
    jd_clean = preprocess_text(jd_text)
    cv_cleaned = [preprocess_text(cv) for cv in cv_texts]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_clean] + cv_cleaned)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]
    
    ranked_results = []
    for idx, cv_text in enumerate(cv_texts):
        experience = extract_experience(cv_text)
        skills_count, matched_skills = extract_skills(cv_text, job_role)
        qualification_count, matched_degrees = extract_qualification(cv_text, job_role)
        hybrid_score = (similarity_scores[idx] * 0.5) + (experience * 0.2) + (skills_count * 0.2) + (qualification_count * 0.1)
        ranked_results.append((cv_text, hybrid_score, experience, matched_skills, matched_degrees))
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    return ranked_results

# Main Execution
if __name__ == "__main__":
    print("Available Job Roles:")
    for role in JOB_ROLES.keys():
        print(f"- {role}")
    
    job_role = input("Enter the job role: ")
    if job_role not in JOB_ROLES:
        print("Invalid job role. Please enter a valid one from the list.")
        exit()
    
    job_description = input("Enter job description: ")
    cv_folder = "cvs/"
    cv_files = [f for f in os.listdir(cv_folder) if f.endswith(".pdf")]
    cv_texts = [extract_text_from_pdf(os.path.join(cv_folder, cv_file)) for cv_file in cv_files]
    ranked_cvs = rank_cvs(job_description, cv_texts, job_role)
    
    print("\n📌 Ranked CVs based on Job Description relevance:")
    for idx, (cv_text, score, exp, skills, degrees) in enumerate(ranked_cvs, 1):
        print(f"\n🔹 Rank {idx} | Score: {score:.4f}")
        print(f"   🏆 Experience: {exp} years")
        print(f"   🔑 Key Skills Matched: {', '.join(skills) if skills else 'None'}")
        print(f"   🎓 Qualifications: {', '.join(degrees) if degrees else 'None'}")
        print(f"   📝 Extracted CV Preview: {cv_text[:300]}...")
