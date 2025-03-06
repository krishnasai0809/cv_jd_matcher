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

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# List of common degree names for qualification extraction
DEGREES = ["bachelor", "master", "phd", "mba", "bsc", "msc", "btech", "mtech", "diploma"]

# List of key skills to look for
KEY_SKILLS = [
    "machine learning", "deep learning", "nlp", "python", "tensorflow",
    "cloud", "data analysis", "artificial intelligence", "sql", "statistics"
]

# Function to extract text from PDF with OCR fallback
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

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters & numbers
    tokens = word_tokenize(text)  # Tokenization
    tokens = [t for t in tokens if t not in stopwords.words("english")]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatization
    return " ".join(tokens)

# Function to extract experience (in years)
def extract_experience(text):
    match = re.search(r"(\d+)\s*(?:years?|yrs?)\s*(?:of experience|experience)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))  # Extract numeric years
    return 0  # Default to 0 if no match found

# Function to extract key skills
def extract_skills(text):
    matched_skills = [skill for skill in KEY_SKILLS if skill in text.lower()]
    return len(matched_skills), matched_skills  # Return count of matched skills and list

# Function to extract qualifications
def extract_qualification(text):
    found_degrees = [degree for degree in DEGREES if degree in text.lower()]
    return len(found_degrees), found_degrees  # Return count and list

# Function to rank CVs based on Job Description
def rank_cvs(jd_text, cv_texts):
    jd_clean = preprocess_text(jd_text)
    cv_cleaned = [preprocess_text(cv) for cv in cv_texts]

    # Convert to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_clean] + cv_cleaned)
    similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])[0]

    ranked_results = []

    for idx, cv_text in enumerate(cv_texts):
        experience = extract_experience(cv_text)
        skills_count, matched_skills = extract_skills(cv_text)
        qualification_count, matched_degrees = extract_qualification(cv_text)

        # Hybrid score calculation
        hybrid_score = (similarity_scores[idx] * 0.5) + (experience * 0.2) + (skills_count * 0.2) + (qualification_count * 0.1)

        ranked_results.append((cv_text, hybrid_score, experience, matched_skills, matched_degrees))

    # Sort by hybrid score (higher is better)
    ranked_results.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_results

# Main Execution
if __name__ == "__main__":
    job_description = """We are looking for a Data Scientist with expertise in Machine Learning, Deep Learning, and NLP. 
                         Experience with Python, TensorFlow, and cloud platforms is a plus."""

    cv_folder = "cvs/"  # Update this to your CV folder path
    cv_files = [f for f in os.listdir(cv_folder) if f.endswith(".pdf")]

    cv_texts = []
    for cv_file in cv_files:
        cv_path = os.path.join(cv_folder, cv_file)
        print(f"Processing: {cv_file} ...")
        extracted_text = extract_text_from_pdf(cv_path)
        cv_texts.append(extracted_text)

    ranked_cvs = rank_cvs(job_description, cv_texts)

    print("\n📌 Ranked CVs based on Job Description relevance:")
    for idx, (cv_text, score, exp, skills, degrees) in enumerate(ranked_cvs, 1):
        print(f"\n🔹 Rank {idx} | Score: {score:.4f}")
        print(f"   🏆 Experience: {exp} years")
        print(f"   🔑 Key Skills Matched: {', '.join(skills) if skills else 'None'}")
        print(f"   🎓 Qualifications: {', '.join(degrees) if degrees else 'None'}")
        print(f"   📝 Extracted CV Preview: {cv_text[:300]}...")  # Show first 300 chars
