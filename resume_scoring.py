import fitz  # PyMuPDF for extracting text from PDFs
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Mistral API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


def extract_skills(text):
    doc = nlp(text)
    # Named Entity Recognition (NER)
    # for ent in doc.ents:
    #  print(f"{ent.text} - {ent.label_}")

    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return skills

def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_score[0][0] * 100  # Convert to percentage



def weighted_score(skill_score, exp_score, edu_score, proj_score):
    return (skill_score * 0.4) + (exp_score * 0.3) + (edu_score * 0.2) + (proj_score * 0.1)


def generate_feedback(resume_text, job_description, prompt):
    if not resume_text or not job_description:
        return "Please provide both resume text and job description."
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    
    # Define prompt for resume and job description comparison
    if not prompt:
        prompt = f"""
        You are an expert career coach and hiring manager. Your task is to analyze the candidate's resume against the given job description and provide **constructive** and **actionable** feedback.
        Please provide feedback in the following format:
        1. Missing skills
        2. Areas for improvement
        3. Additional suggestions

        Resume: {resume_text}
        Job Description: {job_description}
        """
    
    # Choose a Mistral AI Model (Options: "mistral-small", "mistral-medium", "mixtral")
    data = {
        "model": "mistral-medium",  # Best free-tier model
        # "model": "mixtral" # Best for complex tasks,  More powerful (higher accuracy)
        # "model": "mistral-small" # Best for simple tasks
        "messages": [{"role": "system", "content": prompt}]
    }
    
    # Make API request
    response = requests.post(url, headers=headers, json=data)

    # Handle Response
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json()}"
    

#####




def main():
    # parser = argparse.ArgumentParser(description="Resume Scoring and Feedback Generator")
    # parser.add_argument("--resume", required=True, help="Path to the resume PDF")
    # parser.add_argument("--job_description", required=True, help="Path to the job description text file")
    # parser.add_argument("--prompt", help="Custom prompt for the AI model")
    # args = parser.parse_args()


    resume_text = extract_text_from_pdf("./data/neeraj_resume.pdf")
    resume_skills = extract_skills(resume_text)

    # Read job description from a text file
    with open("./data/amazon_SDE1.txt", "r", encoding="utf-8") as file:
        job_description = file.read().strip()  


    resume_score = compute_similarity(resume_text, job_description)
    # print(f"Resume Match Score: {resume_score:.2f}%")

    final_score = weighted_score(80, 75, 90, 60)  # Example scores
    print(f"Final Resume Score: {final_score:.2f}%")

    print("Please wait while the generating feedback...")

    # Generate feedback
    prompt = None
    feedback = generate_feedback(resume_text, job_description, prompt)
    print("\nResume Feedback:\n", feedback)

if __name__ == "__main__":
    main()
import fitz  # PyMuPDF for extracting text from PDFs
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import argparse
import os
from dotenv import load_dotenv
load_dotenv()


# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Mistral API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        text += page.get_text()
    return text


def extract_skills(text):
    doc = nlp(text)
    # Named Entity Recognition (NER)
    # for ent in doc.ents:
    #  print(f"{ent.text} - {ent.label_}")

    skills = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART"]]
    return skills

def compute_similarity(resume_text, job_desc):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_score[0][0] * 100  # Convert to percentage



def weighted_score(skill_score, exp_score, edu_score, proj_score):
    return (skill_score * 0.4) + (exp_score * 0.3) + (edu_score * 0.2) + (proj_score * 0.1)


def generate_feedback(resume_text, job_description, prompt):
    if not resume_text or not job_description:
        return "Please provide both resume text and job description."
    
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    
    # Define prompt for resume and job description comparison
    if not prompt:
        prompt = f"""
        You are an expert career coach and hiring manager. Your task is to analyze the candidate's resume against the given job description and provide **constructive** and **actionable** feedback.
        Please provide feedback in the following format:
        1. Missing skills
        2. Areas for improvement
        3. Additional suggestions

        Resume: {resume_text}
        Job Description: {job_description}
        """
    
    # Choose a Mistral AI Model (Options: "mistral-small", "mistral-medium", "mixtral")
    data = {
        "model": "mistral-medium",  # Best free-tier model
        # "model": "mixtral" # Best for complex tasks,  More powerful (higher accuracy)
        # "model": "mistral-small" # Best for simple tasks
        "messages": [{"role": "system", "content": prompt}]
    }
    
    # Make API request
    response = requests.post(url, headers=headers, json=data)

    # Handle Response
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.json()}"
    

#####




def main():
    # parser = argparse.ArgumentParser(description="Resume Scoring and Feedback Generator")
    # parser.add_argument("--resume", required=True, help="Path to the resume PDF")
    # parser.add_argument("--job_description", required=True, help="Path to the job description text file")
    # parser.add_argument("--prompt", help="Custom prompt for the AI model")
    # args = parser.parse_args()


    resume_text = extract_text_from_pdf("./data/neeraj_resume.pdf")
    resume_skills = extract_skills(resume_text)

    # Read job description from a text file
    with open("./data/amazon_SDE1.txt", "r", encoding="utf-8") as file:
        job_description = file.read().strip()  


    resume_score = compute_similarity(resume_text, job_description)
    # print(f"Resume Match Score: {resume_score:.2f}%")

    final_score = weighted_score(80, 75, 90, 60)  # Example scores
    print(f"Final Resume Score: {final_score:.2f}%")


    prompt = None
    feedback = generate_feedback(resume_text, job_description, prompt)
    print("Resume Feedback:\n", feedback)

if __name__ == "__main__":
    main()