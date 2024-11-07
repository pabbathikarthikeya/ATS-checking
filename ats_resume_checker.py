import streamlit as st
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from docx import Document
import pdfplumber
from spacy.matcher import Matcher

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Define a list of common skills
common_skills = [
    "python", "java", "aws", "docker", "machine learning", "data science",
    "project management", "cloud computing", "databases", "tensorflow",
    "deep learning", "flask", "django", "git", "linux", "devops", "vscode",
    "sql", "etl", "statistics", "mongodb", "power bi", "r", "excel", "hadoop"
]

# Function to parse resume text (with improved email and phone parsing)
def parse_resume(text):
    doc = nlp(text)
    name, email, phone = None, None, None
    skills = []

    # Initialize spaCy matcher for skills extraction
    matcher = Matcher(nlp.vocab)
    for skill in common_skills:
        pattern = [{"LOWER": skill.lower()}]
        matcher.add("SKILL", [pattern])

    matches = matcher(doc)

    # Extract named entities (e.g., person name)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and not name:
            name = ent.text
    
    # Improved email extraction using regular expression
    email_matches = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if email_matches:
        email = email_matches[0]  # Take the first email match
    
    # Improved phone number extraction using regular expression
    phone_matches = re.findall(r'(\+?\d{1,4}[\s-]?)?(?:\(?\d{2,3}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}', text)
    if phone_matches:
        phone = phone_matches[0]  # Take the first phone match
    
    # Skill extraction using Matcher
    for match_id, start, end in matches:
        span = doc[start:end]
        skills.append(span.text)

    # Deduplicate skills and clean the list
    skills = list(set([skill.lower().strip() for skill in skills]))

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": skills
    }

# Function to extract text from uploaded file
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        text = clean_extracted_text(text)
        return text.strip()
    elif file.type == "text/plain":
        return str(file.read(), "utf-8").strip()
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs]).strip()
    else:
        return None

# Function to clean extracted text
def clean_extracted_text(text):
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with a single newline
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text

# TF-IDF Vectorizer for resume scoring
def train_tfidf_model(job_description):
    vectorizer = TfidfVectorizer(stop_words='english')
    job_corpus = [job_description]
    vectorizer.fit(job_corpus)
    return vectorizer

# Calculate similarity score using cosine similarity
def get_similarity_score(resume_text, vectorizer, job_description):
    resume_vec = vectorizer.transform([resume_text])
    job_vec = vectorizer.transform([job_description])
    score = cosine_similarity(resume_vec, job_vec)[0][0]
    return score

# Email sending function
def send_email(email, subject, body):
    sender_email = "pabbathikarthikeya13@gmail.com"
    sender_password = "xjgh ewxb fpmd aquf"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Set up the server connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection
        server.login(sender_email, sender_password)

        # Send the email
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success(f"Email sent to {email}")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Load job description
with open("models/job_description.txt", "r") as file:
    job_description = file.read()

# Load TF-IDF model
vectorizer = train_tfidf_model(job_description)

# Streamlit UI
st.title("Company ATS Resume Checker")

uploaded_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_uploader")

if uploaded_files:
    st.write("### Results")
    
    resumes_above_50 = []
    resumes_below_50 = []
    
    for file in uploaded_files:
        resume_text = extract_text(file)
        if resume_text:
            st.write(f"**Processing resume:** {file.name}")
            parsed_resume = parse_resume(resume_text)
            st.write("Name:", parsed_resume['name'])
            st.write("Email:", parsed_resume['email'])
            st.write("Phone:", parsed_resume['phone'])
            st.write("Skills Extracted:", ", ".join(parsed_resume['skills']))

            # Score resume using similarity
            similarity_score = get_similarity_score(resume_text, vectorizer, job_description)
            st.write(f"**Resume Score:** {similarity_score * 100:.2f}%")

            # Store resumes with valid emails and score >= 50%
            if parsed_resume['email']:
                if similarity_score >= 0.5:
                    resumes_above_50.append(parsed_resume['email'])
                else:
                    resumes_below_50.append(parsed_resume['email'])

    # Display the button for sending emails to resumes with score >= 50%
    if st.button("Send Emails to Resumes with >= 50%"):
        if resumes_above_50:
            for email in resumes_above_50:
                for file in uploaded_files:
                    resume_text = extract_text(file)
                    parsed_resume = parse_resume(resume_text)
                    if parsed_resume['email'] == email:
                        name = parsed_resume['name'] if parsed_resume['name'] else "Candidate"
                        
                        # Send email with the personalized message
                        send_email(
                            email=email,
                            subject="Job Application - Next Steps",
                            body=f'{name}, we are moving forward with your application. We will contact you soon regarding the technical interview.'
                        )
                        break  # Stop searching after finding the match
        else:
            st.warning("No resumes met the 50% threshold.")

    # Optionally, show resumes below 50% if needed
    if resumes_below_50:
        st.write("**Resumes Below 50% Score**")
        st.write(", ".join(resumes_below_50))
