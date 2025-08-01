from flask import Flask, request, render_template, redirect
import pickle
import os
import re
import docx
import PyPDF2
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(__file__), "clf.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "tfidf.pkl")

clf = pickle.load(open(MODEL_PATH, 'rb'))
tfidfd = pickle.load(open(VECTORIZER_PATH, 'rb'))

# Resume cleaning function
def clean_resume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

category_mapping = {
    0: 'Advocate',
    1: 'Arts',
    2: 'Automation Testing',
    3: 'Testing',
    4: 'Business Analyst',
    5: 'Civil Engineer',
    6: 'Data Science',
    7: 'Database',
    8: 'DevOps Engineer',
    9: 'DotNet Developer',
    10: 'ETL Developer',
    11: 'Electrical Engineering',
    12: 'HR',
    13: 'Hadoop',
    14: 'Health and fitness',
    15: 'Java Developer',
    16: 'Mechanical Engineer',
    17: 'Network Security Engineer',
    18: 'Operations Manager',
    19: 'PMO',
    20: 'Python Developer',
    21: 'SAP Developer',
    22: 'Sales',
    23: 'Blockchain',
    24: 'Web Designing'
}

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'resume' not in request.files:
        return "No file part"

    file = request.files['resume']

    if file.filename == '':
        return "No selected file"

    resume_text = ""
    if file.filename.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            resume_text += page.extract_text()

    elif file.filename.endswith('.docx'):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            resume_text += para.text + " "

    elif file.filename.endswith('.txt'):
        resume_bytes = file.read()
        try:
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

    if resume_text.strip() == "":
        return "Could not extract any text from the uploaded file."

    cleaned_resume = clean_resume(resume_text)
    input_features = tfidfd.transform([cleaned_resume])
    prediction_id = clf.predict(input_features)[0]
    category = category_mapping.get(prediction_id, "Unknown")

    return render_template('result.html', category=category, prediction_id=prediction_id)

if __name__ == '__main__':
    app.run(debug=True)
