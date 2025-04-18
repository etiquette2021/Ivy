import os
import json
import logging
from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import fitz  # PyMuPDF for PDF text extraction
import chardet  # Detect text encoding

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the folder exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
def query_chatgpt(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant. **Always respond with valid JSON format only.** "
                               "Do not include explanations, markdown, or any text outside the JSON object."
                },
                {"role": "user", "content": prompt}
            ]
        )

        logging.info("Received OpenAI API response.")

        # Extract response text
        response_text = response.choices[0].message.content.strip()

        # Ensure response is valid JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON received: {response_text}")
            return {"error": "OpenAI returned invalid JSON format."}

    except Exception as e:
        logging.error(f"Error querying OpenAI API: {str(e)}")
        return {"error": f"Error querying OpenAI API: {str(e)}"}

# Disable caching
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/')
def home():
    return render_template('index.html')

# Function to extract text from PDFs using PyMuPDF
def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text("text") for page in doc])

        # If text extraction failed, return a fallback message
        if not text.strip():
            return "No readable text found in PDF."

        # Detect encoding
        detected_encoding = chardet.detect(text.encode())["encoding"]

        # Convert text to UTF-8 safely
        text = text.encode(detected_encoding or "utf-8", errors="replace").decode("utf-8", errors="replace")

        # Strip control characters
        text = "".join(char if char.isprintable() else " " for char in text)

        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return f"Unable to extract text from PDF: {str(e)}"

# Function to safely save uploaded files before processing
def save_uploaded_file(file_obj, save_name):
    """Saves the uploaded file and returns the file path."""
    if file_obj and file_obj.filename:
        file_path = os.path.join(UPLOAD_FOLDER, save_name)
        file_obj.save(file_path)
        return file_path
    return None

# Function to delete files
@app.route('/delete-file', methods=['POST'])
def delete_file():
    try:
        file_type = request.form.get("file_type")
        if file_type == "teacher_feedback":
            file_path = os.path.join(UPLOAD_FOLDER, "teacher_feedback.pdf")
        elif file_type == "report_cards":
            file_path = os.path.join(UPLOAD_FOLDER, "report_cards.pdf")
        else:
            return jsonify({"error": "Invalid file type"}), 400

        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({"status": "success", "message": f"Deleted {file_type}"}), 200
        else:
            return jsonify({"error": "File does not exist"}), 404
    except Exception as e:
        logging.error(f"Error deleting file: {str(e)}")
        return jsonify({"error": f"Error deleting file: {str(e)}"}), 500

# Function to process file uploads and GPT analysis
@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Gather form data
        user_data = {
            "school_type": request.form.get("school_type"),
            "year": request.form.get("year"),
            "gpa_ib": request.form.get("gpa_ib"),
            "predicted_sat": request.form.get("predicted_sat"),
            "school_name": request.form.get("school_name"),
            "extracurriculars": request.form.get("extracurriculars"),
            "jobs_volunteer": request.form.get("jobs_volunteer"),
            "unique_aspects": request.form.get("unique_aspects"),
            "difficulties": request.form.get("difficulties"),
        }

        # **Save and Process Teacher Recommendation File**
        teacher_feedback_text = ""
        teacher_feedback_file = request.files.get("teacher_feedback")
        teacher_feedback_path = save_uploaded_file(teacher_feedback_file, "teacher_feedback.pdf")

        if teacher_feedback_path:
            teacher_feedback_text = extract_text_from_pdf(teacher_feedback_path)

        # **Save and Process Report Card File**
        report_card_text = ""
        report_card_file = request.files.get("report_cards")
        report_card_path = save_uploaded_file(report_card_file, "report_cards.pdf")

        if report_card_path:
            report_card_text = extract_text_from_pdf(report_card_path)

        # Ensure Text is UTF-8
        teacher_feedback_text = teacher_feedback_text.encode("utf-8", errors="replace").decode("utf-8")
        report_card_text = report_card_text.encode("utf-8", errors="replace").decode("utf-8")

        # Build Prompt for LLM
        prompt = f"""
        Profile Information:
        - School Type: {user_data['school_type']}
        - Current Grade/Year: {user_data['year']}
        - GPA or IB Score: {user_data['gpa_ib']}
        - Predicted SAT Score: {user_data['predicted_sat']}
        - School Name: {user_data['school_name']}
        - Extracurricular Activities: {user_data['extracurriculars']}
        - Jobs or Volunteer Activities: {user_data['jobs_volunteer']}
        - Unique Aspects: {user_data['unique_aspects']}
        - Difficulties: {user_data['difficulties']}

        Additional Materials:
        - Teacher Comments: {teacher_feedback_text if teacher_feedback_text else "No teacher feedback provided"}
        - Report Card Summary: {report_card_text if report_card_text else "No report card uploaded"}

        ### **Return JSON format ONLY.**
        Ensure the following structure:
        {{
            "ivy": [
                {{"school": "Harvard University", "probability": "XX%"}},
                {{"school": "Princeton University", "probability": "XX%"}},
                {{"school": "Yale University", "probability": "XX%"}},
                {{"school": "Columbia University", "probability": "XX%"}},
                {{"school": "University of Pennsylvania", "probability": "XX%"}},
                {{"school": "Dartmouth College", "probability": "XX%"}},
                {{"school": "Brown University", "probability": "XX%"}},
                {{"school": "Cornell University", "probability": "XX%"}}
            ],
            "top_10": [
                {{"university": "Stanford University", "probability": "XX%"}},
                {{"university": "Massachusetts Institute of Technology", "probability": "XX%"}},
                {{"university": "California Institute of Technology", "probability": "XX%"}},
                {{"university": "University of Chicago", "probability": "XX%"}},
                {{"university": "Duke University", "probability": "XX%"}},
                {{"university": "Northwestern University", "probability": "XX%"}},
                {{"university": "Johns Hopkins University", "probability": "XX%"}},
                {{"university": "University of California, Berkeley", "probability": "XX%"}},
                {{"university": "University of California, Los Angeles", "probability": "XX%"}},
                {{"university": "University of Michigan", "probability": "XX%"}}
            ],
            "cumulative_probabilities": [
                "Summarize the likelihood of admission to at least one Ivy League or top university."
            ],
            "strengths": [
                "List key strengths relevant to admissions."
            ],
            "weaknesses": [
                "List weaknesses that might impact admissions."
            ],
            "strategies": [
                "Provide 3-5 specific strategies to improve chances."
            ],
            "report_card_analysis": [
                "Summarize insights from teacher comments and report card."
            ]
        }}
        Ensure each Ivy League school and each top university is **always included** in the JSON response with valid probability estimates.
        """

        logging.info("Calling OpenAI API...")
        api_response = query_chatgpt(prompt)
        return render_template('results.html', sections=api_response)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
