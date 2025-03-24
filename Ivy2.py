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
            "early_decision": request.form.get("early_decision") or "early_decision_princeton",
            "schools_visited": request.form.getlist("schools_visited"),
            "schools_contacted": request.form.getlist("schools_contacted"),
            "schools_followed": request.form.getlist("schools_followed"),
            "academic_focus": request.form.getlist("academic_focus")
        }

        # Log the data to verify
        logging.info(f"Schools Visited: {user_data['schools_visited']}")
        logging.info(f"Schools Contacted: {user_data['schools_contacted']}")
        logging.info(f"Schools Followed: {user_data['schools_followed']}")
        logging.info(f"Jobs or Volunteer Activities: {user_data['jobs_volunteer']}")
        logging.info(f"Academic Focus Areas: {user_data['academic_focus']}")
        logging.info(f"Early Decision School: {user_data['early_decision']}")

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
        - Early Decision/Action: {user_data['early_decision']}
        - Schools Visited: {', '.join(user_data['schools_visited'])}
        - Schools Contacted: {', '.join(user_data['schools_contacted'])}
        - Schools Followed on Social Media: {', '.join(user_data['schools_followed'])}
        - Academic Focus Areas: {', '.join(user_data['academic_focus'])}
        - International Background: The student is from Norway, a country known for its strong social responsibility and unique educational system.

        Additional Materials:
        - Teacher Comments: {teacher_feedback_text if teacher_feedback_text else "No teacher feedback provided"}
        - Report Card Summary: {report_card_text if report_card_text else "No report card uploaded"}

        ### **Return JSON format ONLY.**
        Please analyze the provided profile and estimate the probability of admission to each school listed below. Consider all aspects of the profile, including academic performance, extracurricular involvement, personal interactions with schools, and the impact of early decision applications. Provide insights on how each factor influences the probability estimates.

        Additionally, analyze the impact of:
        - Following schools on social media
        - Campus visits and interactions with admissions officers

        Provide a personality profile ranking for the student per school, considering their unique aspects and experiences.

        Rank schools based on alignment with the student's academic focus areas: {', '.join(user_data['academic_focus'])}.

        Provide a detailed analysis of the jobs and volunteer activities, highlighting how these experiences contribute to the student's profile and potential fit with each school.

        **Find and list admissions regional directors covering Northern Europe, including their emails and phone numbers.**

        Ensure the following structure:
        {{
            "ivy": [
                {{"school": "Harvard University", "probability": "XX%", "address": "Cambridge, MA", "student_body": "6700", "freshmen_admitted": "2000"}},
                {{"school": "Princeton University", "probability": "XX%", "address": "Princeton, NJ", "student_body": "5400", "freshmen_admitted": "1300"}},
                {{"school": "Yale University", "probability": "XX%", "address": "New Haven, CT", "student_body": "6000", "freshmen_admitted": "1700"}},
                {{"school": "Columbia University", "probability": "XX%", "address": "New York, NY", "student_body": "6200", "freshmen_admitted": "1400"}},
                {{"school": "University of Pennsylvania", "probability": "XX%", "address": "Philadelphia, PA", "student_body": "10000", "freshmen_admitted": "2500"}},
                {{"school": "Dartmouth College", "probability": "XX%", "address": "Hanover, NH", "student_body": "4400", "freshmen_admitted": "1100"}},
                {{"school": "Brown University", "probability": "XX%", "address": "Providence, RI", "student_body": "7000", "freshmen_admitted": "1700"}},
                {{"school": "Cornell University", "probability": "XX%", "address": "Ithaca, NY", "student_body": "15000", "freshmen_admitted": "3600"}}
            ],
            "top_10": [
                {{"university": "Stanford University", "probability": "XX%", "address": "Stanford, CA", "student_body": "7000", "freshmen_admitted": "1700"}},
                {{"university": "Massachusetts Institute of Technology", "probability": "XX%", "address": "Cambridge, MA", "student_body": "4500", "freshmen_admitted": "1100"}},
                {{"university": "California Institute of Technology", "probability": "XX%", "address": "Pasadena, CA", "student_body": "900", "freshmen_admitted": "235"}},
                {{"university": "University of Chicago", "probability": "XX%", "address": "Chicago, IL", "student_body": "6200", "freshmen_admitted": "1600"}},
                {{"university": "Duke University", "probability": "XX%", "address": "Durham, NC", "student_body": "6600", "freshmen_admitted": "1750"}},
                {{"university": "Northwestern University", "probability": "XX%", "address": "Evanston, IL", "student_body": "8000", "freshmen_admitted": "2000"}},
                {{"university": "Johns Hopkins University", "probability": "XX%", "address": "Baltimore, MD", "student_body": "6000", "freshmen_admitted": "1300"}},
                {{"university": "University of California, Berkeley", "probability": "XX%", "address": "Berkeley, CA", "student_body": "31000", "freshmen_admitted": "9300"}},
                {{"university": "University of California, Los Angeles", "probability": "XX%", "address": "Los Angeles, CA", "student_body": "31000", "freshmen_admitted": "5900"}},
                {{"university": "University of Michigan", "probability": "XX%", "address": "Ann Arbor, MI", "student_body": "29000", "freshmen_admitted": "6900"}},
                {{"university": "Vanderbilt University", "probability": "XX%", "address": "Nashville, TN", "student_body": "7000", "freshmen_admitted": "1600"}}
            ],
            "cumulative_probabilities": [
                "Provide the overall probability of admission to at least one Ivy League school.",
                "Provide the overall probability of admission to at least one top university.",
                "Discuss how the early decision application might impact these probabilities."
            ],
            "top_3_schools": [
                "Rank the top 3 schools with the highest probability of admission."
            ],
            "input_attribution": [
                "Provide attribution of inputs per school to understand which factors are most influential."
            ],
            "personality_profile": [
                "Provide a personality profile ranking for the student per school."
            ],
            "academic_focus_alignment": [
                "List schools that align best with the student's academic focus areas."
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
            ],
            "jobs_analysis": [
                "Analyze the impact of jobs and volunteer activities on the student's profile."
            ],
            "admissions_directors": [
                {{"name": "Director Name", "email": "email@example.com", "phone": "123-456-7890"}}
            ]
        }}
        Ensure each Ivy League school and each top university is **always included** in the JSON response with valid probability estimates.
        """

        logging.info("Calling OpenAI API...")
        api_response = query_chatgpt(prompt)

        # Log the full response for analysis
        logging.info(f"LLM Response: {api_response}")

        # Check if the response contains the expected data
        if "error" in api_response:
            logging.error(f"Error in API response: {api_response['error']}")
            return jsonify(api_response)

        # Pass the user data to the template
        return render_template(
            'results.html',
            sections=api_response,
            schools_visited=user_data['schools_visited'],
            schools_contacted=user_data['schools_contacted'],
            schools_followed=user_data['schools_followed'],
            early_decision=user_data['early_decision'],
            academic_focus=user_data['academic_focus']
        )

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"})

@app.route('/travel-planner')
def travel_planner():
    return render_template('travel_planner.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
