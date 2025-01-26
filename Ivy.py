import os
import json
from flask import Flask, request, render_template, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import logging
from PyPDF2 import PdfReader

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

# Function to query GPT-4 model via OpenAI API
def query_chatgpt(prompt):
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant providing JSON-only responses with realistic probabilities for college admissions."},
                {"role": "user", "content": prompt}
            ]
        )
        logging.info("Received OpenAI API response.")
        return json.loads(response.choices[0].message.content)  # Parse the response content as JSON
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from OpenAI response: {str(e)}")
        return {"error": "Failed to decode JSON from OpenAI response. Ensure the AI is configured for strict JSON output."}
    except Exception as e:
        logging.error(f"Error querying OpenAI API: {str(e)}")
        return {"error": f"Error querying OpenAI API: {str(e)}"}

# Flask route to disable caching
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
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

        teacher_feedback_file = request.files.get("teacher_feedback")
        report_card_file = request.files.get("report_cards")

        teacher_feedback_text = teacher_feedback_file.read().decode("utf-8") if teacher_feedback_file else ""
        report_card_text = ""
        if report_card_file and report_card_file.filename.endswith('.pdf'):
            pdf_reader = PdfReader(report_card_file)
            report_card_text = "\n".join([page.extract_text() for page in pdf_reader.pages])
        elif report_card_file:
            report_card_text = report_card_file.read().decode("utf-8")

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

        Please analyze the provided information and return a **strict JSON response** in the following format:
        {{
            "ivy": [
                {{"school": "School Name", "probability": "Probability in %"}}
            ],
            "top_10": [
                {{"university": "University Name", "probability": "Probability in %"}}
            ],
            "cumulative_probabilities": [
                "Summary of cumulative probabilities, e.g., admission to at least one Ivy League school."
            ],
            "strengths": [
                "List of strengths based on the provided profile."
            ],
            "weaknesses": [
                "List of weaknesses based on the provided profile."
            ],
            "strategies": [
                "Recommended strategies to improve admissions chances."
            ],
            "report_card_analysis": [
                "Summary of insights from teacher comments and report card."
            ]
        }}
        Ensure realistic probability estimates for highly competitive universities like Ivy League schools and ensure the `top_10` list includes exactly 10 universities.
        """
        logging.info("Calling OpenAI API...")
        api_response = query_chatgpt(prompt)

        # Handle and validate API response
        if "error" in api_response:
            logging.error(f"Error in OpenAI API response: {api_response['error']}")
            return jsonify(api_response)

        # Validate top_10 list and enforce exactly 10 universities
        if len(api_response.get("top_10", [])) < 10:
            while len(api_response["top_10"]) < 10:
                api_response["top_10"].append({"university": "Placeholder University", "probability": "N/A"})

        # Validate ivy list and enforce at least 5 Ivy League schools
        if len(api_response.get("ivy", [])) < 5:
            while len(api_response["ivy"]) < 5:
                api_response["ivy"].append({"school": "Placeholder Ivy School", "probability": "N/A"})

        logging.info(f"Final Processed API Response: {api_response}")
        return render_template('results.html', sections=api_response)

    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
