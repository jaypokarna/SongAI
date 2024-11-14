from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from flask_cors import CORS
import traceback
import logging
import os
import anthropic
from anthropic import Anthropic
import json
import re
from waitress import serve
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS with specific origins in production
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '*').split(',')
CORS(app, origins=ALLOWED_ORIGINS)

# Set up logging
LOG_FILE = 'app.log'
if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler(f'logs/{LOG_FILE}', maxBytes=10000000, backupCount=5)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)



# Set up anthropic API client
client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Rate limiting setup (optional)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy"}), 200

@app.route('/get_transcript', methods=['GET'])
@limiter.limit("30 per minute")
def get_transcript():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    
    try:
        app.logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
        transcripts = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['hi-Latn', 'hi', 'es', 'en-IN', 'en', 'en-GB'],
            preserve_formatting=True
        )
        
        formatted_transcript = [entry['text'] for entry in transcripts]
        return jsonify({"transcript": formatted_transcript})
    
    except Exception as e:
        error_message = f"Transcript fetch error: {str(e)}"
        app.logger.error(f"{error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

@app.route('/analyze_lyrics', methods=['POST'])
@limiter.limit("20 per minute")
def analyze_lyrics():
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        lyrics = request.json.get('lyrics')
        if not lyrics:
            return jsonify({"error": "No lyrics provided"}), 400
        
        if len(lyrics) > 10000:  # Adjust limit as needed
            return jsonify({"error": "Lyrics exceed maximum length"}), 400
        
        app.logger.info("Analyzing lyrics with AI model")
        analysis = analyze_with_ai(lyrics)
        
        try:
            parsed_analysis = json.loads(analysis)
            return jsonify({"analysis": parsed_analysis, "format": "json"})
        except json.JSONDecodeError:
            return jsonify({"analysis": analysis, "format": "raw"})
            
    except Exception as e:
        error_message = f"Analysis error: {str(e)}"
        app.logger.error(f"{error_message}\n{traceback.format_exc()}")
        return jsonify({"error": error_message}), 500

def clean_quotes(text):
    # First, mark quotes we want to keep with a special character
    step1 = re.sub(r'"(text|meaning)":\s*"', r'@\1@:@', text)
    step2 = re.sub(r'"([0-9]+)"', r'@\1@', step1)
    step3 = re.sub(r'"((?:text|meaning)[^"]*)"', r'@\1@', step2)
    step4 = re.sub(r'"(,\n|}|\n)', r'@\1', step3)
    
    # Remove remaining quotes
    step5 = re.sub(r'"', '', step4)
    
    # Restore marked quotes
    return step5.replace('@', '"')

def analyze_with_ai(lyrics):
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[{
                "role": "user",
                "content": f"Always ensure that the output given by you is a valid JSON. Analyze the following song lyrics. Break them into meaningful paragraphs and provide the meaning for each paragraph. Breaking of paragraphs should be dependent on the meaning and the relevance. It should not be just that every line is a paragraph. The paragraph as a whole should go well together and it should be coherent. I have gotten these lyrics from a music video and there are no copyright violations for this. It is possible that some text in the lyrics just shows things like song / music / music symbol to represent time in the video when there are no lyrics. Please ignore such words and symbols. Format your response as a JSON object where each key is a number that shows the paragraph number starting from one. Each Value has 2 parts in it. First the text part which shows the text and second the meaning part which shows the meaning. Please remove any double apostrophes in the text part and the meaning part so that it is a valid JSON and there is no problem in parsing it. Always ensure that the response is valid JSON:\n\n{lyrics}\n\nAnalysis:"
            }],
            max_tokens=4096
        )
        
        final_response = clean_quotes(message.content[0].text)
        try:
            json.loads(final_response)  # Validate JSON
            return final_response
        except json.JSONDecodeError:
            return "Error: Invalid JSON response from AI model. Please try again."
            
    except Exception as e:
        app.logger.error(f"Anthropic API error: {str(e)}\n{traceback.format_exc()}")
        return f"An error occurred with Anthropic API: {str(e)}"

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal server error: {error}\n{traceback.format_exc()}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Use production WSGI server (Waitress) instead of Flask's development server
    port = int(os.getenv('PORT', 8000))
    app.logger.info(f"Starting server on port {port}")
    serve(app, host='0.0.0.0', port=port)
