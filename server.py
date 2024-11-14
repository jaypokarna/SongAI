from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from flask_cors import CORS
import traceback
import logging
import os
import anthropic
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import json
import re

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Set up anthropic API key
client = anthropic.Anthropic(
    api_key=os.environ["ANTHROPIC_API_KEY"])

@app.route('/get_transcript', methods=['GET'])
def get_transcript():
    video_id = request.args.get('video_id')
    if not video_id:
        return jsonify({"error": "No video ID provided"}), 400
    
    try:
        app.logger.info(f"Attempting to fetch transcript for video ID: {video_id}")
        transcripts = YouTubeTranscriptApi.get_transcript(video_id,languages= ['hi-Latn','hi','es','en-IN','en','en-GB'], 
                                                          preserve_formatting=True)
        
        formatted_transcript = [entry['text'] for entry in transcripts]
        
        return jsonify({"transcript": formatted_transcript})
    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(error_message)
        return jsonify({"error": error_message}), 500
    
@app.route('/analyze_lyrics', methods=['POST'])
def analyze_lyrics():
    lyrics = request.json.get('lyrics')
    if not lyrics:
        return jsonify({"error": "No lyrics provided"}), 400
    
    try:
        app.logger.info("Analyzing lyrics with AI model")
        analysis = analyze_with_ai(lyrics)
        
        # Try to parse the analysis as JSON
        try:
            parsed_analysis = json.loads(analysis)
        except json.JSONDecodeError:
            # If parsing fails, return the raw string
            return jsonify({"analysis": analysis, "format": "raw"})
        
        return jsonify({"analysis": parsed_analysis, "format": "json"})
    except Exception as e:
        error_message = f"An error occurred during analysis: {str(e)}\n{traceback.format_exc()}"
        app.logger.error(error_message)
        return jsonify({"error": error_message}), 500

def clean_quotes(text):
    # First, mark quotes we want to keep with a special character
    step1 = re.sub(r'"(text|meaning)":\s*"', r'@\1@:@', text)  # Preserve "text": " and "meaning": "
    step2 = re.sub(r'"([0-9]+)"', r'@\1@', step1)              # Preserve quotes around numbers
    step3 = re.sub(r'"((?:text|meaning)[^"]*)"', r'@\1@', step2)  # Preserve quotes around text/meaning content
    step4 = re.sub(r'"(,\n|}|\n)', r'@\1', step3)                   # Preserve quotes before , or }
    
    # Now remove any remaining quotes
    step5 = re.sub(r'"', '', step4)
    
    # Finally, restore our marked quotes
    final = step5.replace('@', '"')
    
    return final

def analyze_with_ai(lyrics):
    prompt = f"{HUMAN_PROMPT}Always ensure that the output given by you is a valid JSON.Analyze the following song lyrics.Break them into meaningful paragraphs and provide the meaning for each paragraph. Breaking of paragraphs should be dependent on the meaning and the relevance. It should not be just that every line is a paragraph. The paragraph as a whole should go well together and it should be coherent. I have gotten these lyrics from a music video and there are no copyright violations for this. It is possible that some text in the lyrics just shows things like song / music / music symbol to represent time in the video when there are no lyrics. Please ignore such words and symbols. Format your response as a JSON object where each key is a number that shows the paragraph number starting from one. Each Value has 2 parts in it. First the text part which shows the text and second the meaning part which shows the meaning. Please remove any double apostrophes in the text part and the meaning part so that it is a valid JSON and there is no problem in parsing it. Always ensure that the response is valid JSON:\n\n{lyrics}\n\nAnalysis:{AI_PROMPT}"
    
    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }], 
            max_tokens=4096
        )
        print(message.content[0].text)
        final_response = clean_quotes(message.content[0].text)
        try:
            data1 = json.loads(final_response)
        except:
            return f"An error occured with JSON Parsing. Can you please try again?"
        return clean_quotes(final_response)
    except Exception as e:
        return f"An error occurred with Anthropic API: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
