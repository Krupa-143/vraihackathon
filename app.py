from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import ollama
import os
import time

app = Flask(__name__)
CORS(app)

@app.route('/generate-story', methods=['POST'])
def generate_story():
    start_time = time.time()

    # Check for image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    
    # Resize image to reduce processing time
    img = Image.open(file)
    img = img.resize((256, 256))  # Resize to smaller dimensions
    file_bytes = BytesIO()
    img.save(file_bytes, format='PNG')

    user_prompt = request.form.get('prompt', '')

    try:
        # Define a generator function to stream the response
        @stream_with_context
        def generate_response():
            res = ollama.chat(
                model='llava',  # Use a smaller model if possible
                messages=[
                    {'role': 'user', 'content': '250 words' + user_prompt, 'images': [file_bytes.getvalue()]}
                ]
            )
            # Simulating tokenized response streaming
            for chunk in res['message']['content'].split():
                yield f"{chunk} "
                time.sleep(0.1)  # Simulate delay for each word

        print(f"Time taken: {time.time() - start_time} seconds")
        return Response(generate_response(), content_type='text/plain')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
