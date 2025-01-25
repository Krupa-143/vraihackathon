from flask import Flask, request, jsonify, Response, render_template, stream_with_context
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import ollama
import time

app = Flask(__name__)
CORS(app)


@app.route('/')
def home():
    return render_template('hello.html')


@app.route('/generate-story', methods=['POST'])
def generate_story():
    start_time = time.time()

    # Check for image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img = Image.open(file)
    img = img.resize((4096, 4096))
    file_bytes = BytesIO()
    img.save(file_bytes, format='PNG')

    user_prompt = request.form.get('prompt', '')

    try:
        @stream_with_context
        def generate_response():
            res = ollama.chat(
                model='llava',
                messages=[
                    {'role': 'user', 'content': '250 words ' + user_prompt, 'images': [file_bytes.getvalue()]}
                ]
            )

            for chunk in res['message']['content'].split():
                yield f"{chunk} "
                time.sleep(0.1)

        print(f"Time taken: {time.time() - start_time} seconds")
        return Response(generate_response(), content_type='text/plain')

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
