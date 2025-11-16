from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from model_loader import load_model, generate_layout

app = Flask(__name__)
CORS(app)

MODEL_PATH = "best_policy_phase.pth"
policy_model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate_layout', methods=['POST'])
def api_generate_layout():
    data = request.get_json() or {}

    # -------------------------------
    # 1. Get inputs from API body
    # -------------------------------
    letter_freqs = data.get("letter_freqs", {})
    bigram_freqs = data.get("bigram_freqs", {})

    # -------------------------------
    # 2. Default fallback values
    # -------------------------------
    default_letter_freqs = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5,
        'i': 7.0, 'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0
    }

    default_bigram_freqs = {
        'th': 3.5, 'he': 2.8, 'in': 2.0, 'er': 1.8,
        'an': 1.6, 're': 1.5, 'ed': 1.4, 'on': 1.3,
        'es': 1.2, 'st': 1.1
    }

    # If user sends nothing → use defaults
    if not letter_freqs:
        letter_freqs = default_letter_freqs
    
    if not bigram_freqs:
        bigram_freqs = default_bigram_freqs

    # -------------------------------
    # 3. Generate layout using model
    # -------------------------------
    mapping, score, layout_str = generate_layout(
        policy_model,
        letter_freqs=letter_freqs,
        bigram_freqs=bigram_freqs,
        steps=data.get('steps', 300),
        start_layout_qwerty=data.get('start_qwerty', True)
    )

    # -------------------------------
    # 4. Return results
    # -------------------------------
    return jsonify({
        'score': score,
        'layout': layout_str,
        'mapping': mapping,
        'input_letter_freqs': letter_freqs,
        'input_bigram_freqs': bigram_freqs
    })


if __name__ == '__main__':
    app.run(debug=True)
