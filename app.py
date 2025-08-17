from flask import Flask, request, jsonify, render_template
from model_loader import load_model, generate_layout

app = Flask(__name__)
MODEL_PATH = "best_policy_phase.pth"
policy_model = load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/generate_layout', methods=['POST'])
def api_generate_layout():
    data = request.get_json()
    
    letter_freqs = {
        'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7,
        's': 6.3, 'h': 6.1, 'r': 6.0
    }
    bigram_freqs = {
        'th': 3.5, 'he': 2.8, 'in': 2.0, 'er': 1.8, 'an': 1.6,
        're': 1.5, 'ed': 1.4, 'on': 1.3, 'es': 1.2, 'st': 1.1
    }
    
    mapping, score, layout_str = generate_layout(
        policy_model,
        letter_freqs,
        bigram_freqs,
        steps=data.get('steps', 300),
        start_layout_qwerty=data.get('start_qwerty', True)
    )
    
    return jsonify({
        'score': score,
        'layout': layout_str,
        'mapping': mapping
    })

if __name__ == '__main__':
    app.run(debug=True)