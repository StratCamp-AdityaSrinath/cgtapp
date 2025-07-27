from flask import Flask, jsonify
from flask_cors import CORS
import sys

app = Flask(__name__)
CORS(app)

@app.route('/api/main', methods=['POST'])
def handle_simulation():
    # This print statement is the only thing we need to see in the Vercel logs.
    print("--- SERVER LOG: The handle_simulation function was reached successfully! ---", file=sys.stderr)
    
    return jsonify({"message": "Success! The backend is connected and running."})