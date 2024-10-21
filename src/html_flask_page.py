from __future__ import annotations

import numpy as np

# Created by storm at 21.10.24

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    print("HIII")
    print(data)
    return None
    # Process the data as needed
    #result = your_processing_function(data)
    #return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
