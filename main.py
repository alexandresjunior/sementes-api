from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)


@app.route("/")
def home():
    return "Sementes API"


@app.route("/sementes", methods=["POST"])
def preverSemente():
    # Get request body input parameters
    area = request.json.get("area")
    perimetro = request.json.get("perimetro")
    compacidade = request.json.get("compacidade")
    comprimento = request.json.get("comprimento")
    largura = request.json.get("largura")
    assimetria = request.json.get("assimetria")
    comprimento_sulco = request.json.get("comprimento_sulco")

    # Load trained neural network model
    model = load_model(os.path.join("./model", "sementes_classificacao.h5"))

    # Tensor Formatting
    input = [[area, perimetro, compacidade, comprimento,
              largura, assimetria, comprimento_sulco]]

    # Prediction Step
    pred = model.predict(input)
    result = np.argmax(pred, axis=1)

    # Set JSON response object
    response = {
        "especie": int(result[0])
    }

    return jsonify(response)


app.run(debug=True)
