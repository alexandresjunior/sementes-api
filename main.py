from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)


@app.get("/sementes")
def obterDadosDaSemente():
    return render_template("index.html")


@app.post("/sementes")
def preverEspecieDaSemente():
    # Get request body (form/json) input parameters
    area = float(request.form.get("area"))
    perimetro = float(request.form.get("perimetro"))
    compacidade = float(request.form.get("compacidade"))
    comprimento = float(request.form.get("comprimento"))
    largura = float(request.form.get("largura"))
    assimetria = float(request.form.get("assimetria"))
    comprimento_sulco = float(request.form.get("comprimento_sulco"))

    # Load trained neural network model
    model = load_model(os.path.join("./model", "sementes_classificacao.h5"))

    # Tensor Formatting
    input = [[area, perimetro, compacidade, comprimento,
              largura, assimetria, comprimento_sulco]]

    # Prediction Step
    pred = model.predict(input)
    result = np.argmax(pred, axis=1)

    # Set JSON response object
    # response = {
    #     "especie": int(result[0])
    # }

    # return str(result[0])

    return render_template("index.html", especie=str(result[0]))


if __name__ == "__main__":
    app.run(debug=True)
