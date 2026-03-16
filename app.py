from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get symptoms from form
    user_symptoms = request.form["symptoms"]

    # Predict disease
    prediction = model.predict([user_symptoms])

    # Get result
    result = prediction[0]

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)