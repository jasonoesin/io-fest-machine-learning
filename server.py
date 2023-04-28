from flask import *
import cv2
from model import Model
import sys


app = Flask(__name__)
cv2.ocl.setUseOpenCL(False)

# Create a new Machine Learning Model

model = Model()

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    high_bp = float(request.form.get('high_bp'))
    high_chol = float(request.form.get('high_chol'))
    chol_check = float(request.form.get('chol_check'))

    bmi = float(request.form.get('bmi'))
    smoker = float(request.form.get('smoker'))
    stroke = float(request.form.get('stroke'))

    heart_disease = float(request.form.get('heart_disease'))
    phys_activity = float(request.form.get('phys_activity'))
    fruits = float(request.form.get('fruits'))

    veggies = float(request.form.get('veggies'))
    heavy_alc = float(request.form.get('heavy_alc'))
    health_insurance = float(request.form.get('health_insurance'))

    no_doc_bc_cost = float(request.form.get('no_doc_bc_cost'))
    gen_health = float(request.form.get('gen_health'))
    mental_health = float(request.form.get('mental_health'))

    phys_health = float(request.form.get('phys_health'))
    diff_walk = float(request.form.get('diff_walk'))
    sex = float(request.form.get('sex'))
    age_category = float(request.form.get('age_category'))

    result = model.predict(
        high_bp, high_chol, chol_check, bmi, smoker, stroke,
        heart_disease, phys_activity, fruits, veggies, heavy_alc,
        health_insurance, no_doc_bc_cost, gen_health, mental_health,
        phys_health, diff_walk, sex, age_category
    )

    print(result, file=sys.stderr)

    return jsonify({
    'status' : "success",
    'result' : result.tolist()
    })

if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)