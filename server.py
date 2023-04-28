import sys

import cv2
from flask import *

from model import Model

app = Flask(__name__)
cv2.ocl.setUseOpenCL(False)

# Create a new Machine Learning Model

model = Model()

# Routes
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    high_bp = float(data['high_bp'])
    high_chol = float(data['high_chol'])
    chol_check = float(data['chol_check'])

    bmi = float(data['bmi'])
    smoker = float(data['smoker'])
    stroke = float(data['stroke'])

    heart_disease = float(data['heart_disease'])
    phys_activity = float(data['phys_activity'])
    fruits = float(data['fruits'])

    veggies = float(data['veggies'])
    heavy_alc = float(data['heavy_alc'])
    health_insurance = float(data['health_insurance'])

    no_doc_bc_cost = float(data['no_doc_bc_cost'])
    gen_health = float(data['gen_health'])
    mental_health = float(data['mental_health'])

    phys_health = float(data['phys_health'])
    diff_walk = float(data['diff_walk'])
    sex = float(data['sex'])
    age_category = float(data['age_category'])

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