import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('stroke_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)

    label_features=["Gender","Age","Hypertension","Avg-Glu","BMI","Smoke"]
    data_patients= pd.DataFrame(final_features,columns=label_features)
    pred_html = "Could you suffer a stroke in the future? {} (1=stroke, 0=no stroke)".format(output)
    
    pred_values="Data Enter:"
    data_patients["Smoke"] = data_patients["Smoke"].map({ 0:'Unknown', 1:'never smoked', 2:'smokes' , 3:'formerly smoked'}) 
    data_patients["Hypertension"] = data_patients["Hypertension"].map({ 0:'No', 1:'Yes'})
    data_patients["Gender"] = data_patients["Gender"].map({ 0:'Female', 1:'Male', 2:'Other'}) 
  
    pred_values = pred_values + " ".join(i + ":" + str(data_patients[i][0]) + " " for i in label_features)

    return render_template('index.html', prediction_text=pred_html, prediction_title="Prediction", prediction_values=pred_values)

if __name__=="__main__":
    app.run(port=6011, debug=True)