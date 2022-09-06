import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('stroke_model2.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  
    try:
        #review the data input, if not complete show message
        required= True
        for x in request.form.values():
            if not x.isdigit():
               required=False     

        """
        0   gender             1277 non-null   int64  
        1   age                1277 non-null   int64  
        2   hypertension       1277 non-null   int64  
        3   heart_disease      1277 non-null   int64  
        4   ever_married       1277 non-null   int64  
        5   work_type          1277 non-null   int64  
        6   Residence_type     1277 non-null   int64  
        7   avg_glucose_level  1277 non-null   float64
        8   bmi                1277 non-null   float64
        9   smoking_status     1277 non-null   int64  
        """

        if required:
            #only run the model if all data required input
            int_features = [int(x) for x in request.form.values()]
            final_features = [np.array(int_features)]
            print(final_features)

            label_features=["Gender","Age","Hypertension","Heart Disease","Ever Married","Work Type","Residence Type","Avg-Glu","BMI","Smoke"]
            data_patients= pd.DataFrame(final_features,columns=label_features)
                    
            pred_values="<span>Data Enter:</span><ul>"
            data_patients["Smoke"] = data_patients["Smoke"].map({ 0:'Unknown', 1:'never smoked', 2:'smokes' , 3:'formerly smoked'}) 
            data_patients["Hypertension"] = data_patients["Hypertension"].map({ 0:'No', 1:'Yes'})
            data_patients["Gender"] = data_patients["Gender"].map({ 0:'Female', 1:'Male', 2:'Other'}) 
            data_patients["Heart Disease"] = data_patients["Heart Disease"].map({ 0:'No', 1:'Yes'}) 
            data_patients["Ever Married"] = data_patients["Ever Married"].map({ 0:'No', 1:'Yes'}) 
            data_patients["Residence Type"] = data_patients["Residence Type"].map({'Urban': 0, 'Rural' : 1}) 
            data_patients["Work Type"] = data_patients["Work Type"].map({'Private' : 0, 'Self-employed': 1, 'children': 2 , 'Govt_job':3, 'Never_worked':4})

            pred_values = pred_values + " ".join( "<li>" + i + " : " + str(data_patients[i][0]) + " </li>" for i in label_features)
   
            pred_values = pred_values + "</ul>"
            #Run the model
            prediction = model.predict(final_features)
            
            #Process the result to show in the page
            output = round(prediction[0], 2)

            pred_html_1=""
            pred_html_0=""

            if output==1:
                pred_html_1 = "The patient can suffer stroke in the future."
            else:
                pred_html_0 = "The patient can NOT suffer stroke in the future."
            
            return render_template('index.html', prediction_text_1=pred_html_1, prediction_text_0=pred_html_0,prediction_title="Prediction", prediction_values=pred_values)
        else:
            return render_template('index.html', prediction_text_1="Review the data required.",prediction_text_0="", prediction_title="Prediction", prediction_values="")
    
    except IndexError:
        return render_template('index.html', prediction_text_1="Review the data input.", prediction_text_0="", prediction_title="Prediction", prediction_values="")
    

if __name__=="__main__":
    app.run(port=6016, debug=True)