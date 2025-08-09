from flask import Flask,render_template,request
import joblib 
import numpy as np

app = Flask(__name__)

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
columns = joblib.load('columns.pkl')

@app.route('/',methods=['GET','POST'])
def index():
    result_text = None
    if request.method == "POST":
       try:
        data = []
        for col in columns:
            value = request.form.get(col,0)
            try:
                data.append(float(value))
            except:
                data.append(0.0)

        final_input = np.array([data])
        final_input_scaled = scaler.transform(final_input)            
        prediction = model.predict(final_input_scaled)[0]
        result_text = "High Risk of Heart disease ⚠️ " if prediction == 1 else  " ✅  Low Risk of Heart disease"

       except Exception as e:
        return f"error : {str(e)}"
       
    return render_template('index.html',columns=columns,result=result_text)   

if __name__ == "__main__":
   app.run(debug=True)
    

                  
       
