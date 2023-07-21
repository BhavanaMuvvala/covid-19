import numpy as np
import pickle
from flask import Flask,render_template,request

#Create a flask object
app=Flask(__name__)
#Loading the pickle files(model)
regmodel=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('19.html')       

@app.route('/predict',methods=['POST'])
def predict():
    #print(request.form)
    location = float(request.form['location'])
    total_cases = float(request.form['total_cases'])
    total_deaths = float(request.form['total_deaths'])
    weekly_cases= float(request.form['weekly_cases'])
    weekly_deaths= float(request.form['weekly_deaths'])
    biweekly_cases= float(request.form['biweekly_cases'])
    biweekly_deaths= float(request.form['weekly_deaths'])
    final_input = np.array([[location,total_cases, total_deaths,weekly_cases,weekly_deaths,biweekly_cases,biweekly_deaths]])
    output=regmodel.predict(final_input)[0]
    print('--------------------------',output)
    return render_template('19.html', prediction_text="The new cases are {}".format(output))

if __name__ == '__main__':
    app.run()
