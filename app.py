import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('lr_assignment1.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("assign.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('assign.html', prediction_text='Regression Model  has predicted price for given square-feet is : {}'.format(prediction))
if __name__ == "__main__":
  app.run(debug=True)
