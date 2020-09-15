import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import datetime

app = Flask(__name__,template_folder='template')
#model = pickle.load(open('modelfile1.pkl', 'rb'))
mod = 'modelfile1.pkl'
with open(mod, 'rb') as d:
    model = pickle.load(d)
    
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST","GET"])
def predict():
    def convert_date_to_ordinal(date):
        return date.toordinal()

    if request.method == "POST":

#        req = request.form

#        signup_date = request.form["signup_date"]
#        ref_date = request.form["ref_date"]
#        money = request.form["money"]
#        time = request.form["time"]
#        country = request.form["country"]
        
        signup_date = request.form.get("signup_date")
        ref_date = request.form.get("ref_date")
        money = request.form.get("money")
        time = request.form.get("time")
        country = request.form.get("country")
        
#        int_features = [convert_date_to_ordinal(datetime.datetime.strptime(signup_date,'%b %d %Y')),convert_date_to_ordinal(datetime.datetime.strptime(ref_date,'%b %d %Y')),int(money),int(time),country]
        int_features = [convert_date_to_ordinal(datetime.datetime.fromisoformat(signup_date)),convert_date_to_ordinal(datetime.datetime.fromisoformat(ref_date)),int(money),int(time),country]
        prediction = model.predict(int_features)
        
        output = datetime.datetime.fromordinal(int(round(prediction)))#[0])
        T = datetime.datetime(2050, 1, 1, 0, 0)
        
        if output == T:
            return render_template('index.html', prediction_text = 'customer does not churn')
        else:
            return render_template('index.html', prediction_text = 'Customer churn date is {}'.format(output))




#@app.route('/predict',methods=['POST',"GET"])
#def predict():
#    '''
#    For rendering results on HTML GUI
#    '''
#    int_features = [x for x in request.form.values()]
#    final_features = [np.array(int_features)]
#    prediction = model.predict(final_features)
#
#    output = round(prediction[0], 2)
#
#    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
#
#@app.route('/predict_api',methods=['POST',"GET"])



if __name__ == "__main__":
    app.run(debug=True)
    
    
    


    
    
