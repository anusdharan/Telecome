from flask import Flask, request, render_template
import pickle as pk
import numpy as np

# Flask constructor
application = Flask(__name__)
@application.route('/')
def main():
    return render_template('index.html')

# A decorator used to tell the application
# which URL is associated function
# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,18)
    loaded_model = pk.load(open(r"finalized_Charge_model.pk", "rb"))
    result = loaded_model.predict(to_predict)
    return result


@application.route('/index', methods=['POST'])
def gfg():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int,to_predict_list))
        result = ValuePredictor(to_predict_list)
    return "Total Charge will be " + str(np.round(result[0], decimals=2)) + "Rupees"
    return render_template("index.html")

if __name__ == '__main__':
    application.run(debug=True)