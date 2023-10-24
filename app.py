import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST','GET'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data_input = {
            "battery_power": [int(request.form.get('battery_power'))],
            "blue": [int(request.form.get('blue'))],
            "clock_speed": [float(request.form.get('clock_speed'))],
            "dual_sim": [int(request.form.get('dual_sim'))],
            "fc": [int(request.form.get('fc'))],
            "four_g": [int(request.form.get('four_g'))],
            "int_memory": [int(request.form.get('int_memory'))],
            "m_dep": [float(request.form.get('m_dep'))],
            "mobile_wt": [int(request.form.get('mobile_wt'))],
            "n_cores": [int(request.form.get('n_cores'))],
            "pc": [int(request.form.get('pc'))],
            "px_height": [int(request.form.get('px_height'))],
            "px_width": [int(request.form.get('px_width'))],
            "ram": [int(request.form.get('ram'))],
            "sc_h": [int(request.form.get('sc_h'))],
            "sc_w": [int(request.form.get('sc_w'))],
            "talk_time": [int(request.form.get('talk_time'))],
            "three_g": [int(request.form.get('three_g'))],
            "touch_screen": [int(request.form.get('touch_screen'))],
            "wifi": [int(request.form.get('wifi'))],
        }

        data = pd.DataFrame(data_input)
        output = model.predict(data)
        return render_template('home.html', results=output[0])

if __name__ == '__main__':
    app.run(debug=True)
