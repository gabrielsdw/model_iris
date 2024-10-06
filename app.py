import numpy as np
import torch
from flask import Flask, render_template, request

model = torch.jit.load('./models/model_scripted.pt')
model.eval()

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')




@app.route('/predict', methods=['POST'])
def predict():
    form = request.form
    sepal_w, sepal_l = float(form.get('sepal_w')), float(form.get('sepal_l')) 
    petal_w, petal_l = float(form.get('petal_w')), float(form.get('petal_l'))

    data = np.array([sepal_w, sepal_l, petal_w, petal_l])
    data_tensor = torch.FloatTensor(data)
    
    y_pred = model(data_tensor)
    result = torch.round(y_pred, decimals=2).tolist()
    
    return render_template('results.html', results=result)


if __name__ == '__main__':
    app.run(debug=True)