from flask import Flask, request, jsonify, send_from_directory, redirect
from flask.ext.cors import CORS
import redis
from time import sleep
from datetime import datetime
import pickle
import uuid
import requests

app = Flask(__name__)
CORS(app)
redis_obj = redis.Redis()

def pushQuery(image_id, question):
    r_id = str(uuid.uuid4())
    payload = "/__/".join([r_id, image_id, question])
    redis_obj.rpush("in", payload)
    redis_obj.rpush("query_log", "|...|".join([r_id, image_id, question, str(datetime.now()).split('.')[0]]))
    while True:
        predictions = redis_obj.hget("out", r_id)
        if predictions:
            redis_obj.hdel("out", r_id)
            return predictions
    
@app.route('/q', methods=['GET'])
def get_query():
    response = {'status': False}
    if len(redis_obj.lrange("in", 0, -1)) > 100:
        response['message'] = "Server is overloaded at the moment. Please try after some time."
        return jsonify(response)

    image_id = str(request.args['img_id'])
    question = str(request.args['q'])
    predictions = pushQuery(image_id, question)
    response['status'] = True
    response['payload'] = pickle.loads(predictions)
    return jsonify(response)

@app.route('/images/<path:path>')
def serve_img(path):
    return send_from_directory('static/images/train2014/', path)

@app.route('/')
def index():
    return redirect('https://anantzoid.github.io/VQA-Keras-Visual-Question-Answering/')
    return 'Redirect to app.'

# endpoint where Qanary will make a call to retreive the answer
@app.route('/annotatequestion')
def annotateQuestion():
    response = {'status': False}
    print request.args
    # Access triplestore to get query params
    data_access_uri = request.args.get('http://qanary/#endpoint')
    data = requests.get(data_access_uri)

    # NOTE assuming these are the key names in the payload
    image_id = data.get("image_id", None)
    question = data.get("question", None)
    if not(image_id and question):
        response['message'] = "Missing image_id or question"
        return jsonify(response)

    predictions = pushQuery(image_id, question)
    predictions = pickle.loads(predictions)
    # TODO @Kuldeep: push predictions to triplestore via SPARQL query
    response['status'] = True
    return jsonify(response)

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
