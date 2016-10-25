from flask import Flask, request, jsonify, send_from_directory, redirect
from flask.ext.cors import CORS
import redis
from time import sleep
from datetime import datetime
import pickle
import uuid

app = Flask(__name__)
CORS(app)
redis_obj = redis.Redis()

@app.route('/q', methods=['GET'])
def get_query():
    if len(redis_obj.lrange("in", 0, -1)) > 100:
        response['message'] = "Server is overloaded at the moment. Please try after some time."
        return jsonify(response)

    image_id = str(request.args['img_id'])
    question = str(request.args['q'])
    r_id = str(uuid.uuid4())
    response = {'status': False}

    payload = "/__/".join([r_id, image_id, question])
    redis_obj.rpush("in", payload)
    redis_obj.rpush("query_log", "|...|".join([r_id, image_id, question, str(datetime.now()).split('.')[0]]))
    while True:
        predictions = redis_obj.hget("out", r_id)
        if predictions:
            predictions = pickle.loads(predictions)
            redis_obj.hdel("out", r_id)
            response['status'] = True
            response['payload'] = predictions
            break
    return jsonify(response)

@app.route('/images/<path:path>')
def serve_img(path):
    return send_from_directory('static/images/train2014/', path)

@app.route('/')
def index():
    return redirect('https://anantzoid.github.io/VQA-Keras-Visual-Question-Answering/')
    return 'Redirect to app.'

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
