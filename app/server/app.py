from flask import Flask, request, jsonify
from flask.ext.cors import CORS
import redis
from time import sleep
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
    while True:
        predictions = redis_obj.hget("out", r_id)
        if predictions:
            predictions = pickle.loads(predictions)
            redis_obj.hdel("out", "1")
            response['status'] = True
            response['payload'] = predictions
            break
    return jsonify(response)

if __name__ == "__main__":
    app.run('0.0.0.0', debug=True)
