'''
Job script to create a connection to Qanary and register VQA as a service.
Keeps polling every 10 sec. to keep the connection alive.
'''

import requests
import json
import time
import redis
from app import app
redis_obj = redis.Redis()

def getAdminUrl():
    return "http://localhost:8080"

def getHTTPHeaders():
    return { 
             'Content-Type': 'application/json',
             'Accept': 'application/json'
            }

def compareAndSetRegisteredId(response_id):
    connection = redis_obj.hget("qanary_instance", response_id)
    connection = int(connection)+1 if connection else 1
    redis_obj.hset("qanary_instance", response_id, connection)
    return True

def register():
    service_url = getAdminUrl()
    headers = getHTTPHeaders()
    response = requests.post(service_url, headers=headers)    

    if response.status_code == 200:
        response_body = json.loads(response.text)
        if compareAndSetRegisteredId(response_body["id"]):
            print "Application registered as %s"%response_body
        else:
            print "Application refreshed itself as %s"%response_body
        return True
    else:
        print "Application failed to register as Visual Question Answering. Response:%s"%response.text
        return False


if __name__ == "__main__":
    while True:
        if not register():
            break
        time.sleep(10)
