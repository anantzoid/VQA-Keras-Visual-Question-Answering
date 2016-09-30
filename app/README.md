### VQA Demo Application
The demo app uses [Flask](flask.pocoo.org) as it's backend for serving the predictions via a background job communicated through [Redis](redis.io) queue. The frontend is built using ReactJs bootstrapper  [create-react-app](https://github.com/facebookincubator/create-react-app).

### Prequisites
 - ReactJS and `create-react-app`
 - Redis
 - Flask
 - NLTK (download punkt tokenizer via `nltk.download()`)
 - Run `npm install`

### Data Resources
 - Checkout `server/resources` docs for the required data files needed.
 - Download [images dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) to serve in the app. 
    
### Running the Application  
To run the appliation in your local server:
1.  Start Redis Server
2.  Run background job: `python server/job.py` 
3.  Run backend server: `python server/app.py`
4.  Run node server: `npm run build`
