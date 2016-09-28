### VQA Demo Application
The demo app uses [Flask](flask.pocoo.org) as it's backend for serving the predictions via a background job communicated through [Redis](redis.io) queue. The frontend is built using ReactJs bootstrapper  [create-react-app](https://github.com/facebookincubator/create-react-app).

### Prequisites
 - ReactJS
 - Redis
 - Flask
 - NLTK (download punkt tokenizer via `nltk.download()`)

### Data Resources
Checkout `server/resources` docs for the required data files needed.
    
### Running the Application  
To run the appliation in your local server:
1.  Start Redis Server
2.  Run background job: `python server/job.py` 
3.  Run backend server: `python server/app.py`
4.  Run node server: `npm run build`
