import React, { Component } from 'react';
import $ from 'jquery';
import './App.css';
import data from './image_list.json'

var App = React.createClass({
   getInitialState() {  
        let host = "https://4cf5d16c.ngrok.io/";
        let image_data = data.images;
        let min = 0;
        let max = image_data.length-1;
        let load_images = 30;
        let n_rand = 0;
        let rand_idx = [];
        while (n_rand < load_images) {
            let rand_int = Math.floor(Math.random() * (max - min + 1)) + min;
            if (rand_idx.indexOf(rand_int) === -1) {
                rand_idx.push(rand_int);
                n_rand ++;
            }
        }
        let image_list = [];
        for (let i in rand_idx) {
            image_list.push(image_data[rand_idx[i]]);
        }
        return  {
            "HOST": host,
            "image_id": 0,
            "question": "",    
            "image_list" : image_list,
            "predictions": [],
            "show_loader": true,
            "lightbox": false,
            "lightbox_src": ""
      }; 
  },
  componentDidMount() {
    let _ = this;
    $(window).on("load", () => _.setState({"show_loader":false}));
  },
  selectImage(image_id, event) {
        event.preventDefault();
        if(this.state.image_id != 0) {
            $('#'+this.state.image_id).addClass("image");
        }
        this.setState({'image_id': image_id});
        $('#'+image_id).removeClass("image");
        $("#"+image_id+" span").css({top: ($('#'+image_id).height()-25)+"px"});
  },
  expandImage(image_id) {
    for(let i in this.state.image_list) {
        if (this.state.image_list[i].key === image_id) {
            this.setState({'lightbox': true, 'lightbox_src':this.state.HOST+this.state.image_list[i].file});
            break;
        }
    } 
  },
  closeLightbox() {
    this.setState({'lightbox': false, 'lightbox_src':''});
  }, 
  updateQuestion(event) {
        this.setState({'question': event.target.value});
  },
  submitQuery(event) {
        if (event.key === 'Enter') {
            this.sendQuery();
        }
  },
  sendQuery() {
        if(!this.state.image_id) {
            alert("Please select an image first");
        }
        else if(!this.state.question) {
            alert("Please enter a question");
        } else {
            $.ajax({
                url: 'https://4cf5d16c.ngrok.io/q',
                data: {
                    'img_id': this.state.image_id,
                    'q': this.state.question
                },
                beforeSend: () => this.setState({'show_loader': true}),
                success:(response) => {
                    if (response.status == true) {
                        this.setState({
                            'show_loader': false,
                            'predictions': response.payload, 
                            'question_previous': capitalizeFirstLetter(this.state.question),
                            'question': ''
                            })
                        this.animateScroll(".answer_row");
                    } else {
                        this.setState({'show_loader': false});
                        alert(response.message);
                    }
                },
                error:() => {
                    this.setState({'show_loader': false});
                    alert("An error occured. Please try again.");
                }
            });
        }
  },
  animateScroll(target) {
        $('html, body').animate({
            scrollTop: $(target).offset().top
        }, 1000);
  },
  render() {
      let _ = this;
      let images = this.state.image_list.map(function(image) {
        return <a className="image"  href="#" key={image.key} id={image.key} onClick={_.selectImage.bind(_, image.key)}><img src={_.state.HOST+image.file} alt={image.key}></img><span className="glyphicon glyphicon-resize-full" aria-hidden="true" onClick={_.expandImage.bind(_, image.key)}></span></a>;
      });
   
    let divStyle =  {
        "width":"100%",
        "height":"600px",
        "overflow": "hidden"
    };

    return (
      <div className="App">
        { this.state.show_loader ?
            <div className="loader-overlay"><img className="inner-loader-img" src= "/images/loading.gif"/></div>
        : null }
        <div className="App-header">
            <div className="App-intro">
                Visual Question Answering
            </div>
            <p>The Deep Learning based task of giving a natural language answer to any question about an image. <a href="#" id="learn_more" onClick={this.animateScroll.bind(this, ".info")}>Learn more</a></p>
        </div>

        <div className="banner_text"><p>To get started, select any image and start asking questions...</p></div> 
        <div className="montage" style={divStyle}>
            <div className="am-container" id="am-container">
                {images}      
            </div> 
       </div> 
       <div className="question_container">
            <input type="text" className="question" value={this.state.question} onChange={this.updateQuestion} onKeyPress={this.submitQuery} placeholder="Ask a Question: e.g. What is the man doing?"/>
        </div>
        { this.state.predictions.length ?
            <div className="row answer_row">
            <div className="col-lg-3"></div>
            <div className="answer_container col-lg-9 col-xs-12">
                <h3>{this.state.question_previous}</h3>
                <ul className="answer_list">
                    {this.state.predictions.map((pred, i) =>  
                        <li className="clearfix" key={i}>
                            <div className="pull-left class">{i+1}. {pred[0]}</div>
                            <div className="pred_percent">
                                <div className="pull-left">{pred[1]}%</div>
                                <div className="progress">
                                    <div className={"progress-bar "+(pred[1]>50?"progress-bar-success":(pred[1]<10?"progress-bar-warning":""))} role="progressbar" aria-valuenow={pred[1]} aria-valuemin="0" aria-valuemax="100" style={{"width":pred[1]+"%"}}></div>
                                </div> 
                            </div> 
                        </li> 
                    )} 
                </ul>
            </div>
            </div>
            : null
        }
        { this.state.lightbox ?
            <div onClick={this.closeLightbox}>
                <div className="loader-overlay lightboxOverlay"></div>
                <div className="lightbox-container"><img src= {this.state.lightbox_src}/></div>
            </div>
        : null }
   </div>
    );
  }
});

function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}


export default App;

