from flask import Flask, request, app, jsonify, render_template

import pandas as pd
import numpy as np
from google_drive_downloader import GoogleDriveDownloader as gdd
import whisper
from moviepy.editor import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # getting the video link as data. that's all i need.
    print("input only file id\n e.g https://drive.google.com/file/d/1EF8zsMYmSeXX9VeKekiEcJX6H8eCA6cL/view?usp=share_link if this is the link, input only this: 1EF8zsMYmSeXX9VeKekiEcJX6H8eCA6cL")
    data = request.json['data']
    file_id = data['file_id']
    video_file = gdd.download_file_from_google_drive(file_id=file_id,
        dest_path='./data/video.mp4', unzip=False)
    
    # tiny, base, small, medium, large: pick one model among these
    model = whisper.load_model("small")
    video_file_name =  './data/video.mp4'
    video = VideoFileClip(video_file_name)
    audio = video.audio
    audio.write_audiofile("./data/audio.mp3")
    output = model.transcribe("./data/audio.mp3", language='en')
    
    # you can print this so that there is no need to upload the file.
    transcript = output['text']
    print(transcript)
    with open('./data/final_transcript.txt', 'w') as f:
        f.write(transcript)
    
    return render_template('main_page.html')


if __name__ == "__main__":
    app.run(debug=True)
