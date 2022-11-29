from flask import Flask, request, app, jsonify, render_template
from google_drive_downloader import GoogleDriveDownloader as gdd
import whisper
import pandas as pd
from moviepy.editor import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # getting the video link as data. that's all i need.
    
    # data = request.json['data']
    # file_id = data['file_id']
    file_id = request.form.values()
    print(file_id)
    video_file = gdd.download_file_from_google_drive(file_id=file_id,
        dest_path='./data/video.mp4', unzip=False, overwrite=True)
    
    # tiny, base, small, medium, large: pick one model among these
    model = whisper.load_model("base")
    video_file_name =  './data/video.mp4'
    video = VideoFileClip(video_file_name)
    audio = video.audio
    audio.write_audiofile("./data/audio.mp3")
    output = model.transcribe("./data/audio.mp3", language='en')
    
    # you can print this so that there is no need to upload the file.
    transcript = output['text']
    transcription_df = pd.json_normalize(output['segments'])

    transcription_df['temp'] = "{"+ transcription_df['start'].astype(str) + ", " + transcription_df['end'].astype(str) +"}" + transcription_df['text'] 

    transcription1 = '\n'.join(transcription_df['temp'].to_list())

    transcription2 = "\n".join(transcription_df['text'])

    print(transcription2)
    with open('./data/final_transcript1.txt', 'w') as f:
        f.write(transcription1)
    with open('./data/final_transcript2.txt', 'w') as f:
        f.write(transcription2)
    return render_template('main_page.html', prediction_text="{}".format(transcription2))


if __name__ == "__main__":
    app.run(port=80)
## problems facing:
## 1. ngrok only running when connected to hotspot not with wifi
## 2. files either self update or file should get deleted once the transcription is ready.