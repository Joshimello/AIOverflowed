import wave
import pyaudio
from faster_whisper import WhisperModel
import os
from openai import OpenAI
import json
import RPi.GPIO as GPIO
import cv2
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
map_key = os.getenv("MAPS_API_KEY")

GPIO.setmode(GPIO.BCM)

TOUCH_PIN = 24

GPIO.setup(TOUCH_PIN, GPIO.IN)

client = OpenAI(api_key=api_key)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def record_chunk(stream, frames):
  data = stream.read(1024, exception_on_overflow = False)
  frames.append(data)

def transcribe_chunk(model, p, frames):
  file_path = "temp_chunk.wav"
  wf = wave.open(file_path, 'wb')
  wf.setnchannels(1)
  wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
  wf.setframerate(16000)
  wf.writeframes(b''.join(frames))
  wf.close()
  
  segments, info = model.transcribe(file_path)
  text = ' '.join(segment.text for segment in segments)
  os.remove(file_path)
  return text

def get_action(transcription):
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "system", "content": ""},
      {"role": "user", "content": transcription}
    ],
    functions = [
      {
        'name': 'get_action',
        'description': 'Get action from user input',
        'parameters': {
          'type': 'object',
          'properties': {
            'action': {
              'type': 'string',
              'description': 'Is the user asking a simple question, asking about the image, or asking for directions? Return "simple" or "image" or "direction", you should answer "image" for questions like "what is this" or "what am i holding" or "where is this", you should reply "location" to questions like "where am i" or "where is this place" or "where is this located"'
            },
            'answer': {
              'type': 'string',
              'description': 'The answer to the question, you must give a response if the user is asking a simple question unrelated to the image or directions'
            },
            'location': {
              'type': 'string',
              'description': 'The location the user is asking about, you must give a response if the user is asking for directions'
            }
          }
        }
      }
    ],
    function_call = 'auto'
  )
  
  return json.loads(response.choices[0].message.function_call.arguments)


def take_image():
  cam = cv2.VideoCapture(0)
  ret, image = cam.read()
  cv2.imwrite('img.jpg', image)
  cam.release()


def get_image(transcription):
  with open('img.jpg', "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
          {
            "type": "text",
            "text": f"{transcription}, give a short answer."
          },
          {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
          ]
        }
      ],
      "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()['choices'][0]['message']['content']


def get_directions(location):
  origin_url = 'https://www.google.com/maps/place/%E6%9D%BE%E5%B1%B1%E6%96%87%E5%89%B5%E5%9C%92%E5%8C%BA/@25.0438366,121.5606383,15z/data=!4m2!3m1!1s0x0:0xc82b0f87ff7df9dc?sa=X&ved=1t:2428&ictx=111'
  destination_url = location
  url = f'https://maps.googleapis.com/maps/api/directions/json?origin={origin_url}&destination={destination_url}&key={map_key}'
  response = requests.get(url)
  directions = response.json()
  htmls = [x['html_instructions'] for x in directions['routes'][0]["legs"][0]['steps']]
  html = ' '.join(html)
  string = html.replace('<b>', '').replace('</b>', '').replace('<div style="font-size:0.9em">', '').replace('</div>', '')
  return string
  
def main():
  model = WhisperModel('tiny.en', device='cpu', compute_type="int8")
  
  p = pyaudio.PyAudio()
  stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=1)
  
  frames = []
  try:
    while True:
      touch_state = GPIO.input(TOUCH_PIN)
      if touch_state == GPIO.HIGH:
        if not frames:
          print("[Recording]")
        record_chunk(stream, frames)
      elif frames:
        take_image()
        print("[Transcribing]")
        transcription = transcribe_chunk(model, p, frames)
        
        if len(transcription) > 0:
          print('YOU: ', end='')
          print(transcription)
          
          response = get_action(transcription)
          print('ASSISTANT: ', end='')
          print(response)

          if response['action'] == 'image':
            image_response = get_image(transcription)
            print(image_response)
            
          elif response['action'] == 'direction':
            directions_response = get_directions(response['location'])
            print(directions_response)

        frames = []
  finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
  main()
