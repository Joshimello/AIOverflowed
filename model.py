import wave
import pyaudio
from faster_whisper import WhisperModel
import os
import keyboard
from openai import OpenAI
import json

client = OpenAI(api_key='')

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def record_chunk(stream, frames):
    data = stream.read(1024)
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
                            'description': 'The location the user is asking about'
                        }
                    }
                }
            }
        ],
        function_call = 'auto'
    )
    
    return json.loads(response.choices[0].message.function_call.arguments)


def main():
    model = WhisperModel('base.en', device='cpu', compute_type="int8")
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    frames = []
    try:
        print("Press and hold 'ALT' to record, release to transcribe.")
        while True:
            if keyboard.is_pressed('ALT'):
                if not frames:
                    print("[Recording]")
                record_chunk(stream, frames)
            elif frames:
                print("[Transcribing]")
                transcription = transcribe_chunk(model, p, frames)
                
                if len(transcription) > 0:
                    print('YOU: ', end='')
                    print(transcription)
                    
                    response = get_action(transcription)
                    print('ASSISTANT: ', end='')
                    print(response)

                frames = []
    except KeyboardInterrupt:
        print('Exiting...')
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    main()