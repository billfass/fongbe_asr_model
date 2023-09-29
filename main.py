from fastapi import FastAPI, UploadFile, File
from utils import ASRInference
import soundfile as sf

app = FastAPI()

asr = ASRInference()

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    try:
        audio, _ = sf.read(file.file)
        transcription = asr.inference(audio)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    
@app.get("/transcribe")
def transcribe(file: UploadFile = File(...)):
    try:
        audio_array, _ = sf.read(file.file)
        transcription = asr.inference(audio_array)
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
