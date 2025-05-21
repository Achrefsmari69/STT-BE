from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from speech_to_text import AdvancedSpeechToText
import shutil
import os
from datetime import datetime

app = FastAPI()
stt = AdvancedSpeechToText()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_ext = os.path.splitext(file.filename)[-1]
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}{file_ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)

        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = stt.process_audio(filepath, language="de")
        return JSONResponse(content={"status": "ok", "results": results})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
