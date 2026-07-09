import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import our AI functions
from src.ai.audio_genre_classifier import predict_genre_from_audio
from src.ai.engagement_predictor import predict_engagement
from src.ai.conversational_assistant import answer_question

app = FastAPI(
    title="Music Platform AI API",
    description="FastAPI Web Service for Genre Classification, Engagement Prediction, and Chatbot Assistant.",
    version="1.0.0"
)

# Allow CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic request models
class EngagementRequest(BaseModel):
    duration: float
    bit_rate: int
    favorites_track: int
    album_tracks_count: int
    album_listens: int
    album_favorites: int
    genre_top: str

class ChatbotRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Music Platform AI API. Go to /docs for interactive Swagger documentation."
    }

@app.post("/predict-genre")
async def classify_audio(file: UploadFile = File(...)):
    """Uploads an audio file (.mp3) and predicts its musical genre using MERT embeddings."""
    if not file.filename.endswith(".mp3"):
        raise HTTPException(status_code=400, detail="Only MP3 files are supported.")
        
    try:
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
            
        # Run prediction
        genre, confidence = predict_genre_from_audio(temp_path)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "filename": file.filename,
            "predicted_genre": genre,
            "confidence": round(confidence, 4)
        }
        
    except Exception as e:
        # Ensure cleanup in case of error
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict-engagement")
def estimate_engagement(req: EngagementRequest):
    """Predicts the estimated listens (engagement) of a track based on metadata."""
    try:
        predicted_listens = predict_engagement(
            duration=req.duration,
            bit_rate=req.bit_rate,
            favorites_track=req.favorites_track,
            album_tracks_count=req.album_tracks_count,
            album_listens=req.album_listens,
            album_favorites=req.album_favorites,
            genre_top=req.genre_top
        )
        return {
            "predicted_listens": round(predicted_listens, 1),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-assistant")
def query_assistant(req: ChatbotRequest):
    """Translates a natural language question into SQL, queries the DB, and returns a response."""
    try:
        response, sql_query = answer_question(req.question)
        return {
            "question": req.question,
            "sql_query": sql_query,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # To run locally: python api.py
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
