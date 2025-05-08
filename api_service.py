from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model_loader import ModelLoader
from inference import InferenceEngine
import uvicorn
import os 
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine on startup
@app.on_event("startup")
async def startup_event():
    global engine
    try:
        engine = InferenceEngine()
        print("Inference engine initialized successfully")
    except Exception as e:
        print(f"Error initializing engine: {str(e)}")
        raise

@app.post("/predict")
async def predict(angles: dict):
    try:
        result = engine.process_frame(angles)
        return {
            "success": True,
            "data": result,
            "engine_status": engine.get_status()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))