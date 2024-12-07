from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from generation.GRU_generation import main_gru_generate_music
from generation.LSTM_generation import main_lstm_generate_music
from generation.Transformer_generation import main_transformer_generate_music
import tempfile
import os

# environment variables
subset_dataset_path = os.getenv("SUBSET_DATASET_PATH", "./generation/subset_dataset")
weights_path = os.getenv("WEIGHTS_PATH", "./generation/trained-models")


app = FastAPI()

# cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://ai-ds-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# basemodel for a request
class MusicGenerationRequest(BaseModel):
    model_type: str
    amount_of_notes: int
    valid_notes: list[str]
    range_lower: int
    range_upper: int
    tempo: float
    temperature: float
    durations: list[float]


@app.post("/generate-music/")
def generate_music_endpoint(request: MusicGenerationRequest):
    try:
        # Create a temporary file for the MIDI output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as temp_file:
            temp_file_path = temp_file.name

        # Select the appropriate generation function based on the model type
        if request.model_type.lower() == "lstm":
            output_path = main_lstm_generate_music(
                amount_of_notes=request.amount_of_notes,
                valid_notes=request.valid_notes,
                range_lower=request.range_lower,
                range_upper=request.range_upper,
                tempo=request.tempo,
                temperature=request.temperature,
                durations=request.durations,
                dataset_path=subset_dataset_path,
                model_weights_path=weights_path,
                output_path=temp_file_path
            )
        elif request.model_type.lower() == "gru":
            output_path = main_gru_generate_music(
                amount_of_notes=request.amount_of_notes,
                valid_notes=request.valid_notes,
                range_lower=request.range_lower,
                range_upper=request.range_upper,
                tempo=request.tempo,
                temperature=request.temperature,
                durations=request.durations,
                dataset_path=subset_dataset_path,
                model_weights_path=weights_path,
                output_path=temp_file_path
            )
        elif request.model_type.lower() == "transformer":
            output_path = main_transformer_generate_music(
                amount_of_notes=request.amount_of_notes,
                valid_notes=request.valid_notes,
                range_lower=request.range_lower,
                range_upper=request.range_upper,
                tempo=request.tempo,
                temperature=request.temperature,
                durations=request.durations,
                dataset_path=subset_dataset_path,
                model_weights_path=weights_path,
                output_path=temp_file_path
            )
        else:
            raise ValueError("Invalid model_typea")

        # Serve the generated file to the user
        return FileResponse(
            output_path,
            media_type="audio/midi",
            filename="generated_music.mid"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
