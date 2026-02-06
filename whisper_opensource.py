from time import perf_counter
import functools
import gc
import torch
import whisper
from tqdm import tqdm

#? Helper function to get the runtime
def timeit(fcn):
    @functools.wraps(fcn)
    def wrapper(*args,**kwargs):
        start_time = perf_counter()
        result = fcn(*args, **kwargs)
        end_time = perf_counter()
        runtime = end_time - start_time
        name = kwargs.get("model_name", args[0] if args else "unknown")
        print(f"{name} executed in {runtime:.6f} seconds")
        return result
    return wrapper

@timeit
def getTranscript(model_name:str, model, audio_path:str, language:str):
    result = model.transcribe(audio_path, language=language, verbose=False, fp16=True)
    print(f">> {result['text']}")
    print("")


# Configuration
DEVICE = "mps"  # "cpu | gpu"
MODEL_DIR = "./models"

model_names = ["large-v3-turbo"]

# Get audio file
audio_path = "./audios/CH_Where_is_toronto.m4a"

# Load and process one model at a time to avoid OOM errors
# for model_name in tqdm(model_names, desc="Processing models"):
for model_name in model_names:
    print(f"\nLoading {model_name}...")
    model = whisper.load_model(
        model_name,
        device=DEVICE,
        download_root=MODEL_DIR
    )

    getTranscript(model_name, model, audio_path, "Zh")

    # Free memory before loading next model
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()