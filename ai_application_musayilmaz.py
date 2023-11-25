# necessary libraries that should be installed
# pip install gradio SpeechRecognition
# !Important ffmpeg needs to be installed in order to recognice non-wav audio

# importing the necessary libraries
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import gradio as gr

# setting device to CUDA to use GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device} device is chosen")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# setting model and processor
model_id = "openai/whisper-base"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# creating a pipeline function
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

# defining a function for gradio to capture the audio recording
def transcribe_audio(audio_clip):
    print(audio_clip)
    if audio_clip is None:
        return "No audio detected"  # Handling the case where the audio clip is None
    result = pipe(audio_clip) 
    return result["text"]

# setting up the interface with gradio
iface = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.Textbox(label="Transcription")
)
iface.launch()

# usage:
#   - open command prompt at the script directory 
#   - run the script "python ai_application_musayilmaz.py"
#   - go to local URL
#   - record the audio
#   - submit

