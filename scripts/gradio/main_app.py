import gradio as gr
import numpy as np
import torch
from src.models.models import ModelCnnLstm
from src.tools.process_data import get_mel_spectrogram
from tools import read_path

MODEL_PATH = "/home/xabierdetoro/xabi/tfm/gits/scripts/models_results/ser_model_best_result.pt"

EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL = ModelCnnLstm(num_classes=len(EMOTIONS))
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)), )
MODEL = MODEL.to(DEVICE)
MODEL.eval()


def transcribe(speech):
    audio = read_path(speech)
    if len(audio) / 16000 < 1:
        return "Por favor grabe un audio de más duración", ""
    # _, y = reformat_freq(*speech)
    mel = get_mel_spectrogram(audio)
    mel = np.stack([mel], axis=0)
    mel = np.expand_dims(mel, axis=0)
    mel_tensor = torch.tensor(mel, device=DEVICE).float()

    with torch.no_grad():
        output_logits, output_softmax, attention_weights_norm = MODEL(mel_tensor)

        predictions = torch.argmax(output_softmax, dim=1)
        predictions = predictions.cpu().numpy()
    return EMOTIONS[int(predictions[0])], np.round(output_softmax.max().item(), 2)


if __name__ == "__main__":

    outputs = [gr.outputs.Textbox(label="Emoción"),
               gr.outputs.Textbox(label="Confianza")]

    title = "Emotion Classifer Demo"
    description = "Please record an audio of at least 2 seconds."

    gr.Interface(
        fn=transcribe,
        title=title,
        description=description,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs=outputs).launch()
