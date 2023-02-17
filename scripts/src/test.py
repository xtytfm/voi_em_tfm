from typing import Dict

import numpy as np
import pandas as pd
import seaborn as sn
import logging
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from models.models import ModelCnnLstm
from tools.process_data import load_audio_duration_fixed
from train import make_validate_fnc, prepare_data, loss_fnc

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger("Test")


MODEL_PATH = "/home/xabierdetoro/xabi/tfm/gits/scripts/models_results/ser_model_my_data_best_result.pt"
# MODEL_PATH = "/home/xabierdetoro/xabi/tfm/gits/output/ser_model_best_result.pt"
# MODEL_PATH = "/home/xabierdetoro/xabi/tfm/gits/scripts/models_results/ser_model_join_best_result.pt"
# MODEL_PATH = "/home/xabierdetoro/xabi/tfm/gits/output/ser_model_join_best_result.pt"
TEST_PATH = "/home/xabierdetoro/xabi/tfm/gits/scripts/src/test.csv"
DEVICE = 'cpu'
OUTPUT_FIG = "/home/xabierdetoro/xabi/tfm/gits/results/my_data/cm_test_join_my_model.png"

from sklearn.metrics import precision_score

# EMOTIONS = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fear', 7: 'disgust', 0: 'surprise'}
EMOTIONS = {0: 'happy', 1: 'angry', 2: 'other', 3: 'sadness'}


def get_confusion_matrix(labels, predictions: np.array, emotions_dict: Dict):
    cm = confusion_matrix(labels, predictions)
    cm_accurracy = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    logger.info(f'Accurracy por clase {cm_accurracy.diagonal()}')
    names = [emotions_dict[ind] for ind in range(len(emotions_dict))]
    df_cm = pd.DataFrame(cm, index=names, columns=names)
    sn.set(font_scale=1)
    sns_heat_map = sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})  # font size
    fig = sns_heat_map.get_figure()
    fig.savefig(OUTPUT_FIG)


if __name__ == "__main__":

    model = ModelCnnLstm(num_classes=len(EMOTIONS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(DEVICE)),)
    model = model.to(DEVICE)
    model.eval()

    validate_fn = make_validate_fnc(model, loss_fnc)

    test = pd.read_csv(TEST_PATH)
    # test = test.head(2)
    test_array = load_audio_duration_fixed(test, duration=3, sample_rate=16000)

    X_test, Y_test = prepare_data(X=test_array, Y=test['label'].values, sample_rate=16000,
                                  augmentation=False, dataset_type='validation')
    # print(X_test.shape)

    X_test_tensor = torch.tensor(X_test, device=DEVICE).float()
    # print(X_test_tensor.shape)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long, device=DEVICE)

    test_loss, test_acc, predictions = validate_fn(X_test_tensor, Y_test_tensor)
    logger.info(f'Test loss is {test_loss:.3f}')
    logger.info(f'Test accuracy is {test_acc:.2f}%')

    predictions = predictions.cpu().numpy()
    f1 = f1_score(Y_test, predictions, average=None)
    f1_total = f1_score(Y_test, predictions, average='micro')
    precision = precision_score(Y_test, predictions, average=None)
    precision_total = precision_score(Y_test, predictions, average='micro')
    recall = recall_score(Y_test, predictions, average=None)
    recall_total = recall_score(Y_test, predictions, average='micro')
    logger.info(f'f1_score: {f1}')
    logger.info(f'f1_total: {f1_total}')
    logger.info(f'precision: {precision}')
    logger.info(f'precison_total: {precision_total}')
    logger.info(f'recall: {recall}')
    logger.info(f'recall_total: {recall_total}')

    get_confusion_matrix(Y_test, predictions, EMOTIONS)
