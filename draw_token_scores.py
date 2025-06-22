from typing import List

import fire
import matplotlib.pyplot as plt
import numpy
import numpy as np

from src.modeling import TOKENIZERS
from src.utils import json_load


def draw_token_scores(tokens: List[str], scores: List[float], vmin=None, vmax=None, save_fig_name=None):
    scores = scores[:len(tokens)]
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 14

    # 创建热力图
    plt.figure(figsize=(len(tokens) // 3, 3))
    plt.imshow([scores], cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
    plt.yticks([])
    plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
    plt.subplots_adjust(bottom=0.35)
    plt.tight_layout()
    plt.show()
    if save_fig_name:
        plt.savefig(save_fig_name)


def draw_scores(data_file: str, model_type: str, tokenizer_file: str, idx: int = 0):
    datalist = json_load(data_file)
    acc = 0
    error_indices = []
    for i, data in enumerate(datalist):
        chosen_score = numpy.mean(data["chosen_token_scores"])
        rejected_score = numpy.mean(data["rejected_token_scores"])
        if chosen_score > rejected_score:
            acc += 1
        else:
            error_indices.append(i)
    print(error_indices)
    acc /= len(datalist)
    print("Accuracy: ", acc)

    tokenizer = TOKENIZERS[model_type](tokenizer_file)
    data = datalist[idx]
    print(data["instruction"])
    vmin = min([*data["chosen_token_scores"], *data["rejected_token_scores"]])
    vmax = max([*data["chosen_token_scores"], *data["rejected_token_scores"]])
    chosen_tokens = tokenizer.tokenize(data["chosen"], True)
    rejected_tokens = tokenizer.tokenize(data["rejected"], True)

    draw_token_scores(chosen_tokens, data["chosen_token_scores"], vmin, vmax, save_fig_name=f"chosen_{idx}.pdf")
    draw_token_scores(rejected_tokens, data["rejected_token_scores"], vmin, vmax, save_fig_name=f"rejected_{idx}.pdf")


if __name__ == '__main__':
    fire.Fire(draw_scores)

