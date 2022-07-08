from Listen import Listen
import random
import json
import torch
from Brain import NeuralNet
from NeuralNetwork import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)


FILE = "TrainData.pth"
data = torch.load(FILE)


model_state = data["model_state"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
# model.load_state_dict(model_state, strict=True)
model.eval()


# ---------------------------------------------
NAME = "Azleen"


def Main():
    sentence = Listen()


Main()
