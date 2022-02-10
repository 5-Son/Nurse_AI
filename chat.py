import random
import json

import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#### Roberta model
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def answer_questions(question_text, context_text):
    question_set = {'question' : question_text, 'context' : context_text}

    results = nlp(question_set)
    if results['score'] >= 0.2:
        print(f"{bot_name}: {results['answer']}")
    else :
        sentence = tokenize(question_text)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.50:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: Pardon? Can you reiterate?")

#### ML Model
with open('intents.json', 'r', encoding='utf8') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Pastor Zimmerman"
print("Start of the simmulation! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break
    
    # Roberta model
    context_text = """
    I am Zimmerman, I work as a pastor. I am at the clinic for follow-up after negative biopsy. I am aware that I drink heavily and tried to stop it in the past. I am having a lot of stress at home and work. I am in multiple co-occuring medical conditions. I recently had a nodule removed in the throat. I've been pastoring the church for the last 25 years and I have built up a nice congregation. My wife's at home with Alzheimer's. I feel very guilty these days because I've been drinking pretty heavily and I don't want to disappoint people. But it’s the only way I’ve been able to cope lately. It's not an easy situation. She has good days and bad days. And on the bad days, I get affected the most. You know my son helps a lot around the house but he is working on his PhD. and he doesn’t have time to help me out that much. It's just very painful to watch it deteriorate and the drinking helps me cope. I am using alcohol as a coping mechanism and I am not proud of it. Alcohol calms me down,helps me sleep, and makes it easier for me to communicate with people. But I know it's not healthy for me but it's the most cost-efficient way for me to calm down these days.
    """
    answer_questions(sentence, context_text)

    # ML model
    # sentence = tokenize(sentence)
    # X = bag_of_words(sentence, all_words)
    # X = X.reshape(1, X.shape[0])
    # X = torch.from_numpy(X).to(device)

    # output = model(X)
    # _, predicted = torch.max(output, dim=1)

    # tag = tags[predicted.item()]

    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]
    # if prob.item() > 0.50:
    #     for intent in intents['intents']:
    #         if tag == intent["tag"]:
    #             print(f"{bot_name}: {random.choice(intent['responses'])}")
    # else:
    #     print(f"{bot_name}: Pardon? Can you reiterate?")