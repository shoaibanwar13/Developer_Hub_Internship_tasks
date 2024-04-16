from django.shortcuts import render
import os
import joblib
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment(text, model, tokenizer, config):
    preprocessed_text = preprocess(text)
    encoded_input = tokenizer(preprocessed_text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)

    ranking = np.argsort(scores)[::-1]
    results = []
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = np.round(float(scores[ranking[i]]), 4)
        results.append((label, score))
    
    return results

def get_model():
    model_path = os.path.join(os.getcwd(), './MLModels/model.joblib')
    loaded_model = joblib.load(model_path)
    return loaded_model
def index(request):
    return render(request,'index.html')
def sentiment_analysis_view(request):
    if request.method == 'POST':
        text_to_analyze = request.POST.get('text', '')

        # Load the model, tokenizer, and config
        model = get_model()
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # Call the sentiment analysis function
        results = analyze_sentiment(text_to_analyze, model, tokenizer, config)

         
        return render(request, 'result.html',{'results': results})

    # Render the initial form or HTML page for GET requests
    return render(request, 'result.html')
