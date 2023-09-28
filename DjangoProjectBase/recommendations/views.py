from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect

from dotenv import load_dotenv, find_dotenv
import json
import os
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import numpy as np

from movie.models import Movie

def create(request):
    
    prompt = request.GET.get('prompt')


    if prompt:
        movie_title = getRecommendation(prompt)
        movie = Movie.objects.filter(title__icontains = movie_title)
        
        return render(request, 'create.html', {'prompt':prompt, 'movies': movie})

    else:
        return render(request, 'create.html', {'movies': None})
    
        


def getRecommendation(prompt):
    
    env_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'openAI.env')
    load_dotenv(env_file_path)
    openai.api_key = os.environ['openAI_api_key']

    json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'movie_descriptions_embeddings.json')
    with open(json_file_path, 'r') as file:
        file_content = file.read()
        movies = json.loads(file_content)
    
    req = prompt
    emb = get_embedding(req,engine='text-embedding-ada-002')

    sim = []
    for i in range(len(movies)):
        sim.append(cosine_similarity(emb,movies[i]['embedding']))
    sim = np.array(sim)
    idx = np.argmax(sim)
    return movies[idx]['title']
