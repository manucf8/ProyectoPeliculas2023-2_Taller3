from django.urls import path
from . import views

urlpatterns = [
    path('obtainRecommendation', views.create, name='create'),
    

]