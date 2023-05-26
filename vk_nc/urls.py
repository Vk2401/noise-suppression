from django.urls import path,include
from . import views

urlpatterns = [
    path('', views.login, name = "login"),
    path('index/', views.index, name = "index"),
    path('p/', views.pai, name = "pai"),
    path('r/', views.rec_py1, name = "rec_py1"),
    path('rc/', views.rec_cl, name = "rec_cl"),
   ]
