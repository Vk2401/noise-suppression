from django import forms
from . models import *

class a_form(forms.ModelForm):
    class Meta:
        model = myad
        fields = "__all__"

# class a_form(forms.Form):
#     audio_file = 
    