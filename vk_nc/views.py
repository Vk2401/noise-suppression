from django.shortcuts import render ,redirect
from django.http import HttpResponse
from django.template import loader


import numpy as np
import numpy as np
from scipy.io import wavfile
import scipy.ndimage
from tqdm import tqdm
import shutil
import pickle
# Packages we're using
import numpy as np
from scipy.io import wavfile
import scipy.ndimage

import glob
import os
from tqdm import tqdm
# plt, plot, tqdm
import warnings
warnings.filterwarnings('ignore')

from scipy.io import wavfile
import scipy.signal as sps
from pydub import AudioSegment
from pydub.utils import make_chunks
import keras
from .all_fn import *
import datetime
from django.core.files.storage import default_storage
from .form1 import *
from django.forms import forms
from .rec import *

# Create your views here
def login(request):
    if request.method == 'POST':
        form = a_form(request.POST, request.FILES)
        if form.is_valid():
            form.save()
             
            return redirect('pai')
    else:
        form = a_form()
    return render(request, 'vk_nc/index.html', {'form': form})




def pai(request):
    my_model = myad.objects.latest('id')
    last_audio = my_model.afile.url
    return render(request, 'vk_nc/check.html',{'audio_files': last_audio } )

def index(request):
 
    try:
        shutil.rmtree('.static/vk_nc/result')
    except:
        pass

    

    last_audio = myad.objects.latest('id')
    audio_file_path = last_audio.afile.path


    
    file_url = audio_file_path
    
    
    now = datetime.datetime.now()  # Get the current time
    a = "a_" + now.strftime("%Y-%m-%d_%H-%M-%S")  # Create a unique filename
    c = os.path.join("./vk_nc/all_a",a)
    os.mkdir(c)
    
    
    
    path_Voice = os.path.join("vk_nc/all_a", a)
  

    new_rate = 16000
    sampling_rate, data = wavfile.read(file_url)
    number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
    data = sps.resample(data, number_of_samples)
    data = np.asarray(data, dtype=np.int16)
    wavfile.write(file_url,new_rate,data)

    new_model = keras.models.load_model("./vk_nc/model.h5")
    

    myaudio = AudioSegment.from_file(file_url) 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    
    for i, chunk in enumerate(chunks):
        
        chunk_name = "chunk{0}.wav".format(i)
        name = path_Voice+'/'+chunk_name
        
        chunk.export(name, format="wav")
        
        rate, data = wavfile.read(path_Voice+'/'+chunk_name)
        
        
        if len(data.shape) >= 2 and data.size > 0:
            if data.shape[-1] > 1:
                data = data.mean(axis=-1)
            else:
                data = np.reshape(data, data.shape[:-1])
        img = generateSpectrogramForWave(data)
        np.save(path_Voice+'/'+chunk_name[:-4]+'.npy', img)
    
    ROW = 257
    COL = 62

    test_pred = []


    count = 0
    rate = 16000

    for file_url in tqdm(glob.glob(os.path.join(path_Voice, '*.npy'))):
        img_test = np.load(file_url)
        row_,col_,_ = img_test.shape
    
        if col_ < COL:
            break
    
        print(img_test.shape)
    
        img_test = img_test/255
        img_test = img_test.reshape(-1, ROW,COL,3)
        decoded_imgs = new_model.predict(img_test) #predict
        decoded_imgs = decoded_imgs.reshape(ROW,COL,3)
        decoded_imgs = decoded_imgs*255
        decoded_imgs = decoded_imgs.astype(np.int16)
        data = recoverSignalFromSpectrogram(decoded_imgs)
        file = './'+"testpred_{}".format(count)+'.wav'
        scipy.io.wavfile.write(file, rate, data)
        test_pred.append(file)
        count = count+1
    
    sound = 0
    pr = os.path.join("./vk_nc/static/vk_nc/result/",a)
    path_recovered= pr
    lr = os.path.join('vk_nc/result/',a)
    for i in range(len(test_pred)):
        print(test_pred[i])
        sound += AudioSegment.from_wav(test_pred[i])
        os.remove(test_pred[i])
    sound.export(path_recovered+"sound.wav", format="wav")   
    pp = path_recovered + "sound.wav"
    gg = "sound.wav"
    ll = lr + gg
    shutil.rmtree(c)
    my_model = myad.objects.latest('id')
    last_audio1 = my_model.afile.url
    
    return render(request, 'result.html', {'v': ll,'audio_files': last_audio1})


def rec_py1(request):

    fl,fl2 = rec_py()
    return  render(request , 'vk_nc/check1.html ' , {'lk1':fl,'lk2': fl2})





def rec_cl(request):
 
    try:
        shutil.rmtree('.static/vk_nc/result')
    except:
        pass

    
    from django.shortcuts import render


    if request.method == 'POST':
        variable_value = request.POST.get('variable_name')
        variable_value1 = request.POST.get('variable_name1')
    

    


    
    file_url = variable_value
    
    
    now = datetime.datetime.now()  # Get the current time
    a = "a_" + now.strftime("%Y-%m-%d_%H-%M-%S")  # Create a unique filename
    c = os.path.join("./vk_nc/all_a",a)
    os.mkdir(c)
    
    
    
    path_Voice = os.path.join("vk_nc/all_a", a)
  

    new_rate = 16000
    sampling_rate, data = wavfile.read(file_url)
    number_of_samples = round(len(data) * float(new_rate) / sampling_rate)
    data = sps.resample(data, number_of_samples)
    data = np.asarray(data, dtype=np.int16)
    wavfile.write(file_url,new_rate,data)

    new_model = keras.models.load_model("./vk_nc/model.h5")
    

    myaudio = AudioSegment.from_file(file_url) 
    chunk_length_ms = 1000 # pydub calculates in millisec
    chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
    
    for i, chunk in enumerate(chunks):
        
        chunk_name = "chunk{0}.wav".format(i)
        name = path_Voice+'/'+chunk_name
        
        chunk.export(name, format="wav")
        
        rate, data = wavfile.read(path_Voice+'/'+chunk_name)
        
        
        if len(data.shape) >= 2 and data.size > 0:
            if data.shape[-1] > 1:
                data = data.mean(axis=-1)
            else:
                data = np.reshape(data, data.shape[:-1])
        img = generateSpectrogramForWave(data)
        np.save(path_Voice+'/'+chunk_name[:-4]+'.npy', img)
    
    ROW = 257
    COL = 62

    test_pred = []


    count = 0
    rate = 16000

    for file_url in tqdm(glob.glob(os.path.join(path_Voice, '*.npy'))):
        img_test = np.load(file_url)
        row_,col_,_ = img_test.shape
    
        if col_ < COL:
            break
    
        print(img_test.shape)
    
        img_test = img_test/255
        img_test = img_test.reshape(-1, ROW,COL,3)
        decoded_imgs = new_model.predict(img_test) #predict
        decoded_imgs = decoded_imgs.reshape(ROW,COL,3)
        decoded_imgs = decoded_imgs*255
        decoded_imgs = decoded_imgs.astype(np.int16)
        data = recoverSignalFromSpectrogram(decoded_imgs)
        file = './'+"testpred_{}".format(count)+'.wav'
        scipy.io.wavfile.write(file, rate, data)
        test_pred.append(file)
        count = count+1
    
    sound = 0
    pr = os.path.join("./vk_nc/static/vk_nc/result/",a)
    path_recovered= pr
    lr = os.path.join('vk_nc/result/',a)
    for i in range(len(test_pred)):
        print(test_pred[i])
        sound += AudioSegment.from_wav(test_pred[i])
        os.remove(test_pred[i])
    sound.export(path_recovered+"sound.wav", format="wav")   
    pp = path_recovered + "sound.wav"
    gg = "sound.wav"
    ll = lr + gg
    shutil.rmtree(c)
    
    
    return render(request, 'result1.html', {'v': ll,'audio_files': variable_value1})
