

import os
import datetime
import wave
import pyaudio


def rec_py():
        # Set the audio parameters
    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second
    seconds = 5  # Record for 10 seconds

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Start Recording
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []  # Initialize array to store frames

    print("Recording...")
    for i in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop Recording
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Finished Recording")

    # Save the recorded audio to a file
    now = datetime.datetime.now()  # Get the current time
    filename = "audio_" + now.strftime("%Y-%m-%d_%H-%M-%S") + ".wav"  # Create a unique filename
    filename1 = "./vk_nc/static/vk_nc/rec_audio/" + filename
    filepath = os.path.join("./vk_nc/static/vk_nc/rec_audio", filename)
    wavefile = wave.open(filepath, 'wb')
    wavefile.setnchannels(channels)
    wavefile.setsampwidth(p.get_sample_size(sample_format))
    wavefile.setframerate(fs)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    print("Saved as", filename, "in", filepath)
    filename = "vk_nc/rec_audio/" + filename 
    # Store the file path in a variable
    r = filepath
    print(filename)
    print("File path saved in variable: file_path_variable")
    # filename1 = "./vk_nc/static/vk_nc/rec_audio/" + filename 

    return  filename , filename1





