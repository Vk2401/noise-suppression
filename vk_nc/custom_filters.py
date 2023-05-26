import os.path

# def basename(value):
#     return os.path.basename(value)

import os

def basename(value):
    file_name = os.path.basename(value)
    return "vk_nc/audio_files/" + file_name

# def basename(value):
#     return os.path.join('vk_nc\\result\\',basename(value))