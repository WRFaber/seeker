from helpers import read_text_file
import os
import random
from datetime import datetime


path = os.getenv("DATA_PATH")

os.chdir(path)

list_of_data_samples = []

# iterate through all file
for file in os.listdir():
    # Check file is in json format 
    if file.endswith(".json"):
        file_path = f"{path}/{file}"
        list_of_data_samples.append(file_path)


get_random_sample = random.sample(list_of_data_samples,1)

read_random_sample = read_text_file(get_random_sample[0])

def strip_fragment_json(file):
    vis_windows = file['timeBasedVisWindows']
    dates = [(datetime.strptime(x['date'],'%Y-%m-%dT%H:%M:%S.%f'), x['availableLooks']) for x in vis_windows]
    dates.sort()
    
    print(dates)

strip_fragment_json(read_random_sample)

print(list_of_data_samples)
