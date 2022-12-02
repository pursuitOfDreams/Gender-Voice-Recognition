import requests

male_url = "https://dagshub.com/kingabzpro/voice_gender_detection/raw/a8728610059108c2e63da8623c02921d1f86c9db/males/"
female_url = "https://dagshub.com/kingabzpro/voice_gender_detection/raw/a8728610059108c2e63da8623c02921d1f86c9db/females/"

file_idx = 1
while True:
    response = requests.get(male_url+str(file_idx)+".wav")
    if response.status_code==200:
        with open("../../data/VoxCeleb/males/"+str(file_idx)+".wav", 'wb') as file:
            file.write(response.content)
        file_idx+=1
    else:
        break 

file_idx = 1982
while True:
    response = requests.get(female_url+str(file_idx)+".wav")
    if response.status_code==200:
        with open("../../data/VoxCeleb/females/"+str(file_idx)+".wav", 'wb') as file:
            file.write(response.content)
        file_idx+=1
    else:
        break
