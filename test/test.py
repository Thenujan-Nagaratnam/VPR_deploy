import requests


# print('Started testing')
# resp = requests.post("http://localhost:5000/", files={'file': open('1.jpg', 'rb')})
resp = requests.post("https://indexvpr-4l2dxaoo7q-uc.a.run.app", files={'file': open('1.jpg', 'rb')})
# print('finished testing')

print(resp.json())
