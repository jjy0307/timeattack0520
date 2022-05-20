from tensorflow.keras.preprocessing import image
import keras
import datetime
#opencv가 설치가 안될때는 터미널에서 pip install opencv-contrib-python으로 해주세요
import cv2
import numpy as np
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

def simpson(img_name, cred):
    #firebase에 접속합니다
    app = firebase_admin.initialize_app(cred, {
        'storageBucket': 'rs-object-recognition.appspot.com'
    }, name='upload-images')
    bucket = storage.bucket(app=app)
    #firebase이미지 불러와서
    blob = bucket.get_blob(img_name) #blob데이터 형식으로 불러옵니다
    url = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET') #blob에서 url을 빼온뒤
    image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8) #이미지로 바꿔줍니다
    c_image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    #이미지를 보고 싶다면 하단 코드의 주석을 풀어주세요
    # c_image.resize(64,64)
    # cv2.imshow('Image from url', c_image)
    # cv2.waitKey(0)

    saved_model = keras.models.load_model('static/model/model.h5')
    c_image = cv2.resize(c_image,(256,256)) #이미지를 데이터에 맞게 리사이즈 해줍니다
    img_array = image.img_to_array(c_image)
    labels=['sideshow_bob', 'mayor_quimby', 'troy_mcclure', 'lisa_simpson', 'moe_szyslak', 'groundskeeper_willie', 'sideshow_mel', 'patty_bouvier', 'waylon_smithers', 'ralph_wiggum', 'chief_wiggum', 'professor_john_frink', 'agnes_skinner', 'rainier_wolfcastle', 'otto_mann', 'miss_hoover', 'charles_montgomery_burns', 'homer_simpson', 'maggie_simpson', 'bart_simpson', 'comic_book_guy', 'martin_prince', 'gil', 'marge_simpson', 'lionel_hutz', 'nelson_muntz', 'snake_jailbird', 'krusty_the_clown', 'lenny_leonard', 'carl_carlson', 'abraham_grampa_simpson', 'milhouse_van_houten', 'kent_brockman', 'disco_stu', 'selma_bouvier', 'apu_nahasapeemapetilon', 'simpsons_dataset', 'fat_tony', 'cletus_spuckler', 'edna_krabappel', 'ned_flanders', 'barney_gumble', 'principal_skinner']
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = saved_model.predict(img_batch)
    prediction = labels[np.argmax(prediction)]
    return prediction