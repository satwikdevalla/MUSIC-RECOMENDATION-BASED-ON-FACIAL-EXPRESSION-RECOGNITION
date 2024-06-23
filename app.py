import numpy as np
import streamlit as st
import pandas as pd
import cv2
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
df = pd.read_csv("muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name','emotional','pleasant','link','artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index()
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]
def fun(list, times=None):
    data = pd.DataFrame()
    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)])
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)])
        elif v == 'fear':
            data = pd.concat([data, df_fear.sample(n=t)])
        elif v == 'happy':
            data = pd.concat([data, df_happy.sample(n=t)])
        else:
            data = pd.concat([data, df_sad.sample(n=t)])
    elif len(list) == 2:
        times = [20,10]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)])
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)])
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)])
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)])
            else:
                data = pd.concat([data, df_sad.sample(n=t)])
    elif len(list) == 3:
        times = [15,10,5]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)])
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)])
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)])
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)])
            else:
                data = pd.concat([data, df_sad.sample(n=t)])
    elif len(list) == 4:
        times = [10,9,8,3]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)])
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)])
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)])
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)])
            else:
                data = pd.concat([data, df_sad.sample(n=t)])
    else:
        if times is None:
            times = [10, 7, 6, 5, 2]
            if len(list) > 5:
                times = [int(len(list) / 5) if i < 4 else len(list) % 4 for i in range(len(list))]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)])
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)])
            elif v == 'fear':
                data = pd.concat([data, df_fear.sample(n=t)])
            elif v == 'happy':
                data = pd.concat([data, df_happy.sample(n=t)])
            else:
                data = pd.concat([data, df_sad.sample(n=t)])
    return data
def pre(l):
    result = [item for items, c in Counter(l).most_common()
              for item in [items] * c]
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (48,48,1)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation= 'softmax'))
model.load_weights('model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
cv2.ocl.setUseOpenCL(False)
cap = cv2.VideoCapture(0)
st.markdown("<h2 style='text-align: center; color: white;'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html = True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Click on the Name of Recommended Song to Reach Website</b></h5>", unsafe_allow_html = True)
col1,col2,col3 = st.columns(3)
list = []
with col1:
    pass
with col2:
    if st.button('SCAN EMOTION(Click Here)'):
        count = 0
        list.clear()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors=5)
            count = count+1
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)
                prediction = model.predict(cropped_img)
                max_index = int(np.argmax(prediction))
                list.append(emotion_dict[max_index])
                cv2.putText(frame, emotion_dict[max_index], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow('Video', cv2.resize(frame, (1000, 700), interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break
            if count >= 20:
                break
        cap.release()
        cv2.destroyAllWindows()
        list = pre(list)
        st.markdown(f"<h5 style='text-align: center; color: red;'><b> {"We Detected: " + emotion_dict[max_index]}</b></h5>", unsafe_allow_html=True)
with col3:
    pass
new_df = fun(list)
st.write("")
if emotion_dict[max_index] == "Happy" or emotion_dict[max_index] == "Sad" or emotion_dict[max_index] == "Surprise":
    st.markdown("<h5 style = 'text-align: center; color: grey;'><b>Recommended Happy Songs </b></h5>", unsafe_allow_html = True)
elif emotion_dict[max_index] == "Angry":
    st.markdown("<h5 style = 'text-align: center; color: grey;'><b>Recommended Calm Songs </b></h5>", unsafe_allow_html = True)
elif emotion_dict[max_index] == "Neutral":
    st.markdown("<h5 style = 'text-align: center; color: grey;'><b>Recommended All Genre Songs </b></h5>", unsafe_allow_html = True)
else:
    st.markdown("<h5 style = 'text-align: center; color: grey;'><b>Recommended Motivation Songs </b></h5>", unsafe_allow_html = True)


st.write("---------------------------------------------------------------------------------------")
try:
    for l,a,n,i in zip(new_df["link"],new_df['artist'],new_df['name'],range(30)):
        st.markdown("""<h4 style ='text-align: center;'><a href={}>{} - {}</a></h4>""" .format(l,i+1,n),unsafe_allow_html = True)
        st.markdown("<h5 style='text-align: center; color: grey;'><i>{}</i></h5>".format(a), unsafe_allow_html= True)
        st.write("---------------------------------------------------------------------------------------")
except:
    pass
