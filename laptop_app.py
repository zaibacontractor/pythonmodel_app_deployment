import streamlit as st
import pickle
import numpy as np

st.title("Laptop Price Predictor App")
st.text("This app is created by training an ML model on about 1200 different laptop models.")
st.text("The ML algorithm used to train this model is Random Forest Regressor.")

pipe=pickle.load(open('pipe.pkl','rb'))
df=pickle.load(open('df.pkl','rb'))

company=st.selectbox("Manufacturer of the laptop",df['Company'].unique(),index=4)
typename = st.selectbox("Type of the laptop",df['TypeName'].unique(),index=1)
cpu = st.selectbox("Processor on the system",df['Cpu'].unique())
ram = st.radio("RAM on the system(in GB)",[4,8,12,16,24,32,64,128],index=1,horizontal=True)
gpu = st.selectbox("Graphics Card on the laptop",df['Gpu'].unique(),index=1)
os = st.radio("Operating System",df['OpSys'].unique(),index=2)
weight = st.slider("Weight of the laptop(in kg)",min_value=0.7,max_value=4.8,
                   value=2.0,step=0.1)
ips = st.radio("IPS Display Panel?",["Yes","No"],index=1,horizontal=True)
touchscreen = st.radio("Touchscreen Display?",["Yes","No"],index=1,horizontal=True)
screen_size = st.slider("Screen Size(in Inches, measured diagonally)", min_value=10.0,
                        max_value=18.5,value=15.6,step=0.1)
screen_resolution = st.selectbox("Screen Resolution of the laptop",
                                 ["2560x1600","1440x900","1920x1080","2880x1800","1366x768",
                                  "2304x1440","3200x1800","1920x1200","2256x1504","3840x2160",
                                  "2160x1440","2560x1440","1600x900","2736x1824","2400x1600"],
                                  index=2)
cpu_speed = st.slider("Clock speed of CPU(in GHz)",min_value=0.9,max_value=3.8,
                      value=2.3,step=0.1)
hdd = st.selectbox("Hard Drive on the system(in GB, select 0 if system has SSD)",
                   [0,512,1024,2048,4096])
ssd = st.selectbox("SSD storage on the system(in GB, if HDD select 0)",
                   [0,256,512,1024,2048],index=2)
if st.button("PREDICT PRICE"):
      if touchscreen == "Yes":
            touchscreen = 1
      else:
            touchscreen = 0
      if ips == "Yes":
            ips = 1
      else:
            ips = 0
      X_res = int(screen_resolution.split("x")[0])
      Y_res = int(screen_resolution.split("x")[1])
      ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size

      query = np.array([[company, typename, cpu, ram, gpu, os, weight, ips,
                         touchscreen, ppi, cpu_speed, hdd, ssd]])
      op = pipe.predict(query)
      st.subheader(f'''The predicted price of the laptop with the above specification is estimated to be ₹{int(round(op[0],-2))}''')