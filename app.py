import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from helper import helper
from helper import preprocessor
from saved_model import model
import os
import subprocess

#for directly access the git lfs file

if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/ksraj/CoughVid/main/saved_model/new_covid_model_15.h5"'], shell=True)



icon = Image.open('./assets/icon.png')


# Page config setup
st.set_page_config(
    page_title="CoughVid",
    page_icon=icon,
    layout="centered",
    menu_items={
        "About": """
        ## Thanks for using the app
        Made with ❤️ by [Kumar](https://github.com/ksraj)
        """,
        "Get Help": "https://twitter.com/kumaronzero",
        "Report a Bug": "https://github.com/ksraj",
    },
)



header = st.container()
recorder = st.container()
result = st.container()



with header:
# 	st.title('Are you Covid-19 Positive?')
	st.markdown(f'<p style="font-family:Garamond, serif;font-weight: bold;color:#B22222;font-size:40px;font-size:calc(100% + 2.8vw);">{"Are you Covid-19 Positive?"}</p>', 
						unsafe_allow_html=True)
	intro = "Check for yourself by just coughing to your screen for 5 seconds!"
	st.markdown(f'<p style="font-family:Courier New, monospace;color:#778899;font-size:10px;font-size:calc(100% + 0.001vw);">{intro}</p>', 
						unsafe_allow_html=True)
#	st.text('Check for yourself by just coughing to your screen for 5 seconds!')
	
	model_load_state = st.info("Loading the pretrained model...")
	
	@st.cache(allow_output_mutation=True)
	def load_saved_model(path):
		loaded_model = model.build_model()
		loaded_model.load_weights(str("./model.h5")) #for directly access the git lfs file
#		loaded_model._make_predict_function()
		loaded_model.summary()
		return loaded_model
		
	seed = 42
	low_mem = False
	num_generated = 0
	model_fpath = Path("./saved_model/new_covid_model_15.h5")
	loaded_model = 0
	loaded_model = load_saved_model(str(model_fpath))
# 	loaded_model = model.build_model()
# 	loaded_model.load_weights(str(model_fpath))
	model_load_state.success("Loaded the model, app is ready for use!")
	
	
with recorder:
#	st.header('Record your cough here:')
	st.markdown(f'<p style="font-family:Tahoma, sans-serif;font-weight: bold;font-size:25px;font-size:calc(100% + 0.7vw);">{"Record your cough here:"}</p>', 
						unsafe_allow_html=True)
	recorder_flag = False
	filename = st.text_input("Please enter your name below")
	st.markdown("""
		<style>.stTextInput > label {font-size:105%;border: 2px;border-radius: 2px;} </style>
		""", unsafe_allow_html=True)
	filename = filename.replace(' ', '_')
	
	if st.button(f"Click to Record"):
		if filename == "":
			st.warning("Please enter your name above.")
		else:
#			record_state = st.text("Recording for 5 seconds... ")
			duration = 5  # seconds
			fs = 44100
			record_state = st.info("Recording for 5 seconds... ")
			cough_state = st.subheader("Cough now!")
			frames = helper.record(duration, fs)
			record_state.text(f"Saving your sample as {filename}.wav")
			path_myrecording = f"./data/samples/{filename}.wav"
			helper.save_record(path_myrecording, frames, fs)
			record_state.text("Done! Saved your sample in our database for analysis.")
			cough_state.empty()
			recorder_flag = True
			
		
with result:
	sample_path = Path(f"./data/samples/{filename}.wav")
	if recorder_flag:
		if sample_path.is_file():
			st.audio(helper.read_audio(path_myrecording))
			result_state = st.text("Please wait while we are preprocessing your sample.")
			feature = preprocessor.sample_preprocess(sample_path)
#			fig1 = plt.figure()
# 			plt.title('scale_aug_feature')
# 			plt.imshow(feature)
# 			st.pyplot(fig1)
			model_prediction = loaded_model.predict(feature)
			final_prediction = np.argmax(model_prediction)
			high_confidence = round(np.max(model_prediction)*100, 2)
			low_confidence = round(np.min(model_prediction)*100, 2)
#			st.header("Here are your results:")
			st.markdown(f'<p style="font-family:Tahoma, sans-serif;font-weight: bold;font-size:25px;font-size:calc(100% + 0.7vw);">{"Here are your results:"}</p>', 
						unsafe_allow_html=True)
			
			tab1, tab2 = st.tabs(["Results", "Spectrogram"])
			
			with tab1:
				if final_prediction:
					positive = "There is a " + str(high_confidence) + "% possibility that you are Covid-19 POSITIVE."
					negative = "Also there is a " + str(low_confidence) + "% possibility that you are Covid-19 NEGATIVE."
					st.markdown(f'<p style="font-family:Times New Roman, serif;;color:Red;font-size:22px;font-size:calc(100% + 0.6vw);">{positive}</p>', 
							unsafe_allow_html=True)
					st.markdown(f'<p style="font-family:Times New Roman, serif;color:Green;font-size:22px;opacity: 0.6;font-size:calc(100% + 0.6vw);">{negative}</p>', 
							unsafe_allow_html=True)
					fig = plt.figure()
					height = [high_confidence, low_confidence]
					bars = ("Covid-19 +ve", "Covid-19 -ve")
					x_pos = np.arange(len(bars))
					plt.bar(x_pos, height, color=['red', 'green'])
					plt.xticks(x_pos, bars)
					st.pyplot(fig)
				else:
					negative = "There is a " + str(high_confidence) + "% possibility that you are Covid-19 NEGATIVE."
					positive = "Also there is a " + str(low_confidence) + "% possibility that you are Covid-19 POSITIVE."
					st.markdown(f'<p style="font-family:Times New Roman, serif;color:Green;font-size:22px;font-size:calc(100% + 0.6vw);">{negative}</p>', 
							unsafe_allow_html=True)
					st.markdown(f'<p style="font-family:Times New Roman, serif;color:Red;font-size:22px;opacity: 0.6;font-size:calc(100% + 0.6vw);">{positive}</p>', 
							unsafe_allow_html=True)
					fig = plt.figure()
					height = [high_confidence, low_confidence]
					bars = ("Covid-19 -ve", "Covid-19 +ve")
					x_pos = np.arange(len(bars))
					plt.bar(x_pos, height, color=['green', 'red'])
					plt.xticks(x_pos, bars)
					st.pyplot(fig)
			
			with tab2:
				fig = helper.create_spectrogram(path_myrecording)
				st.pyplot(fig)
				result_state.empty()
				
				



