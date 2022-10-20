import cv2
import librosa
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, scale




def scale_feature(feature, featureSize):

    widthTarget, heightTarget = featureSize 
    height, width = feature.shape 

    # scale according to factor
    newSize = (int(width / 4),41) 
    #print ('newSize ={}, old size = {}'.format(newSize, feature.shape ))
    feature = cv2.resize(feature, newSize)
    # Normalization
    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)
    feature = np.pad(feature, ((0, 0), (0, widthTarget - feature.shape[1])), 'constant')
    #transpose
    feature = np.transpose(feature)
    
    return feature




def sample_preprocess(sample_path):
	
	file = str(sample_path)
	y,sr=librosa.load(file)
        
	ZCR = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512, center=True)
    	
	mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, S=None, n_fft=2048, 
                                        	hop_length=512, win_length=None, window='hann', 
                                            center=True, pad_mode='reflect', power=2.0)
                                            
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, S=None, norm=None, n_fft=2048, 
                                    		hop_length=512, win_length=None, window='hann', 
                                    		center=True, pad_mode='reflect', tuning=None, n_chroma=12)
                                    		
	MFCC = librosa.feature.mfcc(y=y, sr=sr, S=None, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)

	spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, S=None, n_fft=2048, 
                                          	hop_length=512, freq=None, win_length=None, window='hann', 
                                            center=True, pad_mode='reflect')

	spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, S=None, n_fft=2048, 
                                           	hop_length=512, win_length=None, window='hann', 
                                           	center=True, pad_mode='reflect', freq=None, centroid=None, norm=True, p=2)

	spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, S=None, n_fft=2048, 
                                         	hop_length=512, win_length=None, window='hann', 
                                         	center=True, pad_mode='reflect', freq=None, roll_percent=0.85)
                                         	
	feature = np.concatenate((mel_spec,chroma_stft,MFCC,ZCR,spectral_centroid,spectral_bandwidth,spectral_rolloff), 
                                 axis=0)
	feature = librosa.power_to_db(feature, ref=np.max)
	#length = aug_feature.shape[1]
	max_length = 1292
	scaled_feature = scale_feature(feature,featureSize = (int(max_length/4), 41)) # 323 = 1292/4, 41 = 164/4 (from original data)
	scaled_feature = scaled_feature.reshape(-1, int(max_length/4), 41, 1)
	
	return scaled_feature
	
	
	
	

 
    
    
