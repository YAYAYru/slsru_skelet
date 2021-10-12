import numpy as np
import cv2 as cv
import sys
sys.path.append('../slsru_skelet')
from view_module import MediapipeModel_v08

data_folder = "data/video/"
# filename = data_folder + "S1540022.mp4"
filename = data_folder + "12_s10020_9_1.mp4"

s_m = MediapipeModel_v08(filename, 30, 30, "full")
s_m.processing()
s_m.show(mode=1)
s_m.to_csv(filename + ".csv")