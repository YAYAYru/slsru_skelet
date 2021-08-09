from typing import List
from abc import ABC, abstractmethod
import os
import cv2
from numpy.distutils.conv_template import parse_string
import mediapipe as mp
import numpy as np

"""Вход изображения или видео"""


class SkeletModel(ABC):
    def __init__(self, filename_or_camera, fps, screen_resolution):
        self.filename_or_camera = filename_or_camera
        self.fps = fps
        self.screen_resolution = screen_resolution

    @abstractmethod
    def show(self, mode=0, type_show="OpenCV"):
        """mode=0 - представить только скелетную модель на черном фоне,
        mode=1 - представить скелетную модель и с видео"""
        """type_show="OpenCV" - представить с помощью OpenCV, 
           type_show="Jupyter" - представить на Jupyter,
           type_show="print" - представить на командной строке"""
        pass

    @abstractmethod
    def to_npy(self, npy_filename_path):
        pass

    @abstractmethod
    def to_csv(self, csv_filename_path):
        pass

    @abstractmethod
    def from_npy(self, npy_filename_path):
        pass

    @abstractmethod
    def from_csv(self, csv_filename_path):
        pass