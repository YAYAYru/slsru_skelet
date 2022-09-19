import click
import os
import glob
import pandas as pd
from slsru_skelet.skelet_mediapipe import MediapipeModel_v08, camera
import cv2
import mediapipe as mp
import time

    

@click.command()
# @click.argument("filepath", type=click.Path(exists=True))
@click.option("--videopath", default=None)
@click.option("--from_folder", default=None)
@click.option("--to_folder", default=None)
@click.option("--skelet_part", default="hands+pose")
def main(videopath: str, from_folder: str, to_folder: str, skelet_part: str): 
    """
    Transform video to skelet in format csv
    :param videopath: Path
    :param from_folder: Path
    :param skelet_part: Path
    :return:
    """

    if  videopath is None and from_folder is None and to_folder is None and skelet_part is None:
        camera()

    if videopath:
        s_m = MediapipeModel_v08(videopath, skelet_part)
        s_m.processing()
        #BUG не работает face
        s_m.show(mode=1)
        s_m.to_csv(videopath + ".csv")

    if from_folder and to_folder:
        list_path = glob.glob(from_folder)
        for path in list_path:
            print("path", path)
            s_m = MediapipeModel_v08(path, skelet_part)
            #s_m.processing()
            to_path = to_folder + "/" + os.path.split(path)[-1] + ".csv"
            s_m.to_csv(to_path)
    else:
        if from_folder or to_folder:
            if from_folder is None:
                print("--from_folder=???")
                return 0
            if to_folder is None:
                print("--to_folder=???")
                return 0




if __name__ == '__main__':
    main()