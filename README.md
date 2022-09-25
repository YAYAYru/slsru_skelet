# slru_skelet

### init
- Create env and install pip on Windows 10:
```bash
python -m venv venv
venv\Scripts\activate # for Windows
source venv\bin\activate # for Linux or Mac
pip install -r req.txt
cd cli
```
### Test
- Test mediapipe with web-camera
```bash
python app/mediapipe_with_camera.py
```
- Test mediapipe read from videofile to csv and show video
```bash
mediapipe_video2skelet.py
```
### Feature
- Transform skelet by fps
```bash
python3 trans_by_fps.py --paths_from_folder_csv=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet/*.csv --path_to_folder_csv=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet_30fps/ --to_fps=30 --feature_names="hands21_pose25_xyz"
```

- Transform all csv format file to hands+pose csv `python3 cli/all2handspose.py source_folder trans_folder`, ex: `python cli/all2handspose.py /home/yayay/yayay/dataset/0_sl_dataset/sl_sentence/meta/csv_slsru_skelet_v0_1_0/20210429_DianaB /home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/csv_slsru_skelet_v0_1_0_hands_pose/20210429_DianaB`
- Transform video to skelet in format csv  `python3 cli/video2skelet.py, example: 
```bash
python3 video2skelet.py --videopath=data/video/12_s10020_9_1.mp4 # `data/video/12_s10020_9_1.mp4.csv` should appear default. Default hands+pose
```
Если нужно все части скелета, ставить параметр --skelet_part=full
- Camera `python3 cli/video2skelet.py, example: 
```bash
python3 video2skelet.py
```
- Transform videofiles in folder to skelets in format csv in another folder  `python3 cli/video2skelet.py from_folder to_folder, example:
```bash
python3 video2skelet.py --from_folder=data/video/*.mp4 --to_folder=data/csv # default hands+pose
```  
Если нужно все части скелета, ставить параметр --skelet_part=full
- Tranform skeletal model file by FPS in folder:
```bash
python3 cli/trans_by_fps.py --paths_from_folder=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet/*.csv --path_to_folder=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet_30fps
```
- Visual test interpolation from fps to fps
```bash
python3 cli/visual_test_interpolation.py --path_from_csv=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet/316345_3_1.mp4.csv --path_to_csv=/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet_30fps/316345_3_1.mp4.csv
```
