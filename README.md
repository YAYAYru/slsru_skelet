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

- **show_from_csv**
    - `--skelet_part=hands+pose`
    - `--hand_mirror=left`
```bash
python3 cli/show_from_csv.py --path_csv=data/csv/12_s10020_9_1.mp4.csv 
```