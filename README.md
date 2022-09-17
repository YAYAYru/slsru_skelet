# slru_skelet

### init
- Create env and install pip on Windows 10:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r req.txt
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