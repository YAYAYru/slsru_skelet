import view_opencv as vo
import skelet_mp as mp

def main_file():
    v = vo.ViewOpenCV("I:/Cloud/Google/0_sl_dataset/sl_word/video/burkova1006/x_5_x/63_5_1.mp4")
    #v = vo.ViewOpenCV(0)
    
    #mpp = mp.SkeletPoseMediapipe()
    #mpp = mp.SkeletHandMediapipe()
    mpp = mp.SkeletAllMediapipe()


    while True:
        img = v.part1_process() 


        img = mpp.find_skelet(img, True) 
        #img = mp.find_skelet(img, True)  
        #img = mp.find_skelet(img, True) 
        #print(mpp.extract_keypoints())
        img = v.paint_1tv_process(img)

        v.part2_process(img)

        if v.key_q():
            break







if __name__ == '__main__':
    main_file()