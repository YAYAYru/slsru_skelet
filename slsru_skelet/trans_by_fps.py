import click
import os
import glob
import json

import pandas as pd
import numpy as np
import pose_format.numpy.pose_body as pb

from slsru_skelet import load


def folder_skelet2skelet_folder(paths_from_folder, path_to_folder, to_fps, feature_names):
    list_path = glob.glob(paths_from_folder)
    for path in list_path[:1]:
        print("from", path)
        csv_skelet2skelet_csv(path, path_to_folder+os.path.split(path)[-1], to_fps, feature_names)
        print("to", path_to_folder+os.path.split(path)[-1])
        print("-------------------")


def csv_skelet2skelet_csv(from_path, to_path, to_fps, feature_names):
    df_data, fps, df_conf = load.from_csv_slsru_skelet_v0_1_0_df(from_path, feature_names, fps=True, conf=True)
    df = df_skelet2skelet_df(df_data, fps, to_fps, df_conf, feature_names)
    df.to_csv(to_path, index=False)


def df_skelet2skelet_df(df_data, fps, to_fps, df_conf=None, select_features=None):
    np_data = df_data.to_numpy()
    np_conf = df_conf.to_numpy()  
    feature_names = []
    if select_features=="hands21_pose25_xyz":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_names.append("pose_z" + str(i))
        for i in range(21):
            feature_names.append("lhand_x" + str(i))
            feature_names.append("lhand_y" + str(i))
            feature_names.append("lhand_z" + str(i))
        for i in range(21):
            feature_names.append("rhand_x" + str(i))
            feature_names.append("rhand_y" + str(i))
            feature_names.append("rhand_z" + str(i))        

    if len(feature_names) is None:
        print("feature_names not found")
    else:
        np_p_data = np_skelet2skelet_np(np_data, fps, to_fps, np_conf)
        return pd.DataFrame(np_p_data, columns=feature_names)


def np_skelet2skelet_np(np_data, fps, to_fps, np_conf=None):
    print("np_data", np_data.shape)
    #print("np_conf", np_conf.shape)
    print("fps", fps)
    print("to_fps", to_fps)
    return pose_format_interpolation_XYZ(np_data, np_conf, fps, to_fps)

def pose_format_interpolation_XYZ(np_data, np_conf, fps, to_fps):
    """
    If unknown np_conf then can set the confidence to np.ones
    # https://github.com/AmitMY/pose-format/issues/1
    """
    # BUG
    if np_conf is None:
        print("np_conf is None")
        np_conf = np.ones((np_data.shape[0], int(np_data.shape[1]/3)))
    # print("np_conf", np_conf.shape)
    np_X = np_data[:,0::3]
    np_Y = np_data[:,1::3]
    np_Z = np_data[:,2::3]
    # print("np_X.shape", np_X.shape)
    # print("np_Y.shape", np_Y.shape) 
    # print("np_Z.shape", np_Z.shape) 

    data = np.zeros((np_X.shape[0], 1, np_X.shape[1], 3))
    conf = np.zeros((np_conf.shape[0], 1, np_conf.shape[1]))
    # print("data.shape", data.shape)
    # print("conf.shape", conf.shape)

    for i, n in enumerate(np_X):
        for j, _ in enumerate(n):
            data[i,0,j] = [np_X[i,j], np_Y[i,j], np_Z[i,j]]
            conf[i,0,j] = np_conf[i,j]   
    # print("data", data)
    p = pb.NumPyPoseBody(int(fps), data, conf)

    # print("pose_body_shape", p.data.shape)

    p = p.interpolate(new_fps=to_fps)
    print("pose_body_shape", p.data.shape)

    list_p_data = []
    for n in p.data:
        f = []
        for nn in n[0,:,:]:
            f.append(nn[0])
            f.append(nn[1])
            f.append(nn[2])
        
        list_p_data.append(f)        

    np_p_data = np.array(list_p_data)
    np_p_data = np.around(np_p_data, decimals=5)
    print("np_p_data.shape", np_p_data.shape)
    return np_p_data

def pose_format_interpolation_XY(np_data, np_conf, fps, to_fps):
    print("np_conf&&&&&&&&&&&&&&&&\n", np_conf.shape)
    print("np_conf&&&&&&&&&&&&&&&&------------------")
    """
    If unknown np_conf then can set the confidence to np.ones
    # https://github.com/AmitMY/pose-format/issues/1
    """
    np_X = np_data[:,0::2]
    np_Y = np_data[:,1::2]
    print("np_X.shape", np_X.shape)
    print("np_Y.shape", np_Y.shape) 

    data = np.zeros((np_X.shape[0], 1, np_X.shape[1], 2))
    conf = np.zeros((np_conf.shape[0], 1, np_conf.shape[1]))
    print("data.shape", data.shape)
    print("conf.shape", conf.shape)

    for i, n in enumerate(np_X):
        for j, _ in enumerate(n):
            data[i,0,j] = [np_X[i,j], np_Y[i,j]]
            conf[i,0,j] = np_conf[i,j]   

    p = pb.NumPyPoseBody(int(fps), data, conf)

    print("pose_body_shape", p.data.shape)

    p = p.interpolate(new_fps=to_fps)
    print("pose_body_shape", p.data.shape)

    list_p_data = []
    for n in p.data:
        f = []
        for nn in n[0,:,:]:
            f.append(nn[0])
            f.append(nn[1])
        
        list_p_data.append(f)        

    np_p_data = np.array(list_p_data)
    np_p_data = np.around(np_p_data, decimals=5)
    print("np_p_data.shape", np_p_data.shape)
    return np_p_data


def interpolation_XY_label(df_data, df_conf, np_label, fps, to_fps):
    per = to_fps/fps

    print("per", per)
    if per>0.95 and per<1.05:
        print("fps==to_fps")
        return df_data, np_label

    np_data = df_data.to_numpy()
    np_conf = df_conf.to_numpy()    
    np_p_data = pose_format_interpolation_XY(np_data, np_conf, fps, to_fps)  
    nn = np_label[0]
    j = 0
    np_label_newfps = np.zeros(np_p_data.shape[0]) 

    pers = []
    list_label = []

    np_label_out = []
    if fps>to_fps or fps<to_fps:
        # data
        for i,n in enumerate(np_label):
            #print("aaa", np_label[i]==n)
            if nn==n and i<np_label.shape[0]-1:
                j = j + 1
            else:
                if i==np_label.shape[0]-1:
                    list_label.append([nn,round((j+1)*per)])
                    pers.append(round((j+1)*per))
                else:
                    list_label.append([nn,round(j*per)])
                    pers.append(round(j*per))
                nn = np_label[i]
                j = 1
        s = 0
        for n in list_label:
            s = s + n[1]
        for n in list_label:
            for i in range(n[1]):
                np_label_out.append(n[0])
        d = len(np_label_newfps)-len(np_label_out)
        if d != 0:
            d1 = round(len(np_label_out)/d)
            print("d1", d1)
            print("np_label_newfps.shape", np_label_newfps.shape)
            ss = d1
            for n in range(len(np_label_newfps)):
                if ss==n:
                    #print("np_label_out", np_label_out)
                    #print("len(np_label_out)", len(np_label_out))
                    #print("ss", ss)                    
                    if len(np_label_out)==ss:
                        v = np_label_out[ss-1]
                    else:
                        v = np_label_out[ss]
                    
                    np_label_out.insert(ss,v)
                    ss = ss + d1

        np_label_out = np.array(np_label_out)
        print("np_label_out.shape", np_label_out.shape)            

    for n in list_label:
        s = s + n[1]
    print("list_label", s)
    print("list_label*per", sum(pers))

    # np_p_data - нужно проверить, потом np to df
    feature_names = []
    if load.SELECT_FEATURES=="pose_xy_25":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
    
    if load.SELECT_FEATURES=="hands_pose_xyz":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_names.append("pose_z" + str(i))
        for i in range(21):
            feature_names.append("lhand_x" + str(i))
            feature_names.append("lhand_y" + str(i))
            feature_names.append("lhand_z" + str(i))
        for i in range(21):
            feature_names.append("rhand_x" + str(i))
            feature_names.append("rhand_y" + str(i))
            feature_names.append("rhand_z" + str(i))


    if len(feature_names)==0:
        pass
    else:
        df_p_data=pd.DataFrame(np_p_data, columns=feature_names)

    return df_p_data, np_label_out


def interpolation_XY_label_folder2folder(to_fps, folder_csv, folder_json, to_folder_csv, to_folder_json):
    csvlist = glob.glob(folder_csv + '*.csv')
    jsonlist = glob.glob(folder_json + '*.json')
    videolist = load.set_csv_list(csvlist, jsonlist)

    dict_01234 = {1:"BeginMovement", 2:"TransitionalMovement", 3:"EndMovement"}
    for n in videolist:
        print("n", n)
        df_data, fps, df_conf = load.from_csv_slsru_skelet_v0_1_0_df(folder_csv+n+".csv", load.SELECT_FEATURES,fps=True,conf=True)
        print("df_conf????????:\n", df_conf)
        print("df_conf????????:------------------------------------")
        np_label = load.from_json_supervisely_WLM(folder_json+n+".json")
        df_data, np_label = interpolation_XY_label(df_data,df_conf,np_label,fps,to_fps=to_fps)
        df_data.to_csv(to_folder_csv+n+".csv", index_label="frame")
        
        j = 0
        list_tags = []

        nn = np_label[0]
        for i,nj in enumerate(np_label):
            if nn==nj:
                pass
            else:
                if nn==1 or nn==2 or nn==3:
                    list_tags.append({"name":dict_01234[nn], "frameRange":[j,i-1] })
                nn = nj
                j = i

        dict_label = {"tags":list_tags, "framesCount": np_label.shape[0]}   
        path = to_folder_json+n+".json"
        print("wrote to path", path)
        with open(path, "w") as outfile:
            json.dump(dict_label, outfile, indent = 4)



if __name__ == '__main__':
    #folder2folder()
    print("-------interpolation_XY_label_folder2folder------------")
    #FROM_FOLDER_CSV = "/home/yayay/yayay/cloud/add_to_drive/sl_dataset/sl_word_sentence/meta/csv_slsru_skelet_v0_1_0/fix_gloss/"
    #FROM_FOLDER_JSON = "/home/yayay/yayay/cloud/add_to_drive/sl_dataset/sl_word_sentence/meta/WeaklyLabeledMovement/fix_gloss/"
    FROM_FOLDER_CSV = "/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet/"
    FROM_FOLDER_JSON = "/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_wlm/"
    
    #TO_FOLDER_CSV = "/home/yayay/yayay/cloud/add_to_drive/sl_dataset/sl_word_sentence/meta/csv_slsru_skelet_v0_1_0/fix_gloss_interpol_30FPS/"
    #TO_FOLDER_JSON = "/home/yayay/yayay/cloud/add_to_drive/sl_dataset/sl_word_sentence/meta/WeaklyLabeledMovement/fix_gloss_interpol_30FPS/"
    TO_FOLDER_CSV = "/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_skelet_interpol_30FPS_hands_pose_xyz/"
    TO_FOLDER_JSON = "/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_wlm_interpol_30FPS_hands_pose_xyz/"

    interpolation_XY_label_folder2folder(30, FROM_FOLDER_CSV, FROM_FOLDER_JSON, TO_FOLDER_CSV, TO_FOLDER_JSON)
    # FROM_FOLDER_JSON = "/home/yayay/yayay/dataset/0_sl_dataset/sl_word_sentence/meta/w1006_s142_wlm/*"
    # load.wlm_jsons_validation(FROM_FOLDER_JSON)