# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 23:27:52 2022

@author: Jan Setunský
"""

import os
import json
import shutil as sh
import cv2 as cv
import numpy as np
from matplotlib import pyplot as pt
from PIL import Image as img
from PIL import ImageDraw as img_draw
from itertools import product as prod

class resize_image:
    def __init__(self, path1, path2, save_path, width, height):
        self.path1 = path1
        self.path2 = path2
        self.save_path = save_path
        self.width = width
        self.height = height
    def auto(set):
        path1 = set.path1
        path2 = set.path2
        save_path = set.save_path
        save1 = save_path+"resize_bin.png"
        save2 = save_path+"resize_train.png"
        img_bin = cv.imread(path1, cv.IMREAD_UNCHANGED)
        img_train = cv.imread(path2, cv.IMREAD_UNCHANGED)
        scale_perc = 100
        # Resize bin
        w = int(img_bin.shape[1] * scale_perc/100)
        h = int(img_bin.shape[0] * scale_perc/100)
        d = (w, h)
        resize = cv.resize(img_bin, d, interpolation = cv.INTER_AREA)
        cv.imwrite(save1, resize)
        # Resize train
        w = int(img_train.shape[1]*scale_perc/100)
        h = int(img_train.shape[0]*scale_perc/100)
        d = (w, h)
        resize = cv.resize(img_train, d, interpolation = cv.INTER_AREA)        
        cv.imwrite(save2, resize)
        print("Resized Dimension: ",resize.shape)
        return (save1, save2)
    def fix(set):
        path1 = set.path1
        path2 = set.path2
        save_path = set.save_path
        width = set.width
        height = set.height        
        save1 = save_path+"resize_bin.png"
        save2 = save_path+"resize_train.png"
        img_bin = img.open(path1)
        img_train = img.open(path2)
        # Resize fix
        w = width
        h = height
        resized_bin = img_bin.resize((w, h))
        resized_train = img_train.resize((w, h))
        resized_bin.save(save1)
        resized_train.save(save2)
        img_bin.close()
        img_train.close()
        print("Resized bin: ",resized_bin)
        print("Resized train: ",resized_train)
        return (save1, save2)
    def fix_layout(set):
        
        path1 = set.path1
        path2 = set.path2
        save_path = set.save_path
        width = set.width
        height = set.height        
        save1 = save_path+"resize_bin.png"
        save2 = save_path+"resize_train.png"
        img_bin = img.open(path1)
        img_train = img.open(path2)
        # Resize fix
        w = width
        h = height
        # Add join image to fix template layout
        bin_w = int(img_bin.size[0])
        bin_h = int(img_bin.size[1])
        train_w = int(img_train.size[0])
        train_h = int(img_train.size[1])        
        # Resize in template BIN
        template_layout_bin = img.new(img_bin.mode, (w, h),(0,0,0))
        bin_w_con = False
        bin_h_con = False
        if bin_w == w and bin_h == h:
            template_layout_bin.paste(img_bin, (0, 0))
        else:
            if bin_w == w:
                spec_w = 0
            else:
                if bin_w > w:
                    bin_w_con = True
                else:
                    spec_w = (w - bin_w)/2
            if bin_h == h:
                spec_h = 0
            else:
                if bin_h > h:
                    bin_h_con = True
                else:
                    spec_h = (h - bin_h)/2
            if bin_w_con and bin_h_con:
                template_layout_bin = img_bin.resize((w, h))
            elif bin_w_con:
                template_layout_bin = img_bin.resize((w, h))
            elif bin_h_con:
                template_layout_bin = img_bin.resize((w, h))
            else:
                template_layout_bin.paste(img_bin, (int(spec_w), int(spec_h)))
        img_draw.Draw(template_layout_bin)
        template_layout_bin.save(save1)
        # Resize in template TRAIN
        template_layout_train = img.new(img_train.mode, (w, h),(0,0,0))
        train_w_con = False
        train_h_con = False
        if train_w == w and train_h == h:
            template_layout_train.paste(img_train, (0, 0))
        else:
            if train_w == w:
                spec_w = 0
            else:
                if train_w > w:
                    train_w_con = True
                else:
                    spec_w = (w - train_w)/2
            if train_h == h:
                spec_h = 0
            else:
                if train_h > h:
                    train_h_con = True
                else:
                    spec_h = (h - train_h)/2
            if train_w_con and train_h_con:
                template_layout_train = img_train.resize((w, h))
            elif train_w_con:
                template_layout_train = img_train.resize((w, h))
            elif train_h_con:
                template_layout_train = img_train.resize((w, h))
            else:
                template_layout_train.paste(img_train, (int(spec_w), int(spec_h)))
        img_draw.Draw(template_layout_train)
        template_layout_train.save(save2)        
        print("Resized bin: ",template_layout_bin)
        print("Resized train: ",template_layout_train)
        return (save1, save2)
    def keeping_ratio(set):
        path1 = set.path1
        path2 = set.path2
        save_path = set.save_path
        width = set.width
        height = set.height        
        save1 = save_path+"resize_bin.png"
        save2 = save_path+"resize_train.png"
        img_bin = img.open(path1)
        img_train = img.open(path2)
        # Resize fix
        w = width
        h = height
        # Add join image to fix template layout
        bin_w = int(img_bin.size[0])
        bin_h = int(img_bin.size[1])
        train_w = int(img_train.size[0])
        train_h = int(img_train.size[1])        
        # Resize in template BIN
        template_layout_bin = img.new(img_bin.mode, (w, h),(0,0,0))
        bin_w_con = False
        bin_h_con = False
        if bin_w == w and bin_h == h:
            template_layout_bin.paste(img_bin, (0, 0))
        else:
            if bin_w == w:
                spec_w = 0
            else:
                if bin_w > w:
                    bin_w_con = True
                else:
                    spec_w = w/bin_w
            if bin_h == h:
                spec_h = 0
            else:
                if bin_h > h:
                    bin_h_con = True
                else:
                    spec_h = h/bin_h
            if bin_w_con and bin_h_con:
                template_layout_bin = img_bin.resize((w, h))
            elif bin_w_con:
                template_layout_bin = img_bin.resize((w, h))
            elif bin_h_con:
                template_layout_bin = img_bin.resize((w, h))
            else:
                if spec_w > spec_h:
                    bin_ratio = int(bin_w*spec_h)
                    bin_point = int((w - bin_ratio)/2)
                    template_layout_ratio_bin = img_bin.resize((bin_ratio, h))
                    template_layout_bin.paste(template_layout_ratio_bin, (bin_point, 0))
                elif spec_w < spec_h:
                    bin_ratio = int(bin_h*spec_w)
                    bin_point = int((h - bin_ratio)/2)
                    template_layout_ratio_bin = img_bin.resize((w, bin_ratio))
                    template_layout_bin.paste(template_layout_ratio_bin, (0, bin_point))
                else:
                    template_layout_bin = img_bin.resize((w, h))
                    
        img_draw.Draw(template_layout_bin)
        template_layout_bin.save(save1)
        # Resize in template TRAIN
        template_layout_train = img.new(img_train.mode, (w, h),(0,0,0))
        train_w_con = False
        train_h_con = False
        if train_w == w and train_h == h:
            template_layout_train.paste(img_train, (0, 0))
        else:
            if train_w == w:
                spec_w = 0
            else:
                if train_w > w:
                    train_w_con = True
                else:
                    spec_w = w/train_w
            if train_h == h:
                spec_h = 0
            else:
                if train_h > h:
                    train_h_con = True
                else:
                    spec_h = h/train_h
            if train_w_con and train_h_con:
                template_layout_train = img_train.resize((w, h))
            elif train_w_con:
                template_layout_train = img_train.resize((w, h))
            elif train_h_con:
                template_layout_train = img_train.resize((w, h))
            else:
                if spec_w > spec_h:
                    train_ratio = int(train_w*spec_h)
                    train_point = int((w - train_ratio)/2)
                    template_layout_ratio_train = img_train.resize((train_ratio, h))
                    template_layout_train.paste(template_layout_ratio_train, (train_point, 0))
                elif spec_w < spec_h:
                    train_ratio = int(train_h*spec_w)
                    train_point = int((h - train_ratio)/2)
                    template_layout_ratio_train = img_train.resize((w, train_ratio))
                    template_layout_train.paste(template_layout_ratio_train, (0, train_point))
                else:
                    template_layout_train = img_train.resize((w, h))
        img_draw.Draw(template_layout_train)
        template_layout_train.save(save2)        
        print("Resized bin: ",template_layout_bin)
        print("Resized train: ",template_layout_train)
        return (save1, save2)
    
class split_image:
    def __init__(self, filename, dir_in, dir_out, split_perc):
        self.filename = filename
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.split_perc = split_perc
    def auto(set):
        filename = set.filename
        dir_in = set.dir_in
        dir_out = set.dir_out
        split_perc = set.split_perc
        n, e = os.path.splitext(filename)
        img_open = img.open(os.path.join(dir_in, filename))
        w, h = img_open.size
        d_w = int(w * split_perc)
        d_h = int(h * split_perc)
        grid = prod(range(0, h-h%d_h, d_h), range(0, w-w%d_w, d_w))
        for x, y in grid:
            box = (y, x, y+d_w, x+d_h)
            out = os.path.join(dir_out, f'{n}_{x}_{y}{e}')
            img_open.crop(box).save(out)
        img_open.close()

class join_image:
    def __init__(self,img_path , img_name, img_ext, grid_name, list_in, dir_out, template_param, template_position):
        self.img_path = img_path
        self.img_name = img_name
        self.img_ext = img_ext
        self.grid_name = grid_name
        self.list_in = list_in
        self.dir_out = dir_out
        self.template_param = template_param
        self.template_position = template_position
    def auto(set):
        img_path = set.img_path
        img_name = set.img_name
        img_ext = set.img_ext
        grid_name = set.grid_name
        list_in = set.list_in
        dir_out = set.dir_out
        template_param = set.template_param
        template_position = set.template_position
        crop_paths = []
        # Horizontal CROP
        for index in range(len(list_in)):
            lst = list_in[index]
            list_paths = [img.open(img_path+img_name+img_part+img_ext) for img_part in lst]
            w, h = list_paths[0].size
            mode = list_paths[0].mode
            template = img.new(mode, (w*len(list_paths), h),(0,0,0))
            for i in range(len(list_paths)):
                if i == 0:
                    template.paste(list_paths[i],(0,0))
                else:
                    template.paste(list_paths[i],(w*i, 0))
            img_draw.Draw(template)
            template.save(dir_out+grid_name+"-"+str(index)+img_ext)
            crop_paths.append(dir_out+grid_name+"-"+str(index)+img_ext)

        # Vertical CROP
        list_paths = [img.open(path) for path in crop_paths]
        w, h = list_paths[0].size
        mode = list_paths[0].mode
        template = img.new(mode, (w, h*len(list_paths)),(0,0,0))
        for i in range(len(list_paths)):
            if i == 0:
                template.paste(list_paths[i],(0,0))
            else:
                template.paste(list_paths[i],(0, h*i))
        img_draw.Draw(template)
        
        # Add join images to template layout
        tw, th = template_param
        template_layout = img.new(mode, (tw, th),(0,0,0))        
        template_layout.paste(template,(template_position))
        template_layout.save(dir_out+grid_name+"-full"+img_ext)
        template.close()
        template_layout.close()

        # Remove cuttings
        for path in crop_paths:
            if os.path.exists(path):
                os.remove(path)
            else:
                print("Error: {} is not exist".format(path))
        
class analysis_train:
    def __init__(self, dataset, dataset_format):
        self.dataset = dataset
        self.dataset_format = dataset_format
    def add_to_list(set):
        dataset = set.dataset
        dataset_format = set.dataset_format
        if dataset_format == "UINT8":
            for idt in dataset:
                mtch_g_kp = idt["k"]
                mtch_g_des = idt["d"]
                
                con_kp = mtch_g_kp in mtchs_g_list_kp_UINT8
                con_des = mtch_g_des in mtchs_g_list_des_UINT8
                if con_kp:
                    if len(mtchs_g_dlist_kp_UINT8) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_kp_UINT8))):
                            dlist_item_k = mtchs_g_dlist_kp_UINT8[dlist_i]["k"]
                            if mtch_g_kp == dlist_item_k:
                                format_dict = {"k":mtch_g_kp, "c":mtchs_g_dlist_kp_UINT8[dlist_i]["c"]+1}
                                mtchs_g_dlist_kp_UINT8.pop(dlist_i)
                                mtchs_g_dlist_kp_UINT8.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_kp, "c":2}
                                mtchs_g_dlist_kp_UINT8.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_kp, "c":2}
                        mtchs_g_dlist_kp_UINT8.append(format_dict)
                else:
                    mtchs_g_list_kp_UINT8.append(mtch_g_kp)
                if con_des:
                    if len(mtchs_g_dlist_des_UINT8) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_des_UINT8))):
                            dlist_item_k = mtchs_g_dlist_des_UINT8[dlist_i]["k"]
                            if mtch_g_des == dlist_item_k:
                                format_dict = {"k":mtch_g_des, "c":mtchs_g_dlist_des_UINT8[dlist_i]["c"]+1}
                                mtchs_g_dlist_des_UINT8.pop(dlist_i)
                                mtchs_g_dlist_des_UINT8.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_des, "c":2}
                                mtchs_g_dlist_des_UINT8.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_des, "c":2}
                        mtchs_g_dlist_des_UINT8.append(format_dict)
                else:
                    mtchs_g_list_des_UINT8.append(mtch_g_des)
                
        elif dataset_format == "FLOAT32":
            for idt in dataset:
                mtch_g_kp = idt["k"]
                mtch_g_des = idt["d"]
                
                con_kp = mtch_g_kp in mtchs_g_list_kp_FLOAT32
                con_des = mtch_g_des in mtchs_g_list_des_FLOAT32
                if con_kp:
                    if len(mtchs_g_dlist_kp_FLOAT32) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_kp_FLOAT32))):
                            dlist_item_k = mtchs_g_dlist_kp_FLOAT32[dlist_i]["k"]
                            if mtch_g_kp == dlist_item_k:
                                format_dict = {"k":mtch_g_kp, "c":mtchs_g_dlist_kp_FLOAT32[dlist_i]["c"]+1}
                                mtchs_g_dlist_kp_FLOAT32.pop(dlist_i)
                                mtchs_g_dlist_kp_FLOAT32.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_kp, "c":2}
                                mtchs_g_dlist_kp_FLOAT32.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_kp, "c":2}
                        mtchs_g_dlist_kp_FLOAT32.append(format_dict)
                else:
                    mtchs_g_list_kp_FLOAT32.append(mtch_g_kp)
                if con_des:
                    if len(mtchs_g_dlist_des_FLOAT32) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_des_FLOAT32))):
                            dlist_item_k = mtchs_g_dlist_des_FLOAT32[dlist_i]["k"]
                            if mtch_g_des == dlist_item_k:
                                format_dict = {"k":mtch_g_des, "c":mtchs_g_dlist_des_FLOAT32[dlist_i]["c"]+1}
                                mtchs_g_dlist_des_FLOAT32.pop(dlist_i)
                                mtchs_g_dlist_des_FLOAT32.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_des, "c":2}
                                mtchs_g_dlist_des_FLOAT32.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_des, "c":2}
                        mtchs_g_dlist_des_FLOAT32.append(format_dict)
                else:
                    mtchs_g_list_des_FLOAT32.append(mtch_g_des)
    def add_to_list_by_face_type(set, face_type):
        dataset = set.dataset
        dataset_format = set.dataset_format
        mtchs_g_dlist_kp_UINT8_face_type = []
        mtchs_g_dlist_des_UINT8_face_type = []
        mtchs_g_dlist_kp_FLOAT32_face_type = []
        mtchs_g_dlist_des_FLOAT32_face_type = []        
        
        mtchs_g_list_kp_UINT8_face_type = []
        mtchs_g_list_des_UINT8_face_type = []
        mtchs_g_list_kp_FLOAT32_face_type = []
        mtchs_g_list_des_FLOAT32_face_type = []        
        if dataset_format == "UINT8":
            for idt in dataset:
                mtch_g_kp = idt["k"]
                mtch_g_des = idt["d"]
                
                con_kp = mtch_g_kp in mtchs_g_list_kp_UINT8_face_type
                con_des = mtch_g_des in mtchs_g_list_des_UINT8_face_type
                if con_kp:
                    if len(mtchs_g_dlist_kp_UINT8_face_type) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_kp_UINT8_face_type))):
                            dlist_item_k = mtchs_g_dlist_kp_UINT8_face_type[dlist_i]["k"]
                            if mtch_g_kp == dlist_item_k:
                                format_dict = {"k":mtch_g_kp, "c":mtchs_g_dlist_kp_UINT8_face_type[dlist_i]["c"]+1}
                                mtchs_g_dlist_kp_UINT8_face_type.pop(dlist_i)
                                mtchs_g_dlist_kp_UINT8_face_type.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_kp, "c":2}
                                mtchs_g_dlist_kp_UINT8_face_type.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_kp, "c":2}
                        mtchs_g_dlist_kp_UINT8_face_type.append(format_dict)
                else:
                    mtchs_g_list_kp_UINT8_face_type.append(mtch_g_kp)
                if con_des:
                    if len(mtchs_g_dlist_des_UINT8_face_type) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_des_UINT8_face_type))):
                            dlist_item_k = mtchs_g_dlist_des_UINT8_face_type[dlist_i]["k"]
                            if mtch_g_des == dlist_item_k:
                                format_dict = {"k":mtch_g_des, "c":mtchs_g_dlist_des_UINT8_face_type[dlist_i]["c"]+1}
                                mtchs_g_dlist_des_UINT8_face_type.pop(dlist_i)
                                mtchs_g_dlist_des_UINT8_face_type.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_des, "c":2}
                                mtchs_g_dlist_des_UINT8_face_type.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_des, "c":2}
                        mtchs_g_dlist_des_UINT8_face_type.append(format_dict)
                else:
                    mtchs_g_list_des_UINT8_face_type.append(mtch_g_des)
            if face_type == "forehead":
                mtchs_g_dlist_kp_UINT8_forehead.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_forehead.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_forehead.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_forehead.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "left_eye":
                mtchs_g_dlist_kp_UINT8_left_eye.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_left_eye.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_left_eye.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_left_eye.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "right_eye":
                mtchs_g_dlist_kp_UINT8_right_eye.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_right_eye.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_right_eye.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_right_eye.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "eyes":
                mtchs_g_dlist_kp_UINT8_eyes.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_eyes.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_eyes.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_eyes.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "nose":
                mtchs_g_dlist_kp_UINT8_nose.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_nose.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_nose.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_nose.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "mouth_chin":
                mtchs_g_dlist_kp_UINT8_mouth_chin.append(mtchs_g_dlist_kp_UINT8_face_type)
                mtchs_g_dlist_des_UINT8_mouth_chin.append(mtchs_g_dlist_des_UINT8_face_type)
                mtchs_g_list_kp_UINT8_mouth_chin.append(mtchs_g_list_kp_UINT8_face_type)
                mtchs_g_list_des_UINT8_mouth_chin.append(mtchs_g_list_des_UINT8_face_type)
                
        elif dataset_format == "FLOAT32":
            for idt in dataset:
                mtch_g_kp = idt["k"]
                mtch_g_des = idt["d"]
                
                con_kp = mtch_g_kp in mtchs_g_list_kp_FLOAT32_face_type
                con_des = mtch_g_des in mtchs_g_list_des_FLOAT32_face_type
                if con_kp:
                    if len(mtchs_g_dlist_kp_FLOAT32_face_type) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_kp_FLOAT32_face_type))):
                            dlist_item_k = mtchs_g_dlist_kp_FLOAT32_face_type[dlist_i]["k"]
                            if mtch_g_kp == dlist_item_k:
                                format_dict = {"k":mtch_g_kp, "c":mtchs_g_dlist_kp_FLOAT32_face_type[dlist_i]["c"]+1}
                                mtchs_g_dlist_kp_FLOAT32_face_type.pop(dlist_i)
                                mtchs_g_dlist_kp_FLOAT32_face_type.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_kp, "c":2}
                                mtchs_g_dlist_kp_FLOAT32_face_type.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_kp, "c":2}
                        mtchs_g_dlist_kp_FLOAT32_face_type.append(format_dict)
                else:
                    mtchs_g_list_kp_FLOAT32_face_type.append(mtch_g_kp)
                if con_des:
                    if len(mtchs_g_dlist_des_FLOAT32_face_type) > 1:
                        for dlist_i in range(len(dict(mtchs_g_dlist_des_FLOAT32_face_type))):
                            dlist_item_k = mtchs_g_dlist_des_FLOAT32_face_type[dlist_i]["k"]
                            if mtch_g_des == dlist_item_k:
                                format_dict = {"k":mtch_g_des, "c":mtchs_g_dlist_des_FLOAT32_face_type[dlist_i]["c"]+1}
                                mtchs_g_dlist_des_FLOAT32_face_type.pop(dlist_i)
                                mtchs_g_dlist_des_FLOAT32_face_type.append(format_dict)
                            else:
                                format_dict = {"k":mtch_g_des, "c":2}
                                mtchs_g_dlist_des_FLOAT32_face_type.append(format_dict)
                    else:
                        format_dict = {"k":mtch_g_des, "c":2}
                        mtchs_g_dlist_des_FLOAT32_face_type.append(format_dict)
                else:
                    mtchs_g_list_des_FLOAT32_face_type.append(mtch_g_des)
            
            if face_type == "forehead":
                mtchs_g_dlist_kp_FLOAT32_forehead.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_forehead.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_forehead.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_forehead.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "left_eye":
                mtchs_g_dlist_kp_FLOAT32_left_eye.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_left_eye.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_left_eye.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_left_eye.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "right_eye":
                mtchs_g_dlist_kp_FLOAT32_right_eye.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_right_eye.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_right_eye.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_right_eye.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "eyes":
                mtchs_g_dlist_kp_FLOAT32_eyes.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_eyes.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_eyes.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_eyes.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "nose":
                mtchs_g_dlist_kp_FLOAT32_nose.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_nose.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_nose.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_nose.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "mouth_chin":
                mtchs_g_dlist_kp_FLOAT32_mouth_chin.append(mtchs_g_dlist_kp_FLOAT32_face_type)
                mtchs_g_dlist_des_FLOAT32_mouth_chin.append(mtchs_g_dlist_des_FLOAT32_face_type)
                mtchs_g_list_kp_FLOAT32_mouth_chin.append(mtchs_g_list_kp_FLOAT32_face_type)
                mtchs_g_list_des_FLOAT32_mouth_chin.append(mtchs_g_list_des_FLOAT32_face_type)
    def add_to_list_by_face_type_only_des(set, face_type):
        dataset = set.dataset
        dataset_format = set.dataset_format
        mtchs_g_dlist_des_UINT8_face_type = []
        mtchs_g_dlist_des_FLOAT32_face_type = []        
        mtchs_g_list_des_UINT8_face_type = []
        mtchs_g_list_des_FLOAT32_face_type = []        
        if dataset_format == "UINT8":
            for idt in dataset:
                mtch_g_des = idt
                mtchs_g_list_des_UINT8_face_type.append(mtch_g_des)
            if face_type == "forehead":
                mtchs_g_list_des_UINT8_forehead.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "left_eye":
                mtchs_g_list_des_UINT8_left_eye.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "right_eye":
                mtchs_g_list_des_UINT8_right_eye.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "eyes":
                mtchs_g_list_des_UINT8_eyes.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "nose":
                mtchs_g_list_des_UINT8_nose.append(mtchs_g_list_des_UINT8_face_type)
            elif face_type == "mouth_chin":
                mtchs_g_list_des_UINT8_mouth_chin.append(mtchs_g_list_des_UINT8_face_type)
                
        elif dataset_format == "FLOAT32":
            for idt in dataset:
                mtch_g_des = idt
                mtchs_g_list_des_FLOAT32_face_type.append(mtch_g_des)
            if face_type == "forehead":
                mtchs_g_list_des_FLOAT32_forehead.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "left_eye":
                mtchs_g_list_des_FLOAT32_left_eye.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "right_eye":
                mtchs_g_list_des_FLOAT32_right_eye.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "eyes":
                mtchs_g_list_des_FLOAT32_eyes.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "nose":
                mtchs_g_list_des_FLOAT32_nose.append(mtchs_g_list_des_FLOAT32_face_type)
            elif face_type == "mouth_chin":
                mtchs_g_list_des_FLOAT32_mouth_chin.append(mtchs_g_list_des_FLOAT32_face_type)
                    
class dtc_train:
    def __init__(self, path1, path2, save_path):
        self.path1 = path1
        self.path2 = path2
        self.save_path = save_path

    def algorithm_FLANN_INDEX_LSH(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        img_train = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(img_bin, None)
        kp2 , des2 = sf.detectAndCompute(img_train, None)
        FLANN_INDEX_LSH = 1
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)
        #FLANN_INDEX_KDTREE = 1
        #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
        search_params = dict(checks = checks)
        fl = cv.FlannBasedMatcher(index_params, search_params)
        try:
            mtchs = fl.knnMatch(des1, des2, k = 2)
        except:
            mtchs = []
        mtchsMask = [[0, 0] for i in range(len(mtchs))]
        MIN_MATCH_COUNT = MIN_MATCH_COUNT
        mtchs_g = []
        mtchs_g_r = []
        i = 0
        for m, n in mtchs:
            i+=1
            if m.distance < dist*n.distance:
                mtchs_g.append(m)
                mtchs_g_r.append(i)
        mtchs_g_kp_des = []
        mtchs_g_kp = []
        mtchs_g_des = []
        img_bin_kp = cv.KeyPoint_convert(kp1)
        img_bin_des = des1
        i = 0
        for i in mtchs_g_r:
            try:
                img_bin_kp_i = img_bin_kp[i]
                img_bin_des_i = img_bin_des[i]
                format_dict = {"k":str(img_bin_kp_i), "d":str(img_bin_des_i).replace(". ", ".,")}
                mtchs_g_kp_des.append(format_dict)
            except:
                pass
        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = img_bin.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_train = cv.polylines(img_train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None

        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mtchsMask, flags = flags)
        img_mtchs = cv.drawMatches(img_bin, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        cv.imwrite(save_path+"TRAIN-{}-algorithm_FLANN_INDEX_LSH.png".format(filename), img_mtchs)            
        return mtchs_g_kp_des
    
    def algorithm_FLANN_INDEX_KDTREE(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        img_train = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(img_bin, None)
        kp2 , des2 = sf.detectAndCompute(img_train, None)
        #FLANN_INDEX_LSH = 1
        #index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
        search_params = dict(checks = checks)
        fl = cv.FlannBasedMatcher(index_params, search_params)
        try:
            mtchs = fl.knnMatch(des1, des2, k = 2)
        except:
            mtchs = []
        mtchsMask = [[0, 0] for i in range(len(mtchs))]
        MIN_MATCH_COUNT = MIN_MATCH_COUNT
        mtchs_g = []
        mtchs_g_r = []
        i = 0
        for m, n in mtchs:
            i+=1
            if m.distance < dist*n.distance:
                mtchs_g.append(m)
                mtchs_g_r.append(i)
        mtchs_g_kp_des = []
        mtchs_g_kp = []
        mtchs_g_des = []
        img_bin_kp = cv.KeyPoint_convert(kp1)
        img_bin_des = des1
        i = 0
        for i in mtchs_g_r:
            try:
                img_bin_kp_i = img_bin_kp[i]
                img_bin_des_i = img_bin_des[i]
                format_dict = {"k":str(img_bin_kp_i), "d":str(img_bin_des_i).replace(". ", ".,")}
                mtchs_g_kp_des.append(format_dict)
            except:
                pass
        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = img_bin.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_train = cv.polylines(img_train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None

        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mtchsMask, flags = flags)
        img_mtchs = cv.drawMatches(img_bin, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        cv.imwrite(save_path+"TRAIN-{}-algorithm_FLANN_INDEX_KDTREE.png".format(filename), img_mtchs)            
        return mtchs_g_kp_des    
    
    def algorithm_BFMatcher_NONE(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        img_train = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(img_bin, None)
        kp2 , des2 = sf.detectAndCompute(img_train, None)
        bf = cv.BFMatcher()
        try:
            mtchs = bf.knnMatch(des1, des2, k = 2)
            mtchsMask = [[0, 0] for i in range(len(mtchs))]
            MIN_MATCH_COUNT = MIN_MATCH_COUNT
            mtchs_g = []
            mtchs_g_r = []
            i = 0
            for m, n in mtchs:
                i+=1
                if m.distance < dist*n.distance:
                    mtchs_g.append(m)
                    mtchs_g_r.append(i)
            mtchs_g_kp_des = []
            mtchs_g_kp = []
            mtchs_g_des = []
            img_bin_kp = cv.KeyPoint_convert(kp1)
            img_bin_des = des1
            i = 0
            for i in mtchs_g_r:
                try:
                    img_bin_kp_i = img_bin_kp[i]
                    img_bin_des_i = img_bin_des[i]
                    format_dict = {"k":str(img_bin_kp_i), "d":str(img_bin_des_i).replace(". ", ".,")}
                    mtchs_g_kp_des.append(format_dict)
                except:
                    pass            
        except:
            mtchs = []
            mtchsMask = [[0, 0] for i in range(len(mtchs))]
            MIN_MATCH_COUNT = MIN_MATCH_COUNT
            mtchs_g = []
            mtchs_g_r = []            
            mtchs_g_kp_des = []
            mtchs_g_kp = []
            mtchs_g_des = []            

        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = img_bin.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_train = cv.polylines(img_train, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None

        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mtchsMask, flags = flags)
        img_mtchs = cv.drawMatches(img_bin, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        cv.imwrite(save_path+"TRAIN-{}-algorithm_BFMatcher_NONE.png".format(filename), img_mtchs)            
        return mtchs_g_kp_des

    def algorithm_BFMatcher_NORM_HAMMING(set, read_type, filename):
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        img_bin = cv.imread(img_1_path, read_type)
        img_train = cv.imread(img_2_path, read_type)
        orb = cv.ORB_create()
        kp1 , des1 = orb.detectAndCompute(img_bin, None)
        kp2 , des2 = orb.detectAndCompute(img_train, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        mtchs = bf.match(des1, des2)
        #mtchs = sorted(mtchs, key = lambda x:x.distance)
        mtchs_g_r = []
        i = 0
        for mtchs_val in mtchs:
            mtchs_g_r.append(mtchs_val)
        mtchs_g_kp_des = []
        mtchs_g_kp = []
        mtchs_g_des = []
        img_bin_kp = cv.KeyPoint_convert(kp1)
        img_bin_des = des1
        i = 0
        for i in range(len(mtchs_g_r)):
            try:
                img_bin_kp_i = img_bin_kp[i]
                img_bin_des_i = img_bin_des[i]
                format_dict = {"k":str(img_bin_kp_i), "d":str(img_bin_des_i).replace(". ", ".,")}
                mtchs_g_kp_des.append(format_dict)
            except:
                pass
        print("Matches found - {}".format(len(mtchs_g_r)))
        img_mtchs = cv.drawMatches(img_bin, kp1, img_train, kp2, mtchs[:0], None, flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite(save_path+"TRAIN-{}-algorithm_BFMatcher_NORM_HAMMING.png".format(filename), img_mtchs)
        return mtchs_g_kp_des
    def get_des_by_SIFT(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(img_bin, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list
    
    def get_des_by_SIFT_erode(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, None)
        dst_gray, dst_color = cv.pencilSketch(img_bin, sigma_s = 70, sigma_r = 0.08, shade_factor = 0.05)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(dst_gray, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list
    
    def get_des_by_SIFT_oil_painting(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, None)
        res = cv.xphoto.oilPainting(img_bin, 9, 2)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(res, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list    
    
    def get_des_by_SIFT_sketch_gray(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, None)
        dst_gray, dst_color = cv.pencilSketch(img_bin, sigma_s = 70, sigma_r = 0.08, shade_factor = 0.05)
        sf = cv.SIFT_create()
        kp1 , des1 = sf.detectAndCompute(dst_gray, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list    
    
    def get_des_by_ORB(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        orb = cv.ORB_create()
        kp1 , des1 = orb.detectAndCompute(img_bin, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list
    
    def get_des_by_ORB_erode(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        kernel = np.ones((5,5), np.float32)
        img_bin_erode = cv.erode(img_bin, kernel, iterations = 1)        
        orb = cv.ORB_create()
        kp1 , des1 = orb.detectAndCompute(img_bin_erode, None)
        des_list = []
        for des in des1:
            des_list.append(str(des))
        return des_list    
    
class dtc_test:
    def __init__(self, path1, path2, save_path):
        self.path1 = path1
        self.path2 = path2
        self.save_path = save_path        
    def algorithm_FLANN_INDEX_LSH(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, dataset, filename):
        # Class parameters
        save_path = set.save_path
        img_2_path = set.path2
        # Source
        img_test = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 ,des1 = dataset
        kp2 , des2 = sf.detectAndCompute(img_test, None)
        FLANN_INDEX_LSH = 1
        index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
        #FLANN_INDEX_KDTREE = 1
        #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
        search_params = dict(checks = checks)
        fl = cv.FlannBasedMatcher(index_params, search_params)
        try:
            mtchs = fl.knnMatch(des1, des2, k = 2)
        except:
            mtchs = []
        mtchsMask = [[0, 0] for i in range(len(mtchs))]
        MIN_MATCH_COUNT = MIN_MATCH_COUNT
        mtchs_g = []
        for m, n in mtchs:
            if m.distance < dist*n.distance:
                mtchs_g.append(m)
        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            except:
                src_pts = None
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            mtchs_g_points = []
            for idst_pts in dst_pts:
                mtch_g_pts = (str(idst_pts).replace("[[       ", "").replace("[[      ", "").replace("[[     ", "").replace("[[    ", "").replace("[[   ", "").replace("[[  ", "").replace("[[ ", "").replace("[[", "").replace("       ]]", "").replace("      ]]", "").replace("     ]]", "").replace("    ]]", "").replace("   ]]", "").replace("  ]]", "").replace(" ]]", "").replace("]]", "").replace("        ", ",").replace("       ", ",").replace("      ", ",").replace("     ", ",").replace("    ", ",").replace("   ", ",").replace("  ", ",").replace(" ", ",").split(","))
                mtchs_g_points.append(mtch_g_pts)

            mtchs_g_points_float32 = np.asarray(mtchs_g_points, dtype="float32")
            mtchs_g_points_coords = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtchs_g_points_float32]
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = 200,200
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_test = cv.polylines(img_test, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None
            mtchs_g_points_coords = None
        """
        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, flags = flags)
        img_test_draw = cv.drawMatches(img_test, kp1, img_test, kp2, mtchs_g, None, flags = 2)
        img_test_draw = cv.drawMatches(img_test, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        """
        img_test_draw = cv.drawKeypoints(img_test, mtchs_g_points_coords, None, color=(0, 255, 0), flags=flags)
        
        cv.imwrite(save_path+"TEST-{}-algorithm_FLANN_INDEX_LSH.png".format(filename), img_test_draw)
        return len(mtchs_g)
    
    def algorithm_FLANN_INDEX_KDTREE(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, dataset, filename):
        # Class parameters
        save_path = set.save_path
        img_2_path = set.path2
        # Source
        img_test = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 ,des1 = dataset
        kp2 , des2 = sf.detectAndCompute(img_test, None)
        #FLANN_INDEX_LSH = 1
        #index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
        search_params = dict(checks = checks)
        fl = cv.FlannBasedMatcher(index_params, search_params)
        try:
            mtchs = fl.knnMatch(des1, des2, k = 2)
        except:
            mtchs = []
        mtchsMask = [[0, 0] for i in range(len(mtchs))]
        MIN_MATCH_COUNT = MIN_MATCH_COUNT
        mtchs_g = []
        for m, n in mtchs:
            if m.distance < dist*n.distance:
                mtchs_g.append(m)
        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            except:
                src_pts = None
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            mtchs_g_points = []
            for idst_pts in dst_pts:
                mtch_g_pts = (str(idst_pts).replace("[[       ", "").replace("[[      ", "").replace("[[     ", "").replace("[[    ", "").replace("[[   ", "").replace("[[  ", "").replace("[[ ", "").replace("[[", "").replace("       ]]", "").replace("      ]]", "").replace("     ]]", "").replace("    ]]", "").replace("   ]]", "").replace("  ]]", "").replace(" ]]", "").replace("]]", "").replace("        ", ",").replace("       ", ",").replace("      ", ",").replace("     ", ",").replace("    ", ",").replace("   ", ",").replace("  ", ",").replace(" ", ",").split(","))
                mtchs_g_points.append(mtch_g_pts)

            mtchs_g_points_float32 = np.asarray(mtchs_g_points, dtype="float32")
            mtchs_g_points_coords = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtchs_g_points_float32]
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = 200,200
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_test = cv.polylines(img_test, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None
            mtchs_g_points_coords = None
        """
        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, flags = flags)
        img_test_draw = cv.drawMatches(img_test, kp1, img_test, kp2, mtchs_g, None, flags = 2)
        img_test_draw = cv.drawMatches(img_test, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        """
        img_test_draw = cv.drawKeypoints(img_test, mtchs_g_points_coords, None, color=(0, 255, 0), flags=flags)
        
        cv.imwrite(save_path+"TEST-{}-algorithm_FLANN_INDEX_KDTREE.png".format(filename), img_test_draw)
        return len(mtchs_g)      

    def algorithm_BFMatcher_NONE(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, dataset, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        img_test = cv.imread(img_2_path, read_type)
        sf = cv.SIFT_create()
        kp1 ,des1 = dataset
        kp2, des2 = sf.detectAndCompute(img_test, None)
        bf = cv.BFMatcher()
        try:
            mtchs = bf.knnMatch(des1, des2, k = 2)
            mtchsMask = [[0, 0] for i in range(len(mtchs))]
            MIN_MATCH_COUNT = MIN_MATCH_COUNT
            mtchs_g = []
            for m, n in mtchs:
                if m.distance < dist*n.distance:
                    mtchs_g.append(m)            
        except:
            mtchs = []
            mtchsMask = [[0, 0] for i in range(len(mtchs))]
            MIN_MATCH_COUNT = MIN_MATCH_COUNT
            mtchs_g = []            

        if len(mtchs_g) > MIN_MATCH_COUNT:
            print("Matches found - {}".format(len(mtchs_g)))
            try:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            except:
                src_pts = None
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in mtchs_g]).reshape(-1, 1, 2)
            mtchs_g_points = []
            for idst_pts in dst_pts:
                mtch_g_pts = (str(idst_pts).replace("[[    ", "").replace("[[   ", "").replace("[[  ", "").replace("[[ ", "").replace("[[", "").replace("    ]]", "").replace("   ]]", "").replace("  ]]", "").replace(" ]]", "").replace("]]", "").replace("      ", ",").replace("     ", ",").replace("    ", ",").replace("   ", ",").replace("  ", ",").replace(" ", ",").split(","))
                mtchs_g_points.append(mtch_g_pts)
            mtchs_g_points_float32 = np.asarray(mtchs_g_points, dtype="float32")
            mtchs_g_points_coords = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtchs_g_points_float32]
            try:
                mm, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
                mtchsMask = mask.ravel().tolist()
                h, w = 200,200
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv.perspectiveTransform(pts, mm)
                img_test = cv.polylines(img_test, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
            except:
                mtchsMask = None
        else:
            print("Not enough matches are found - {}/{}".format(len(mtchs_g), MIN_MATCH_COUNT))
            mtchsMask = None
            mtchs_g_points_coords = None
        """
        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, flags = flags)
        img_test_draw = cv.drawMatches(img_test, kp1, img_test, kp2, mtchs_g, None, flags = 2)
        img_test_draw = cv.drawMatches(img_test, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        """
        img_test_draw = cv.drawKeypoints(img_test, mtchs_g_points_coords, None, color=(255, 255, 0), flags=flags)
        
        cv.imwrite(save_path+"TEST-{}-algorithm_BFMatcher_NONE.png".format(filename), img_test_draw)
        return len(mtchs_g)
    
    def algorithm_BFMatcher_NORM_HAMMING(set,trees, checks, MIN_MATCH_COUNT, dist, flags, read_type, dataset, filename):
        # Class parameters
        save_path = set.save_path
        img_1_path = set.path1
        img_2_path = set.path2
        # Source
        img_bin = cv.imread(img_1_path, read_type)
        img_test = cv.imread(img_2_path, read_type)
        kp1 ,des1 = dataset
        orb = cv.ORB_create()
        kp2 , des2 = orb.detectAndCompute(img_test, None)
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        mtchs = bf.match(des1, des2)
        mtchs = sorted(mtchs, key = lambda x:x.distance)
        mtchsMask = [[0, 0] for i in range(len(mtchs))]
        MIN_MATCH_COUNT = MIN_MATCH_COUNT
        mtchs_g = mtchs
        print("Matches found - {}".format(len(mtchs_g)))
        """
        draw_params = dict(matchColor = (0, 255, 0), singlePointColor = None, flags = flags)
        img_test_draw = cv.drawMatches(img_test, kp1, img_test, kp2, mtchs_g, None, flags = 2)
        img_test_draw = cv.drawMatches(img_test, kp1, img_train, kp2, mtchs_g, None, **draw_params)
        """
        img_test_draw = cv.drawMatches(img_bin, kp1, img_test, kp2, mtchs[:0], None, flags = flags)
        
        #img_test_draw = cv.drawKeypoints(img_test, mtchs_g_points_coords, None, color=(0, 255, 0), flags=flags)
        
        cv.imwrite(save_path+"TEST-{}-algorithm_BFMatcher_NORM_HAMMING.png".format(filename), img_test_draw)
        return len(mtchs_g)    
  
class dtc_process:
    def __init__(self, root_path):
        self.root_path = root_path
    def test_images_dataset(set):
        root_path = set.root_path
        face_detect_path = root_path+'face_detector\\img_in\\face_detect.png'
        img_result_path = root_path+'face_detector\\img_result\\'
        img_out_path = root_path+'face_detector\\img_out\\'
        img_resize_path = root_path+'face_detector\\img_resize\\'
        img_split_path = root_path+'face_detector\\img_split\\'
        img_split_in_path = root_path+'face_detector\\img_split\\in\\'
        img_split_out_path = root_path+'face_detector\\img_split\\out\\'
        img_split_out_bin_path = root_path+'face_detector\\img_split\\out\\bin\\'
        img_split_out_train_path = root_path+'face_detector\\img_split\\out\\train\\'
        img_join_path = root_path+'face_detector\\img_join\\'
        img_join_bin_path = root_path+'face_detector\\img_join\\bin\\'
        img_join_train_path = root_path+'face_detector\\img_join\\train\\'
        import_img_path = root_path+'face_detector\\img_in\\dataset\\'
        
        if os.path.exists(face_detect_path):
            pass
        else:
            os.mkdir(face_detect_path)
        if os.path.exists(img_result_path):
            pass
        else:
            os.mkdir(img_result_path)
        if os.path.exists(img_out_path):
            pass
        else:
            os.mkdir(img_out_path)    
        if os.path.exists(img_resize_path):
            pass
        else:
            os.mkdir(img_resize_path)
        if os.path.exists(img_split_path):
            pass
        else:
            os.mkdir(img_split_path)               
        if os.path.exists(img_split_in_path):
            pass
        else:
            os.mkdir(img_split_in_path)
        if os.path.exists(img_split_out_path):
            pass
        else:
            os.mkdir(img_split_out_path)
        if os.path.exists(img_split_out_bin_path):
            pass
        else:
            os.mkdir(img_split_out_bin_path)
        if os.path.exists(img_split_out_train_path):
            pass
        else:
            os.mkdir(img_split_out_train_path)
        if os.path.exists(img_join_path):
            pass
        else:
            os.mkdir(img_join_path)            
        if os.path.exists(img_join_bin_path):
            pass
        else:
            os.mkdir(img_join_bin_path)
        if os.path.exists(img_join_train_path):
            pass
        else:
            os.mkdir(img_join_train_path)
        if os.path.exists(import_img_path):
            pass
        else:
            os.mkdir(import_img_path)
            
        # Parameters data path
        param_02k_0_path = root_path+'face_detector\\parameters\\param_0.2k_0.json'
        param_02k_18_path = root_path+'face_detector\\parameters\\param_0.2k_18.json'
        param_02k_19_path = root_path+'face_detector\\parameters\\param_0.2k_19.json'
        param_02k_128_path = root_path+'face_detector\\parameters\\param_0.2k_128.json'
        param_025k_0_path = root_path+'face_detector\\parameters\\param_0.25k_0.json'
        param_025k_18_path = root_path+'face_detector\\parameters\\param_0.25k_18.json'
        param_025k_19_path = root_path+'face_detector\\parameters\\param_0.25k_19.json'
        param_025k_128_path = root_path+'face_detector\\parameters\\param_0.25k_128.json'
        
        # Parameters data variables
        param_02k_0_data = []
        param_02k_18_data = []
        param_02k_19_data = []
        param_02k_128_data = []
        param_025k_0_data = []
        param_025k_18_data = []
        param_025k_19_data = []
        param_025k_128_data = []
        
        # Loading dataset type parameters
        if os.path.exists(param_02k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_0_data.append(data)
        else:
            print("Error: {0}".format(param_02k_0_path))
        if os.path.exists(param_02k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_18_data.append(data)
        else:
            print("Error: {0}".format(param_02k_18_path))
        if os.path.exists(param_02k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_19_data.append(data)
        else:
            print("Error: {0}".format(param_02k_19_path))
        if os.path.exists(param_02k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_128_data.append(data)
        else:
            print("Error: {0}".format(param_02k_128_path))
        if os.path.exists(param_025k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_0_data.append(data)
        else:
            print("Error: {0}".format(param_025k_0_path))
        if os.path.exists(param_025k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_18_data.append(data)
        else:
            print("Error: {0}".format(param_025k_18_path))
        if os.path.exists(param_025k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_19_data.append(data)
        else:
            print("Error: {0}".format(param_025k_19_path))
        if os.path.exists(param_025k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_128_data.append(data)
        else:
            print("Error: {0}".format(param_025k_128_path))            
        
        # List result
        result_list = []
        
        # Import faces datasets
        img_paths = os.listdir(import_img_path)
        
        for img_filename in img_paths:
            print("Train image process for: {}".format(img_filename))
            # Default file paths
            img_ext = ".png"
            img_in_path1 = face_detect_path
            img_in_path2 = import_img_path+img_filename
            f = img_in_path1, img_in_path2, img_result_path
           
            # Train parameters
            train_steps = 1
            # LSH abd KDTREE
            p1 = (3, 25, 2, 0.683, 2000)
            # BF Matcher
            p2 = (1, 9, 2, 0.590, 2000)
            # GRAY SCALE
            rt = 19
            # GRAY
            #rt = 18
            # COLOR
            #rt = 19
            
            # Test parameters
            test_steps = 2
            # LSH abd KDTREE
            p3 = (3, 25, 2, 0.683, 2000)
            # BF Matcher
            p4 = (1, 9, 2, 0.590, 2000)
            
            # Resize param
            resize_param = "0.2k"
            #resize_param = "0.25k"
            #resize_param = "0.5k"
            #resize_param = "1k"
            #resize_param = "2k"
            #resize_param = "3k"
            #resize_param = "4k"
            #resize_param = "8k"
            
            if resize_param == "0.2k" and rt == 0:
                p = param_02k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 18:
                p = param_02k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 19:
                p = param_02k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 128:
                p = param_02k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())             
            elif resize_param == "0.25k" and rt == 0:
                p = param_025k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 18:
                p = param_025k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 19:
                p = param_025k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 128:
                p = param_025k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            
            # Detected matches as dtc
            # FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32
            global mtchs_g_dlist_des_FLOAT32
            global mtchs_g_list_kp_FLOAT32
            global mtchs_g_list_des_FLOAT32
            mtchs_g_dlist_kp_FLOAT32 = []
            mtchs_g_dlist_des_FLOAT32 = []
            mtchs_g_list_kp_FLOAT32 = []
            mtchs_g_list_des_FLOAT32 = []
            mtchs_g_list_kp_test_FLOAT32 = []
            mtchs_g_list_des_test_FLOAT32 = []
        
            # FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8
            global mtchs_g_dlist_des_UINT8
            global mtchs_g_list_kp_UINT8
            global mtchs_g_list_des_UINT8
            mtchs_g_dlist_kp_UINT8 = []
            mtchs_g_dlist_des_UINT8 = []
            mtchs_g_list_kp_UINT8 = []
            mtchs_g_list_des_UINT8 = []
            mtchs_g_list_kp_test_UINT8 = []
            mtchs_g_list_des_test_UINT8 = []    
            
            # Is exist path process
            if os.path.exists(img_in_path1):
                pass
            else:
                print("Image path 1 is not exist")
            if os.path.exists(img_in_path2):
                pass
            else:
                print("Image path 2 is not exist")
        
            if resize_param == "0.2k":
                # Resize process
                resize_width = 200
                resize_height = 200
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.25k":
                # Resize process
                resize_width = 250
                resize_height = 250
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.5k":
                # Resize process
                resize_width = 500
                resize_height = 500
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]  
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]       
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "1k":
                # Resize process
                resize_width = 1000
                resize_height = 1000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "4k":
                # Resize process
                resize_width = 4000
                resize_height = 4000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"] 
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]   
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "8k":
                # Resize process
                resize_width = 8000
                resize_height = 8000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]         
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()        
                
            
            face_type_list = "forehead","left_eye","right_eye","eyes","nose","mouth_chin"
            #face_type_list = "forehead"
            dt = "0"
            
            # Face detector
            for i in range(train_steps):
                print("#################### TRAIN Steps: {} ####################".format(i+1))
                
                # Scan method
                scan_method = "FLOAT32"
                
                if type(face_type_list) is str:
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list()
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list()            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type_list))
                    ratio = p2[3]+float((i/1000))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list()
                else:
                    for face_type in face_type_list:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type+"-full"+img_ext
                        # Compare - Face type                
                        # algorithm_FLANN_INDEX_LSH
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list()
                        # algorithm_FLANN_INDEX_KDTREE
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list()                
                        # algorithm_BFMatcher_NONE
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                        ratio = p2[3]+float((i/1000))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list()
        
                # Scan method
                scan_method = "UINT8"
                    
                if resize_param == "0.25k" or resize_param == "0.2k":
                    pass
                else:        
                    if type(face_type_list) is str:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(rt,face_type_list)
                        analysis_train(dt, scan_method).add_to_list()
                    else:
                        for face_type in face_type_list:
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(rt,face_type)
                            analysis_train(dt, scan_method).add_to_list()
                    
            
            print("Total unique matches found: {}".format(len(dt)))
        
            # FORMAT FLOAT 32        
            for idst in mtchs_g_list_kp_FLOAT32:
                mtch_g_kp = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                mtchs_g_list_kp_test_FLOAT32.append(mtch_g_kp)
            for idst in mtchs_g_list_des_FLOAT32:
                mtch_g_des = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                mtchs_g_list_des_test_FLOAT32.append(mtch_g_des)
            mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp_test_FLOAT32, dtype="float32")
            mtch_g_des_float32 = np.asarray(mtchs_g_list_des_test_FLOAT32, dtype="float32")        
            format_FLOAT32 = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtch_g_kp_float32], mtch_g_des_float32    
            dataset_mtchs = format_FLOAT32
            
            # Test matches
            test_mtchs = 0
            
            # Compare train dataset type 1
            for i in range(test_steps):
                print("#################### FORMAT FLOAT32 TEST Steps: {} ####################".format(i+1))
                for face_type in face_type_list:
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {}  ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {}  ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt            
                # Compare - Resized image method 1
                # algorithm_FLANN_INDEX_LSH
                print("Face type: Full Resized ### Algorithm: algorithm_FLANN_INDEX_LSH")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt
                # algorithm_FLANN_INDEX_KDTREE
                print("Face type: Full Resized ### Algorithm: algorithm_FLANN_INDEX_KDTREE")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt        
                # algorithm_BFMatcher_NONE
                print("Face type: Full Resized ### Algorithm: algorithm_BFMatcher_NONE")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt
                # Compare - Original image method 1
                # algorithm_FLANN_INDEX_LSH
                print("Face type: Full Original ### Algorithm: algorithm_FLANN_INDEX_LSH")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt
                # algorithm_FLANN_INDEX_KDTREE
                print("Face type: Full Original ### Algorithm: algorithm_FLANN_INDEX_KDTREE")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt        
                # algorithm_BFMatcher_NONE
                print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NONE")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt        
        
            # FORMAT UINT8
            for idst in mtchs_g_list_kp_UINT8:
                mtch_g_kp = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                mtchs_g_list_kp_test_UINT8.append(mtch_g_kp)
            for idst in mtchs_g_list_des_UINT8:
                mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                mtchs_g_list_des_test_UINT8.append(mtch_g_des)
            mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp_test_UINT8, dtype="float32")
            mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
            format_UNIT8 = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtch_g_kp_float32], mtch_g_des_uint8
            dataset_mtchs = format_UNIT8
        
            if resize_param == "0.25k" or resize_param == "0.2k":
                pass
            else:
                # Compare train dataset type 2
                for i in range(test_steps):
                    print("#################### FORMAT UINT8 TEST Steps: {} ####################".format(i+1))
                    if type(face_type_list) is str:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type_list)
                        #test_mtchs+=dt            
                    else:
                        for face_type in face_type_list:
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                            #test_mtchs+=dt
                    # Compare - Resized image
                    # algorithm_BFMatcher_NORM_HAMMING
                    print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NORM_HAMMING")
                    dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                    #test_mtchs+=dt
                    # Compare - Original image
                    # algorithm_BFMatcher_NORM_HAMMING
                    print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NORM_HAMMING")
                    dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                    #test_mtchs+=dt        
        
            # Result data
            result_data = os.listdir(img_result_path)
            # Send detected data matches to img_out
            for match_name in result_data:
                old_path = img_result_path+match_name
                new_path = img_out_path+"\\{0}\\{0}-{1}".format(img_filename, match_name)
                new_path_dir = img_out_path+"\\{0}\\".format(img_filename)
                if os.path.exists(new_path_dir):
                    sh.copyfile(old_path, new_path)
                else:
                    os.mkdir(new_path_dir)
                    sh.copyfile(old_path, new_path)
            # Remove detected data matches
            for match_name in result_data:
                old_path = img_result_path+match_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            # Remove detected split data
            split_bin_data = os.listdir(img_split_out_bin_path)
            split_train_data = os.listdir(img_split_out_train_path)
            for split_name in split_bin_data:
                old_path = img_split_out_bin_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            for split_name in split_train_data:
                old_path = img_split_out_train_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))            
        
            if test_mtchs > 70:
                print("--- TEST SUCCESS! ---")
                print("Total matches count: {}".format(test_mtchs))
                """
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                
                # SAVE DATASET
                if open(fp):
                    fo = open(fp, "w")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                else:
                    fo = open(fp, "xw")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                
                
                # OPEN DATASET
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                print("Analysis dataset path: {}".format(fp))
                fo = open(fp, "r")
                fr = fo.read().replace("\\n", ",")
                dst = json.loads(fr.replace("'", "\""))
                fo.close()
                print("Dataset matches count: {}".format(len(dst)))
                
                mtchs_g_list_kp = []
                mtchs_g_list_des = []
                for idst in dst:
                    mtch_g_kp = ("["+str(idst["k"].replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                    mtch_g_des = idst["d"].replace("[", "").replace("]", "").split(",")
                    mtchs_g_list_kp.append(mtch_g_kp)
                    mtchs_g_list_des.append(mtch_g_des)
                    
                mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp, dtype="float32")
                mtch_g_des_float32 = np.asarray(mtchs_g_list_des, dtype="float32")    
                  
                """    
                
                
            else:
                print("--- TEST FAILED! ---")
                print("Total matches count: {}".format(test_mtchs))
            
            # Add to result List
            result_list.append(("Name: {0} Result: {1} pts".format(img_filename, test_mtchs)))
        
        print("\n")
        print("Result list:")
        for result in result_list:
            print(result)
    def create_new_dataset(set, img_filename, resize_type, read_type):
        root_path = set.root_path
        face_detect_path = root_path+'face_detector\\img_in\\face_detect.png'
        img_result_path = root_path+'face_detector\\img_result\\'
        img_out_path = root_path+'face_detector\\img_out\\'
        img_resize_path = root_path+'face_detector\\img_resize\\'
        img_split_in_path = root_path+'face_detector\\img_split\\in\\'
        img_split_out_path = root_path+'face_detector\\img_split\\out\\'
        img_split_out_bin_path = root_path+'face_detector\\img_split\\out\\bin\\'
        img_split_out_train_path = root_path+'face_detector\\img_split\\out\\train\\'
        img_join_bin_path = root_path+'face_detector\\img_join\\bin\\'
        img_join_train_path = root_path+'face_detector\\img_join\\train\\'
        import_img_path = root_path+'face_detector\\img_in\\dataset\\'
        datasets_path = root_path+'face_detector\\datasets\\'
        filename_dataset_path = root_path+'face_detector\\datasets\\{0}\\'.format(img_filename)
        
        # Validation paths exists
        if os.path.exists(face_detect_path):
            pass
        else:
            os.mkdir(face_detect_path)
        if os.path.exists(img_result_path):
            pass
        else:
            os.mkdir(img_result_path)
        if os.path.exists(img_out_path):
            pass
        else:
            os.mkdir(img_out_path)    
        if os.path.exists(img_resize_path):
            pass
        else:
            os.mkdir(img_resize_path)
        if os.path.exists(img_split_in_path):
            pass
        else:
            os.mkdir(img_split_in_path)
        if os.path.exists(img_split_out_path):
            pass
        else:
            os.mkdir(img_split_out_path)
        if os.path.exists(img_split_out_bin_path):
            pass
        else:
            os.mkdir(img_split_out_bin_path)
        if os.path.exists(img_split_out_train_path):
            pass
        else:
            os.mkdir(img_split_out_train_path)
        if os.path.exists(img_join_bin_path):
            pass
        else:
            os.mkdir(img_join_bin_path)
        if os.path.exists(img_join_train_path):
            pass
        else:
            os.mkdir(img_join_train_path)
        if os.path.exists(import_img_path):
            pass
        else:
            os.mkdir(import_img_path)
        if os.path.exists(datasets_path):
            pass
        else:
            os.mkdir(datasets_path)
        if os.path.exists(filename_dataset_path):
            pass
        else:
            os.mkdir(filename_dataset_path)
          
        # Parameters data path
        param_02k_0_path = root_path+'face_detector\\parameters\\param_0.2k_0.json'
        param_02k_18_path = root_path+'face_detector\\parameters\\param_0.2k_18.json'
        param_02k_19_path = root_path+'face_detector\\parameters\\param_0.2k_19.json'
        param_02k_128_path = root_path+'face_detector\\parameters\\param_0.2k_128.json'
        param_025k_0_path = root_path+'face_detector\\parameters\\param_0.25k_0.json'
        param_025k_18_path = root_path+'face_detector\\parameters\\param_0.25k_18.json'
        param_025k_19_path = root_path+'face_detector\\parameters\\param_0.25k_19.json'
        param_025k_128_path = root_path+'face_detector\\parameters\\param_0.25k_128.json'
        
        # Parameters data variables
        param_02k_0_data = []
        param_02k_18_data = []
        param_02k_19_data = []
        param_02k_128_data = []
        param_025k_0_data = []
        param_025k_18_data = []
        param_025k_19_data = []
        param_025k_128_data = []
        
        # Loading dataset type parameters
        if os.path.exists(param_02k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_0_data.append(data)
        else:
            print("Error: {0}".format(param_02k_0_path))
        if os.path.exists(param_02k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_18_data.append(data)
        else:
            print("Error: {0}".format(param_02k_18_path))
        if os.path.exists(param_02k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_19_data.append(data)
        else:
            print("Error: {0}".format(param_02k_19_path))
        if os.path.exists(param_02k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_128_data.append(data)
        else:
            print("Error: {0}".format(param_02k_128_path))
        if os.path.exists(param_025k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_0_data.append(data)
        else:
            print("Error: {0}".format(param_025k_0_path))
        if os.path.exists(param_025k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_18_data.append(data)
        else:
            print("Error: {0}".format(param_025k_18_path))
        if os.path.exists(param_025k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_19_data.append(data)
        else:
            print("Error: {0}".format(param_025k_19_path))
        if os.path.exists(param_025k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_128_data.append(data)
        else:
            print("Error: {0}".format(param_025k_128_path))
          
        
        # List result
        result_list = []
        
        for face_detect_process in range(1):
            print("Create dataset: {}".format(face_detect_path))
            # Default file paths
            img_ext = ".png"
            img_in_path1 = face_detect_path
            img_in_path2 = face_detect_path
            f = img_in_path1, img_in_path2, img_result_path
           
            # Read tyoe
            rt = read_type
            
            # Resize param
            resize_param = resize_type

            if resize_param == "0.2k" and rt == 0:
                p = param_02k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 18:
                p = param_02k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 19:
                p = param_02k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 128:
                p = param_02k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())             
            elif resize_param == "0.25k" and rt == 0:
                p = param_025k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 18:
                p = param_025k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 19:
                p = param_025k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 128:
                p = param_025k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            
            # Detected matches as dtc
            # FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32
            global mtchs_g_dlist_des_FLOAT32
            global mtchs_g_list_kp_FLOAT32
            global mtchs_g_list_des_FLOAT32
            mtchs_g_dlist_kp_FLOAT32 = []
            mtchs_g_dlist_des_FLOAT32 = []
            mtchs_g_list_kp_FLOAT32 = []
            mtchs_g_list_des_FLOAT32 = []
            mtchs_g_list_kp_test_FLOAT32 = []
            mtchs_g_list_des_test_FLOAT32 = []
        
            # FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8
            global mtchs_g_dlist_des_UINT8
            global mtchs_g_list_kp_UINT8
            global mtchs_g_list_des_UINT8
            mtchs_g_dlist_kp_UINT8 = []
            mtchs_g_dlist_des_UINT8 = []
            mtchs_g_list_kp_UINT8 = []
            mtchs_g_list_des_UINT8 = []
            mtchs_g_list_kp_test_UINT8 = []
            mtchs_g_list_des_test_UINT8 = []
            
            # Face types format matches variables FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32_forehead
            global mtchs_g_dlist_kp_FLOAT32_left_eye
            global mtchs_g_dlist_kp_FLOAT32_right_eye
            global mtchs_g_dlist_kp_FLOAT32_eyes
            global mtchs_g_dlist_kp_FLOAT32_nose
            global mtchs_g_dlist_kp_FLOAT32_mouth_chin
            global mtchs_g_dlist_des_FLOAT32_forehead
            global mtchs_g_dlist_des_FLOAT32_left_eye
            global mtchs_g_dlist_des_FLOAT32_right_eye
            global mtchs_g_dlist_des_FLOAT32_eyes
            global mtchs_g_dlist_des_FLOAT32_nose
            global mtchs_g_dlist_des_FLOAT32_mouth_chin

            global mtchs_g_list_kp_FLOAT32_forehead
            global mtchs_g_list_kp_FLOAT32_left_eye
            global mtchs_g_list_kp_FLOAT32_right_eye
            global mtchs_g_list_kp_FLOAT32_eyes
            global mtchs_g_list_kp_FLOAT32_nose
            global mtchs_g_list_kp_FLOAT32_mouth_chin
            global mtchs_g_list_des_FLOAT32_forehead
            global mtchs_g_list_des_FLOAT32_left_eye
            global mtchs_g_list_des_FLOAT32_right_eye
            global mtchs_g_list_des_FLOAT32_eyes
            global mtchs_g_list_des_FLOAT32_nose
            global mtchs_g_list_des_FLOAT32_mouth_chin

            mtchs_g_dlist_kp_FLOAT32_forehead = []
            mtchs_g_dlist_kp_FLOAT32_left_eye = []
            mtchs_g_dlist_kp_FLOAT32_right_eye = []
            mtchs_g_dlist_kp_FLOAT32_eyes = []
            mtchs_g_dlist_kp_FLOAT32_nose = []
            mtchs_g_dlist_kp_FLOAT32_mouth_chin = []
            mtchs_g_dlist_des_FLOAT32_forehead = []
            mtchs_g_dlist_des_FLOAT32_left_eye = []
            mtchs_g_dlist_des_FLOAT32_right_eye = []
            mtchs_g_dlist_des_FLOAT32_eyes = []
            mtchs_g_dlist_des_FLOAT32_nose = []
            mtchs_g_dlist_des_FLOAT32_mouth_chin = []

            mtchs_g_list_kp_FLOAT32_forehead = []
            mtchs_g_list_kp_FLOAT32_left_eye = []
            mtchs_g_list_kp_FLOAT32_right_eye = []
            mtchs_g_list_kp_FLOAT32_eyes = []
            mtchs_g_list_kp_FLOAT32_nose = []
            mtchs_g_list_kp_FLOAT32_mouth_chin = []
            mtchs_g_list_des_FLOAT32_forehead = []
            mtchs_g_list_des_FLOAT32_left_eye = []
            mtchs_g_list_des_FLOAT32_right_eye = []
            mtchs_g_list_des_FLOAT32_eyes = []
            mtchs_g_list_des_FLOAT32_nose = []
            mtchs_g_list_des_FLOAT32_mouth_chin = []            
            
            # Face types format matches variables FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8_forehead
            global mtchs_g_dlist_kp_UINT8_left_eye
            global mtchs_g_dlist_kp_UINT8_right_eye
            global mtchs_g_dlist_kp_UINT8_eyes
            global mtchs_g_dlist_kp_UINT8_nose
            global mtchs_g_dlist_kp_UINT8_mouth_chin
            global mtchs_g_dlist_des_UINT8_forehead
            global mtchs_g_dlist_des_UINT8_left_eye
            global mtchs_g_dlist_des_UINT8_right_eye
            global mtchs_g_dlist_des_UINT8_eyes
            global mtchs_g_dlist_des_UINT8_nose
            global mtchs_g_dlist_des_UINT8_mouth_chin

            global mtchs_g_list_kp_UINT8_forehead
            global mtchs_g_list_kp_UINT8_left_eye
            global mtchs_g_list_kp_UINT8_right_eye
            global mtchs_g_list_kp_UINT8_eyes
            global mtchs_g_list_kp_UINT8_nose
            global mtchs_g_list_kp_UINT8_mouth_chin
            global mtchs_g_list_des_UINT8_forehead
            global mtchs_g_list_des_UINT8_left_eye
            global mtchs_g_list_des_UINT8_right_eye
            global mtchs_g_list_des_UINT8_eyes
            global mtchs_g_list_des_UINT8_nose
            global mtchs_g_list_des_UINT8_mouth_chin

            mtchs_g_dlist_kp_UINT8_forehead = []
            mtchs_g_dlist_kp_UINT8_left_eye = []
            mtchs_g_dlist_kp_UINT8_right_eye = []
            mtchs_g_dlist_kp_UINT8_eyes = []
            mtchs_g_dlist_kp_UINT8_nose = []
            mtchs_g_dlist_kp_UINT8_mouth_chin = []
            mtchs_g_dlist_des_UINT8_forehead = []
            mtchs_g_dlist_des_UINT8_left_eye = []
            mtchs_g_dlist_des_UINT8_right_eye = []
            mtchs_g_dlist_des_UINT8_eyes = []
            mtchs_g_dlist_des_UINT8_nose = []
            mtchs_g_dlist_des_UINT8_mouth_chin = []

            mtchs_g_list_kp_UINT8_forehead = []
            mtchs_g_list_kp_UINT8_left_eye = []
            mtchs_g_list_kp_UINT8_right_eye = []
            mtchs_g_list_kp_UINT8_eyes = []
            mtchs_g_list_kp_UINT8_nose = []
            mtchs_g_list_kp_UINT8_mouth_chin = []
            mtchs_g_list_des_UINT8_forehead = []
            mtchs_g_list_des_UINT8_left_eye = []
            mtchs_g_list_des_UINT8_right_eye = []
            mtchs_g_list_des_UINT8_eyes = []
            mtchs_g_list_des_UINT8_nose = []
            mtchs_g_list_des_UINT8_mouth_chin = []
            
            # Is exist path process
            if os.path.exists(img_in_path1):
                pass
            else:
                print("Image path 1 is not exist")
            if os.path.exists(img_in_path2):
                pass
            else:
                print("Image path 2 is not exist")
        
            if resize_param == "0.2k":
                # Resize process
                resize_width = 200
                resize_height = 200
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.25k":
                # Resize process
                resize_width = 250
                resize_height = 250
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.5k":
                # Resize process
                resize_width = 500
                resize_height = 500
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]  
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]       
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "1k":
                # Resize process
                resize_width = 1000
                resize_height = 1000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "4k":
                # Resize process
                resize_width = 4000
                resize_height = 4000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"] 
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]   
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "8k":
                # Resize process
                resize_width = 8000
                resize_height = 8000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]         
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()        
                
            
            face_type_list = "forehead","left_eye","right_eye","eyes","nose","mouth_chin"
            #face_type_list = "forehead"
            dt = "0"
            
            # Face detector
            for i in range(train_steps):
                print("#################### TRAIN Steps: {} ####################".format(i+1))
                
                # Scan method
                scan_method = "FLOAT32"
                
                if type(face_type_list) is str:
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type(face_type_list)
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type(face_type_list)            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type_list))
                    ratio = p2[3]+float((i/1000))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type(face_type_list)
                else:
                    for face_type in face_type_list:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type+"-full"+img_ext
                        # Compare - Face type                
                        # algorithm_FLANN_INDEX_LSH
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type(face_type)
                        # algorithm_FLANN_INDEX_KDTREE
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type(face_type)                
                        # algorithm_BFMatcher_NONE
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                        ratio = p2[3]+float((i/1000))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type(face_type)
        
                # Scan method
                scan_method = "UINT8"
                    
                if resize_param == "0.25k" or resize_param == "0.2k":
                    pass
                else:        
                    if type(face_type_list) is str:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(rt,face_type_list)
                        analysis_train(dt, scan_method).add_to_list_by_face_type(face_type_list)
                    else:
                        for face_type in face_type_list:
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_train(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(rt,face_type)
                            analysis_train(dt, scan_method).add_to_list_by_face_type(face_type)
                    
            
            print("Total unique matches found: {}".format(len(dt)))
        
            # Test matches
            test_mtchs = 0
            
            # Patch identificator name
            patch_id = "{0}-{1}".format(resize_param, rt)
            
            # Compare train dataset type 1
            for i in range(test_steps):
                print("#################### FORMAT FLOAT32 TEST Steps: {} ####################".format(i+1))
                for face_type in face_type_list:
                    # Face type distributor matches list
                    if face_type == "forehead":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_forehead
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_forehead
                    elif face_type == "left_eye":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_left_eye
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_left_eye
                    elif face_type == "right_eye":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_right_eye
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_right_eye
                    elif face_type == "eyes":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_eyes
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_eyes
                    elif face_type == "nose":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_nose
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_nose
                    elif face_type == "mouth_chin":
                        mtchs_g_list_kp_FLOAT32 = mtchs_g_list_kp_FLOAT32_mouth_chin
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_mouth_chin
                    # Save face type dataset    
                    ftp = filename_dataset_path+patch_id
                    fp_kp = ftp+"\\FLOAT32-kp-"+face_type+".txt"
                    fp_des = ftp+"\\FLOAT32-des-"+face_type+".txt"
                    # Create resize param dir
                    if os.path.exists(ftp):
                        pass
                    else:
                        os.mkdir(ftp) 
                    # Create kp file
                    if os.path.exists(fp_kp):
                        fo = open(fp_kp, "w")
                        fo.write(str(mtchs_g_list_kp_FLOAT32[0]))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_kp))
                    else:
                        fo = open(fp_kp, "x")
                        fo.write(str(mtchs_g_list_kp_FLOAT32[0]))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_kp))
                    # Create des file
                    if os.path.exists(fp_des):
                        fo = open(fp_des, "w")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))
                    else:
                        fo = open(fp_des, "x")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))                         
                    # FORMAT FLOAT 32
                    for idst in mtchs_g_list_kp_FLOAT32[0]:
                        mtch_g_kp = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                        mtchs_g_list_kp_test_FLOAT32.append(mtch_g_kp)
                    for idst in mtchs_g_list_des_FLOAT32[0]:
                        mtch_g_des = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                        mtchs_g_list_des_test_FLOAT32.append(mtch_g_des)
                    mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp_test_FLOAT32, dtype="float32")
                    mtch_g_des_float32 = np.asarray(mtchs_g_list_des_test_FLOAT32, dtype="float32")
                    format_FLOAT32 = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtch_g_kp_float32], mtch_g_des_float32
                    dataset_mtchs = format_FLOAT32
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {}  ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {}  ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                    dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, face_type)
                    test_mtchs+=dt            
                # Compare - Resized image method 1
                # algorithm_FLANN_INDEX_LSH
                print("Face type: Full Resized ### Algorithm: algorithm_FLANN_INDEX_LSH")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt
                # algorithm_FLANN_INDEX_KDTREE
                print("Face type: Full Resized ### Algorithm: algorithm_FLANN_INDEX_KDTREE")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt        
                # algorithm_BFMatcher_NONE
                print("Face type: Full Resized ### Algorithm: algorithm_BFMatcher_NONE")
                dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, "resized")
                test_mtchs+=dt
                # Compare - Original image method 1
                # algorithm_FLANN_INDEX_LSH
                print("Face type: Full Original ### Algorithm: algorithm_FLANN_INDEX_LSH")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_FLANN_INDEX_LSH(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt
                # algorithm_FLANN_INDEX_KDTREE
                print("Face type: Full Original ### Algorithm: algorithm_FLANN_INDEX_KDTREE")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_FLANN_INDEX_KDTREE(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt        
                # algorithm_BFMatcher_NONE
                print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NONE")
                dt = dtc_test(f[0], f[1], f[2]).algorithm_BFMatcher_NONE(p4[0],p4[1],p4[2],p4[3],p4[4], rt, dataset_mtchs, "original")
                test_mtchs+=dt        
        
            if resize_param == "0.25k" or resize_param == "0.2k":
                pass
            else:
                # Compare train dataset type 2
                for i in range(test_steps):
                    print("#################### FORMAT UINT8 TEST Steps: {} ####################".format(i+1))
                    if type(face_type_list) is str:
                        # Face type distributor matches list
                        if face_type == "forehead":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_forehead
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                        elif face_type == "left_eye":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_left_eye
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                        elif face_type == "right_eye":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_right_eye
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                        elif face_type == "eyes":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_eyes
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                        elif face_type == "nose":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_nose
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                        elif face_type == "mouth_chin":
                            mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_mouth_chin
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                        # Save face type dataset    
                        ftp = filename_dataset_path+patch_id
                        fp_kp = ftp+"\\UINT8-kp-"+face_type+".txt"
                        fp_des = ftp+"\\UINT8-des-"+face_type+".txt"
                        # Create resize param dir
                        if os.path.exists(ftp):
                            pass
                        else:
                            os.mkdir(ftp) 
                        # Create kp file
                        if os.path.exists(fp_kp):
                            fo = open(fp_kp, "w")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        else:
                            fo = open(fp_kp, "x")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        # Create des file
                        if os.path.exists(fp_des):
                            fo = open(fp_des, "w")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        else:
                            fo = open(fp_des, "x")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        # FORMAT UINT8
                        for idst in mtchs_g_list_kp_UINT8[0]:
                            mtch_g_kp = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                            mtchs_g_list_kp_test_UINT8.append(mtch_g_kp)
                        for idst in mtchs_g_list_des_UINT8[0]:
                            mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                            mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                        mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp_test_UINT8, dtype="float32")
                        mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
                        format_UNIT8 = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtch_g_kp_float32], mtch_g_des_uint8
                        dataset_mtchs = format_UNIT8                        
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type_list)
                        #test_mtchs+=dt            
                    else:
                        for face_type in face_type_list:
                            # Face type distributor matches list
                            if face_type == "forehead":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_forehead
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                            elif face_type == "left_eye":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_left_eye
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                            elif face_type == "right_eye":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_right_eye
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                            elif face_type == "eyes":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_eyes
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                            elif face_type == "nose":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_nose
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                            elif face_type == "mouth_chin":
                                mtchs_g_list_kp_UINT8 = mtchs_g_list_kp_UINT8_mouth_chin
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                            # Save face type dataset    
                            ftp = filename_dataset_path+patch_id
                            fp_kp = ftp+"\\"+scan_method+"-kp"+"-"+face_type+".txt"
                            fp_des = ftp+"\\"+scan_method+"-des"+"-"+face_type+".txt"
                            # Create resize param dir
                            if os.path.exists(ftp):
                                pass
                            else:
                                os.mkdir(ftp)
                            # Create kp file
                            if os.path.exists(fp_kp):
                                fo = open(fp_kp, "w")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            else:
                                fo = open(fp_kp, "x")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            # Create des file
                            if os.path.exists(fp_des):
                                fo = open(fp_des, "w")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))
                            else:
                                fo = open(fp_des, "x")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))                                
                            # FORMAT UINT8
                            for idst in mtchs_g_list_kp_UINT8[0]:
                                mtch_g_kp = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                                mtchs_g_list_kp_test_UINT8.append(mtch_g_kp)
                            for idst in mtchs_g_list_des_UINT8[0]:
                                mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                                mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                            mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp_test_UINT8, dtype="float32")
                            mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
                            format_UNIT8 = [cv.KeyPoint(coord[0], coord[1], 2) for coord in mtch_g_kp_float32], mtch_g_des_uint8
                            dataset_mtchs = format_UNIT8
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {}  ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_test(face_bin_path, face_train_path, f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, face_type)
                            #test_mtchs+=dt
                    # Compare - Resized image
                    # algorithm_BFMatcher_NORM_HAMMING
                    print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NORM_HAMMING")
                    dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                    #test_mtchs+=dt
                    # Compare - Original image
                    # algorithm_BFMatcher_NORM_HAMMING
                    print("Face type: Full Original ### Algorithm: algorithm_BFMatcher_NORM_HAMMING")
                    dt = dtc_test((img_split_in_path+"resize_bin.png"), img_split_in_path+"resize_train.png", f[2]).algorithm_BFMatcher_NORM_HAMMING(p3[0],p3[1],p3[2],p3[3],p3[4], rt, dataset_mtchs, "resized")
                    #test_mtchs+=dt        
        
            # Result data
            result_data = os.listdir(img_result_path)
            # Send detected data matches to img_out
            for match_name in result_data:
                old_path = img_result_path+match_name
                new_path = img_out_path+"\\{0}\\{0}-{1}".format(img_filename, match_name)
                new_path_dir = img_out_path+"\\{0}\\".format(img_filename)
                if os.path.exists(new_path_dir):
                    sh.copyfile(old_path, new_path)
                else:
                    os.mkdir(new_path_dir)
                    sh.copyfile(old_path, new_path)
            # Remove detected data matches
            for match_name in result_data:
                old_path = img_result_path+match_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            # Remove detected split data
            split_bin_data = os.listdir(img_split_out_bin_path)
            split_train_data = os.listdir(img_split_out_train_path)
            for split_name in split_bin_data:
                old_path = img_split_out_bin_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            for split_name in split_train_data:
                old_path = img_split_out_train_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))            
        
            if test_mtchs > 70:
                print("--- TEST SUCCESS! ---")
                print("Total matches count: {}".format(test_mtchs))
                """
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                
                # SAVE DATASET
                if open(fp):
                    fo = open(fp, "w")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                else:
                    fo = open(fp, "xw")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                
                
                # OPEN DATASET
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                print("Analysis dataset path: {}".format(fp))
                fo = open(fp, "r")
                fr = fo.read().replace("\\n", ",")
                dst = json.loads(fr.replace("'", "\""))
                fo.close()
                print("Dataset matches count: {}".format(len(dst)))
                
                mtchs_g_list_kp = []
                mtchs_g_list_des = []
                for idst in dst:
                    mtch_g_kp = ("["+str(idst["k"].replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                    mtch_g_des = idst["d"].replace("[", "").replace("]", "").split(",")
                    mtchs_g_list_kp.append(mtch_g_kp)
                    mtchs_g_list_des.append(mtch_g_des)
                    
                mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp, dtype="float32")
                mtch_g_des_float32 = np.asarray(mtchs_g_list_des, dtype="float32")    
                  
                """    
                
                
            else:
                print("--- TEST FAILED! ---")
                print("Total matches count: {}".format(test_mtchs))
            
            # Add to result List
            result_list.append(("Name: {0} Result: {1} pts".format(img_filename, test_mtchs)))
        
        print("\n")
        print("Result list:")
        for result in result_list:
            print(result)
    def create_new_dataset_only_des(set, img_filename, resize_type, read_type):
        root_path = set.root_path
        face_detect_path = root_path+'face_detector\\img_in\\face_detect.png'
        img_result_path = root_path+'face_detector\\img_result\\'
        img_out_path = root_path+'face_detector\\img_out\\'
        img_resize_path = root_path+'face_detector\\img_resize\\'
        img_split_in_path = root_path+'face_detector\\img_split\\in\\'
        img_split_out_path = root_path+'face_detector\\img_split\\out\\'
        img_split_out_bin_path = root_path+'face_detector\\img_split\\out\\bin\\'
        img_split_out_train_path = root_path+'face_detector\\img_split\\out\\train\\'
        img_join_bin_path = root_path+'face_detector\\img_join\\bin\\'
        img_join_train_path = root_path+'face_detector\\img_join\\train\\'
        import_img_path = root_path+'face_detector\\img_in\\dataset\\'
        datasets_path = root_path+'face_detector\\datasets\\'
        filename_dataset_path = root_path+'face_detector\\datasets\\{0}\\'.format(img_filename)
        
        # Validation paths exists
        if os.path.exists(face_detect_path):
            pass
        else:
            os.mkdir(face_detect_path)
        if os.path.exists(img_result_path):
            pass
        else:
            os.mkdir(img_result_path)
        if os.path.exists(img_out_path):
            pass
        else:
            os.mkdir(img_out_path)    
        if os.path.exists(img_resize_path):
            pass
        else:
            os.mkdir(img_resize_path)
        if os.path.exists(img_split_in_path):
            pass
        else:
            os.mkdir(img_split_in_path)
        if os.path.exists(img_split_out_path):
            pass
        else:
            os.mkdir(img_split_out_path)
        if os.path.exists(img_split_out_bin_path):
            pass
        else:
            os.mkdir(img_split_out_bin_path)
        if os.path.exists(img_split_out_train_path):
            pass
        else:
            os.mkdir(img_split_out_train_path)
        if os.path.exists(img_join_bin_path):
            pass
        else:
            os.mkdir(img_join_bin_path)
        if os.path.exists(img_join_train_path):
            pass
        else:
            os.mkdir(img_join_train_path)
        if os.path.exists(import_img_path):
            pass
        else:
            os.mkdir(import_img_path)
        if os.path.exists(datasets_path):
            pass
        else:
            os.mkdir(datasets_path)
        if os.path.exists(filename_dataset_path):
            pass
        else:
            os.mkdir(filename_dataset_path)
          
        # Parameters data path
        param_02k_0_path = root_path+'face_detector\\parameters\\param_0.2k_0.json'
        param_02k_18_path = root_path+'face_detector\\parameters\\param_0.2k_18.json'
        param_02k_19_path = root_path+'face_detector\\parameters\\param_0.2k_19.json'
        param_02k_128_path = root_path+'face_detector\\parameters\\param_0.2k_128.json'
        param_025k_0_path = root_path+'face_detector\\parameters\\param_0.25k_0.json'
        param_025k_18_path = root_path+'face_detector\\parameters\\param_0.25k_18.json'
        param_025k_19_path = root_path+'face_detector\\parameters\\param_0.25k_19.json'
        param_025k_128_path = root_path+'face_detector\\parameters\\param_0.25k_128.json'
        
        # Parameters data variables
        param_02k_0_data = []
        param_02k_18_data = []
        param_02k_19_data = []
        param_02k_128_data = []
        param_025k_0_data = []
        param_025k_18_data = []
        param_025k_19_data = []
        param_025k_128_data = []
        
        # Loading dataset type parameters
        if os.path.exists(param_02k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_0_data.append(data)
        else:
            print("Error: {0}".format(param_02k_0_path))
        if os.path.exists(param_02k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_18_data.append(data)
        else:
            print("Error: {0}".format(param_02k_18_path))
        if os.path.exists(param_02k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_19_data.append(data)
        else:
            print("Error: {0}".format(param_02k_19_path))
        if os.path.exists(param_02k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_128_data.append(data)
        else:
            print("Error: {0}".format(param_02k_128_path))
        if os.path.exists(param_025k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_0_data.append(data)
        else:
            print("Error: {0}".format(param_025k_0_path))
        if os.path.exists(param_025k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_18_data.append(data)
        else:
            print("Error: {0}".format(param_025k_18_path))
        if os.path.exists(param_025k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_19_data.append(data)
        else:
            print("Error: {0}".format(param_025k_19_path))
        if os.path.exists(param_025k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_128_data.append(data)
        else:
            print("Error: {0}".format(param_025k_128_path))
          
        
        # List result
        result_list = []
        
        for face_detect_process in range(1):
            print("Create dataset: {}".format(face_detect_path))
            # Default file paths
            img_ext = ".png"
            img_in_path1 = face_detect_path
            img_in_path2 = face_detect_path
            f = img_in_path1, img_in_path2, img_result_path
           
            # Read tyoe
            rt = read_type
            
            # Resize param
            resize_param = resize_type

            if resize_param == "0.2k" and rt == 0:
                p = param_02k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 18:
                p = param_02k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 19:
                p = param_02k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.2k" and rt == 128:
                p = param_02k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())             
            elif resize_param == "0.25k" and rt == 0:
                p = param_025k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 18:
                p = param_025k_18_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 19:
                p = param_025k_19_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            elif resize_param == "0.25k" and rt == 128:
                p = param_025k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            
            # Detected matches as dtc
            # FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32
            global mtchs_g_dlist_des_FLOAT32
            global mtchs_g_list_kp_FLOAT32
            global mtchs_g_list_des_FLOAT32
            mtchs_g_dlist_kp_FLOAT32 = []
            mtchs_g_dlist_des_FLOAT32 = []
            mtchs_g_list_kp_FLOAT32 = []
            mtchs_g_list_des_FLOAT32 = []
            mtchs_g_list_kp_test_FLOAT32 = []
            mtchs_g_list_des_test_FLOAT32 = []
        
            # FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8
            global mtchs_g_dlist_des_UINT8
            global mtchs_g_list_kp_UINT8
            global mtchs_g_list_des_UINT8
            mtchs_g_dlist_kp_UINT8 = []
            mtchs_g_dlist_des_UINT8 = []
            mtchs_g_list_kp_UINT8 = []
            mtchs_g_list_des_UINT8 = []
            mtchs_g_list_kp_test_UINT8 = []
            mtchs_g_list_des_test_UINT8 = []
            
            # Face types format matches variables FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32_forehead
            global mtchs_g_dlist_kp_FLOAT32_left_eye
            global mtchs_g_dlist_kp_FLOAT32_right_eye
            global mtchs_g_dlist_kp_FLOAT32_eyes
            global mtchs_g_dlist_kp_FLOAT32_nose
            global mtchs_g_dlist_kp_FLOAT32_mouth_chin
            global mtchs_g_dlist_des_FLOAT32_forehead
            global mtchs_g_dlist_des_FLOAT32_left_eye
            global mtchs_g_dlist_des_FLOAT32_right_eye
            global mtchs_g_dlist_des_FLOAT32_eyes
            global mtchs_g_dlist_des_FLOAT32_nose
            global mtchs_g_dlist_des_FLOAT32_mouth_chin

            global mtchs_g_list_kp_FLOAT32_forehead
            global mtchs_g_list_kp_FLOAT32_left_eye
            global mtchs_g_list_kp_FLOAT32_right_eye
            global mtchs_g_list_kp_FLOAT32_eyes
            global mtchs_g_list_kp_FLOAT32_nose
            global mtchs_g_list_kp_FLOAT32_mouth_chin
            global mtchs_g_list_des_FLOAT32_forehead
            global mtchs_g_list_des_FLOAT32_left_eye
            global mtchs_g_list_des_FLOAT32_right_eye
            global mtchs_g_list_des_FLOAT32_eyes
            global mtchs_g_list_des_FLOAT32_nose
            global mtchs_g_list_des_FLOAT32_mouth_chin

            mtchs_g_dlist_kp_FLOAT32_forehead = []
            mtchs_g_dlist_kp_FLOAT32_left_eye = []
            mtchs_g_dlist_kp_FLOAT32_right_eye = []
            mtchs_g_dlist_kp_FLOAT32_eyes = []
            mtchs_g_dlist_kp_FLOAT32_nose = []
            mtchs_g_dlist_kp_FLOAT32_mouth_chin = []
            mtchs_g_dlist_des_FLOAT32_forehead = []
            mtchs_g_dlist_des_FLOAT32_left_eye = []
            mtchs_g_dlist_des_FLOAT32_right_eye = []
            mtchs_g_dlist_des_FLOAT32_eyes = []
            mtchs_g_dlist_des_FLOAT32_nose = []
            mtchs_g_dlist_des_FLOAT32_mouth_chin = []

            mtchs_g_list_kp_FLOAT32_forehead = []
            mtchs_g_list_kp_FLOAT32_left_eye = []
            mtchs_g_list_kp_FLOAT32_right_eye = []
            mtchs_g_list_kp_FLOAT32_eyes = []
            mtchs_g_list_kp_FLOAT32_nose = []
            mtchs_g_list_kp_FLOAT32_mouth_chin = []
            mtchs_g_list_des_FLOAT32_forehead = []
            mtchs_g_list_des_FLOAT32_left_eye = []
            mtchs_g_list_des_FLOAT32_right_eye = []
            mtchs_g_list_des_FLOAT32_eyes = []
            mtchs_g_list_des_FLOAT32_nose = []
            mtchs_g_list_des_FLOAT32_mouth_chin = []            
            
            # Face types format matches variables FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8_forehead
            global mtchs_g_dlist_kp_UINT8_left_eye
            global mtchs_g_dlist_kp_UINT8_right_eye
            global mtchs_g_dlist_kp_UINT8_eyes
            global mtchs_g_dlist_kp_UINT8_nose
            global mtchs_g_dlist_kp_UINT8_mouth_chin
            global mtchs_g_dlist_des_UINT8_forehead
            global mtchs_g_dlist_des_UINT8_left_eye
            global mtchs_g_dlist_des_UINT8_right_eye
            global mtchs_g_dlist_des_UINT8_eyes
            global mtchs_g_dlist_des_UINT8_nose
            global mtchs_g_dlist_des_UINT8_mouth_chin

            global mtchs_g_list_kp_UINT8_forehead
            global mtchs_g_list_kp_UINT8_left_eye
            global mtchs_g_list_kp_UINT8_right_eye
            global mtchs_g_list_kp_UINT8_eyes
            global mtchs_g_list_kp_UINT8_nose
            global mtchs_g_list_kp_UINT8_mouth_chin
            global mtchs_g_list_des_UINT8_forehead
            global mtchs_g_list_des_UINT8_left_eye
            global mtchs_g_list_des_UINT8_right_eye
            global mtchs_g_list_des_UINT8_eyes
            global mtchs_g_list_des_UINT8_nose
            global mtchs_g_list_des_UINT8_mouth_chin

            mtchs_g_dlist_kp_UINT8_forehead = []
            mtchs_g_dlist_kp_UINT8_left_eye = []
            mtchs_g_dlist_kp_UINT8_right_eye = []
            mtchs_g_dlist_kp_UINT8_eyes = []
            mtchs_g_dlist_kp_UINT8_nose = []
            mtchs_g_dlist_kp_UINT8_mouth_chin = []
            mtchs_g_dlist_des_UINT8_forehead = []
            mtchs_g_dlist_des_UINT8_left_eye = []
            mtchs_g_dlist_des_UINT8_right_eye = []
            mtchs_g_dlist_des_UINT8_eyes = []
            mtchs_g_dlist_des_UINT8_nose = []
            mtchs_g_dlist_des_UINT8_mouth_chin = []

            mtchs_g_list_kp_UINT8_forehead = []
            mtchs_g_list_kp_UINT8_left_eye = []
            mtchs_g_list_kp_UINT8_right_eye = []
            mtchs_g_list_kp_UINT8_eyes = []
            mtchs_g_list_kp_UINT8_nose = []
            mtchs_g_list_kp_UINT8_mouth_chin = []
            mtchs_g_list_des_UINT8_forehead = []
            mtchs_g_list_des_UINT8_left_eye = []
            mtchs_g_list_des_UINT8_right_eye = []
            mtchs_g_list_des_UINT8_eyes = []
            mtchs_g_list_des_UINT8_nose = []
            mtchs_g_list_des_UINT8_mouth_chin = []
            
            # Is exist path process
            if os.path.exists(img_in_path1):
                pass
            else:
                print("Image path 1 is not exist")
            if os.path.exists(img_in_path2):
                pass
            else:
                print("Image path 2 is not exist")
        
            if resize_param == "0.2k":
                # Resize process
                resize_width = 200
                resize_height = 200
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.25k":
                # Resize process
                resize_width = 250
                resize_height = 250
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.5k":
                # Resize process
                resize_width = 500
                resize_height = 500
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]  
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]       
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "1k":
                # Resize process
                resize_width = 1000
                resize_height = 1000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "4k":
                # Resize process
                resize_width = 4000
                resize_height = 4000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"] 
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]   
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "8k":
                # Resize process
                resize_width = 8000
                resize_height = 8000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]         
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()        
                
            
            face_type_list = "forehead","left_eye","right_eye","eyes","nose","mouth_chin"
            #face_type_list = "forehead"
            dt = "0"
            
            # Face detector
            for i in range(train_steps):
                print("#################### TRAIN Steps: {} ####################".format(i+1))
                
                # Scan method
                scan_method = "FLOAT32"
                
                if type(face_type_list) is str:
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type_list))
                    ratio = p2[3]+float((i/1000))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                else:
                    for face_type in face_type_list:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type+"-full"+img_ext
                        # Compare - Face type                
                        # algorithm_FLANN_INDEX_LSH
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
                        # algorithm_FLANN_INDEX_KDTREE
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)                
                        # algorithm_BFMatcher_NONE
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                        ratio = p2[3]+float((i/1000))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
        
                # Scan method
                scan_method = "UINT8"
                    
                if resize_param == "0.25k" or resize_param == "0.2k":
                    pass
                else:        
                    if type(face_type_list) is str:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_ORB(rt,face_type_list)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                    else:
                        for face_type in face_type_list:
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_ORB(rt,face_type)
                            analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
                    
            
            print("Total unique matches found: {}".format(len(dt)))
        
            # Test matches
            test_mtchs = 0
            
            # Patch identificator name
            patch_id = "{0}-{1}".format(resize_param, rt)
            
            # Compare train dataset type 1
            for i in range(test_steps):
                print("#################### FORMAT FLOAT32 TEST Steps: {} ####################".format(i+1))
                for face_type in face_type_list:
                    # Face type distributor matches list
                    if face_type == "forehead":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_forehead
                    elif face_type == "left_eye":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_left_eye
                    elif face_type == "right_eye":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_right_eye
                    elif face_type == "eyes":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_eyes
                    elif face_type == "nose":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_nose
                    elif face_type == "mouth_chin":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_mouth_chin
                    # Save face type dataset    
                    ftp = filename_dataset_path+patch_id
                    fp_des = ftp+"\\FLOAT32-des-"+face_type+".txt"
                    # Create resize param dir
                    if os.path.exists(ftp):
                        pass
                    else:
                        os.mkdir(ftp) 
                    # Create des file
                    if os.path.exists(fp_des):
                        fo = open(fp_des, "w")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))
                    else:
                        fo = open(fp_des, "x")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))                         
                    # FORMAT FLOAT 32
                    for idst in mtchs_g_list_des_FLOAT32[0]:
                        idst = str(idst)
                        mtch_g_des = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                        mtchs_g_list_des_test_FLOAT32.append(mtch_g_des)
                    mtch_g_des_float32 = np.asarray(mtchs_g_list_des_test_FLOAT32, dtype="float32")
       
            if resize_param == "0.25k" or resize_param == "0.2k":
                pass
            else:
                # Compare train dataset type 2
                for i in range(test_steps):
                    print("#################### FORMAT UINT8 TEST Steps: {} ####################".format(i+1))
                    if type(face_type_list) is str:
                        # Face type distributor matches list
                        if face_type == "forehead":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                        elif face_type == "left_eye":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                        elif face_type == "right_eye":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                        elif face_type == "eyes":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                        elif face_type == "nose":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                        elif face_type == "mouth_chin":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                        # Save face type dataset    
                        ftp = filename_dataset_path+patch_id
                        fp_kp = ftp+"\\UINT8-kp-"+face_type+".txt"
                        fp_des = ftp+"\\UINT8-des-"+face_type+".txt"
                        # Create resize param dir
                        if os.path.exists(ftp):
                            pass
                        else:
                            os.mkdir(ftp) 
                        # Create kp file
                        if os.path.exists(fp_kp):
                            fo = open(fp_kp, "w")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        else:
                            fo = open(fp_kp, "x")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        # Create des file
                        if os.path.exists(fp_des):
                            fo = open(fp_des, "w")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        else:
                            fo = open(fp_des, "x")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        # FORMAT UINT8
                        for idst in mtchs_g_list_des_UINT8[0]:
                            mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                            mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                        mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
                    else:
                        for face_type in face_type_list:
                            # Face type distributor matches list
                            if face_type == "forehead":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                            elif face_type == "left_eye":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                            elif face_type == "right_eye":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                            elif face_type == "eyes":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                            elif face_type == "nose":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                            elif face_type == "mouth_chin":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                            # Save face type dataset    
                            ftp = filename_dataset_path+patch_id
                            fp_kp = ftp+"\\"+scan_method+"-kp"+"-"+face_type+".txt"
                            fp_des = ftp+"\\"+scan_method+"-des"+"-"+face_type+".txt"
                            # Create resize param dir
                            if os.path.exists(ftp):
                                pass
                            else:
                                os.mkdir(ftp)
                            # Create kp file
                            if os.path.exists(fp_kp):
                                fo = open(fp_kp, "w")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            else:
                                fo = open(fp_kp, "x")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            # Create des file
                            if os.path.exists(fp_des):
                                fo = open(fp_des, "w")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))
                            else:
                                fo = open(fp_des, "x")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))                                
                            # FORMAT UINT8
                            for idst in mtchs_g_list_des_UINT8[0]:
                                mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                                mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                            mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
            # Result data
            result_data = os.listdir(img_result_path)
            # Remove detected data matches
            for match_name in result_data:
                old_path = img_result_path+match_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            # Remove detected split data
            split_bin_data = os.listdir(img_split_out_bin_path)
            split_train_data = os.listdir(img_split_out_train_path)
            for split_name in split_bin_data:
                old_path = img_split_out_bin_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            for split_name in split_train_data:
                old_path = img_split_out_train_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))            
        
            if test_mtchs > 70:
                print("--- TEST SUCCESS! ---")
                print("Total matches count: {}".format(test_mtchs))
                """
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                
                # SAVE DATASET
                if open(fp):
                    fo = open(fp, "w")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                else:
                    fo = open(fp, "xw")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                
                
                # OPEN DATASET
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                print("Analysis dataset path: {}".format(fp))
                fo = open(fp, "r")
                fr = fo.read().replace("\\n", ",")
                dst = json.loads(fr.replace("'", "\""))
                fo.close()
                print("Dataset matches count: {}".format(len(dst)))
                
                mtchs_g_list_kp = []
                mtchs_g_list_des = []
                for idst in dst:
                    mtch_g_kp = ("["+str(idst["k"].replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                    mtch_g_des = idst["d"].replace("[", "").replace("]", "").split(",")
                    mtchs_g_list_kp.append(mtch_g_kp)
                    mtchs_g_list_des.append(mtch_g_des)
                    
                mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp, dtype="float32")
                mtch_g_des_float32 = np.asarray(mtchs_g_list_des, dtype="float32")    
                  
                """    
                
                
            else:
                print("--- TEST FAILED! ---")
                print("Total matches count: {}".format(test_mtchs))
            
            # Add to result List
            result_list.append(("Name: {0} Result: {1} pts".format(img_filename, test_mtchs)))
        
        print("\n")
        print("Result list:")
        for result in result_list:
            print(result)        

    def create_new_dataset_only_des_sketch_gray(set, img_filename, resize_type, read_type):
        root_path = set.root_path
        face_detect_path = root_path+'face_detector\\img_in\\face_detect.png'
        img_result_path = root_path+'face_detector\\img_result\\'
        img_out_path = root_path+'face_detector\\img_out\\'
        img_resize_path = root_path+'face_detector\\img_resize\\'
        img_split_in_path = root_path+'face_detector\\img_split\\in\\'
        img_split_out_path = root_path+'face_detector\\img_split\\out\\'
        img_split_out_bin_path = root_path+'face_detector\\img_split\\out\\bin\\'
        img_split_out_train_path = root_path+'face_detector\\img_split\\out\\train\\'
        img_join_bin_path = root_path+'face_detector\\img_join\\bin\\'
        img_join_train_path = root_path+'face_detector\\img_join\\train\\'
        import_img_path = root_path+'face_detector\\img_in\\dataset\\'
        datasets_path = root_path+'face_detector\\datasets\\'
        filename_dataset_path = root_path+'face_detector\\datasets\\{0}\\'.format(img_filename)
        
        # Validation paths exists
        if os.path.exists(face_detect_path):
            pass
        else:
            os.mkdir(face_detect_path)
        if os.path.exists(img_result_path):
            pass
        else:
            os.mkdir(img_result_path)
        if os.path.exists(img_out_path):
            pass
        else:
            os.mkdir(img_out_path)    
        if os.path.exists(img_resize_path):
            pass
        else:
            os.mkdir(img_resize_path)
        if os.path.exists(img_split_in_path):
            pass
        else:
            os.mkdir(img_split_in_path)
        if os.path.exists(img_split_out_path):
            pass
        else:
            os.mkdir(img_split_out_path)
        if os.path.exists(img_split_out_bin_path):
            pass
        else:
            os.mkdir(img_split_out_bin_path)
        if os.path.exists(img_split_out_train_path):
            pass
        else:
            os.mkdir(img_split_out_train_path)
        if os.path.exists(img_join_bin_path):
            pass
        else:
            os.mkdir(img_join_bin_path)
        if os.path.exists(img_join_train_path):
            pass
        else:
            os.mkdir(img_join_train_path)
        if os.path.exists(import_img_path):
            pass
        else:
            os.mkdir(import_img_path)
        if os.path.exists(datasets_path):
            pass
        else:
            os.mkdir(datasets_path)
        if os.path.exists(filename_dataset_path):
            pass
        else:
            os.mkdir(filename_dataset_path)
          
        # Parameters data path
        param_02k_0_path = root_path+'face_detector\\parameters\\param_0.2k_0.json'
        param_02k_18_path = root_path+'face_detector\\parameters\\param_0.2k_18.json'
        param_02k_19_path = root_path+'face_detector\\parameters\\param_0.2k_19.json'
        param_02k_128_path = root_path+'face_detector\\parameters\\param_0.2k_128.json'
        param_025k_0_path = root_path+'face_detector\\parameters\\param_0.25k_0.json'
        param_025k_18_path = root_path+'face_detector\\parameters\\param_0.25k_18.json'
        param_025k_19_path = root_path+'face_detector\\parameters\\param_0.25k_19.json'
        param_025k_128_path = root_path+'face_detector\\parameters\\param_0.25k_128.json'
        
        # Parameters data variables
        param_02k_0_data = []
        param_02k_18_data = []
        param_02k_19_data = []
        param_02k_128_data = []
        param_025k_0_data = []
        param_025k_18_data = []
        param_025k_19_data = []
        param_025k_128_data = []
        
        # Loading dataset type parameters
        if os.path.exists(param_02k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_0_data.append(data)
        else:
            print("Error: {0}".format(param_02k_0_path))
        if os.path.exists(param_02k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_18_data.append(data)
        else:
            print("Error: {0}".format(param_02k_18_path))
        if os.path.exists(param_02k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_19_data.append(data)
        else:
            print("Error: {0}".format(param_02k_19_path))
        if os.path.exists(param_02k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_128_data.append(data)
        else:
            print("Error: {0}".format(param_02k_128_path))
        if os.path.exists(param_025k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_0_data.append(data)
        else:
            print("Error: {0}".format(param_025k_0_path))
        if os.path.exists(param_025k_18_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_18_data.append(data)
        else:
            print("Error: {0}".format(param_025k_18_path))
        if os.path.exists(param_025k_19_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_19_data.append(data)
        else:
            print("Error: {0}".format(param_025k_19_path))
        if os.path.exists(param_025k_128_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_128_data.append(data)
        else:
            print("Error: {0}".format(param_025k_128_path))
          
        
        # List result
        result_list = []
        
        for face_detect_process in range(1):
            print("Create dataset: {}".format(face_detect_path))
            # Default file paths
            img_ext = ".png"
            img_in_path1 = face_detect_path
            img_in_path2 = face_detect_path
            f = img_in_path1, img_in_path2, img_result_path
           
            # Read tyoe
            rt = read_type
            
            # Resize param
            resize_param = resize_type

            if resize_param == "0.2k" and rt == "None":
                p = param_02k_128_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())             
            elif resize_param == "0.25k" and rt == "None":
                p = param_025k_0_data[0][0]
                # Train parameters
                train_steps = p["train_steps"]
                # LSH abd KDTREE
                p1 = list(tuple(p["p1"])[0].values())
                # BF Matcher
                p2 = list(tuple(p["p2"])[0].values())
                
                # Test parameters
                test_steps = p["test_steps"]
                # LSH abd KDTREE
                p3 = list(tuple(p["p3"])[0].values())
                # BF Matcher
                p4 = list(tuple(p["p4"])[0].values())
            
            # Detected matches as dtc
            # FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32
            global mtchs_g_dlist_des_FLOAT32
            global mtchs_g_list_kp_FLOAT32
            global mtchs_g_list_des_FLOAT32
            mtchs_g_dlist_kp_FLOAT32 = []
            mtchs_g_dlist_des_FLOAT32 = []
            mtchs_g_list_kp_FLOAT32 = []
            mtchs_g_list_des_FLOAT32 = []
            mtchs_g_list_kp_test_FLOAT32 = []
            mtchs_g_list_des_test_FLOAT32 = []
        
            # FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8
            global mtchs_g_dlist_des_UINT8
            global mtchs_g_list_kp_UINT8
            global mtchs_g_list_des_UINT8
            mtchs_g_dlist_kp_UINT8 = []
            mtchs_g_dlist_des_UINT8 = []
            mtchs_g_list_kp_UINT8 = []
            mtchs_g_list_des_UINT8 = []
            mtchs_g_list_kp_test_UINT8 = []
            mtchs_g_list_des_test_UINT8 = []
            
            # Face types format matches variables FORMAT FLOAT32
            global mtchs_g_dlist_kp_FLOAT32_forehead
            global mtchs_g_dlist_kp_FLOAT32_left_eye
            global mtchs_g_dlist_kp_FLOAT32_right_eye
            global mtchs_g_dlist_kp_FLOAT32_eyes
            global mtchs_g_dlist_kp_FLOAT32_nose
            global mtchs_g_dlist_kp_FLOAT32_mouth_chin
            global mtchs_g_dlist_des_FLOAT32_forehead
            global mtchs_g_dlist_des_FLOAT32_left_eye
            global mtchs_g_dlist_des_FLOAT32_right_eye
            global mtchs_g_dlist_des_FLOAT32_eyes
            global mtchs_g_dlist_des_FLOAT32_nose
            global mtchs_g_dlist_des_FLOAT32_mouth_chin

            global mtchs_g_list_kp_FLOAT32_forehead
            global mtchs_g_list_kp_FLOAT32_left_eye
            global mtchs_g_list_kp_FLOAT32_right_eye
            global mtchs_g_list_kp_FLOAT32_eyes
            global mtchs_g_list_kp_FLOAT32_nose
            global mtchs_g_list_kp_FLOAT32_mouth_chin
            global mtchs_g_list_des_FLOAT32_forehead
            global mtchs_g_list_des_FLOAT32_left_eye
            global mtchs_g_list_des_FLOAT32_right_eye
            global mtchs_g_list_des_FLOAT32_eyes
            global mtchs_g_list_des_FLOAT32_nose
            global mtchs_g_list_des_FLOAT32_mouth_chin

            mtchs_g_dlist_kp_FLOAT32_forehead = []
            mtchs_g_dlist_kp_FLOAT32_left_eye = []
            mtchs_g_dlist_kp_FLOAT32_right_eye = []
            mtchs_g_dlist_kp_FLOAT32_eyes = []
            mtchs_g_dlist_kp_FLOAT32_nose = []
            mtchs_g_dlist_kp_FLOAT32_mouth_chin = []
            mtchs_g_dlist_des_FLOAT32_forehead = []
            mtchs_g_dlist_des_FLOAT32_left_eye = []
            mtchs_g_dlist_des_FLOAT32_right_eye = []
            mtchs_g_dlist_des_FLOAT32_eyes = []
            mtchs_g_dlist_des_FLOAT32_nose = []
            mtchs_g_dlist_des_FLOAT32_mouth_chin = []

            mtchs_g_list_kp_FLOAT32_forehead = []
            mtchs_g_list_kp_FLOAT32_left_eye = []
            mtchs_g_list_kp_FLOAT32_right_eye = []
            mtchs_g_list_kp_FLOAT32_eyes = []
            mtchs_g_list_kp_FLOAT32_nose = []
            mtchs_g_list_kp_FLOAT32_mouth_chin = []
            mtchs_g_list_des_FLOAT32_forehead = []
            mtchs_g_list_des_FLOAT32_left_eye = []
            mtchs_g_list_des_FLOAT32_right_eye = []
            mtchs_g_list_des_FLOAT32_eyes = []
            mtchs_g_list_des_FLOAT32_nose = []
            mtchs_g_list_des_FLOAT32_mouth_chin = []            
            
            # Face types format matches variables FORMAT UINT8
            global mtchs_g_dlist_kp_UINT8_forehead
            global mtchs_g_dlist_kp_UINT8_left_eye
            global mtchs_g_dlist_kp_UINT8_right_eye
            global mtchs_g_dlist_kp_UINT8_eyes
            global mtchs_g_dlist_kp_UINT8_nose
            global mtchs_g_dlist_kp_UINT8_mouth_chin
            global mtchs_g_dlist_des_UINT8_forehead
            global mtchs_g_dlist_des_UINT8_left_eye
            global mtchs_g_dlist_des_UINT8_right_eye
            global mtchs_g_dlist_des_UINT8_eyes
            global mtchs_g_dlist_des_UINT8_nose
            global mtchs_g_dlist_des_UINT8_mouth_chin

            global mtchs_g_list_kp_UINT8_forehead
            global mtchs_g_list_kp_UINT8_left_eye
            global mtchs_g_list_kp_UINT8_right_eye
            global mtchs_g_list_kp_UINT8_eyes
            global mtchs_g_list_kp_UINT8_nose
            global mtchs_g_list_kp_UINT8_mouth_chin
            global mtchs_g_list_des_UINT8_forehead
            global mtchs_g_list_des_UINT8_left_eye
            global mtchs_g_list_des_UINT8_right_eye
            global mtchs_g_list_des_UINT8_eyes
            global mtchs_g_list_des_UINT8_nose
            global mtchs_g_list_des_UINT8_mouth_chin

            mtchs_g_dlist_kp_UINT8_forehead = []
            mtchs_g_dlist_kp_UINT8_left_eye = []
            mtchs_g_dlist_kp_UINT8_right_eye = []
            mtchs_g_dlist_kp_UINT8_eyes = []
            mtchs_g_dlist_kp_UINT8_nose = []
            mtchs_g_dlist_kp_UINT8_mouth_chin = []
            mtchs_g_dlist_des_UINT8_forehead = []
            mtchs_g_dlist_des_UINT8_left_eye = []
            mtchs_g_dlist_des_UINT8_right_eye = []
            mtchs_g_dlist_des_UINT8_eyes = []
            mtchs_g_dlist_des_UINT8_nose = []
            mtchs_g_dlist_des_UINT8_mouth_chin = []

            mtchs_g_list_kp_UINT8_forehead = []
            mtchs_g_list_kp_UINT8_left_eye = []
            mtchs_g_list_kp_UINT8_right_eye = []
            mtchs_g_list_kp_UINT8_eyes = []
            mtchs_g_list_kp_UINT8_nose = []
            mtchs_g_list_kp_UINT8_mouth_chin = []
            mtchs_g_list_des_UINT8_forehead = []
            mtchs_g_list_des_UINT8_left_eye = []
            mtchs_g_list_des_UINT8_right_eye = []
            mtchs_g_list_des_UINT8_eyes = []
            mtchs_g_list_des_UINT8_nose = []
            mtchs_g_list_des_UINT8_mouth_chin = []
            
            # Is exist path process
            if os.path.exists(img_in_path1):
                pass
            else:
                print("Image path 1 is not exist")
            if os.path.exists(img_in_path2):
                pass
            else:
                print("Image path 2 is not exist")
        
            if resize_param == "0.2k":
                # Resize process
                resize_width = 200
                resize_height = 200
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 20,20
                list_par_1 = ["20_20","20_40","20_60","20_80","20_100","20_120","20_140","20_160"]
                list_par_2 = ["40_20","40_40","40_60","40_80","40_100","40_120","40_140","40_160"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80"]
                list_par_2 = ["60_40","60_60","60_80"]
                list_par_3 = ["80_40","80_60","80_80"]
                list_par_4 = ["100_40","100_60","100_80"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 100,40
                list_par_1 = ["40_100","40_120","40_140"]
                list_par_2 = ["60_100","60_120","60_140"]
                list_par_3 = ["80_100","80_120","80_140"]
                list_par_4 = ["100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 40,40
                list_par_1 = ["40_40","40_60","40_80","40_100","40_120","40_140"]
                list_par_2 = ["60_40","60_60","60_80","60_100","60_120","60_140"]
                list_par_3 = ["80_40","80_60","80_80","80_100","80_120","80_140"]
                list_par_4 = ["100_40","100_60","100_80","100_100","100_120","100_140"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 80,60
                list_par_1 = ["60_80","60_100"]
                list_par_2 = ["80_80","80_100"]
                list_par_3 = ["100_80","100_100"]
                list_par_4 = ["120_80","120_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 60,140
                list_par_1 = ["140_60","140_80","140_100","140_120"]
                list_par_2 = ["160_60","160_80","160_100","160_120"]
                list_par_3 = ["180_60","180_80","180_100","180_120"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.25k":
                # Resize process
                resize_width = 250
                resize_height = 250
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 25,25
                list_par_1 = ["25_25","25_50","25_75","25_100","25_125","25_150","25_175","25_200"]
                list_par_2 = ["50_25","50_50","50_75","50_100","50_125","50_150","50_175","50_200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100"]
                list_par_2 = ["75_50","75_75","75_100"]
                list_par_3 = ["100_50","100_75","100_100"]
                list_par_4 = ["125_50","125_75","125_100"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 125,50
                list_par_1 = ["50_125","50_150","50_175"]
                list_par_2 = ["75_125","75_150","75_175"]
                list_par_3 = ["100_125","100_150","100_175"]
                list_par_4 = ["125_125","125_150","125_175"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 50,50
                list_par_1 = ["50_50","50_75","50_100","50_125","50_150","50_175"]
                list_par_2 = ["75_50","75_75","75_100","75_125","75_150","75_175"]
                list_par_3 = ["100_50","100_75","100_100","100_125","100_150","100_175"]
                list_par_4 = ["125_50","125_75","125_100","125_125","125_150","125_175"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 100,75
                list_par_1 = ["75_100","75_125"]
                list_par_2 = ["100_100","100_125"]
                list_par_3 = ["125_100","125_125"]
                list_par_4 = ["150_100","150_125"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 75,175
                list_par_1 = ["175_75","175_100","175_125","175_150"]
                list_par_2 = ["200_75","200_100","200_125","200_150"]
                list_par_3 = ["225_75","225_100","225_125","225_150"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "0.5k":
                # Resize process
                resize_width = 500
                resize_height = 500
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 50,50
                list_par_1 = ["50_50","50_100","50_150","50_200","50_250","50_300","50_350","50_400"]
                list_par_2 = ["100_50","100_100","100_150","100_200","100_250","100_300","100_350","100_400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200"]
                list_par_2 = ["150_100","150_150","150_200"]
                list_par_3 = ["200_100","200_150","200_200"]
                list_par_4 = ["250_100","250_150","250_200"]  
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 250,100
                list_par_1 = ["100_250","100_300","100_350"]
                list_par_2 = ["150_250","150_300","150_350"]
                list_par_3 = ["200_250","200_300","200_350"]
                list_par_4 = ["250_250","250_300","250_350"]       
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 100,100
                list_par_1 = ["100_100","100_150","100_200","100_250","100_300","100_350"]
                list_par_2 = ["150_100","150_150","150_200","150_250","150_300","150_350"]
                list_par_3 = ["200_100","200_150","200_200","200_250","200_300","200_350"]
                list_par_4 = ["250_100","250_150","250_200","250_250","250_300","250_350"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 200,150
                list_par_1 = ["150_200","150_250"]
                list_par_2 = ["200_200","200_250"]
                list_par_3 = ["250_200","250_250"]
                list_par_4 = ["300_200","300_250"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 150,350
                list_par_1 = ["350_150","350_200","350_250","350_300"]
                list_par_2 = ["400_150","400_200","400_250","400_300"]
                list_par_3 = ["450_150","450_200","450_250","450_300"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "1k":
                # Resize process
                resize_width = 1000
                resize_height = 1000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 100,100
                list_par_1 = ["100_100","100_200","100_300","100_400","100_500","100_600","100_700","100_800"]
                list_par_2 = ["200_100","200_200","200_300","200_400","200_500","200_600","200_700","200_800"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400"]
                list_par_2 = ["300_200","300_300","300_400"]
                list_par_3 = ["400_200","400_300","400_400"]
                list_par_4 = ["500_200","500_300","500_400"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 500,200
                list_par_1 = ["200_500","200_600","200_700"]
                list_par_2 = ["300_500","300_600","300_700"]
                list_par_3 = ["400_500","400_600","400_700"]
                list_par_4 = ["500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 200,200
                list_par_1 = ["200_200","200_300","200_400","200_500","200_600","200_700"]
                list_par_2 = ["300_200","300_300","300_400","300_500","300_600","300_700"]
                list_par_3 = ["400_200","400_300","400_400","400_500","400_600","400_700"]
                list_par_4 = ["500_200","500_300","500_400","500_500","500_600","500_700"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 400,300
                list_par_1 = ["300_400","300_500"]
                list_par_2 = ["400_400","400_500"]
                list_par_3 = ["500_400","500_500"]
                list_par_4 = ["600_400","600_500"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 300,700
                list_par_1 = ["700_300","700_400","700_500","700_600"]
                list_par_2 = ["800_300","800_400","800_500","800_600"]
                list_par_3 = ["900_300","900_400","900_500","900_600"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "4k":
                # Resize process
                resize_width = 4000
                resize_height = 4000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 400,400
                list_par_1 = ["400_400","400_800","400_1200","400_1600","400_2000","400_2400","400_2800","400_3200"]
                list_par_2 = ["800_400","800_800","800_1200","800_1600","800_2000","800_2400","800_2800","800_3200"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600"]
                list_par_2 = ["1200_800","1200_1200","1200_1600"]
                list_par_3 = ["1600_800","1600_1200","1600_1600"]
                list_par_4 = ["2000_800","2000_1200","2000_1600"] 
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 2000,800
                list_par_1 = ["800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_2000","2000_2400","2000_2800"]      
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 800,800
                list_par_1 = ["800_800","800_1200","800_1600","800_2000","800_2400","800_2800"]
                list_par_2 = ["1200_800","1200_1200","1200_1600","1200_2000","1200_2400","1200_2800"]
                list_par_3 = ["1600_800","1600_1200","1600_1600","1600_2000","1600_2400","1600_2800"]
                list_par_4 = ["2000_800","2000_1200","2000_1600","2000_2000","2000_2400","2000_2800"]   
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 1600,1200
                list_par_1 = ["1200_1600","1200_2000"]
                list_par_2 = ["1600_1600","1600_2000"]
                list_par_3 = ["2000_1600","2000_2000"]
                list_par_4 = ["2400_1600","2400_2000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 1200,2800
                list_par_1 = ["2800_1200","2800_1600","2800_2000","2800_2400"]
                list_par_2 = ["3200_1200","3200_1600","3200_2000","3200_2400"]
                list_par_3 = ["3600_1200","3600_1600","3600_2000","3600_2400"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
            elif resize_param == "8k":
                # Resize process
                resize_width = 8000
                resize_height = 8000
                rs = resize_image(f[0], f[1], img_resize_path, resize_width, resize_height).keeping_ratio()
                img_bin_copy = sh.copy(rs[0], img_split_in_path)
                img_train_copy = sh.copy(rs[1], img_split_in_path)
                
                # Split BIN image
                img_bin_file_name = os.path.basename(rs[0])
                spl = split_image(img_bin_file_name, img_split_in_path, img_split_out_bin_path, 0.1).auto()
                # Split TRAIN image
                img_train_file_name = os.path.basename(rs[1])
                spl = split_image(img_train_file_name, img_split_in_path, img_split_out_train_path, 0.1).auto()
                
                # Join BIN image
                img_name = "resize_bin_"
                split_path = img_split_out_bin_path
                joint_path = img_join_bin_path
                template_param = resize_width, resize_height
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]    
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                
                # Join TRAIN image
                img_name = "resize_train_"
                split_path = img_split_out_train_path
                joint_path = img_join_train_path
                # Grid 8x2 - Forehead
                grid_name = "forehead"
                template_position = 800,800
                list_par_1 = ["800_800","800_1600","800_2400","800_3200","800_4000","800_4800","800_5600","800_6400"]
                list_par_2 = ["1600_800","1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600","1600_6400"]
                list_full = list_par_1, list_par_2
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Left eye
                grid_name = "left_eye"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x3 - Right eye
                grid_name = "right_eye"
                template_position = 4000,1600
                list_par_1 = ["1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_4000","4000_4800","4000_5600"]         
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 6x3 - Eyes
                grid_name = "eyes"
                template_position = 1600,1600
                list_par_1 = ["1600_1600","1600_2400","1600_3200","1600_4000","1600_4800","1600_5600"]
                list_par_2 = ["2400_1600","2400_2400","2400_3200","2400_4000","2400_4800","2400_5600"]
                list_par_3 = ["3200_1600","3200_2400","3200_3200","3200_4000","3200_4800","3200_5600"]
                list_par_4 = ["4000_1600","4000_2400","4000_3200","4000_4000","4000_4800","4000_5600"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 4x3 - Nose
                grid_name = "nose"
                template_position = 3200,2400
                list_par_1 = ["2400_3200","2400_4000"]
                list_par_2 = ["3200_3200","3200_4000"]
                list_par_3 = ["4000_3200","4000_4000"]
                list_par_4 = ["4800_3200","4800_4000"]
                list_full = list_par_1, list_par_2, list_par_3, list_par_4
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()
                # Grid 3x4 - Mouth and Chin
                grid_name = "mouth_chin"
                template_position = 2400,5600
                list_par_1 = ["5600_2400","5600_3200","5600_4000","5600_4800"]
                list_par_2 = ["6400_2400","6400_3200","6400_4000","6400_4800"]
                list_par_3 = ["7200_2400","7200_3200","7200_4000","7200_4800"]
                list_full = list_par_1, list_par_2, list_par_3
                join = join_image(split_path, img_name, img_ext, grid_name, list_full, joint_path, template_param, template_position).auto()        
                
            
            face_type_list = "forehead","left_eye","right_eye","eyes","nose","mouth_chin"
            #face_type_list = "forehead"
            dt = "0"
            
            # Face detector
            for i in range(train_steps):
                print("#################### TRAIN Steps: {} ####################".format(i+1))
                
                # Scan method
                scan_method = "FLOAT32"
                
                if type(face_type_list) is str:
                    # Face paths
                    face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                    face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                    # Compare - Face type
                    # algorithm_FLANN_INDEX_LSH
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                    # algorithm_FLANN_INDEX_KDTREE
                    print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type_list))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)            
                    # algorithm_BFMatcher_NONE
                    print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type_list))
                    ratio = p2[3]+float((i/1000))
                    dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type_list)
                    analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                else:
                    for face_type in face_type_list:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type+"-full"+img_ext
                        # Compare - Face type                
                        # algorithm_FLANN_INDEX_LSH
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_LSH".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
                        # algorithm_FLANN_INDEX_KDTREE
                        print("Face type: {} ### Algorithm: algorithm_FLANN_INDEX_KDTREE".format(face_type))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p1[0],p1[1],p1[2],p1[3],p1[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)                
                        # algorithm_BFMatcher_NONE
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NONE".format(face_type))
                        ratio = p2[3]+float((i/1000))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_SIFT_sketch_gray(p2[0],p2[1],p2[2],ratio,p2[4],rt,face_type)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
        
                # Scan method
                scan_method = "UINT8"
                    
                if resize_param == "0.25k" or resize_param == "0.2k":
                    pass
                else:        
                    if type(face_type_list) is str:
                        # Face paths
                        face_bin_path = img_join_bin_path+face_type_list+"-full"+img_ext
                        face_train_path = img_join_train_path+face_type_list+"-full"+img_ext
                        # Compare - Face type
                        # algorithm_BFMatcher_NORM_HAMMING
                        print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type_list))
                        dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_ORB_sketch_gray(rt,face_type_list)
                        analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type_list)
                    else:
                        for face_type in face_type_list:
                            # Face paths
                            face_bin_path = img_join_bin_path+face_type+"-full"+img_ext
                            face_train_path = img_join_train_path+face_type+"-full"+img_ext
                            # Compare - Face type
                            # algorithm_BFMatcher_NORM_HAMMING
                            print("Face type: {} ### Algorithm: algorithm_BFMatcher_NORM_HAMMING".format(face_type))
                            dt = dtc_train(face_bin_path, face_train_path, f[2]).get_des_by_ORB_sketch_gray(rt,face_type)
                            analysis_train(dt, scan_method).add_to_list_by_face_type_only_des(face_type)
                    
            
            print("Total unique matches found: {}".format(len(dt)))
        
            # Test matches
            test_mtchs = 0
            
            # Patch identificator name
            patch_id = "{0}-{1}".format(resize_param, rt)
            
            # Compare train dataset type 1
            for i in range(test_steps):
                print("#################### FORMAT FLOAT32 TEST Steps: {} ####################".format(i+1))
                for face_type in face_type_list:
                    # Face type distributor matches list
                    if face_type == "forehead":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_forehead
                    elif face_type == "left_eye":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_left_eye
                    elif face_type == "right_eye":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_right_eye
                    elif face_type == "eyes":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_eyes
                    elif face_type == "nose":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_nose
                    elif face_type == "mouth_chin":
                        mtchs_g_list_des_FLOAT32 = mtchs_g_list_des_FLOAT32_mouth_chin
                    # Save face type dataset    
                    ftp = filename_dataset_path+patch_id
                    fp_des = ftp+"\\FLOAT32-des-"+face_type+".txt"
                    # Create resize param dir
                    if os.path.exists(ftp):
                        pass
                    else:
                        os.mkdir(ftp) 
                    # Create des file
                    if os.path.exists(fp_des):
                        fo = open(fp_des, "w")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))
                    else:
                        fo = open(fp_des, "x")
                        fo.write(str(mtchs_g_list_des_FLOAT32[0]).replace("\\n",","))
                        fo.close()
                        print("Dataset save to path: {}".format(fp_des))                         
                    # FORMAT FLOAT 32
                    for idst in mtchs_g_list_des_FLOAT32[0]:
                        idst = str(idst)
                        mtch_g_des = ("["+str(idst.replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                        mtchs_g_list_des_test_FLOAT32.append(mtch_g_des)
                    mtch_g_des_float32 = np.asarray(mtchs_g_list_des_test_FLOAT32, dtype="float32")
       
            if resize_param == "0.25k" or resize_param == "0.2k":
                pass
            else:
                # Compare train dataset type 2
                for i in range(test_steps):
                    print("#################### FORMAT UINT8 TEST Steps: {} ####################".format(i+1))
                    if type(face_type_list) is str:
                        # Face type distributor matches list
                        if face_type == "forehead":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                        elif face_type == "left_eye":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                        elif face_type == "right_eye":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                        elif face_type == "eyes":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                        elif face_type == "nose":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                        elif face_type == "mouth_chin":
                            mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                        # Save face type dataset    
                        ftp = filename_dataset_path+patch_id
                        fp_kp = ftp+"\\UINT8-kp-"+face_type+".txt"
                        fp_des = ftp+"\\UINT8-des-"+face_type+".txt"
                        # Create resize param dir
                        if os.path.exists(ftp):
                            pass
                        else:
                            os.mkdir(ftp) 
                        # Create kp file
                        if os.path.exists(fp_kp):
                            fo = open(fp_kp, "w")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        else:
                            fo = open(fp_kp, "x")
                            fo.write(str(mtchs_g_list_kp_UINT8[0]))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_kp))
                        # Create des file
                        if os.path.exists(fp_des):
                            fo = open(fp_des, "w")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        else:
                            fo = open(fp_des, "x")
                            fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                            fo.close()
                            print("Dataset save to path: {}".format(fp_des))
                        # FORMAT UINT8
                        for idst in mtchs_g_list_des_UINT8[0]:
                            mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                            mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                        mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
                    else:
                        for face_type in face_type_list:
                            # Face type distributor matches list
                            if face_type == "forehead":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_forehead
                            elif face_type == "left_eye":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_left_eye
                            elif face_type == "right_eye":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_right_eye
                            elif face_type == "eyes":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_eyes
                            elif face_type == "nose":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_nose
                            elif face_type == "mouth_chin":
                                mtchs_g_list_des_UINT8 = mtchs_g_list_des_UINT8_mouth_chin
                            # Save face type dataset    
                            ftp = filename_dataset_path+patch_id
                            fp_kp = ftp+"\\"+scan_method+"-kp"+"-"+face_type+".txt"
                            fp_des = ftp+"\\"+scan_method+"-des"+"-"+face_type+".txt"
                            # Create resize param dir
                            if os.path.exists(ftp):
                                pass
                            else:
                                os.mkdir(ftp)
                            # Create kp file
                            if os.path.exists(fp_kp):
                                fo = open(fp_kp, "w")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            else:
                                fo = open(fp_kp, "x")
                                fo.write(str(mtchs_g_list_kp_UINT8[0]))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_kp))
                            # Create des file
                            if os.path.exists(fp_des):
                                fo = open(fp_des, "w")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))
                            else:
                                fo = open(fp_des, "x")
                                fo.write(str(mtchs_g_list_des_UINT8[0]).replace("\\n",","))
                                fo.close()
                                print("Dataset save to path: {}".format(fp_des))                                
                            # FORMAT UINT8
                            for idst in mtchs_g_list_des_UINT8[0]:
                                mtch_g_des = ("["+str(idst.replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                                mtchs_g_list_des_test_UINT8.append(mtch_g_des)
                            mtch_g_des_uint8 = np.asarray(mtchs_g_list_des_test_UINT8, dtype="uint8")
            # Result data
            result_data = os.listdir(img_result_path)
            # Remove detected data matches
            for match_name in result_data:
                old_path = img_result_path+match_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            # Remove detected split data
            split_bin_data = os.listdir(img_split_out_bin_path)
            split_train_data = os.listdir(img_split_out_train_path)
            for split_name in split_bin_data:
                old_path = img_split_out_bin_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))
            for split_name in split_train_data:
                old_path = img_split_out_train_path+split_name
                if os.path.exists(old_path):
                    os.remove(old_path)
                else:
                    print("Error: {} is not exist".format(old_path))            
        
            if test_mtchs > 70:
                print("--- TEST SUCCESS! ---")
                print("Total matches count: {}".format(test_mtchs))
                """
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                
                # SAVE DATASET
                if open(fp):
                    fo = open(fp, "w")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                else:
                    fo = open(fp, "xw")
                    fo.write(str(dt))
                    fo.close()
                    print("Dataset save to path: {}".format(fp))
                
                
                # OPEN DATASET
                fp = ('d:\\python_cv\\train\\'+'test.txt')
                print("Analysis dataset path: {}".format(fp))
                fo = open(fp, "r")
                fr = fo.read().replace("\\n", ",")
                dst = json.loads(fr.replace("'", "\""))
                fo.close()
                print("Dataset matches count: {}".format(len(dst)))
                
                mtchs_g_list_kp = []
                mtchs_g_list_des = []
                for idst in dst:
                    mtch_g_kp = ("["+str(idst["k"].replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ","))+"]").replace("[", "").replace("]", "").split(",")
                    mtch_g_des = idst["d"].replace("[", "").replace("]", "").split(",")
                    mtchs_g_list_kp.append(mtch_g_kp)
                    mtchs_g_list_des.append(mtch_g_des)
                    
                mtch_g_kp_float32 = np.asarray(mtchs_g_list_kp, dtype="float32")
                mtch_g_des_float32 = np.asarray(mtchs_g_list_des, dtype="float32")    
                  
                """    
                
                
            else:
                print("--- TEST FAILED! ---")
                print("Total matches count: {}".format(test_mtchs))
            
            # Add to result List
            result_list.append(("Name: {0} Result: {1} pts".format(img_filename, test_mtchs)))
        
        print("\n")
        print("Result list:")
        for result in result_list:
            print(result)                    
    
    def create_all_datasets(set):
        root_path = set.root_path
        # Create test datasets database
        datasets_path = root_path+"face_detector\\datasets\\"
        face_path_png = root_path+"face_detector\\img_in\\face_detect.png"
        face_path = root_path+"face_detector\\img_in\\"
        dst_path = root_path+"face_detector\\img_in\\dataset\\"
        dst_list = os.listdir(dst_path)
        i=0
        for dst in dst_list:
            old_path = dst_path+dst
            new_path = face_path+"face_detect.png"
            face_con = os.path.exists(face_path_png)
            if face_con:
                os.remove(face_path_png)
                sh.copyfile(old_path, new_path)
            else:
                sh.copyfile(old_path, new_path)
            
            # Create person
            person_name = "person-{}".format(i)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.2k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.2k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.2k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.2k", 128)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.25k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.25k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.25k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset(person_name, "0.25k", 128)
            dataset_face_path = datasets_path+person_name+"\\face.png"
            dst_face_con = os.path.exists(dataset_face_path)
            if dst_face_con:
                os.remove(dataset_face_path)
                sh.copyfile(old_path, dataset_face_path)
            else:
                sh.copyfile(old_path, dataset_face_path)            
            i+=1

    def create_all_datasets_only_des(set):
        root_path = set.root_path
        # Create test datasets database
        datasets_path = root_path+"face_detector\\datasets\\"
        face_path_png = root_path+"face_detector\\img_in\\face_detect.png"
        face_path = root_path+"face_detector\\img_in\\"
        dst_path = root_path+"face_detector\\img_in\\dataset\\"
        dst_list = os.listdir(dst_path)
        i=0
        for dst in dst_list:
            old_path = dst_path+dst
            new_path = face_path+"face_detect.png"
            face_con = os.path.exists(face_path_png)
            if face_con:
                os.remove(face_path_png)
                sh.copyfile(old_path, new_path)
            else:
                sh.copyfile(old_path, new_path)
            
            # Create person
            person_name = "person-{}".format(i)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 128)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 128)
            dataset_face_path = datasets_path+person_name+"\\face.png"
            dst_face_con = os.path.exists(dataset_face_path)
            if dst_face_con:
                os.remove(dataset_face_path)
                sh.copyfile(old_path, dataset_face_path)
            else:
                sh.copyfile(old_path, dataset_face_path)            
            i+=1
            
    def create_all_datasets_only_des_sketch_gray(set):
        root_path = set.root_path
        # Create test datasets database
        datasets_path = root_path+"face_detector\\datasets\\"
        face_path_png = root_path+"face_detector\\img_in\\face_detect.png"
        face_path = root_path+"face_detector\\img_in\\"
        dst_path = root_path+"face_detector\\img_in\\dataset\\"
        dst_list = os.listdir(dst_path)
        i=0
        for dst in dst_list:
            old_path = dst_path+dst
            new_path = face_path+"face_detect.png"
            face_con = os.path.exists(face_path_png)
            if face_con:
                os.remove(face_path_png)
                sh.copyfile(old_path, new_path)
            else:
                sh.copyfile(old_path, new_path)
            
            # Create person
            person_name = "person-{}".format(i)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.2k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.2k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.2k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.2k", 128)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.25k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.25k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.25k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.25k", 128)
            dataset_face_path = datasets_path+person_name+"\\face.png"
            dst_face_con = os.path.exists(dataset_face_path)
            if dst_face_con:
                os.remove(dataset_face_path)
                sh.copyfile(old_path, dataset_face_path)
            else:
                sh.copyfile(old_path, dataset_face_path)            
            i+=1
            
    def create_all_datasets_only_des_MULTI_SCALE(set):
        root_path = set.root_path
        # Create test datasets database
        datasets_path = root_path+"face_detector\\datasets\\"
        face_path_png = root_path+"face_detector\\img_in\\face_detect.png"
        face_path = root_path+"face_detector\\img_in\\"
        dst_path = root_path+"face_detector\\img_in\\dataset\\"
        dst_list = os.listdir(dst_path)
        i=0
        for dst in dst_list:
            old_path = dst_path+dst
            new_path = face_path+"face_detect.png"
            face_con = os.path.exists(face_path_png)
            if face_con:
                os.remove(face_path_png)
                sh.copyfile(old_path, new_path)
            else:
                sh.copyfile(old_path, new_path)
            
            # Create person
            person_name = "person-{}".format(i)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.2k", 128)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 0)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 18)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 19)
            create_dataset = dtc_process(root_path).create_new_dataset_only_des(person_name, "0.25k", 128)
            # SKETCH GRAY
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.2k", "None")
            create_dataset = dtc_process(root_path).create_new_dataset_only_des_sketch_gray(person_name, "0.25k", "None")
            dataset_face_path = datasets_path+person_name+"\\face.png"
            dst_face_con = os.path.exists(dataset_face_path)
            if dst_face_con:
                os.remove(dataset_face_path)
                sh.copyfile(old_path, dataset_face_path)
            else:
                sh.copyfile(old_path, dataset_face_path)            
            i+=1
            
class compare_process:
    def __init__(self):
        pass
    def matching_and_detected(set, index_dataset, detect_method_1, detect_method_2, detect_count):
        set.index_dataset = index_dataset
        
        # DATA FORMAT FLOAT32
        # 02k_0_kp
        data_FLOAT32_02k_0_kp_forehead = []
        data_FLOAT32_02k_0_kp_left_eye = []
        data_FLOAT32_02k_0_kp_right_eye = []
        data_FLOAT32_02k_0_kp_eyes = []
        data_FLOAT32_02k_0_kp_nose = []
        data_FLOAT32_02k_0_kp_mouth_chin = []
        # 02k_18_kp
        data_FLOAT32_02k_18_kp_forehead = []
        data_FLOAT32_02k_18_kp_left_eye = []
        data_FLOAT32_02k_18_kp_right_eye = []
        data_FLOAT32_02k_18_kp_eyes = []
        data_FLOAT32_02k_18_kp_nose = []
        data_FLOAT32_02k_18_kp_mouth_chin = []
        # 02k_19_kp
        data_FLOAT32_02k_19_kp_forehead = []
        data_FLOAT32_02k_19_kp_left_eye = []
        data_FLOAT32_02k_19_kp_right_eye = []
        data_FLOAT32_02k_19_kp_eyes = []
        data_FLOAT32_02k_19_kp_nose = []
        data_FLOAT32_02k_19_kp_mouth_chin = []
        # 02k_128_kp
        data_FLOAT32_02k_128_kp_forehead = []
        data_FLOAT32_02k_128_kp_left_eye = []
        data_FLOAT32_02k_128_kp_right_eye = []
        data_FLOAT32_02k_128_kp_eyes = []
        data_FLOAT32_02k_128_kp_nose = []
        data_FLOAT32_02k_128_kp_mouth_chin = []
        # 025k_0_kp
        data_FLOAT32_025k_0_kp_forehead = []
        data_FLOAT32_025k_0_kp_left_eye = []
        data_FLOAT32_025k_0_kp_right_eye = []
        data_FLOAT32_025k_0_kp_eyes = []
        data_FLOAT32_025k_0_kp_nose = []
        data_FLOAT32_025k_0_kp_mouth_chin = []
        # 025k_18_kp
        data_FLOAT32_025k_18_kp_forehead = []
        data_FLOAT32_025k_18_kp_left_eye = []
        data_FLOAT32_025k_18_kp_right_eye = []
        data_FLOAT32_025k_18_kp_eyes = []
        data_FLOAT32_025k_18_kp_nose = []
        data_FLOAT32_025k_18_kp_mouth_chin = []
        # 025k_19_kp
        data_FLOAT32_025k_19_kp_forehead = []
        data_FLOAT32_025k_19_kp_left_eye = []
        data_FLOAT32_025k_19_kp_right_eye = []
        data_FLOAT32_025k_19_kp_eyes = []
        data_FLOAT32_025k_19_kp_nose = []
        data_FLOAT32_025k_19_kp_mouth_chin = []
        # 025k_128_kp
        data_FLOAT32_025k_128_kp_forehead = []
        data_FLOAT32_025k_128_kp_left_eye = []
        data_FLOAT32_025k_128_kp_right_eye = []
        data_FLOAT32_025k_128_kp_eyes = []
        data_FLOAT32_025k_128_kp_nose = []
        data_FLOAT32_025k_128_kp_mouth_chin = []
        
        # 02k_0_des
        data_FLOAT32_02k_0_des_forehead = []
        data_FLOAT32_02k_0_des_left_eye = []
        data_FLOAT32_02k_0_des_right_eye = []
        data_FLOAT32_02k_0_des_eyes = []
        data_FLOAT32_02k_0_des_nose = []
        data_FLOAT32_02k_0_des_mouth_chin = []
        data_FLOAT32_02k_0_des_face = []
        # 02k_18_des
        data_FLOAT32_02k_18_des_forehead = []
        data_FLOAT32_02k_18_des_left_eye = []
        data_FLOAT32_02k_18_des_right_eye = []
        data_FLOAT32_02k_18_des_eyes = []
        data_FLOAT32_02k_18_des_nose = []
        data_FLOAT32_02k_18_des_mouth_chin = []
        data_FLOAT32_02k_18_des_face = []
        # 02k_19_des
        data_FLOAT32_02k_19_des_forehead = []
        data_FLOAT32_02k_19_des_left_eye = []
        data_FLOAT32_02k_19_des_right_eye = []
        data_FLOAT32_02k_19_des_eyes = []
        data_FLOAT32_02k_19_des_nose = []
        data_FLOAT32_02k_19_des_mouth_chin = []
        data_FLOAT32_02k_19_des_face = []
        # 02k_128_des
        data_FLOAT32_02k_128_des_forehead = []
        data_FLOAT32_02k_128_des_left_eye = []
        data_FLOAT32_02k_128_des_right_eye = []
        data_FLOAT32_02k_128_des_eyes = []
        data_FLOAT32_02k_128_des_nose = []
        data_FLOAT32_02k_128_des_mouth_chin = []
        data_FLOAT32_02k_128_des_face = []
        # 025k_0_des
        data_FLOAT32_025k_0_des_forehead = []
        data_FLOAT32_025k_0_des_left_eye = []
        data_FLOAT32_025k_0_des_right_eye = []
        data_FLOAT32_025k_0_des_eyes = []
        data_FLOAT32_025k_0_des_nose = []
        data_FLOAT32_025k_0_des_mouth_chin = []
        data_FLOAT32_025k_0_des_face = []
        # 025k_18_des
        data_FLOAT32_025k_18_des_forehead = []
        data_FLOAT32_025k_18_des_left_eye = []
        data_FLOAT32_025k_18_des_right_eye = []
        data_FLOAT32_025k_18_des_eyes = []
        data_FLOAT32_025k_18_des_nose = []
        data_FLOAT32_025k_18_des_mouth_chin = []
        data_FLOAT32_025k_18_des_face = []
        # 025k_19_des
        data_FLOAT32_025k_19_des_forehead = []
        data_FLOAT32_025k_19_des_left_eye = []
        data_FLOAT32_025k_19_des_right_eye = []
        data_FLOAT32_025k_19_des_eyes = []
        data_FLOAT32_025k_19_des_nose = []
        data_FLOAT32_025k_19_des_mouth_chin = []
        data_FLOAT32_025k_19_des_face = []
        # 025k_128_des
        data_FLOAT32_025k_128_des_forehead = []
        data_FLOAT32_025k_128_des_left_eye = []
        data_FLOAT32_025k_128_des_right_eye = []
        data_FLOAT32_025k_128_des_eyes = []
        data_FLOAT32_025k_128_des_nose = []
        data_FLOAT32_025k_128_des_mouth_chin = []
        data_FLOAT32_025k_128_des_face = []
        # 02k_none_des
        data_FLOAT32_02k_none_des_forehead = []
        data_FLOAT32_02k_none_des_left_eye = []
        data_FLOAT32_02k_none_des_right_eye = []
        data_FLOAT32_02k_none_des_eyes = []
        data_FLOAT32_02k_none_des_nose = []
        data_FLOAT32_02k_none_des_mouth_chin = []
        data_FLOAT32_02k_none_des_face = []
        # 025k_none_des
        data_FLOAT32_025k_none_des_forehead = []
        data_FLOAT32_025k_none_des_left_eye = []
        data_FLOAT32_025k_none_des_right_eye = []
        data_FLOAT32_025k_none_des_eyes = []
        data_FLOAT32_025k_none_des_nose = []
        data_FLOAT32_025k_none_des_mouth_chin = []
        data_FLOAT32_025k_none_des_face = []
        
        
        datasets_path = root_path+'face_detector\\datasets\\'
        datasets_list = os.listdir(datasets_path)
        for dataset in datasets_list:
            dataset_path = datasets_path+dataset+"\\"
            datasets_inner_list = os.listdir(dataset_path)
            for dataset_type in datasets_inner_list:
                if dataset_type == "face.png":
                    pass
                else:
                    dataset_type_path = datasets_path+dataset+"\\"+dataset_type+"\\"
                    dataset_type_data_list = os.listdir(dataset_type_path)        
                    if dataset_type == "0.2k-0":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_nose.append(data)
                                con_5_data = data
                                con_5 = True
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_0_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_02k_0_des_face.append(face)
                        
                    elif dataset_type == "0.2k-18":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_18_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_02k_18_des_face.append(face)
                    elif dataset_type == "0.2k-19":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                   
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_nose.append(data)
                                con_5_data = data
                                con_5 = True
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_19_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_02k_19_des_face.append(face)
                    elif dataset_type == "0.2k-128":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                   
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                   
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                   
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                   
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                   
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_128_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                   
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_02k_128_des_face.append(face)
                    elif dataset_type == "0.25k-0":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_0_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_025k_0_des_face.append(face)
                    elif dataset_type == "0.25k-18":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_18_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_025k_18_des_face.append(face)
                    elif dataset_type == "0.25k-19":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_19_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_025k_19_des_face.append(face)
                    elif dataset_type == "0.25k-128":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_128_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_025k_128_des_face.append(face)
                    elif dataset_type == "0.2k-None":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_02k_none_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_02k_none_des_face.append(face)
                    elif dataset_type == "0.25k-None":
                        con_1 = False
                        con_2 = False
                        con_3 = False
                        con_4 = False
                        con_5 = False
                        con_6 = False
                        con_1_data = []
                        con_2_data = []
                        con_3_data = []
                        con_4_data = []
                        con_5_data = []
                        con_6_data = []
                        for dataset_type_data in dataset_type_data_list:
                            if dataset_type_data == "FLOAT32-des-forehead.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_forehead.append(data)
                                con_1_data = data
                                con_1 = True                                
                            elif dataset_type_data == "FLOAT32-des-left_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_left_eye.append(data)
                                con_2_data = data
                                con_2 = True                                
                            elif dataset_type_data == "FLOAT32-des-right_eye.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_right_eye.append(data)
                                con_3_data = data
                                con_3 = True                                
                            elif dataset_type_data == "FLOAT32-des-eyes.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_eyes.append(data)
                                con_4_data = data
                                con_4 = True                                
                            elif dataset_type_data == "FLOAT32-des-nose.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_nose.append(data)
                                con_5_data = data
                                con_5 = True                                
                            elif dataset_type_data == "FLOAT32-des-mouth_chin.txt":
                                fp = dataset_type_path+dataset_type_data
                                fo = open(fp, "r")
                                fr = fo.read()
                                data = json.loads(fr.replace("'", "\""))
                                data_FLOAT32_025k_none_des_mouth_chin.append(data)
                                con_6_data = data
                                con_6 = True                                
                            if con_1 == True and con_2 == True and con_3 == True and con_4 == True and con_5 == True and con_6 == True:
                                face = []
                                for data in con_1_data:
                                    face.append(data)
                                for data in con_2_data:
                                    face.append(data)
                                for data in con_3_data:
                                    face.append(data)
                                for data in con_4_data:
                                    face.append(data)
                                for data in con_5_data:
                                    face.append(data)
                                for data in con_6_data:
                                    face.append(data)
                                data_FLOAT32_025k_none_des_face.append(face)
        
        
        # Define float 32 datasets list
        FLOAT32_02k_0_des_forehead = []
        FLOAT32_02k_0_des_left_eye = []
        FLOAT32_02k_0_des_right_eye = []
        FLOAT32_02k_0_des_eyes = []
        FLOAT32_02k_0_des_nose = []
        FLOAT32_02k_0_des_mouth_chin = []
        FLOAT32_02k_0_des_face = []
        
        FLOAT32_02k_18_des_forehead = []
        FLOAT32_02k_18_des_left_eye = []
        FLOAT32_02k_18_des_right_eye = []
        FLOAT32_02k_18_des_eyes = []
        FLOAT32_02k_18_des_nose = []
        FLOAT32_02k_18_des_mouth_chin = []
        FLOAT32_02k_18_des_face = []

        FLOAT32_02k_19_des_forehead = []
        FLOAT32_02k_19_des_left_eye = []
        FLOAT32_02k_19_des_right_eye = []
        FLOAT32_02k_19_des_eyes = []
        FLOAT32_02k_19_des_nose = []
        FLOAT32_02k_19_des_mouth_chin = []
        FLOAT32_02k_19_des_face = []

        FLOAT32_02k_128_des_forehead = []
        FLOAT32_02k_128_des_left_eye = []
        FLOAT32_02k_128_des_right_eye = []
        FLOAT32_02k_128_des_eyes = []
        FLOAT32_02k_128_des_nose = []
        FLOAT32_02k_128_des_mouth_chin = []
        FLOAT32_02k_128_des_face = []
        
        FLOAT32_025k_0_des_forehead = []
        FLOAT32_025k_0_des_left_eye = []
        FLOAT32_025k_0_des_right_eye = []
        FLOAT32_025k_0_des_eyes = []
        FLOAT32_025k_0_des_nose = []
        FLOAT32_025k_0_des_mouth_chin = []
        FLOAT32_025k_0_des_face = []
        
        FLOAT32_025k_18_des_forehead = []
        FLOAT32_025k_18_des_left_eye = []
        FLOAT32_025k_18_des_right_eye = []
        FLOAT32_025k_18_des_eyes = []
        FLOAT32_025k_18_des_nose = []
        FLOAT32_025k_18_des_mouth_chin = []
        FLOAT32_025k_18_des_face = []

        FLOAT32_025k_19_des_forehead = []
        FLOAT32_025k_19_des_left_eye = []
        FLOAT32_025k_19_des_right_eye = []
        FLOAT32_025k_19_des_eyes = []
        FLOAT32_025k_19_des_nose = []
        FLOAT32_025k_19_des_mouth_chin = []
        FLOAT32_025k_19_des_face = []

        FLOAT32_025k_128_des_forehead = []
        FLOAT32_025k_128_des_left_eye = []
        FLOAT32_025k_128_des_right_eye = []
        FLOAT32_025k_128_des_eyes = []
        FLOAT32_025k_128_des_nose = []
        FLOAT32_025k_128_des_mouth_chin = []
        FLOAT32_025k_128_des_face = []
        
        FLOAT32_02k_none_des_forehead = []
        FLOAT32_02k_none_des_left_eye = []
        FLOAT32_02k_none_des_right_eye = []
        FLOAT32_02k_none_des_eyes = []
        FLOAT32_02k_none_des_nose = []
        FLOAT32_02k_none_des_mouth_chin = []
        FLOAT32_02k_none_des_face = []

        FLOAT32_025k_none_des_forehead = []
        FLOAT32_025k_none_des_left_eye = []
        FLOAT32_025k_none_des_right_eye = []
        FLOAT32_025k_none_des_eyes = []
        FLOAT32_025k_none_des_nose = []
        FLOAT32_025k_none_des_mouth_chin = []
        FLOAT32_025k_none_des_face = []
        
        # 02k_0_des
        dst_list = data_FLOAT32_02k_0_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_0_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_0_des_face.append(np.asarray(new_list, dtype="float32"))
        # 02k_18_des
        dst_list = data_FLOAT32_02k_18_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_18_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_18_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 02k_19_des
        dst_list = data_FLOAT32_02k_19_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_19_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_19_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 02k_128_des
        dst_list = data_FLOAT32_02k_128_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_128_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_128_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 025k_0_des
        dst_list = data_FLOAT32_025k_0_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_0_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_0_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 025k_18_des
        dst_list = data_FLOAT32_025k_18_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_18_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_18_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 025k_19_des
        dst_list = data_FLOAT32_025k_19_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_19_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_19_des_face.append(np.asarray(new_list, dtype="float32"))            
        # 025k_128_des
        dst_list = data_FLOAT32_025k_128_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_128_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_128_des_face.append(np.asarray(new_list, dtype="float32"))
        # 02k_none_des
        dst_list = data_FLOAT32_02k_none_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_02k_none_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_02k_none_des_face.append(np.asarray(new_list, dtype="float32"))
        # 025k_none_des
        dst_list = data_FLOAT32_025k_none_des_forehead
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_forehead.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_left_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_left_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_right_eye
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_right_eye.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_eyes
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_eyes.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_nose
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_nose.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_mouth_chin
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_mouth_chin.append(np.asarray(new_list, dtype="float32"))
        dst_list = data_FLOAT32_025k_none_des_face
        for idst in dst_list:
            new_list = []
            for idst_item in idst:
                r = ("["+str(idst_item).replace("'", "").replace("[       ","").replace("[      ","").replace("[     ","").replace("[    ","").replace("[   ","").replace("[  ","").replace("[ ","").replace("[","").replace("       ]","").replace("      ]","").replace("     ]","").replace("    ]","").replace("   ]","").replace("  ]","").replace(" ]","").replace("]","").replace("       ",",").replace("      ",",").replace("     ",",").replace("    ",",").replace("   ",",").replace("  ",",").replace(" ",",").replace(",,", ",")+"]").replace("[", "").replace("]", "").split(",")
                new_list.append(r)
            FLOAT32_025k_none_des_face.append(np.asarray(new_list, dtype="float32"))
        

        # Parameters data path
        param_02k_0_path = root_path+'face_detector\\parameters\\param_0.2k_0.json'
        param_02k_18_path = root_path+'face_detector\\parameters\\param_0.2k_18.json'
        param_02k_19_path = root_path+'face_detector\\parameters\\param_0.2k_19.json'
        param_02k_128_path = root_path+'face_detector\\parameters\\param_0.2k_128.json'
        param_025k_0_path = root_path+'face_detector\\parameters\\param_0.25k_0.json'
        param_025k_18_path = root_path+'face_detector\\parameters\\param_0.25k_18.json'
        param_025k_19_path = root_path+'face_detector\\parameters\\param_0.25k_19.json'
        param_025k_128_path = root_path+'face_detector\\parameters\\param_0.25k_128.json'
        param_02k_none_path = root_path+'face_detector\\parameters\\param_0.2k_none.json'
        param_025k_none_path = root_path+'face_detector\\parameters\\param_0.25k_none.json'        
        
        # Parameters data variables
        param_02k_0_data = []
        param_02k_18_data = []
        param_02k_19_data = []
        param_02k_128_data = []
        param_025k_0_data = []
        param_025k_18_data = []
        param_025k_19_data = []
        param_025k_128_data = []
        param_02k_none_data = []
        param_025k_none_data = []
        
        # Loading dataset type parameters
        if os.path.exists(param_02k_0_path):
            fp = param_02k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_0_data.append(data)
        else:
            print("Error: {0}".format(param_02k_0_path))
        if os.path.exists(param_02k_18_path):
            fp = param_02k_18_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_18_data.append(data)
        else:
            print("Error: {0}".format(param_02k_18_path))
        if os.path.exists(param_02k_19_path):
            fp = param_02k_19_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_19_data.append(data)
        else:
            print("Error: {0}".format(param_02k_19_path))
        if os.path.exists(param_02k_128_path):
            fp = param_02k_128_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_128_data.append(data)
        else:
            print("Error: {0}".format(param_02k_128_path))
        if os.path.exists(param_025k_0_path):
            fp = param_025k_0_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_0_data.append(data)
        else:
            print("Error: {0}".format(param_025k_0_path))
        if os.path.exists(param_025k_18_path):
            fp = param_025k_18_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_18_data.append(data)
        else:
            print("Error: {0}".format(param_025k_18_path))
        if os.path.exists(param_025k_19_path):
            fp = param_025k_19_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_19_data.append(data)
        else:
            print("Error: {0}".format(param_025k_19_path))
        if os.path.exists(param_025k_128_path):
            fp = param_025k_128_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_128_data.append(data)
        else:
            print("Error: {0}".format(param_025k_128_path))
        if os.path.exists(param_02k_none_path):
            fp = param_02k_none_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_02k_none_data.append(data)
        else:
            print("Error: {0}".format(param_02k_none_path))
        if os.path.exists(param_025k_none_path):
            fp = param_025k_none_path
            fo = open(fp, "r")
            fr = fo.read()
            data = json.loads(fr.replace("'", "\""))
            param_025k_none_data.append(data)
        else:
            print("Error: {0}".format(param_025k_none_path))

        # Score list for export data to return
        export_score_list = []

        for detect in range(detect_count):
    
            # Total score variable
            score_list_02k_0_des_forehead = []
            score_list_02k_0_des_left_eye = []
            score_list_02k_0_des_right_eye = []
            score_list_02k_0_des_eyes = []
            score_list_02k_0_des_nose = []
            score_list_02k_0_des_mouth_chin = []
            score_list_02k_0_des_face = []
            
            score_list_02k_18_des_forehead = []
            score_list_02k_18_des_left_eye = []
            score_list_02k_18_des_right_eye = []
            score_list_02k_18_des_eyes = []
            score_list_02k_18_des_nose = []
            score_list_02k_18_des_mouth_chin = []        
            score_list_02k_18_des_face = []
            
            score_list_02k_19_des_forehead = []
            score_list_02k_19_des_left_eye = []
            score_list_02k_19_des_right_eye = []
            score_list_02k_19_des_eyes = []
            score_list_02k_19_des_nose = []
            score_list_02k_19_des_mouth_chin = []
            score_list_02k_19_des_face = []
    
            score_list_02k_128_des_forehead = []
            score_list_02k_128_des_left_eye = []
            score_list_02k_128_des_right_eye = []
            score_list_02k_128_des_eyes = []
            score_list_02k_128_des_nose = []
            score_list_02k_128_des_mouth_chin = []
            score_list_02k_128_des_face = []
    
            score_list_025k_0_des_forehead = []
            score_list_025k_0_des_left_eye = []
            score_list_025k_0_des_right_eye = []
            score_list_025k_0_des_eyes = []
            score_list_025k_0_des_nose = []
            score_list_025k_0_des_mouth_chin = []
            score_list_025k_0_des_face = []
    
            score_list_025k_18_des_forehead = []
            score_list_025k_18_des_left_eye = []
            score_list_025k_18_des_right_eye = []
            score_list_025k_18_des_eyes = []
            score_list_025k_18_des_nose = []
            score_list_025k_18_des_mouth_chin = []
            score_list_025k_18_des_face = []
            
            score_list_025k_19_des_forehead = []
            score_list_025k_19_des_left_eye = []
            score_list_025k_19_des_right_eye = []
            score_list_025k_19_des_eyes = []
            score_list_025k_19_des_nose = []
            score_list_025k_19_des_mouth_chin = []
            score_list_025k_19_des_face = []
    
            score_list_025k_128_des_forehead = []
            score_list_025k_128_des_left_eye = []
            score_list_025k_128_des_right_eye = []
            score_list_025k_128_des_eyes = []
            score_list_025k_128_des_nose = []
            score_list_025k_128_des_mouth_chin = []
            score_list_025k_128_des_face = []
            
            score_list_02k_none_des_forehead = []
            score_list_02k_none_des_left_eye = []
            score_list_02k_none_des_right_eye = []
            score_list_02k_none_des_eyes = []
            score_list_02k_none_des_nose = []
            score_list_02k_none_des_mouth_chin = []
            score_list_02k_none_des_face = []
            
            score_list_025k_none_des_forehead = []
            score_list_025k_none_des_left_eye = []
            score_list_025k_none_des_right_eye = []
            score_list_025k_none_des_eyes = []
            score_list_025k_none_des_nose = []
            score_list_025k_none_des_mouth_chin = []
            score_list_025k_none_des_face = []        
    
            # MATCHING DATASET NATE
            datasets_name = datasets_list[index_dataset]
            print("Compare: {}".format(datasets_name))
            global matching_dataset_name
            matching_dataset_name = datasets_name
           
            detect_method_con_1 = detect_method_1
            detect_method_con_2 = detect_method_2
    
            counter = len(datasets_list)
    
            # Detect Method 1
    
            if detect_method_con_1 == True:
                ########################################
                # FLANN Based Matcher param for -> 02k_0
                ########################################
                dist = param_02k_0_data[0][0]["p1"][0]["distance"]
                trees = param_02k_0_data[0][0]["p1"][0]["trees"]
                checks = param_02k_0_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_forehead[index_dataset]
                    d = FLOAT32_02k_0_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_left_eye[index_dataset]
                    d = FLOAT32_02k_0_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_right_eye[index_dataset]
                    d = FLOAT32_02k_0_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_eyes[index_dataset]
                    d = FLOAT32_02k_0_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_nose[index_dataset]
                    d = FLOAT32_02k_0_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_0_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_face[index_dataset]
                    d = FLOAT32_02k_0_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_face.append(len(m_g_r))
    
                ########################################
                # FLANN Based Matcher param for -> 02k_18
                ########################################
                dist = param_02k_18_data[0][0]["p1"][0]["distance"]
                trees = param_02k_18_data[0][0]["p1"][0]["trees"]
                checks = param_02k_18_data[0][0]["p1"][0]["matches"]        
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_forehead[index_dataset]
                    d = FLOAT32_02k_18_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_left_eye[index_dataset]
                    d = FLOAT32_02k_18_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_right_eye[index_dataset]
                    d = FLOAT32_02k_18_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_eyes[index_dataset]
                    d = FLOAT32_02k_18_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_nose[index_dataset]
                    d = FLOAT32_02k_18_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_18_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_face[index_dataset]
                    d = FLOAT32_02k_18_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 02k_19
                ########################################
                dist = param_02k_19_data[0][0]["p1"][0]["distance"]
                trees = param_02k_19_data[0][0]["p1"][0]["trees"]
                checks = param_02k_19_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_forehead[index_dataset]
                    d = FLOAT32_02k_19_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_left_eye[index_dataset]
                    d = FLOAT32_02k_19_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_right_eye[index_dataset]
                    d = FLOAT32_02k_19_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_eyes[index_dataset]
                    d = FLOAT32_02k_19_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_nose[index_dataset]
                    d = FLOAT32_02k_19_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    try:
                        mtchs = fl.knnMatch(f, d, k = 2)            
                        m_g_r = []
                        im = 0
                        for m, n in mtchs:
                            im+=1
                            if m.distance < dist*n.distance:
                                m_g_r.append(im)
                    except:
                        m_g_r.append(0)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    try:
                        mtchs = fl.knnMatch(f, d, k = 2)
                        m_g_r = []
                        im = 0
                        for m, n in mtchs:
                            im+=1
                            if m.distance < dist*n.distance:
                                m_g_r.append(im)
                    except:
                        m_g_r.append(0)
                    # Add to score list
                    score_list_02k_19_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_19_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_face[index_dataset]
                    d = FLOAT32_02k_19_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 02k_128
                ########################################
                dist = param_02k_128_data[0][0]["p1"][0]["distance"]
                trees = param_02k_128_data[0][0]["p1"][0]["trees"]
                checks = param_02k_128_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_forehead[index_dataset]
                    d = FLOAT32_02k_128_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_left_eye[index_dataset]
                    d = FLOAT32_02k_128_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_right_eye[index_dataset]
                    d = FLOAT32_02k_128_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_eyes[index_dataset]
                    d = FLOAT32_02k_128_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_nose[index_dataset]
                    d = FLOAT32_02k_128_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_128_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_face[index_dataset]
                    d = FLOAT32_02k_128_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 025k_0
                ########################################
                dist = param_025k_0_data[0][0]["p1"][0]["distance"]
                trees = param_025k_0_data[0][0]["p1"][0]["trees"]
                checks = param_025k_0_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_forehead[index_dataset]
                    d = FLOAT32_025k_0_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_left_eye[index_dataset]
                    d = FLOAT32_025k_0_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_right_eye[index_dataset]
                    d = FLOAT32_025k_0_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_eyes[index_dataset]
                    d = FLOAT32_025k_0_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_nose[index_dataset]
                    d = FLOAT32_025k_0_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_0_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_face[index_dataset]
                    d = FLOAT32_025k_0_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_face.append(len(m_g_r))
                    
                ########################################
                # FLANN Based Matcher param for -> 025k_18
                ########################################
                dist = param_025k_18_data[0][0]["p1"][0]["distance"]
                trees = param_025k_18_data[0][0]["p1"][0]["trees"]
                checks = param_025k_18_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_forehead[index_dataset]
                    d = FLOAT32_025k_18_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_left_eye[index_dataset]
                    d = FLOAT32_025k_18_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_right_eye[index_dataset]
                    d = FLOAT32_025k_18_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_eyes[index_dataset]
                    d = FLOAT32_025k_18_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_nose[index_dataset]
                    d = FLOAT32_025k_18_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_18_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_face[index_dataset]
                    d = FLOAT32_025k_18_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 025k_19
                ########################################
                dist = param_025k_19_data[0][0]["p1"][0]["distance"]
                trees = param_025k_19_data[0][0]["p1"][0]["trees"]
                checks = param_025k_19_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_forehead[index_dataset]
                    d = FLOAT32_025k_19_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_left_eye[index_dataset]
                    d = FLOAT32_025k_19_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_right_eye[index_dataset]
                    d = FLOAT32_025k_19_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_eyes[index_dataset]
                    d = FLOAT32_025k_19_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1            # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_nose[index_dataset]
                    d = FLOAT32_025k_19_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_19_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_face[index_dataset]
                    d = FLOAT32_025k_19_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 025k_128
                ########################################
                dist = param_025k_128_data[0][0]["p1"][0]["distance"]
                trees = param_025k_128_data[0][0]["p1"][0]["trees"]
                checks = param_025k_128_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_forehead[index_dataset]
                    d = FLOAT32_025k_128_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_left_eye[index_dataset]
                    d = FLOAT32_025k_128_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_right_eye[index_dataset]
                    d = FLOAT32_025k_128_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_eyes[index_dataset]
                    d = FLOAT32_025k_128_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_nose[index_dataset]
                    d = FLOAT32_025k_128_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_128_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_face[index_dataset]
                    d = FLOAT32_025k_128_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_face.append(len(m_g_r))
                    
                ########################################
                # FLANN Based Matcher param for -> 02k_none
                ########################################
                dist = param_02k_none_data[0][0]["p1"][0]["distance"]
                trees = param_02k_none_data[0][0]["p1"][0]["trees"]
                checks = param_02k_none_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_forehead[index_dataset]
                    d = FLOAT32_02k_none_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_left_eye[index_dataset]
                    d = FLOAT32_02k_none_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_right_eye[index_dataset]
                    d = FLOAT32_02k_none_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_eyes[index_dataset]
                    d = FLOAT32_02k_none_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_nose[index_dataset]
                    d = FLOAT32_02k_none_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_none_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_face[index_dataset]
                    d = FLOAT32_02k_none_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_face.append(len(m_g_r))                
                    
                ########################################
                # FLANN Based Matcher param for -> 025k_none
                ########################################
                dist = param_025k_none_data[0][0]["p1"][0]["distance"]
                trees = param_025k_none_data[0][0]["p1"][0]["trees"]
                checks = param_025k_none_data[0][0]["p1"][0]["matches"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_forehead[index_dataset]
                    d = FLOAT32_025k_none_des_forehead[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_left_eye[index_dataset]
                    d = FLOAT32_025k_none_des_left_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_right_eye[index_dataset]
                    d = FLOAT32_025k_none_des_right_eye[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_eyes[index_dataset]
                    d = FLOAT32_025k_none_des_eyes[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_nose[index_dataset]
                    d = FLOAT32_025k_none_des_nose[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_none_des_mouth_chin[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_face[index_dataset]
                    d = FLOAT32_025k_none_des_face[i]
                    # Find match
                    FLANN_INDEX_LSH = 1
                    index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 2, key_size = 6, multi_probe_level = 1)        
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = trees)
                    search_params = dict(checks = checks)
                    fl = cv.FlannBasedMatcher(index_params, search_params)
                    mtchs = fl.knnMatch(f, d, k = 2)            
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_face.append(len(m_g_r))
            else:
                pass
    
            # Detect Method 2
    
            if detect_method_con_2 == True:
                ########################################
                # BF Matcher distance for -> 02k_0
                ########################################
                dist = param_02k_0_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_forehead[index_dataset]
                    d = FLOAT32_02k_0_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_left_eye[index_dataset]
                    d = FLOAT32_02k_0_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_right_eye[index_dataset]
                    d = FLOAT32_02k_0_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_eyes[index_dataset]
                    d = FLOAT32_02k_0_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_nose[index_dataset]
                    d = FLOAT32_02k_0_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_0_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_0_des_face[index_dataset]
                    d = FLOAT32_02k_0_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_0_des_face.append(len(m_g_r))                
                    
                    
                ########################################
                # BF Matcher distance for -> 02k_18
                ########################################
                dist = param_02k_18_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_forehead[index_dataset]
                    d = FLOAT32_02k_18_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_left_eye[index_dataset]
                    d = FLOAT32_02k_18_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_right_eye[index_dataset]
                    d = FLOAT32_02k_18_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_eyes[index_dataset]
                    d = FLOAT32_02k_18_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_nose[index_dataset]
                    d = FLOAT32_02k_18_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_18_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_18_des_face[index_dataset]
                    d = FLOAT32_02k_18_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_18_des_face.append(len(m_g_r))                    
                    
                ########################################
                # BF Matcher distance for -> 02k_19
                ########################################
                dist = param_02k_19_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_forehead[index_dataset]
                    d = FLOAT32_02k_19_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_left_eye[index_dataset]
                    d = FLOAT32_02k_19_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_right_eye[index_dataset]
                    d = FLOAT32_02k_19_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_eyes[index_dataset]
                    d = FLOAT32_02k_19_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_nose[index_dataset]
                    d = FLOAT32_02k_19_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    try:
                        im = 0
                        for m, n in mtchs:
                            im+=1
                            if m.distance < dist*n.distance:
                                m_g_r.append(im)
                    except:
                        m_g_r.append(0)
                    # Add to score list
                    score_list_02k_19_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_19_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_19_des_face[index_dataset]
                    d = FLOAT32_02k_19_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_19_des_face.append(len(m_g_r))                    
                    
                ########################################
                # BF Matcher distance for -> 02k_128
                ########################################
                dist = param_02k_128_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_forehead[index_dataset]
                    d = FLOAT32_02k_128_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_left_eye[index_dataset]
                    d = FLOAT32_02k_128_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_right_eye[index_dataset]
                    d = FLOAT32_02k_128_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_eyes[index_dataset]
                    d = FLOAT32_02k_128_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_nose[index_dataset]
                    d = FLOAT32_02k_128_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_128_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_128_des_face[index_dataset]
                    d = FLOAT32_02k_128_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_128_des_face.append(len(m_g_r))                    
                    
                ########################################
                # BF Matcher distance for -> 025k_0
                ########################################
                dist = param_025k_0_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_forehead[index_dataset]
                    d = FLOAT32_025k_0_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_left_eye[index_dataset]
                    d = FLOAT32_025k_0_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_right_eye[index_dataset]
                    d = FLOAT32_025k_0_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_eyes[index_dataset]
                    d = FLOAT32_025k_0_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_nose[index_dataset]
                    d = FLOAT32_025k_0_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_0_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_0_des_face[index_dataset]
                    d = FLOAT32_025k_0_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_0_des_face.append(len(m_g_r))
                    
                ########################################
                # BF Matcher distance for -> 025k_18
                ########################################
                dist = param_025k_18_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_forehead[index_dataset]
                    d = FLOAT32_025k_18_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_left_eye[index_dataset]
                    d = FLOAT32_025k_18_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_right_eye[index_dataset]
                    d = FLOAT32_025k_18_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_eyes[index_dataset]
                    d = FLOAT32_025k_18_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_nose[index_dataset]
                    d = FLOAT32_025k_18_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_18_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_18_des_face[index_dataset]
                    d = FLOAT32_025k_18_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_18_des_face.append(len(m_g_r))                
                    
                ########################################
                # BF Matcher distance for -> 025k_19
                ########################################
                dist = param_025k_19_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_forehead[index_dataset]
                    d = FLOAT32_025k_19_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_left_eye[index_dataset]
                    d = FLOAT32_025k_19_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_right_eye[index_dataset]
                    d = FLOAT32_025k_19_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_eyes[index_dataset]
                    d = FLOAT32_025k_19_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_nose[index_dataset]
                    d = FLOAT32_025k_19_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_19_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_19_des_face[index_dataset]
                    d = FLOAT32_025k_19_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_19_des_face.append(len(m_g_r))                
                    
                ########################################
                # BF Matcher distance for -> 025k_128
                ########################################
                dist = param_025k_128_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_forehead[index_dataset]
                    d = FLOAT32_025k_128_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_left_eye[index_dataset]
                    d = FLOAT32_025k_128_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_right_eye[index_dataset]
                    d = FLOAT32_025k_128_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_eyes[index_dataset]
                    d = FLOAT32_025k_128_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_nose[index_dataset]
                    d = FLOAT32_025k_128_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_128_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_128_des_face[index_dataset]
                    d = FLOAT32_025k_128_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_128_des_face.append(len(m_g_r))
                
                ########################################
                # BF Matcher distance for -> 02k_none
                ########################################
                dist = param_02k_none_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_forehead[index_dataset]
                    d = FLOAT32_02k_none_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_left_eye[index_dataset]
                    d = FLOAT32_02k_none_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_right_eye[index_dataset]
                    d = FLOAT32_02k_none_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_eyes[index_dataset]
                    d = FLOAT32_02k_none_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_nose[index_dataset]
                    d = FLOAT32_02k_none_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_mouth_chin[index_dataset]
                    d = FLOAT32_02k_none_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_02k_none_des_face[index_dataset]
                    d = FLOAT32_02k_none_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_02k_none_des_face.append(len(m_g_r))                
                
                ########################################
                # BF Matcher distance for -> 025k_none
                ########################################
                dist = param_025k_none_data[0][0]["p2"][0]["distance"]
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_forehead[index_dataset]
                    d = FLOAT32_025k_none_des_forehead[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_forehead.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_left_eye[index_dataset]
                    d = FLOAT32_025k_none_des_left_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_left_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_right_eye[index_dataset]
                    d = FLOAT32_025k_none_des_right_eye[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_right_eye.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_eyes[index_dataset]
                    d = FLOAT32_025k_none_des_eyes[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_eyes.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_nose[index_dataset]
                    d = FLOAT32_025k_none_des_nose[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_nose.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_mouth_chin[index_dataset]
                    d = FLOAT32_025k_none_des_mouth_chin[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_mouth_chin.append(len(m_g_r))
                for i in range(len(datasets_list)):
                    f = FLOAT32_025k_none_des_face[index_dataset]
                    d = FLOAT32_025k_none_des_face[i]
                    # Find match
                    bf = cv.BFMatcher()
                    mtchs = bf.knnMatch(f, d, k = 2)
                    m_g_r = []
                    im = 0
                    for m, n in mtchs:
                        im+=1
                        if m.distance < dist*n.distance:
                            m_g_r.append(im)
                    # Add to score list
                    score_list_025k_none_des_face.append(len(m_g_r))
            else:
                pass
    
            if (detect_method_con_1 == True and detect_method_con_2 == False) or (detect_method_con_1 == False and detect_method_con_2 == True):
                # Total score - forehead
                score_list_forehead = []
                for i in range(len(score_list_02k_0_des_forehead)):
                    score = 0
                    score += score_list_02k_0_des_forehead[i]
                    score += score_list_02k_18_des_forehead[i]
                    score += score_list_02k_19_des_forehead[i]
                    score += score_list_02k_128_des_forehead[i]
                    score += score_list_025k_0_des_forehead[i]
                    score += score_list_025k_18_des_forehead[i]
                    score += score_list_025k_19_des_forehead[i]
                    score += score_list_025k_128_des_forehead[i]
                    score += score_list_02k_none_des_forehead[i]
                    score += score_list_025k_none_des_forehead[i]
                    score_list_forehead.append(score)
                    
                # Total score - left_eye
                score_list_left_eye = []
                for i in range(len(score_list_02k_0_des_left_eye)):
                    score = 0
                    score += score_list_02k_0_des_left_eye[i]
                    score += score_list_02k_18_des_left_eye[i]
                    score += score_list_02k_19_des_left_eye[i]
                    score += score_list_02k_128_des_left_eye[i]
                    score += score_list_025k_0_des_left_eye[i]
                    score += score_list_025k_18_des_left_eye[i]
                    score += score_list_025k_19_des_left_eye[i]
                    score += score_list_025k_128_des_left_eye[i]
                    score += score_list_02k_none_des_left_eye[i]
                    score += score_list_025k_none_des_left_eye[i]
                    score_list_left_eye.append(score)
                    
                # Total score - right_eye
                score_list_right_eye = []
                for i in range(len(score_list_02k_0_des_right_eye)):
                    score = 0
                    score += score_list_02k_0_des_right_eye[i]
                    score += score_list_02k_18_des_right_eye[i]
                    score += score_list_02k_19_des_right_eye[i]
                    score += score_list_02k_128_des_right_eye[i]
                    score += score_list_025k_0_des_right_eye[i]
                    score += score_list_025k_18_des_right_eye[i]
                    score += score_list_025k_19_des_right_eye[i]
                    score += score_list_025k_128_des_right_eye[i]
                    score += score_list_02k_none_des_right_eye[i]
                    score += score_list_025k_none_des_right_eye[i]
                    score_list_right_eye.append(score)
                    
                # Total score - eyes
                score_list_eyes = []
                for i in range(len(score_list_02k_0_des_eyes)):
                    score = 0
                    score += score_list_02k_0_des_eyes[i]
                    score += score_list_02k_18_des_eyes[i]
                    score += score_list_02k_19_des_eyes[i]
                    score += score_list_02k_128_des_eyes[i]
                    score += score_list_025k_0_des_eyes[i]
                    score += score_list_025k_18_des_eyes[i]
                    score += score_list_025k_19_des_eyes[i]
                    score += score_list_025k_128_des_eyes[i]
                    score += score_list_02k_none_des_eyes[i]
                    score += score_list_025k_none_des_eyes[i]
                    score_list_eyes.append(score)
                    
                # Total score - nose
                score_list_nose = []
                for i in range(len(score_list_02k_0_des_nose)):
                    score = 0
                    score += score_list_02k_0_des_nose[i]
                    score += score_list_02k_18_des_nose[i]
                    score += score_list_02k_19_des_nose[i]
                    score += score_list_02k_128_des_nose[i]
                    score += score_list_025k_0_des_nose[i]
                    score += score_list_025k_18_des_nose[i]
                    score += score_list_025k_19_des_nose[i]
                    score += score_list_025k_128_des_nose[i]
                    score += score_list_02k_none_des_nose[i]
                    score += score_list_025k_none_des_nose[i]
                    score_list_nose.append(score)
                    
                # Total score - mouth_chin
                score_list_mouth_chin = []
                for i in range(len(score_list_02k_0_des_mouth_chin)):
                    score = 0
                    score += score_list_02k_0_des_mouth_chin[i]
                    score += score_list_02k_18_des_mouth_chin[i]
                    score += score_list_02k_19_des_mouth_chin[i]
                    score += score_list_02k_128_des_mouth_chin[i]
                    score += score_list_025k_0_des_mouth_chin[i]
                    score += score_list_025k_18_des_mouth_chin[i]
                    score += score_list_025k_19_des_mouth_chin[i]
                    score += score_list_025k_128_des_mouth_chin[i]
                    score += score_list_02k_none_des_mouth_chin[i]
                    score += score_list_025k_none_des_mouth_chin[i]
                    score_list_mouth_chin.append(score)
                    
                # Total score - face
                score_list_face = []
                for i in range(len(score_list_02k_0_des_face)):
                    score = 0
                    score += score_list_02k_0_des_face[i]
                    score += score_list_02k_18_des_face[i]
                    score += score_list_02k_19_des_face[i]
                    score += score_list_02k_128_des_face[i]
                    score += score_list_025k_0_des_face[i]
                    score += score_list_025k_18_des_face[i]
                    score += score_list_025k_19_des_face[i]
                    score += score_list_025k_128_des_face[i]
                    score += score_list_02k_none_des_face[i]
                    score += score_list_025k_none_des_face[i]
                    score_list_face.append(score)
            elif detect_method_con_1 == True and detect_method_con_2 == True:
                # Total score - forehead
                score_list_forehead = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_forehead[i]
                    score += score_list_02k_18_des_forehead[i]
                    score += score_list_02k_19_des_forehead[i]
                    score += score_list_02k_128_des_forehead[i]
                    score += score_list_025k_0_des_forehead[i]
                    score += score_list_025k_18_des_forehead[i]
                    score += score_list_025k_19_des_forehead[i]
                    score += score_list_025k_128_des_forehead[i]
                    score += score_list_02k_none_des_forehead[i]
                    score += score_list_025k_none_des_forehead[i]                
                    score += score_list_02k_0_des_forehead[i_d]
                    score += score_list_02k_18_des_forehead[i_d]
                    score += score_list_02k_19_des_forehead[i_d]
                    score += score_list_02k_128_des_forehead[i_d]
                    score += score_list_025k_0_des_forehead[i_d]
                    score += score_list_025k_18_des_forehead[i_d]
                    score += score_list_025k_19_des_forehead[i_d]
                    score += score_list_025k_128_des_forehead[i_d]
                    score += score_list_02k_none_des_forehead[i_d]
                    score += score_list_025k_none_des_forehead[i_d]                
                    score_list_forehead.append(score)
                    
                # Total score - left_eye
                score_list_left_eye = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_left_eye[i]
                    score += score_list_02k_18_des_left_eye[i]
                    score += score_list_02k_19_des_left_eye[i]
                    score += score_list_02k_128_des_left_eye[i]
                    score += score_list_025k_0_des_left_eye[i]
                    score += score_list_025k_18_des_left_eye[i]
                    score += score_list_025k_19_des_left_eye[i]
                    score += score_list_025k_128_des_left_eye[i]
                    score += score_list_02k_none_des_left_eye[i]
                    score += score_list_025k_none_des_left_eye[i]                
                    score += score_list_02k_0_des_left_eye[i_d]
                    score += score_list_02k_18_des_left_eye[i_d]
                    score += score_list_02k_19_des_left_eye[i_d]
                    score += score_list_02k_128_des_left_eye[i_d]
                    score += score_list_025k_0_des_left_eye[i_d]
                    score += score_list_025k_18_des_left_eye[i_d]
                    score += score_list_025k_19_des_left_eye[i_d]
                    score += score_list_025k_128_des_left_eye[i_d]
                    score += score_list_02k_none_des_left_eye[i_d]
                    score += score_list_025k_none_des_left_eye[i_d]                
                    score_list_left_eye.append(score)
                    
                # Total score - right_eye
                score_list_right_eye = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_right_eye[i]
                    score += score_list_02k_18_des_right_eye[i]
                    score += score_list_02k_19_des_right_eye[i]
                    score += score_list_02k_128_des_right_eye[i]
                    score += score_list_025k_0_des_right_eye[i]
                    score += score_list_025k_18_des_right_eye[i]
                    score += score_list_025k_19_des_right_eye[i]
                    score += score_list_025k_128_des_right_eye[i]
                    score += score_list_02k_none_des_right_eye[i]
                    score += score_list_025k_none_des_right_eye[i]                
                    score += score_list_02k_0_des_right_eye[i_d]
                    score += score_list_02k_18_des_right_eye[i_d]
                    score += score_list_02k_19_des_right_eye[i_d]
                    score += score_list_02k_128_des_right_eye[i_d]
                    score += score_list_025k_0_des_right_eye[i_d]
                    score += score_list_025k_18_des_right_eye[i_d]
                    score += score_list_025k_19_des_right_eye[i_d]
                    score += score_list_025k_128_des_right_eye[i_d]
                    score += score_list_02k_none_des_right_eye[i_d]
                    score += score_list_025k_none_des_right_eye[i_d]
                    score_list_right_eye.append(score)
                    
                # Total score - eyes
                score_list_eyes = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_eyes[i]
                    score += score_list_02k_18_des_eyes[i]
                    score += score_list_02k_19_des_eyes[i]
                    score += score_list_02k_128_des_eyes[i]
                    score += score_list_025k_0_des_eyes[i]
                    score += score_list_025k_18_des_eyes[i]
                    score += score_list_025k_19_des_eyes[i]
                    score += score_list_025k_128_des_eyes[i]
                    score += score_list_02k_none_des_eyes[i]
                    score += score_list_025k_none_des_eyes[i]                
                    score += score_list_02k_0_des_eyes[i_d]
                    score += score_list_02k_18_des_eyes[i_d]
                    score += score_list_02k_19_des_eyes[i_d]
                    score += score_list_02k_128_des_eyes[i_d]
                    score += score_list_025k_0_des_eyes[i_d]
                    score += score_list_025k_18_des_eyes[i_d]
                    score += score_list_025k_19_des_eyes[i_d]
                    score += score_list_025k_128_des_eyes[i_d]
                    score += score_list_02k_none_des_eyes[i_d]
                    score += score_list_025k_none_des_eyes[i_d]
                    score_list_eyes.append(score)
                    
                # Total score - nose
                score_list_nose = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_nose[i]
                    score += score_list_02k_18_des_nose[i]
                    score += score_list_02k_19_des_nose[i]
                    score += score_list_02k_128_des_nose[i]
                    score += score_list_025k_0_des_nose[i]
                    score += score_list_025k_18_des_nose[i]
                    score += score_list_025k_19_des_nose[i]
                    score += score_list_025k_128_des_nose[i]
                    score += score_list_02k_none_des_nose[i]
                    score += score_list_025k_none_des_nose[i]                
                    score += score_list_02k_0_des_nose[i_d]
                    score += score_list_02k_18_des_nose[i_d]
                    score += score_list_02k_19_des_nose[i_d]
                    score += score_list_02k_128_des_nose[i_d]
                    score += score_list_025k_0_des_nose[i_d]
                    score += score_list_025k_18_des_nose[i_d]
                    score += score_list_025k_19_des_nose[i_d]
                    score += score_list_025k_128_des_nose[i_d]
                    score += score_list_02k_none_des_nose[i_d]
                    score += score_list_025k_none_des_nose[i_d]                
                    score_list_nose.append(score)
                    
                # Total score - mouth_chin
                score_list_mouth_chin = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_mouth_chin[i]
                    score += score_list_02k_18_des_mouth_chin[i]
                    score += score_list_02k_19_des_mouth_chin[i]
                    score += score_list_02k_128_des_mouth_chin[i]
                    score += score_list_025k_0_des_mouth_chin[i]
                    score += score_list_025k_18_des_mouth_chin[i]
                    score += score_list_025k_19_des_mouth_chin[i]
                    score += score_list_025k_128_des_mouth_chin[i]
                    score += score_list_02k_none_des_mouth_chin[i]
                    score += score_list_025k_none_des_mouth_chin[i]                
                    score += score_list_02k_0_des_mouth_chin[i_d]
                    score += score_list_02k_18_des_mouth_chin[i_d]
                    score += score_list_02k_19_des_mouth_chin[i_d]
                    score += score_list_02k_128_des_mouth_chin[i_d]
                    score += score_list_025k_0_des_mouth_chin[i_d]
                    score += score_list_025k_18_des_mouth_chin[i_d]
                    score += score_list_025k_19_des_mouth_chin[i_d]
                    score += score_list_025k_128_des_mouth_chin[i_d]
                    score += score_list_02k_none_des_mouth_chin[i_d]
                    score += score_list_025k_none_des_mouth_chin[i_d]                
                    score_list_mouth_chin.append(score)
                    
                # Total score - face
                score_list_face = []
                i_d = 0
                for i in range(counter):
                    i_d = i+counter
                    score = 0
                    score += score_list_02k_0_des_face[i]
                    score += score_list_02k_18_des_face[i]
                    score += score_list_02k_19_des_face[i]
                    score += score_list_02k_128_des_face[i]
                    score += score_list_025k_0_des_face[i]
                    score += score_list_025k_18_des_face[i]
                    score += score_list_025k_19_des_face[i]
                    score += score_list_025k_128_des_face[i]
                    score += score_list_02k_none_des_face[i]
                    score += score_list_025k_none_des_face[i]                
                    score += score_list_02k_0_des_face[i_d]
                    score += score_list_02k_18_des_face[i_d]
                    score += score_list_02k_19_des_face[i_d]
                    score += score_list_02k_128_des_face[i_d]
                    score += score_list_025k_0_des_face[i_d]
                    score += score_list_025k_18_des_face[i_d]
                    score += score_list_025k_19_des_face[i_d]
                    score += score_list_025k_128_des_face[i_d]                
                    score += score_list_02k_none_des_face[i_d]
                    score += score_list_025k_none_des_face[i_d]                
                    score_list_face.append(score)                    
            
            # Total score - Face
            total_score_face = []
            for i in range(counter):
                score = 0
                score += score_list_forehead[i]
                score += score_list_left_eye[i]
                score += score_list_right_eye[i]
                score += score_list_eyes[i]
                score += score_list_nose[i]
                score += score_list_mouth_chin[i]
                score += score_list_face[i]
                dataset_i = datasets_list[i]
                format_dict = {"person":dataset_i,"forehead":score_list_forehead[i],"left_eye":score_list_left_eye[i],"right_eye":score_list_right_eye[i],"eyes":score_list_eyes[i],"nose":score_list_nose[i],"mouth_chin":score_list_mouth_chin[i],"face":score_list_face[i],"total_score":score}
                total_score_face.append(format_dict)
    
            export_score_list.append(total_score_face)
    
        return export_score_list
            
    
"""
DESCRIPTION:

In the folder d:\python_cv\face_detector\img_in\, put the face you want to find in the database under the name face_detect.png.
In the d:\python_cv\face_detector\img_in\dataset\ folder, put the faces you want to compare.

The process of creating accelerated datasets can take several minutes, depending on how many images you have in the main database.
The scan itself then takes a few seconds, because only DES are compared and unique points on the face are searched for.

When comparing, there are many ways to compare faces.
There are many algorithms implemented in this script, but in the end I only used a few methods that I activated for the final comparison.
It is important to note that before the final implementation of face scanning, it is important to properly test the combination of different algorithms provided by this script on as large a sample of face sets as possible for the highest comparison accuracy and to test the success of the comparison sets.
It is convenient to use multiple methods at the same time for higher accuracy, which is why this Face Detector is marked with version 0.9.1, because there is still a lot of room to improve the accuracy of face matching.
"""
    
# Root path where the application is located
# It is necessary to preserve the double slash syntax as shown in the example
root_path = 'd:\\github\\python-face-detector\\'

if os.path.exists(root_path):
    # Create test datasets and create all paths
    test_images_dataset = dtc_process(root_path).test_images_dataset()
    
    # Create datasets only destionation multi-scale procedures for fast detected for repeat find face
    dtc_process(root_path).create_all_datasets_only_des_MULTI_SCALE()
    
    # Multi-Scale Fast Detected Procedure - Creating a database into an array
    datasets_path = root_path+'face_detector\\datasets\\'
    results = compare_process().matching_and_detected(8, True, False, 1)
    
    # Multi-Scale Fast Detected Procedure - Creating score evaluation
    person_used_list = []
    score_array = []
    compute_results = []
    for result_array in results:
        for result in result_array:
            person_name_1 = result["person"]
            person_con = person_name_1 in person_used_list
            if person_con:
                pass
            else:
                person_score_forehead = 0
                person_score_left_eye = 0
                person_score_right_eye = 0
                person_score_eyes = 0
                person_score_nose = 0
                person_score_mouth_chin = 0
                person_score_face = 0
                person_total_score = 0
                person_used_list.append(person_name_1)
                for person_result_array in results:
                    for person_result in person_result_array:
                        person_name_2 = person_result["person"]
                        if person_name_1 == person_name_2:
                            person_score_forehead += person_result["forehead"]
                            person_score_left_eye += person_result["left_eye"]
                            person_score_right_eye += person_result["right_eye"]
                            person_score_eyes += person_result["eyes"]
                            person_score_nose += person_result["nose"]
                            person_score_mouth_chin += person_result["mouth_chin"]
                            person_score_face += person_result["face"]
                            person_total_score += person_result["total_score"]
                score_array.append(person_total_score)
                format_dict = {"person":person_name_1,"forehead":person_score_forehead,"left_eye":person_score_left_eye,"right_eye":person_score_right_eye,"eyes":person_score_eyes,"nose":person_score_nose,"mouth_chin":person_score_mouth_chin,"face":person_score_face,"total_score":person_total_score}
                compute_results.append(format_dict)            
            
                
    # Multi-Scale Fast Detected Procedure - Creating sorted score
    score_array_sorted = sorted(score_array)
    results_array_sorted = []
    for score_sorted in score_array_sorted:
        for result in compute_results:
            if score_sorted == result["total_score"]:
                results_array_sorted.append("{0} : {1}".format(result["person"], result["total_score"]))
                img_1_path = datasets_path+"{0}\\face.png".format(matching_dataset_name)
                img_2_path = datasets_path+"{0}\\face.png".format(result["person"])
                img_1_bin = cv.imread(img_1_path, 0)
                img_2_bin = cv.imread(img_2_path, 0)
                w, h = img_2_bin.shape[::-1]
                pt.subplot(121), pt.imshow(img_1_bin, cmap = 'gray')
                pt.title('Wanted'), pt.xticks([]), pt.yticks([])
                pt.subplot(122), pt.imshow(img_2_bin, cmap = 'gray')
                pt.title('Detected'), pt.xticks([]), pt.yticks([])
                pt.suptitle("Points: {0}".format(result["total_score"]))
                pt.show()
    
    for result in results_array_sorted:
        print(result)

else:
    print('Your root path is not exist, please repair root path variable.')

