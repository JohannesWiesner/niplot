#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A module for plotting information extracted from Nifti-Images.

@author: jwiesner
"""

import numpy as np
import pandas as pd

from nilearn.image import resample_to_img
from nilearn.masking import apply_mask

from itertools import combinations
from nicalc import get_similarity_matrix,calculate_overlap

import seaborn as sns
import matplotlib.pyplot as plt

# TODO: add docstring.
def distplot_imgs(imgs_dict,mask_img,ignore_zeros=True,xlabel='Value',
                  title=None,dst_dir=None,filename=None,**kwargs):

    # extract data
    imgs_names = []
    imgs_data = []
    
    for name,img in imgs_dict.items():
        
        imgs_names.append(name)
        
        try:
            imgs_data.append(apply_mask(img,mask_img))
        except ValueError:
            img = resample_to_img(img,mask_img)
            imgs_data.append(apply_mask(img,mask_img))
            print('Images where resampled to mask resolution')
            
    if ignore_zeros == True:
        imgs_data = [img_data[img_data != 0] for img_data in imgs_data]
    
    # plot data
    plt.figure()
    for name,img_data in zip(imgs_names,imgs_data):
        sns.distplot(img_data,label=name,**kwargs)
    
    # add plot information
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)
    
    # only plot legend if multiple images are given
    if len(imgs_dict) > 1:
        plt.legend()
    
    # set figure background transparent
    plt.gcf().patch.set_alpha(0.0)
    plt.tight_layout()
    
    if dst_dir:
        
        if not filename:
            raise ValueError('Please provide a filename')
        
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)
    
    plt.show()

# TODO: add docstring.
def barplot_mask_imgs(mask_imgs_dict,mask_img,xlabel='Mask Image',ylabel='Size',
                      title=None,dst_dir=None,filename=None,**kwargs):
    
    # extract data
    mask_imgs_names = []
    mask_imgs_data = []
    
    for name,img in mask_imgs_dict.items():
        mask_imgs_names.append(name)
        mask_imgs_data.append(apply_mask(img,mask_img))
    
    # count number of nonzero voxels (aka. size of the masks)
    n_nonzero_voxels = [np.count_nonzero(mask_img_data) for mask_img_data in mask_imgs_data]
    
    # plot data
    plt.figure()
    sns.barplot(x=mask_imgs_names,y=n_nonzero_voxels,**kwargs)
    
    # add plot information
    plt.xlabel('Mask Image')
    plt.ylabel('Size')
    plt.title(title)
    
    # set figure background transparent
    plt.gcf().patch.set_alpha(0.0)
    plt.tight_layout()
    
    if dst_dir:
        
        if not filename:
            raise ValueError('Please provide a filename')
        
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)
    
    plt.show()
    
# TODO. add docstring
def plot_img_similarity(img_dict,mask_img,similarity_type='cosine_similarity',
                        show_spines=False,dst_dir=None,filename=None):
    
    similarity_matrix = get_similarity_matrix(img_dict,mask_img,similarity_type)
    
    if similarity_type == 'cosine_similarity':
        vmin_vmax = {'vmin':0,'vmax':1}
    else:
        vmin_vmax = {'vmin':-1,'vmax':1}
    
    # create heatmap
    ax = sns.heatmap(similarity_matrix,
                     square=True,
                     cmap='Blues',
                     annot=True,
                     **vmin_vmax)
    
    if show_spines == True:
        for _, spine in ax.spines.items():
            spine.set_visible(True)
    
    plt.yticks(rotation=0) 
    
    if dst_dir:
        if not filename:
            raise ValueError('Please provide a filename')
    
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)

# TODO: add docstring.
# FIMXE: Follow same structure as in plot_img_similarity
# Create get_mask_img_overlap_matrix function
def plot_mask_img_overlap(mask_img_dict,proportion_type='first',dst_dir=None,filename=None):
    
    # get keys and values from dictionary as lists
    mask_img_dict_keys = list(mask_img_dict.keys())
    mask_img_dict_values = list(mask_img_dict.values())
    
    # calculate overlaps between all mask images
    overlap_proportions = [calculate_overlap(first_mask,second_mask,proportion_type=proportion_type) for first_mask,second_mask in combinations(mask_img_dict_values,2)]
    mask_img_dict_keys_combis = [(first_key,second_key) for first_key,second_key in combinations(mask_img_dict_keys,2)]
    
    # create empty data frame
    n_imgs = len(mask_img_dict_keys)
    df = pd.DataFrame(np.zeros((n_imgs,n_imgs)))
    df = df.reindex(mask_img_dict_keys)
    df.columns = mask_img_dict_keys
    
    # fill data frame with overlap values
    for idx,combi in enumerate(mask_img_dict_keys_combis):
        df.loc[combi[0],combi[1]] = overlap_proportions[idx]
        
    # create heatmap
    ax = sns.heatmap(df,square=True,cmap='Blues',annot=True,vmin=0,vmax=100)
    
    if proportion_type == 'first':
        ax.set_xlabel('Reference Image',fontsize=12,labelpad=15)
    
    plt.tight_layout()
    
    if dst_dir:
        if not filename:
            raise ValueError('Please provide a filename')
    
        dst_path = dst_dir + filename
        plt.savefig(dst_path,dpi=600)