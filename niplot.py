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

import holoviews as hv
from holoviews import opts, dim
from bokeh.io import show
hv.extension('bokeh')
hv.output(size=200)


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
        
        
# TO-DO: Implement rotated labels
# https://stackoverflow.com/questions/65561927/inverted-label-text-half-turn-for-chord-diagram-on-holoviews-with-bokeh
# TO-DO: Make plotting work (I can only save the plot as html but can't plot it directly, at least in spyder...)
def plot_connectogram(connectivity_matrix,atlas_labels,atlas_indices,threshold=None,chord_type=int,dst_dir=None,filename=None):
    '''Plot a connectivity matrix as a connectogram.
    

    Parameters
    ----------
    connectivity_matrix : np.array
        A symmetric connectivity matrix.
    atlas_labels : pd.Series or list
        A list-like object providing names of each atlas region.
    atlas_indices : pd.Series or list
        A list-like object providing indices of each atlas region.
    threshold : float or int, optional
        Apply a threshold to the connectivity matrix before plotting. Only connectvity
        values that are greater or equal than this threshold are visualized. 
    chord_type : int or float, optional
        Convert the connectivity values to float or int type. If the weight values 
        are integers, they define the number of chords to 
        be drawn between the source and target nodes directly. If the weights 
        are floating point values, they are normalized to a default of 500 chords, 
        which are divided up among the edges. Any non-zero weight will be assigned 
        at least one chord. The default is int.
    dst_dir : str, optional
        Name of the output directory. The default is None.
    filename : str, optional
        Name of the file (must be provided including the extenstion). 
        The default is None.


    Returns
    -------
    connectogram_plot : holoviews.element.graphs.Chord
        The connectogram plot object.

    '''
    
    # copy matrix
    connectivity_matrix = connectivity_matrix.copy()
    
    # set lower triangle to NaN (since matrix is symmetric we want to remove duplicates)
    il = np.tril_indices(len(connectivity_matrix))
    connectivity_matrix[il] = np.nan
    
    # convert to pd.DataFrame for further processing
    connectivity_matrix_df = pd.DataFrame(data=connectivity_matrix,
                                          columns=atlas_indices,
                                          index=atlas_indices)
    
    # Ensure that index name has the default name 'Index'
    if connectivity_matrix_df.index.name:
        connectivity_matrix_df.index.name = None
    
    # stack connectivity_matrix
    connectivity_matrix_stacked = connectivity_matrix_df.stack().reset_index()
    connectivity_matrix_stacked.columns=['source','target','value']
    
    if chord_type == int:
        connectivity_matrix_stacked = connectivity_matrix_stacked.astype(int)
    
    # reduce to only connections that are not 0
    connectivity_matrix_stacked = connectivity_matrix_stacked.loc[connectivity_matrix_stacked['value'] != 0,:]
    
    # Optional: reduce to only connections >= threshold
    if threshold:
        connectivity_matrix_stacked = connectivity_matrix_stacked.loc[connectivity_matrix_stacked['value'] >= threshold,:]
    
    # add node infos and show only nodes that also have a connection after subsetting to 
    # connections that are not zero and (optionally) connections that pass the specified threshold
    atlas_df = pd.DataFrame({'region_id':atlas_indices,'label':atlas_labels})
    nodes_to_show = np.unique(connectivity_matrix_stacked[['source','target']].values)
    atlas_df = atlas_df.loc[atlas_df['region_id'].isin(nodes_to_show)]
    nodes = hv.Dataset(atlas_df,'region_id','label')
    
    # create plot 
    connectogram_plot = hv.Chord((connectivity_matrix_stacked,nodes),['source','target'],['value'])
    connectogram_plot.opts(opts.Chord(cmap='Category20',
                                      edge_cmap='Category20',
                                      edge_color=dim('source').str(),
                                      node_color=dim('region_id').str(),
                                      labels='label'
                                      ))
    
    # save plot 
    if dst_dir:
        if not filename:
            raise ValueError('Please provide a filename')

        dst_path = dst_dir + filename
        hv.save(connectogram_plot,dst_path)
    
    # FIXME: this doesn't work for me in Spyder
    show(hv.render(connectogram_plot))
    
    return connectogram_plot

def plot_connectivity_matrix(connectivity_matrix,atlas_labels,threshold=None,dst_dir=None,filename=None):
    '''Plot a connectivity matrix. This function is very similar to nilearn.plotting.plot_matrix
    but uses the bokeh backend and is therefore interactive
    
    Parameters
    ----------
    connectivity_matrix : np.array
        A symmetric connectivity matrix.
    atlas_labels : pd.Series or list
        A list-like object providing names of each atlas region.
    threshold : float or int, optional
        Apply a threshold to the connectivity matrix before plotting. Values 
        lower than this threshold will be set to 0.
    dst_dir : str, optional
        Name of the output directory. The default is None.
    filename : str, optional
        Name of the file (must be provided including the extenstion). 
        The default is None.


    Returns
    -------
    connectogram_plot : holoviews.element.raster.HeatMap
        The the connectivity matrix plot object.


    '''

    
    # copy matrix
    connectivity_matrix = connectivity_matrix.copy()

    # convert to pd.DataFrame for further processing
    connectivity_matrix_df = pd.DataFrame(data=connectivity_matrix,
                                          columns=atlas_labels,
                                          index=atlas_labels)
    
    # Ensure that index name has the default name 'Index'
    if connectivity_matrix_df.index.name:
        connectivity_matrix_df.index.name = None
    
    # stack connectivity_matrix
    connectivity_matrix_stacked = connectivity_matrix_df.stack().reset_index()
    connectivity_matrix_stacked.columns=['source','target','value']
    
    if threshold:
        connectivity_matrix_stacked['value'].where(connectivity_matrix_stacked['value'] >= threshold,0,inplace=True)
    
    connectivity_matrix_stacked_ds = hv.Dataset(connectivity_matrix_stacked,['source','target'])
    heatmap = hv.HeatMap(connectivity_matrix_stacked_ds)
    heatmap.opts(opts.HeatMap(tools=['hover'],
                              colorbar=True,
                              xaxis='bare',
                              yaxis='bare',
                              cmap='blues_r'))
    
    # save plot 
    if dst_dir:
        if not filename:
            raise ValueError('Please provide a filename')

        dst_path = dst_dir + filename
        hv.save(heatmap,dst_path)
    
    # FIXME: this doesn't work for me in Spyder
    show(hv.render(heatmap))
    
    return heatmap