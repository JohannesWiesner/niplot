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
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import holoviews as hv
from holoviews import opts, dim
from bokeh.io import show
hv.extension('bokeh')
hv.output(size=200)

import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.express as px

####################################################################################################
## Statistical images  #############################################################################
####################################################################################################

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

####################################################################################################
## Binary MRI-images ###############################################################################
####################################################################################################

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
        
####################################################################################################
## DTI #############################################################################################
####################################################################################################

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

####################################################################################################
## Network Control Theory ##########################################################################
####################################################################################################

def plot_state_to_state_transitions_heat(E,node_names,transition_names):
    '''Plot all state-to-state-transitions as heatmaps with slider'''

    n_transitions = E.shape[2]
    
    data = [go.Heatmap(visible=False,z=E[:,:,n].T,y=node_names) for n in range(n_transitions)]
    steps= [{'label':name,'method':'update','args': [{'visible':[v == n for v in range(n_transitions)]}]} for n,name in enumerate(transition_names)]
    
    fig = go.Figure(data=data)
    fig = fig.update_layout(
        xaxis_title='T',
        yaxis_title='Area',
        sliders=[{"active":0,"steps":steps}]
        )
    
    # TO-DO: is the following line right? (Show first state-to-state-transition)
    fig.data[0].visible = True

    return fig

def plot_state_to_state_transitions_line(E,node_names,transition_names):
    '''Plot all state-to-state-transitions as lineplots with slider'''
    
    n_nodes = E.shape[1]
    n_transitions = E.shape[2]
    
    # create list of traces 
    # FIXME: could be optimized, because we know the number of traces in advance
    traces = []
    
    for t in range(n_transitions):
        for n,node_name in enumerate(node_names):
            traces.append(go.Scattergl(y=E[:,n,t],name=node_name,showlegend=False))
    
    # create a list of step dictionaries as required for slider
    steps = []
    current_min = 0
    current_max = n_nodes
    
    for name in transition_names:
    
        visible = [False for idx in range(n_transitions*n_nodes)]    
    
        for idx in range(current_min,current_max):
            visible[idx] = True
            
        step = {'label':name,'method':'update','args':[{'visible':visible}]}
        steps.append(step)
             
        current_min = current_max
        current_max = current_max + n_nodes
    
    # create figure
    fig = go.Figure(data=traces)
    fig = fig.update_layout(
        xaxis_title='T',
        yaxis_title='u',
        sliders=[{"active":0,"steps":steps}]
        )
    
    # TO-DO: is the following line right? (Show first state-to-state-transition)
    fig.data[1].visible = True

    return fig

####################################################################################################
## fMRI ############################################################################################
####################################################################################################

# TODO: Should include something like n_keys which means that a random set
# of keys is selected from the matrix in order to
# FIXME: n_cols should be named n_random_matrix_columns
def plot_matrix_dict(matrix_dict,labels,subplot_cols,subplot_rows,n_cols=None,rng=None,colors=None):
    '''Plot a dictionary of matrices as lineplots. Common usecase: Each key is a subject id
    and every matrix represents the fmri time series of that subject (where the
    rows in the matrix represent time and the columns represent regions). Each
    subplot corresponds to one matrix.

    Parameters
    ----------
    matrix_dict : dict
        A dictionary of 2D numpy arrays.
    labels : list-like
        The column labels for the 2D array. When plotting fmri time series,
        this corresponds to the names of the brain regions
    subplot_cols : int
        Number of subplot columns. The product of subplot_cols *
        subplot_rows has to equal the number of subplots.
    subplot_rows : int
        Number of subplot rows. The product of subplot_cols *
        subplot_rows has to equal the number of subplots.
    n_cols : int, optional
        If specified, a random set of columns of that size is sampled from the matrices.
        This helps, when the number of columns is very big. The default is None.
    rng : numpy.random.default_rng, optional
        A random number generator. Needed when n_cols is specified. The default is None.
    colors: plotly.colors palette, optional
        A color palette from plotly. If not specified uses default colors.

    Returns
    -------
    fig : plotly.figure

    '''


    # if n_cols is specified get a random sample of indices for n_cols
    # otherwise plot all matrix columns
    if n_cols:
        try:
            column_indices = rng.choice(range(len(labels)),n_cols,replace=False)
        except AttributeError:
            print('Please provide a random number generator')
            return
    else:
        column_indices = range(len(labels))

    # check that each matrix gets one subplot
    n_subplots = subplot_rows * subplot_cols
    if n_subplots != len(matrix_dict):
      raise ValueError('The number of rows and columns must equal the number of provided matrices')

    # get all possible subplot_indices for the subplots
    # TODO: There might be a more elegant way to do this. Can't we just
    # return all the axes like we can do with matplotlib so we can just
    # iterate over a list of axes?
    subplot_indices = []
    for row in range(1, subplot_rows + 1):
         for col in range(1, subplot_cols + 1):
            subplot_indices.append([row, col])

    # create subplot for each matrix in the list
    if not colors:
        colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    fig = make_subplots(rows=subplot_rows,cols=subplot_cols,subplot_titles=list(matrix_dict.keys()))

    # fill each subplot with the time series of that matrix using n_cols
    for matrix,(row,col) in zip(matrix_dict.values(),subplot_indices):
        for idx,col_idx in enumerate(column_indices):
            trace = go.Scattergl(y=matrix[:,col_idx],
                                 line=dict(color=colors[idx]),
                                 name=labels[col_idx],
                                 legendgroup=labels[col_idx]
                                 )
            fig.append_trace(trace,row,col)

    # delete duplicate legend entries
    names = set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))

    # cosmetics
    fig.update_xaxes(title='Scan Number')
    fig.update_yaxes(title='Signal')

    fig.show()

    return fig
