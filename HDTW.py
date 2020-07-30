# -*- coding: utf-8 -*-
"""
Copyright 2020 Yuval Burstyn & Asaf Gazit

Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
import seaborn as sns
from itertools import groupby

class HDTW(object):
    def __init__(self, signals_list=[], 
                 dtw_function_reference=None, **dtw_function_kwargs):
        self.signals_list = signals_list
        self.dtw_function_reference = dtw_function_reference
        self.dtw_function_kwargs = dtw_function_kwargs
        self.signals_for_input = None
        self.signals_for_output = None
        self.___path_matrix = None
        self.___hierarchy_paths = None
        self.___hierarchy_distances = None
    
    @property
    def path_matrix(self):
        return self.___path_matrix
    
    @property
    def hierarchy_paths(self):
        return self.___hierarchy_paths
    
    @property
    def hierarchy_distances(self):
        return self.___hierarchy_distances
    
    @property
    def ___signals_valid(self):
        if(len(self.signals_list)>1):
            return True
        else:
            return False

    @property
    def ___dtw_function_reference_valid(self):
        if(self.dtw_function_reference is not None):
            if(callable(self.dtw_function_reference)):
                return True
        return False

    @property
    def suggested_power(self):
        if(self.___signals_valid):
            return int(len(self.signals_list)).bit_length()-1
        else:
            raise Exception('Cannot make suggested_power. Signals not valid: %s found.'%len(self.signals_list))
            
    @property
    def signals(self):
        return self.signals_list
    @signals.setter
    def signals(self, new_signals_list):
        if(type(new_signals_list) is list):
            self.signals_list = new_signals_list
        else:
            raise Exception('HTDW :: signals_list must be list.')
    
    @property
    def dtw_function(self):
        return self.dtw_function_reference
    @dtw_function.setter
    def dtw_function(self, new_dtw_function_reference):
        if(callable(self.dtw_function_reference)):
            self.dtw_function_reference = new_dtw_function_reference
        else:
            raise Exception('HTDW :: dtw_function_reference must be callable (function).')
    
    # single dtw process
    def ___sdtw(self,signalA,signalB):
    # if same signal 
        if(list(signalA)==list(signalB)):
            path = [(i,i) for i in range(len(signalA))]
            distance = 0
        else:
            distance, path = self.dtw_function(signalA, signalB, **self.dtw_function_kwargs)    
        np_path = np.array(path)
        signalA_star=signalA[np_path[:,0]]
        signalB_star=signalB[np_path[:,1]]
        stacked_signals = np.vstack((signalA_star, signalB_star))
        signals_mean = np.mean(stacked_signals, axis=0)   
        return distance, np_path, signals_mean

    # level hdtw process
    def ___ldtw(self,level_signals_list):
        level_output=[]
        level_paths=[]
        level_distances=[]
        for pair_index in range(0,len(level_signals_list),2):
            signalA=level_signals_list[pair_index]
            signalB=level_signals_list[pair_index+1]
            distance, np_path, signals_mean=self.___sdtw(signalA,signalB)
            level_output.append(signals_mean)
            level_paths.append(np_path)
            level_distances.append(distance)
        return level_output, level_paths, level_distances
        
    def execute(self, verbose=True):
        if(not self.___signals_valid):
            raise Exception('Signals not valid: %s found.'%len(self.signals))
        if(not self.___dtw_function_reference_valid):
            raise Exception('Function reference not valid: "%s".'%len(self.dtw_function_reference))
        # handle 2,3,many signals
        if(len(self.signals)<2):
            raise Exception('Not enough input signals. Found: {}.'.format(len(self.signals)))   
        #no exception
        elif(len(self.signals)==2):
            if(verbose):
                print('HDTW :: 2 input signals. Executing DTW.'%len(self.signals))
            return self.signals, self.___sdtw(self.signals[0],self.signals[1])
        elif(len(self.signals)==3):
            if(verbose):
                print('HDTW :: 3 input signals. Doubling last signal.')
            #case 3 signals
            signals_for_input = self.signals.copy()
            signals_for_input.append(self.signals[-1])
            signals_for_output = self.signals.copy()
            sugg_power = 2
            #general case (4 or more signals)
        else:
            # more than 2 signals:
            # HDTW takes 2^sugg_power signals that fits in input signals
            # trim signals according to the suggested power found
            signals_for_input=self.signals.copy()[:2**self.suggested_power]
            signals_for_output=signals_for_input.copy()
            sugg_power = self.suggested_power
            if(verbose):
                print('HDTW :: Executing %s signals.'%len(signals_for_input))
        # init agg lists
        hierarchy_paths=[]
        hierarchy_distances=[]
        # exec level mapping ("way up") (signals for input propogation)
        for hierarchy_level in range(sugg_power):
            if(verbose):
                print('HDTW :: mapping level %s.'%hierarchy_level)
            signals_for_input,level_paths,level_distances=self.___ldtw(signals_for_input)
            hierarchy_paths.append(level_paths)
            hierarchy_distances.append(level_distances)
        # unfold paths ("way down")
        if(verbose):
            print('HDTW :: doubling top level (%s).'%(sugg_power-1))
        d_last_path=hierarchy_paths[-1][0]
        last_path=[d_last_path[:,0],d_last_path[:,1]]
        for hindex in range(sugg_power-1):
            current_h_level=sugg_power-hindex-1
            if(verbose):
                print('HDTW :: aggregating level %s.'%current_h_level)
            current_h_level=sugg_power-hindex-1
            current_h_level_paths=hierarchy_paths[current_h_level-1]
            #current path to single parts
            current_h_paths=[]
            for path in current_h_level_paths:
                current_h_paths.append(path[:,0])
                current_h_paths.append(path[:,1])
            #list current level paths (doubled to match current path)
            last_path_doubled=[]
            for path in last_path:
                last_path_doubled.append(path)
                last_path_doubled.append(path)
            #index respectively
            h_level_output=[]
            for iindex in range(len(last_path_doubled)):
                lp=last_path_doubled[iindex]
                cp=current_h_paths[iindex]
                h_level_output.append(cp[lp])
            last_path=h_level_output
        # transform to array
        path_matrix=np.array(last_path)[:len(self.signals)]
        self.signals_for_input = signals_for_input
        self.signals_for_output = signals_for_output
        self.___path_matrix = path_matrix
        self.___hierarchy_paths = hierarchy_paths
        self.___hierarchy_distances = hierarchy_distances
        # save
        if(verbose):
            print('HDTW :: Done.')
        return True
    
    def save(self, filename):
        if(self.___path_matrix is None):
            raise Exception('HTDW :: Cannot save. Must execute first.')
        else:
            pickle.dump([self.signals_for_output, self.path_matrix, 
                         self.hierarchy_paths, self.hierarchy_distances],
                    open( filename, 'wb' ))
            return True
    
    def load(self, filename):
        signals_for_output, path_matrix, hierarchy_paths, hierarchy_distances=\
            pickle.load(open(filename,'rb'))
        self.signals_for_output = signals_for_output
        self.signals_list = signals_for_output
        self.___path_matrix = path_matrix
        self.___hierarchy_paths = hierarchy_paths
        self.___hierarchy_distances = hierarchy_distances
        return True

    def ___pad_signals_tomartix(self, input_signals):
        maxlen=max([len(si) for si in input_signals])
        output=np.zeros((len(input_signals),maxlen))
        for idx,si in enumerate(input_signals):
            output[idx,:len(si)]=si
        return output

    @property
    def distances_matrix(self):
        res_matrix = []
        for level, distances in enumerate(self.___hierarchy_distances):
            row_values = []
            for dist in distances:
                for iteration in range(2**level):
                    row_values.append(dist)
            res_matrix.append(row_values)
        return pd.DataFrame(data=np.vstack(res_matrix[::-1]), 
                            index=np.arange(len(res_matrix))[::-1])

    def distances_plot(self, fig_size=(18,6),
                   plot_title = 'Hierarchical distances stacking', plot_title_size = 30,
                   x_axis_title = 'DTW process', x_axis_title_size = 25,
                   y_axis_title = 'Distance (stacked)', y_axis_title_size = 25,
                   major_tick_size=18, minor_tick_size=None,
                   ax_object = None, save_filename=None, color_legend=True):
          
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
            
        ax = pd.DataFrame(data=self.distances_matrix).T.plot(kind='bar', stacked=True, ax=ax, legend=True)

        ax.legend(title='Hierarchy levels')
        
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)

        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    def index_to_connection_plot(self, fig_size=(18,6),
                   plot_title = 'Record-connection plot', plot_title_size = 30,
                   x_axis_title = 'Path matrix connection index', x_axis_title_size = 25,
                   y_axis_title = 'Original record index', y_axis_title_size = 25,
                   major_tick_size=18, minor_tick_size=None,
                   ax_object = None, save_filename=None, color_legend=True):
          
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        
        ax.margins(x=0)
        for pt_id, path_traverse in enumerate(self.path_matrix):
            ax.plot(np.arange(len(path_traverse)),
                    path_traverse, label='Signal %s'%pt_id)
        
        ax.legend(loc='upper left')
        
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)

        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax
        
    @property
    def path_matrix_normalised(self):
        return self.path_matrix/[[len(s)] for s in self.signals]

    @property
    def signals_on_path_matrix(self):
        signal_on_traverse=[]
        for input_signal_id in range(len(self.signals)):
            input_signal = self.signals[input_signal_id]
            traverse = self.path_matrix[input_signal_id]
            signal_on_traverse.append(input_signal[traverse])
        return np.array(signal_on_traverse)
    
    def representative_noise_matrix(self, representative_signal):
        # ensure representative_signal is path lenth
        if(self.signals_on_path_matrix.shape[1]!=\
           len(representative_signal)):
            raise Exception('Representative_signal must be the same'+\
                            'length as the path matrix (%s != %s).'%\
                            (len(representative_signal),
                             self.signals_on_path_matrix.shape[1]))
        delta_array=[]
        for sig_op in self.signals_on_path_matrix:
            delta_array.append(sig_op-representative_signal)
        return np.array(delta_array)
    
    def representative_noise_plot(self, representative_signal, fig_size=(18,6),
                                  colour_map="YlGnBu",
                                  plot_title = 'Noise values', 
                                  plot_title_size = 25,
                                  x_axis_title = 'Alignment connection index', 
                                  x_axis_title_size = 18,
                                  y_axis_title = 'Delta from representative signal', y_axis_title_size = 18,
                                  major_tick_size=10, minor_tick_size=None,
                                  transpose_signals = True, 
                                  ax_object = None, save_filename=None, 
                                  color_legend=True, legend_signal_names=None):
        
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        ax = sns.heatmap(self.representative_noise_matrix(representative_signal),
                         cmap=colour_map,ax=ax,cbar=color_legend,
                         cbar_kws={'label': 'delta from representative signal'})
        
        ax.hlines(np.arange(1,self.path_matrix.shape[0]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)

        
        if(legend_signal_names is not None):
            legend_labels=['Signal '+str(idx)+': '+name for idx,name in 
                           enumerate(legend_signal_names)]
            ax.legend([Rectangle((0, 0), 1, 1, fc="w", fill=False,
                                 edgecolor='none', linewidth=0) for _ in legend_labels],legend_labels, 
                      bbox_to_anchor=(1.135, 1), loc='upper left')
            
        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    def representative_noise_distributions_plot(self, representative_signal, fig_size=(18,6),
                                               colour_map="YlGnBu",
                                               plot_title = 'Noise distributions', 
                                               plot_title_size = 30,
                                               x_axis_title = 'Delta from representative signal', 
                                               x_axis_title_size = 25,
                                               y_axis_title = 'Observations',
                                               y_axis_title_size = 25,
                                               transpose_signals = True,
                                               major_tick_size=18, minor_tick_size=None,
                                               ax_object = None, save_filename=None, 
                                               color_legend=True, legend_signal_names=None,
                                               **dist_plot_kwargs):
        if(legend_signal_names is None):
            legend_signal_names=np.arange(self.signals_on_path_matrix.shape[0]).astype(str)
        
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        
        for sig_delta, sig_label in zip(self.representative_noise_matrix(representative_signal),
                           legend_signal_names):
            sns.distplot(sig_delta,ax=ax,label=str(sig_label),**dist_plot_kwargs)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)

        ax.legend(title='Signals')
        
        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    @property
    def temporal_warping_matrix(self):
        return (self.path_matrix_normalised- \
                self.path_matrix_normalised.mean(axis=0)).T * 100
    
    def temporal_warping_plot(self, fig_size=(18,6),
                   colour_map="YlGnBu",
                   plot_title = 'Temporal warping indication', plot_title_size = 30,
                   x_axis_title = 'Alignment connection index', x_axis_title_size = 25,
                   y_axis_title = 'Signal', y_axis_title_size = 25,
                   major_tick_size=18, minor_tick_size=None,
                   ax_object = None, save_filename=None, 
                   color_legend=True, legend_signal_names=None):
        padded_signals = self.___pad_signals_tomartix(self.signals)
        
        lag_from_mean = self.temporal_warping_matrix
        
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        ax = sns.heatmap(lag_from_mean.T,cmap=colour_map,ax=ax,cbar=color_legend,
                         cbar_kws={'label': '% deviation from the mean time progression'})
        
        ax.hlines(np.arange(1,padded_signals.shape[1]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)

        if(legend_signal_names is not None):
            legend_labels=['Signal '+str(idx)+': '+name for idx,name in 
                           enumerate(legend_signal_names)]
            ax.legend([Rectangle((0, 0), 1, 1, fc="w", fill=False,
                                 edgecolor='none', linewidth=0) for _ in legend_labels],legend_labels, 
                      bbox_to_anchor=(1.135, 1), loc='upper left')
            
        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax
    
    @property
    def warp_factor_matrix(self):
        count_list = []
        for signal_id in range(len(self.signals)):
            u, c = np.unique(self.path_matrix[signal_id], return_counts=True)
            count_list.append(c[self.path_matrix[signal_id]])
        return np.array(count_list)

    def warp_factor_plot(self, fig_size=(18,6), 
                   colour_map="Blues",
                   plot_title = 'Warp factor (index count)', plot_title_size = 30,
                   x_axis_title = 'Alignment connection index', x_axis_title_size = 25,
                   y_axis_title = 'Signal', y_axis_title_size = 25,
                   major_tick_size=10, minor_tick_size=None,
                   transpose_signals = True, ax_object = None, save_filename=None, 
                   color_legend=True, legend_signal_names=None):
        padded_signals = self.___pad_signals_tomartix(self.signals)
        if(transpose_signals):
            padded_signals=padded_signals.T
        
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        count_list = self.warp_factor_matrix
        ax = sns.heatmap(np.array(count_list),cmap=colour_map,ax=ax,cbar=color_legend,
                         cbar_kws={'label': 'Count of index reference'})        
        
        ax.hlines(np.arange(1,padded_signals.shape[1]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
        
        if(legend_signal_names is not None):
            legend_labels=['Signal '+str(idx)+': '+name for idx,name in 
                           enumerate(legend_signal_names)]
            ax.legend([Rectangle((0, 0), 1, 1, fc="w", fill=False,
                                 edgecolor='none', linewidth=0) for _ in legend_labels],legend_labels, 
                      bbox_to_anchor=(1.135, 1), loc='upper left')
        
        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    def indications_plot(self, fig_size=(18,6), number_of_lines=100, 
                   colour_map="YlGnBu",
                   plot_title = 'Input & HDTW indications', plot_title_size = 30,
                   x_axis_title = 'Original records index', x_axis_title_size = 25,
                   y_axis_title = 'Signal', y_axis_title_size = 25,
                   major_tick_size=18, minor_tick_size=None,
                   transpose_signals = True, ax_object = None, save_filename=None,
                   color_legend=True, legend_signal_names=None):
        path_matrix = self.path_matrix.copy()
        padded_signals = self.___pad_signals_tomartix(self.signals)
        if(transpose_signals):
            padded_signals=padded_signals.T
            path_matrix=path_matrix.T
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
        else:
            ax=ax_object
        ygrid=np.linspace(0,path_matrix.shape[0]-1,number_of_lines).astype(int)
        linesarray = np.zeros((path_matrix.shape[1],ygrid.shape[0]))
        for idx in range(path_matrix.shape[1]):
            linesarray[idx,:]= path_matrix[ygrid,idx]
        ax = sns.heatmap(padded_signals.T,cmap=colour_map,ax=ax,cbar=color_legend,
                         cbar_kws={'label': 'Signal value'})
        for lidx in range(linesarray.shape[1]):
            ax.plot(linesarray[:,lidx],np.arange(0.5,padded_signals.shape[1]),alpha=0.5,c='k')
        ax.hlines(np.arange(1,padded_signals.shape[1]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
        
        if(legend_signal_names is not None):
            legend_labels=['Signal %s: %s'%(idx,name) for idx,name in 
                           enumerate(legend_signal_names)]
            ax.legend([Rectangle((0, 0), 1, 1, fc="w", fill=False,
                                 edgecolor='none', linewidth=0) for _ in legend_labels],legend_labels, 
                      bbox_to_anchor=(1.15, 1), loc='upper left')

        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    def aligned_output_plot(self, fig_size=(18,6),  
                   colour_map="YlGnBu",
                   plot_title = 'HDTW aligned output', plot_title_size = 30,
                   x_axis_title = 'Alignment connection index', x_axis_title_size = 25,
                   y_axis_title = 'Signal', y_axis_title_size = 25,
                   major_tick_size=18, minor_tick_size=None,
                   transpose_signals = True, ax_object = None, save_filename=None,
                   color_legend=True, legend_signal_names=None):
        path_matrix = self.path_matrix.copy()
        padded_signals = self.___pad_signals_tomartix(self.signals)
        if(transpose_signals):
            padded_signals=padded_signals.T
            path_matrix=path_matrix.T
            
        HDTW_on_signals = []
        for sidx in range(len(self.signals)):
            track = path_matrix.T[sidx,:].astype(int)
            tsignal = self.signals[sidx]
            hdtw_track = tsignal[track]
            HDTW_on_signals.append(hdtw_track)
        
        if(ax_object is None):
            fig, ax = plt.subplots(figsize=fig_size)
            fig.tight_layout()
        else:
            ax=ax_object
        ax = sns.heatmap(np.array(HDTW_on_signals),
                         cmap=colour_map, ax=ax,cbar=color_legend,
                         cbar_kws={'label': 'Signal value'})
        
        ax.hlines(np.arange(1,padded_signals.shape[1]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
        ax.grid(False)
        ax.set_title(plot_title,fontsize=plot_title_size)
        ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
        ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
        
        if(type(major_tick_size) is int):
            ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
        if(type(minor_tick_size) is int):
            ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
        
        if(legend_signal_names is not None):
            legend_labels=['Signal '+str(idx)+': '+name for idx,name in 
                           enumerate(legend_signal_names)]
            ax.legend([Rectangle((0, 0), 1, 1, fc="w", fill=False,
                                 edgecolor='none', linewidth=0) for _ in legend_labels],legend_labels, 
                      bbox_to_anchor=(1.135, 1), loc='upper left')
        
        if(ax_object is None):
            if(save_filename is not None):
                fig.savefig(save_filename, dpi=300, bbox_inches="tight")
            else:
                plt.show()
            plt.close()
        else:
            return ax

    def signal_features_alignment(self, 
                                  features_mapping_list, 
                                  threshold_value=0,
                                  threshold_function=None,
                                  show_plot=True,
                                  save_filename=None,
                                  fig_size=(18,6),
                                  colour_map="YlGnBu",
                                  plot_title = 'Features on aligned signal', plot_title_size = 25,
                                  x_axis_title = '', x_axis_title_size = 18,
                                  y_axis_title = '', y_axis_title_size = 18,
                                  major_tick_size=10, minor_tick_size=None,
                                  transpose_signals = True, 
                                  **threshold_function_kwargs):
        # ensure function paramater are valid :
        # check features_mapping_list has items as signals
        if(len(self.signals) != len(features_mapping_list)):
            raise Exception('features_mapping_list must be same size as signals.')

        # if threshold value - int
        if(threshold_value is not None):
            if(not (isinstance(threshold_value, int) or isinstance(threshold_value, float))):
                raise Exception('threshold_value must be a number.')

        # if threshold function - callable
        if(threshold_function is not None):
            if(not callable(threshold_function)):
                raise Exception('threshold_function must be callable (function).')

        # signal features alignment: features list (item per signal)
        peak_traverses = np.zeros(self.path_matrix.shape)
        for signal_id in range(len(self.signals)):
            peak_traverses[signal_id, 
                       np.where(np.isin(self.path_matrix[signal_id], 
                                features_mapping_list[signal_id]))] = 1
        # thresholding 
        agreement_signal=peak_traverses.sum(axis=0)
        if(threshold_function is not None):
            threshold_peaks = threshold_function(agreement_signal,
                                                 **threshold_function_kwargs)
        else:
            agree_signal_majority=agreement_signal>=threshold_value
            grpsoutput = np.array([(key, len(list(group))) for key, group\
                                   in groupby(agree_signal_majority==True)])
            csgrpsoutput = np.cumsum(grpsoutput[:,1])-1
            groupedarray = np.zeros((csgrpsoutput.shape[0],3)).astype(int)
            groupedarray[:,:2] = grpsoutput
            groupedarray[:,2] = csgrpsoutput
            threshold_peaks=\
                groupedarray[groupedarray[:,0]==1,2]-\
                (groupedarray[groupedarray[:,0]==1,1]/2).astype(int)
            
        thresholded_peaks = threshold_peaks[agreement_signal[threshold_peaks]>=threshold_value]
        if(show_plot):
            fig, ax = plt.subplots(figsize=fig_size)
            ax.plot(np.arange(len(agreement_signal)),agreement_signal,alpha=0.8,c='k')
            ax.scatter(thresholded_peaks,agreement_signal[thresholded_peaks])
            ax.fill_between(np.arange(len(agreement_signal)),
                            0, 
                            agreement_signal,
                            facecolor='black', interpolate=True)
            ax.margins(x=0)
            ax.hlines(threshold_value, *ax.get_xlim(),colors='k',
                linestyles='dashed',linewidth=0.4,alpha=0.7, label='Applied \nthreshold')
            ax.legend(bbox_to_anchor=(1.18,1.02))
            fig.tight_layout()
            ax.hlines(np.arange(1,self.path_matrix.shape[0]), *ax.get_xlim(),colors='w',linewidth=0.4,alpha=0.2)
            ax.grid(False)
            ax.set_title(plot_title,fontsize=plot_title_size)
            ax.set_xlabel(x_axis_title,fontsize=x_axis_title_size)
            ax.set_ylabel(y_axis_title,fontsize=y_axis_title_size)
            if(type(major_tick_size) is int):
                ax.tick_params(axis='both', which='major', labelsize=major_tick_size)
            if(type(minor_tick_size) is int):
                ax.tick_params(axis='both', which='minor', labelsize=minor_tick_size)
            if(show_plot):
                if(save_filename is not None):
                    fig.savefig(save_filename, dpi=300, bbox_inches="tight")
                plt.close()
        return thresholded_peaks

    def HDTW_report_figure(self, save_filename='HDTW_report.pdf',fig_size=(25, 31),
                           traverses_names=None):
        if(traverses_names is None):
            traverses_names = np.arange(len(self.signals)).astype(str)
        
        fig = plt.figure(figsize=fig_size) 
        gs = gridspec.GridSpec(4, 2, width_ratios=[3, 1]) 
        ax0 = plt.subplot(gs[0])
        ax0 = self.indications_plot(legend_signal_names=traverses_names,
                                    ax_object=ax0)
        ax1 = plt.subplot(gs[1])
        ax1 = self.distances_plot(ax_object=ax1)
        
        ax2 = plt.subplot(gs[2])
        ax2 = self.aligned_output_plot(legend_signal_names=traverses_names,
                                       ax_object=ax2)
        ax3 = plt.subplot(gs[3])
        ax3 = self.index_to_connection_plot(ax_object=ax3)
        
        ax4 = plt.subplot(gs[4])
        ax4 = self.temporal_warping_plot(legend_signal_names=traverses_names,
                                         ax_object=ax4)
        ax5 = plt.subplot(gs[5])
        ax5 = self.representative_noise_distributions_plot(np.median(self.signals_on_path_matrix,axis=0),
                                                           ax_object=ax5,
                                                           plot_title = 'Noise Distributions (median)')
        ax6 = plt.subplot(gs[6])
        ax6 = self.warp_factor_plot(legend_signal_names=traverses_names,
                                    ax_object=ax6)
        ax7 = plt.subplot(gs[7])
        ax7 = self.representative_noise_distributions_plot(np.mean(self.signals_on_path_matrix,axis=0),
                                                           ax_object=ax7,
                                                           plot_title = 'Noise Distributions (mean)')
        plt.tight_layout()
        plt.savefig(save_filename,dpi=300, bbox_inches="tight")
        plt.close()

#eof