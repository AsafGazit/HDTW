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

#imports 

from HDTW import HDTW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from dtaidistance import dtw
from pyampd.ampd import find_peaks
from scipy import signal
from PIL import ImageFilter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
# support functions

# loading laser ablation
def LA_to_PCA(sample_xlfile, sample_sheetname=False,PCA_Columns=
              ['Mg_ppm_m24','Sr_ppm_m88','Ba_ppm_m137','U_ppm_m238']):
    '''
    The function reads .xlsx file, performs PCA and returns the 1st 
     principle component.
     
    :param sample_xlfile: .xlsx file name
    :param sample_sheetname: sheet name in xlsx file (optional)
    :PCA_Columns: columns to include in PCA
    '''
    if(sample_sheetname!=False):
        # load file
        sampledf = pd.read_excel(sample_xlfile, sheet_name=sample_sheetname)
    else:
        sampledf = pd.read_excel(sample_xlfile)
    sampledf=sampledf.loc[:,PCA_Columns] # keep only relevant data
    colmask = np.ones((len(sampledf))) # mask nans
    colmask[np.where(np.isnan(sampledf)==True)[0]]=0
    # remove nan's for standartisation and PCA
    sampledf = sampledf.loc[colmask.astype(bool),:]
    PCA_data_standard = StandardScaler().fit_transform(sampledf.values.T) # 
    pcao = PCA(n_components=len(PCA_Columns)) # define PCA components
    samplePCA = pcao.fit_transform(PCA_data_standard.T) # perform PCA
    #return samplePCA, pcao, PCA_Columns, sampledf # signal for methodolegy(PC1)
    return samplePCA[:,0]

# loading confocal fluorescence image
def image_to_signals(image_file_name, n_signals=4, blur_radius=5):
    '''
    The function reads an image file, performs a gaussian blur and samples
     the required number of traverses (n_signals).
     
    :param image_file_name: image file name
    :param n_signals: number of signals to sample
    :param blur_radius: radius (in pixels) of gaussian blur to apply
    '''
    image_array = Image.open(image_file_name)
    image_array.load()
    image_data = np.asarray(image_array, dtype='int32')
    Image.fromarray(np.uint8(image_data))
    image_array = image_array.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    image_array.load()
    image_data = np.asarray( image_array, dtype='int32')
    # image_data = image_data.mean(axis=2)
    green_channel=image_data[:,:,1]/image_data[:,:,1].max()
    current_signals=[]
    sampling_width=int(green_channel.shape[1]/n_signals)
    for samplesignal_left in range(0,green_channel.shape[1],sampling_width):
        current_signals.append(green_channel[:,samplesignal_left:samplesignal_left+sampling_width].mean(axis=1))
    return current_signals[:n_signals]

# underlying DTW function for HDTW
def dta_dtw(signalA, signalB, **dtw_kwargs):
    '''
    The function bundles the path and distance of the dtaidistance package.
    This is the underlying process to be applied in the HDTW process.
     
    :param signalA: The first signal to apply DTW on
    :param signalB: The second signal to apply DTW on
    :param **dtw_kwargs: any key-word arguments to be propogated to the functions.
    '''
    return dtw.distance_fast(signalA, signalB,**dtw_kwargs), \
           dtw.warping_path(signalA, signalB, **dtw_kwargs)


def find_peaks_SG(sample_data,window_size):
    '''
    The function takes the sample_data and returns suspected peaks indecies.
    This function utilises the 1st and 2nd derivative approximation by applying
     a Savitzky-Golay filter and comparing those to 0 and negative, respectively.
    The suspected peaks are then fine tuned to address the bias of the derivative
     approximation.
     
    :param sample_data: the signal to find peaks on
    :param window_size: window size for the derivative approximation
    '''
    # adjust window size    
    if(int(window_size)%2==0):
        window_size=int(window_size)-1
    else:
        window_size=int(window_size)
    # get 1st and 2nd derivative approximation
    sample_SG_D1=signal.savgol_filter(sample_data, window_size, 
                                      min(5,int(window_size)-1), deriv=1)
    sample_SG_D2=signal.savgol_filter(sample_data, window_size, 
                                      min(5,int(window_size)-1), deriv=2)
    # derivative 1 changes sign (~0)
    der1zero=np.where(np.diff(np.sign(sample_SG_D1)))[0]
    # derivative 2 negative (<0)
    der2negative=np.where(sample_SG_D2<0)[0]
    # find poi with both conditions
    derpeaksmask=np.isin(der1zero,der2negative)
    derpeaks=der1zero[derpeaksmask]
    # fine tune location of found peaks
    fine_tuned_peaks=[]
    ft_half_window_size=max(2,int(round((window_size**0.5),0)))
    for peak_sindex in derpeaks:
        ft_peak=sample_data[max(0,peak_sindex-ft_half_window_size):\
                            min(len(sample_data),peak_sindex+ft_half_window_size+1)].argmax()
        ft_peak_loc=peak_sindex-ft_half_window_size+ft_peak
        fine_tuned_peaks.append(ft_peak_loc)
    return np.unique(fine_tuned_peaks)

def age_scan(HDTW_object, age_estimation, age_estimation_deviation,
             age_scan_range, age_iterations,
             threshold_value, threshold_function,
             adjust_age_estimation=None,
             save_plots=False, save_name_suffix='', skip_plots=False,
             plt_axes=None, **peak_finding_kwargs):
    '''
    The function takes the sample_data and returns suspected peaks indices.
    This function utilises the 1st and 2nd derivative approximation by applying
     a Savitzky-Golay filter and comparing those to 0 and negative, respectively.
    The suspected peaks are then fine tuned to address the bias of the derivative
     approximation.
     
     :param  HDTW_object (HDTW object): the relevant (executed) HDTW object to 
         base the age scan on.
     :param  Age_estimation (int): the age estimation of the sample.Used for 
         the range of values to test and as the target of the age scan (to 
         threshold the ranked peaks). If this is not the desired value, please 
         use ‘Adjust_age_estimation’.
     :param age_estimation_deviation(int): the age estimation deviation (+/-) 
         from ‘Age_estimation’  that represents the age uncertainty.
     :param Age_scan_range (float): this ratio expands the scan range (both 
         upper and lower limits).
     :param Age_iterations (int): the number of tests to perform in the test 
         range.
     :param Threshold_value (int): for peak alignment, to recognise a feature 
         as non-local.
     :param Threshold_function (function, callable): for peak alignment, to 
         recognise a feature as non-local (custom filter function).
     :param Adjust_age_estimation: if the age estimation is not the desired threshold value target, use this variable to indicate a different value.
     :param Save_plots (bool, default False): to save the plot generated.
     :param Save_name_suffix (str): to apply when saving (must have graphic file suffix).
     :param Skip_plots (bool, default False): indicates if to skip the plot generation and return the function data.
     :param Plt_axes (matplotlib.axes object, default None): if passed, the plot is generated on the Plt_axes and returned by the function.
     :param **threshold_function_kwargs (dict, default empty): any key-word arguments to pass to the threshold_function.
     
    '''
    
    age_iterations_results=[]
    iteration_lin_space=np.linspace((age_estimation-age_estimation_deviation)*(1-age_scan_range),\
                               (age_estimation+age_estimation_deviation)*(1+age_scan_range),\
                               age_iterations)
    
    for age_est in iteration_lin_space:
        signal_peaks = [find_peaks_SG(sig, int(len(sig)/age_est)) for sig in HDTW_object.signals]
        age_iterations_results.append(HDTW_object.signal_features_alignment(signal_peaks,
                                         threshold_value=threshold_value, 
                                         threshold_function=threshold_function,
                                         show_plot=True,**peak_finding_kwargs))
    
    long_usets=np.zeros((len(age_iterations_results),HDTW_object.path_matrix.shape[1]))
    for ids,s in enumerate(age_iterations_results):
        long_uset = np.array([0]*HDTW_object.path_matrix.shape[1])
        long_uset[s]=1
        long_usets[ids,:]=long_uset
    unique, counts = np.unique(long_usets.sum(axis=0), return_counts=True)
    unique=unique[::-1][:-1]
    counts=counts[::-1][:-1]
    counts_cumsum=np.cumsum(counts)
    
    if(adjust_age_estimation is not None):
        best_fit_index= np.abs(counts_cumsum - (adjust_age_estimation)).argmin()
    else:
        best_fit_index= np.abs(counts_cumsum - (age_estimation)).argmin()

#    unique_space=np.linspace(unique.max(),unique.min(),20)
#    z = np.polyfit(unique, counts_cumsum, 2)
#    p = np.poly1d(z)
#    cs_mean=counts_cumsum.mean()
#    cs_std=counts_cumsum.std()

    ag_peaks = find_peaks(long_usets.sum(axis=0), scale=10)
    # remove y=min
    ag_peaks_mask = long_usets.sum(axis=0)[ag_peaks]!=long_usets.sum(axis=0).min()
    ag_peaks=ag_peaks[ag_peaks_mask]
    
    if(not skip_plots):
        if(plt_axes is None):
            fig, ax = plt.subplots(figsize=(18,9))
        else:
            ax=plt_axes[0]

        ax.plot(np.arange(long_usets.shape[1]),long_usets.sum(axis=0))
        ax.scatter(ag_peaks,long_usets.sum(axis=0)[ag_peaks], label='Identified stacked peak')
        ax.axhline(y=unique[best_fit_index],linewidth=1, color='r', label='Estimated age threshold')
        ax.set_xlim(xmin=0)
        
        ax.set_title('Windows peak stacking',fontsize=30)
        ax.set_xlabel('Alignment connection index',fontsize=25)
        ax.set_ylabel('Total stacked windows',fontsize=25)
        
        ax.legend(bbox_to_anchor=(1.1,1.02))
        
        if(save_plots):
            fig.savefig('count_windows_'+save_name_suffix, dpi=300, bbox_inches="tight")
            plt.close()
        elif(plt_axes is None):
            plt.show()
            
        if(plt_axes is None):
            fig, ax2 = plt.subplots(figsize=(18,9))
        else:
            ax2=plt_axes[1]
        
        ax2.plot(unique, counts_cumsum)
#        ax2.plot(unique_space,p(unique_space),"r--")
        ax2.scatter([unique[best_fit_index]],[counts_cumsum[best_fit_index]],
                    label='Estimated age threshold')
#        ax2.axhline(y=cs_mean,linewidth=1, color='g')
#        ax2.axhline(y=cs_mean+cs_std,linewidth=2, color='g', alpha=0.5)
#        ax2.axhline(y=cs_mean-cs_std,linewidth=1, color='g', alpha=0.5)
        ax2.set_xlim(xmin=0)
        ax2.grid()
        ax2.set_title('Cumulative window peaks',fontsize=30)
        ax2.set_xlabel('Stacked windows (sum)',fontsize=25)
        ax2.set_ylabel('Count of peaks',fontsize=25)
        ax2.legend(bbox_to_anchor=(1.45,1.02))
        plt.gca().invert_xaxis()
        if(save_plots):
            fig.savefig('count_thersholds_'+save_name_suffix, dpi=300, bbox_inches="tight")
            plt.close()
        elif(plt_axes is None):
            plt.show()
    else:
        ax=None
        ax2=None
    return np.array([ag_peaks,long_usets.sum(axis=0)[ag_peaks]]), unique[best_fit_index], (ax,ax2), (unique,counts_cumsum)
    

def sub_annual_age_model(HDTW_object, peaks_array, threshold, end_year, 
                         reverse_yesrs=True, save_plots=False, 
                         save_name_suffix='', plt_axes=None):
    '''
    This function makes a sub-annual report figure for the HDTW object given
     the peaks_array and the parameters below.

    :param HDTW_object (HDTW object): the relevant (and executed) HDTW object 
       to base the sub annual age mobel on.
    :param Peaks_array (list): a list of peaks found for each of the input
       signals in HDTW_object.
    :param Threshold (int): aligned peaks above this value are considered an 
       indication.
    :param End_year (int): the year the sample chronology ends at.
    :param Reverse_yesrs (bool, default True): sets the order of the years.
    :param Save_plots (bool, default False): indicates if the plot is saved.
    :param Save_name_suffix (str): to apply when saving (must have graphic file
       suffix).
    :param Plt_axes (matplotlib.axes object, default None): if passed, the plot 
       is generated on the Plt_axes and returned by the function.

    '''
    age_best_fit=peaks_array[0][np.where(peaks_array[1]>=threshold)[0]]
    age_best_fit_rest=peaks_array[0][np.where(peaks_array[1]<threshold)[0]]
    age_best_fit_rest_alpha=peaks_array[1][np.where(peaks_array[1]<threshold)[0]]
    
    start_year = end_year - len(age_best_fit) + 1
    index_series = pd.Series(data=np.zeros(HDTW_object.path_matrix.shape[1]))
    
    direction=1
    if(reverse_yesrs):
        direction = -1
        start_year = end_year 
    periods_list=[]
    peak_dates=[]
    for peak_idx, peak_value in enumerate(age_best_fit[:-1]):
        index_series.loc[age_best_fit[peak_idx]:age_best_fit[peak_idx+1]]=\
            pd.date_range(start='1/1/'+str(start_year+peak_idx*direction),
                          end='1/1/'+str(start_year+peak_idx*direction+1*direction),
                          periods=age_best_fit[peak_idx+1]-peak_value+1)
        periods_list.append(age_best_fit[peak_idx+1]-peak_value+1)
        peak_dates.append('1/1/'+str(start_year+peak_idx*direction))
    peak_dates.append('1/1/'+str(start_year+peak_idx*direction+1*direction))
    
    df = pd.DataFrame(data = HDTW_object.signals_on_path_matrix.T)
    
    if(plt_axes is None):
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        ax= plt_axes[0]
    ax.margins(x=0)
    df.plot(ax=ax, alpha=0.5)
    df.loc[:,'median_signal'] = df.median(axis=1).values
    ax.plot(df.loc[:,'median_signal'].index,
            df.loc[:,'median_signal'].values,
            alpha=1,c='k', label = 'Median signal',linewidth=1)

    ax.vlines(age_best_fit, *ax.get_ylim(),colors='k',
        linestyles='dashed',linewidth=0.6,alpha=0.7)
    
    for sc, sc_alpha in zip(age_best_fit_rest,
                            age_best_fit_rest_alpha/100):
        ax.scatter(sc,df.loc[sc,'median_signal'],c='r', s=200, alpha=sc_alpha)    
    
    ax.set_title('Indications',fontsize=30)
    ax.set_xlabel('Alignment connection index',fontsize=25)
    ax.set_ylabel('Signal value',fontsize=25)
    handles, labels = ax.get_legend_handles_labels()
    # manually define a new patch 
    patch = mpatches.Patch(color='red', label='Potential indications')
    # handles is a list, so append manual patch
    handles.append(patch) 
    #plot the legend
    #plt.legend(handles=handles, loc='upper center')
    ax.legend(bbox_to_anchor=(1.1,1.02), title='Annual indications',
               handles=handles)
    if(save_plots):
        fig.savefig('annual_indications_'+save_name_suffix, dpi=300, bbox_inches="tight")
        plt.close()
    elif(plt_axes is None):
        plt.show()
    
    df=pd.DataFrame(data = HDTW_object.signals_on_path_matrix.T, 
                      index = index_series.values)

    rest_dates = index_series.values[age_best_fit_rest.astype(int)]
    rest_alpha = age_best_fit_rest_alpha[rest_dates!=0]
    rest_dates = rest_dates[rest_dates!=0]
    rel_df = df.loc[df.index.values != 0.0,:]

    rel_df.index = pd.to_datetime(rel_df.index)
    
    if(plt_axes is None):
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        ax2= plt_axes[1]
    ax2.margins(x=0)
    rel_df.plot(ax=ax2, alpha=0.5)
    rel_df.loc[:,'median_signal'] = rel_df.median(axis=1).values
    ax2.plot(rel_df.loc[:,'median_signal'].index,
            rel_df.loc[:,'median_signal'].values,
            alpha=1,c='k', label = 'Median signal',linewidth=1)
    for sc, sc_alpha in zip(rest_dates,
                            rest_alpha/100):
        ax2.scatter(sc,rel_df.loc[sc,'median_signal'],c='r', s=200,
                                alpha=sc_alpha)
    ax2.vlines(peak_dates, *ax.get_ylim(),colors='k',
        linestyles='dashed',linewidth=0.4,alpha=0.7)
    
    ax2.set_title('Indications (linear time)',fontsize=30)
    ax2.set_xlabel('Year',fontsize=25)
    ax2.set_ylabel('Signal value',fontsize=25)
    ax2.invert_xaxis()
    
    # where some data has already been plotted to ax
    handles, labels = ax2.get_legend_handles_labels()
    # manually define a new patch 
    patch = mpatches.Patch(color='red', label='Potential indications')
    # handles is a list, so append manual patch
    handles.append(patch) 
    #plot the legend
    #plt.legend(handles=handles, loc='upper center')
    ax2.legend(bbox_to_anchor=(1.1,1.02), title='Annual indications',
               handles=handles)
    
    if(len(peak_dates)<30):
        ax2.xaxis.set_major_locator(mdates.YearLocator())
    else:
        ax2.xaxis.set_major_locator(plt.MaxNLocator(35))
    
    if(reverse_yesrs):
        plt.gca().invert_xaxis()
    plt.xticks(rotation=45)
    if(save_plots):
        fig.savefig('linear_time'+save_name_suffix, dpi=300, bbox_inches="tight")
        plt.close()
    elif(plt_axes is None):
        plt.show()
    
    return ax, ax2

def age_dist_plot(HDTW_object, peaks_array, best_fit_threshold,
                  savename='', sample_size=None, sample_size_unit='',
                  x_grid_interval=1,
                  y_grid_interval=5, ax_object=None):
    '''
    :param HDTW_object (HDTW object): the relevant (and executed) HDTW object 
        to base the age scan on.
    :param age_estimation_deviation(int): the age estimation deviation (+/-)
        from ‘Age_estimation’  that represents the age uncertainty.
    :param Age_scan_range (float): this ratio expands the scan range (both
        upper and lower limits).
    :param Threshold_value (int): for peak alignment, to recognise a feature 
        as non-local.
    :param Threshold_function (function, callable): for peak alignment, to
        recognise a feature as non-local (custom filter function).
    :param End_year (int): the year the sample chronology ends at.
    :param Reverse_yesrs (bool, default True): indicates the order or the years
        in the plot.
    :param Age_iterations (int): the number of tests to perform in the test
        range.
    :param figsize(tuple, default (25, 31)): propagation of the 
        matplotlib.pyplot.figure fig_size parameter to determine the size of the
        plot generated.
    :param adjust_best_fit_threshold (int): any value to adjust the threshold
        to (default threshold is the one to result in the closest solution to the
        age estimation).
    :param Adjust_age_estimation (int): any value to adjust the threshold
        target to.
    :param Sample_size_list (list): list containing the length of each input
        signal (default will show records).
    :param Sample_size_unit (str, default “”): adds the string the the axis
        title to indicate the unit of “Sample_size”.
    :param Save_name (str, default 'default.pdf'): the name of the file to save. 
    :param **threshold_function_kwargs (dict, default empty): any key-word
        arguments to pass to the threshold_function.

    '''
    peak_connections = peaks_array[0][peaks_array[1]>=best_fit_threshold].astype(int)
    if(sample_size is None):
        traverses_peaks = HDTW_object.path_matrix[:,peak_connections]
    else:
        traverses_peaks = (HDTW_object.path_matrix_normalised[:,peak_connections].T*np.array(sample_size)).T
    
    if(ax_object is None):
        fig, ax = plt.subplots(figsize=(16, 9))
    else:
        ax=ax_object
    
    deltas=[]
    for signal_id,age_model in enumerate(traverses_peaks):
        ax.plot(age_model, np.arange(len(age_model)))
        ax.scatter(age_model, np.arange(len(age_model)), label='Signal '+str(signal_id))
        deltas.append(np.cumsum(age_model[::-1])[::-1] )
    ax.set_title('Age model',fontsize=30)
    ax.set_xlabel('Distance on sample %s'%sample_size_unit,fontsize=25)
    ax.set_ylabel('Age',fontsize=25)
    print('traverses_peaks',traverses_peaks)
    print('deltas',deltas)
    ax.grid(axis='both', linestyle='-')

    ax.legend(bbox_to_anchor=(1.1,1.02))
    if(ax_object is None):
        fig.savefig(savename+'_age_model.pdf', dpi=300, bbox_inches="tight")
        plt.close()
    else:
        return ax

def age_scan_tests(HDTW_object, sample_name,
                   age_estimation_parameter_list,
                   age_deviation_parameter_list,
                   scan_ranges_parameter_list,
                   save=True):
    '''
    Performs a parameter sweep over the possible combinations of:
    1 - Age_estimation_parameter_list 
    2 - Age_deviation_parameter_list 
    3 - Scan_ranges_parameter_list 
    As no thresholding is done, the propagated value is the average of the 
        peaks and their rank and therefore should be considered the upper limit 
        representing the found set of possible indications.

    :param HDTW_object (HDTW object): the relevant (and executed) HDTW object 
        to base the age scan on.
    :param Sample_name (str): the name of the sample (for saving).
    :param Age_estimation_parameter_list (list): the list of age estimation 
        values to test.
    :param Age_deviation_parameter_list (list): the list of age deviation 
        values to test.
    :param Scan_ranges_parameter_list (list): the list of scan range values 
        to test

    '''
    if(save):
        # exec tests
        ac_arrays_list=[]
        ac_arrays_settings_list=[]
        print('** Total tests :', len(age_estimation_parameter_list)*\
                                  len(age_deviation_parameter_list)*\
                                  len(scan_ranges_parameter_list))
        count=0
        for age_estimation_parameter in age_estimation_parameter_list:
            for age_deviation_parameter in age_deviation_parameter_list:
                for scan_ranges_parameter in scan_ranges_parameter_list:
                    peaks_array, best_fit_threshold, axes, uc_array =age_scan(HDTW_object, 
                             age_estimation= age_estimation_parameter, 
                             age_estimation_deviation= age_deviation_parameter,
                             age_scan_range= scan_ranges_parameter, 
                             age_iterations= 100,
                             threshold_value= 2, threshold_function= find_peaks, scale=50,
                             skip_plots=True)
                    ac_arrays_list.append(uc_array)
                    ac_arrays_settings_list.append((age_estimation_parameter,
                                                    age_deviation_parameter,
                                                    scan_ranges_parameter))
                    print('Current test: %s'%count)
                    count=count+1
        # make stats
        ac_stats=[]
        for ac_id in range(len(ac_arrays_list)):
            sum_of_numbers = sum(ac_arrays_list[ac_id][0]*ac_arrays_list[ac_id][1])
            count = sum(ac_arrays_list[ac_id][0])
            mean = sum_of_numbers / count
        
            total_squares = sum(ac_arrays_list[ac_id][0]*ac_arrays_list[ac_id][1]*ac_arrays_list[ac_id][1])
            mean_of_squares = total_squares / count
            variance = mean_of_squares - mean * mean
            std_dev = np.sqrt(variance)
            
            ac_stats.append((mean, std_dev))

        # collect to dataframe
        ac_df_stats = pd.DataFrame(data=np.array(ac_stats))
        ac_df_settings = pd.DataFrame(data=np.array(ac_arrays_settings_list))
        
        ac_df=pd.concat([ac_df_stats,ac_df_settings], axis=1)
        ac_df.columns = ['Mean (of tests results)', 'Standard deviation (of tests results)',
                         'Age estimation parameter', 'Estimation deviation parameter', 
                         'Extended range parameter']
        # save dataframe
        ac_df.to_pickle(sample_name+'_tests_dataframe.pkl')
    else:
        ac_df = pd.read_pickle(sample_name+'_tests_dataframe.pkl')
    
    ac_mean = ac_df.loc[:,['Mean (of tests results)', 'Standard deviation (of tests results)']].mean().values
    
    
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.scatterplot(ax=ax,x='Mean (of tests results)', 
                    y='Standard deviation (of tests results)',
                    hue='Estimation deviation parameter', 
                    style='Age estimation parameter',
                    size='Extended range parameter', data=ac_df,
                    legend="full")
    ax.legend(bbox_to_anchor=(1.1,1))
    
    ax.scatter(ac_mean[0],ac_mean[1], label='Mean')
    
    ax.axhline(y=ac_mean[1], xmin=0.0, xmax=ac_mean[0], color='g')
    ax.axvline(x=ac_mean[0], ymin=0.0, ymax=ac_mean[1], color='g')
    
    
    fig.savefig(sample_name+'_tests.pdf', dpi=300, bbox_inches="tight")
    #plt.show()
    plt.close()
    return ac_df

def sub_annual_age_model_figure(HDTW_object, age_estimation,
                                 age_estimation_deviation, age_scan_range, 
                                 threshold_value, end_year,
                                 threshold_function=None, reverse_yesrs=True,
                                 age_iterations= 100, figsize=(25, 31), 
                                 adjust_best_fit_threshold=None,
                                 adjust_age_estimation=None,
                                 sample_size_list=None,
                                 sample_size_unit='',
                                 save_name='default.pdf',
                                 **threshold_function_kwargs):
    '''
    :param HDTW_object (HDTW object): the relevant (and executed) HDTW object
        to base the age scan on.
    :param age_estimation_deviation(int): the age estimation deviation 
        (+/-) from ‘Age_estimation’  that represents the age uncertainty.
    :param Age_scan_range (float): this ratio expands the scan range (both
                          upper and lower limits).
    :param Threshold_value (int): for peak alignment, to recognise a feature 
        as non-local.
    :param Threshold_function (function, callable): for peak alignment, to
        recognise a feature as non-local (custom filter function).
    :param End_year (int): the year the sample chronology ends at.
    :param Reverse_yesrs (bool, default True): indicates the order or the
        years in the plot.
    :param Age_iterations (int): the number of tests to perform in the test 
        range.
    :param figsize(tuple, default (25, 31)): propagation of the
        matplotlib.pyplot.figure fig_size parameter to determine the size of the 
        plot generated.
    :param adjust_best_fit_threshold (int): any value to adjust the threshold
        to (default threshold is the one to result in the closest solution to the
        age estimation).
    :param Adjust_age_estimation (int): any value to adjust the threshold
        target to.
    :param Sample_size_list (list): list containing the length of each input
        signal (default will show records).
    :param Sample_size_unit (str, default “”): adds the string the the axis
        title to indicate the unit of “Sample_size”.
    :param Save_name (str, default 'default.pdf'): the name of the file to save. 
    :param **threshold_function_kwargs (dict, default empty): any key-word
        arguments to pass to the threshold_function.
    '''
    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(5, 1)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])
    ax4 = plt.subplot(gs[4])
    if(adjust_age_estimation is None):
        adjust_age_estimation=age_estimation
    peaks_array, best_fit_threshold, (ax0,ax1), uc_array =age_scan(HDTW_object, 
                 age_estimation= age_estimation, 
                 age_estimation_deviation= age_estimation_deviation,
                 age_scan_range= age_scan_range, age_iterations= age_iterations,
                 adjust_age_estimation=adjust_age_estimation,
                 threshold_value= threshold_value, 
                 threshold_function= threshold_function, plt_axes=(ax0,ax1),
                 **threshold_function_kwargs)

    if(adjust_best_fit_threshold is not None):
        best_fit_threshold=adjust_best_fit_threshold

    (ax2,ax3) = sub_annual_age_model(HDTW_object=HDTW_object, peaks_array=peaks_array, 
                         threshold=best_fit_threshold, end_year=end_year, 
                         reverse_yesrs=reverse_yesrs,
                         plt_axes=(ax2,ax3))
    age_dist_plot(ax_object=ax4, HDTW_object=HDTW_object, 
                  peaks_array=peaks_array, 
                  best_fit_threshold=best_fit_threshold,
                  sample_size=sample_size_list,
                  sample_size_unit=sample_size_unit)
    
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")





# sample testing

## MNDS1
MNDS1_results=[]
MNDS1_traverses_names=[]
MNDS1_HDTW_signals=[]

sample_xlfile = 'MNDS1.xlsx'
sample_sheetname = 'TraverseA'
MNDS1_HDTW_signals.append(LA_to_PCA(sample_xlfile, sample_sheetname))
MNDS1_traverses_names.append(sample_sheetname)

sample_sheetname = 'TraverseB'
MNDS1_HDTW_signals.append(LA_to_PCA(sample_xlfile, sample_sheetname))
MNDS1_traverses_names.append(sample_sheetname)

sample_sheetname = 'TraverseC'
MNDS1_HDTW_signals.append(LA_to_PCA(sample_xlfile, sample_sheetname))
MNDS1_traverses_names.append(sample_sheetname)

MNDS_HDTW = HDTW(MNDS1_HDTW_signals, dtw_function_reference=dta_dtw)
MNDS_HDTW.execute()
MNDS_HDTW.save('hdtw_mnds1.pkl')

MNDS_HDTW=HDTW()
MNDS_HDTW.load('hdtw_mnds1.pkl')
MNDS_HDTW.indications_plot(legend_signal_names=MNDS1_traverses_names,save_filename='MNDS_indications_plot.pdf')
MNDS_HDTW.aligned_output_plot(legend_signal_names=MNDS1_traverses_names,save_filename='MNDS_aligned_output_plot.pdf')
MNDS_HDTW.temporal_warping_plot(legend_signal_names=MNDS1_traverses_names,save_filename='MNDS_temporal_warping_plot.pdf')
MNDS_HDTW.warp_factor_plot(legend_signal_names=MNDS1_traverses_names,save_filename='MNDS_warp_factor_plot.pdf')
MNDS_HDTW.distances_plot(save_filename='MNDS_distances_plot.pdf')
MNDS_HDTW.representative_noise_plot(np.median(MNDS_HDTW.signals_on_path_matrix,axis=0),
                                    save_filename='MNDS_delta_median.pdf')
MNDS_HDTW.representative_noise_distributions_plot(np.median(MNDS_HDTW.signals_on_path_matrix,axis=0),
                                                  save_filename='MNDS_delta_mean.pdf')
sns.set(font_scale=2,style="whitegrid")
MNDS_HDTW.HDTW_report_figure(save_filename='MNDS_HDTW_report.pdf',fig_size=(25, 31),
                           traverses_names=MNDS1_traverses_names)

sub_annual_age_model_figure(HDTW_object=MNDS_HDTW, 
                             age_estimation= 80,
                             age_estimation_deviation= 0,
                             age_scan_range= 0.1, 
                             threshold_value= 2, end_year=1991,
                             threshold_function=find_peaks, reverse_yesrs=True,
                             age_iterations= 100, figsize=(25, 31), 
                             adjust_best_fit_threshold=None,
                             sample_size_list=[34.313,34.091,33.691],
                             sample_size_unit='(mm)',
                             save_name='MNDS_HDTW_sub_annual_age_model.pdf',
                             scale=50)

### so38 test - different speeds and sampling prop
SO38_results=[]
SO38_traverses_names=[]
SO38_HDTW_signals=[]

sample_xlfile = 'SO-38_3m_clean.xlsx'
sample_sheetname = 'Sheet1'
input_signal = LA_to_PCA(sample_xlfile)#, sample_sheetname)
SO38_HDTW_signals.append(input_signal)
SO38_traverses_names.append('3 micron')

sample_xlfile = 'SO-38_2micron-sec_clean.xlsx'
sample_sheetname = 'Sheet1'
input_signal = LA_to_PCA(sample_xlfile, sample_sheetname)
SO38_HDTW_signals.append(input_signal)
SO38_traverses_names.append('2 micron')

sample_xlfile = 'SO-38_WO-trench.xlsx'
sample_sheetname = 'Sheet1'
input_signal = LA_to_PCA(sample_xlfile)#, sample_sheetname)
SO38_HDTW_signals.append(input_signal)
SO38_traverses_names.append('WO-trench')

sample_xlfile = 'SO-38_W-trench.xlsx'
input_signal = LA_to_PCA(sample_xlfile)
SO38_HDTW_signals.append(input_signal)
SO38_traverses_names.append('W-trench')

SO38_HDTW_4signals = HDTW(SO38_HDTW_signals, dtw_function_reference=dta_dtw)
SO38_HDTW_4signals.execute()
SO38_HDTW_4signals.save('HDTW_SO38_4signals.pkl')

SO38_HDTW_4signals=HDTW()
SO38_HDTW_4signals.load('HDTW_SO38_4signals.pkl')
age_scan(SO38_HDTW_4signals, 
         age_estimation= 190, 
         age_estimation_deviation= 20,
         age_scan_range= 0.1, age_iterations= 100,
         threshold_value= 2, threshold_function= find_peaks, scale=50)
SO38_HDTW_4signals.HDTW_report_figure(save_filename='SO38_HDTW_report.pdf',fig_size=(25, 31),
                                      traverses_names=SO38_traverses_names)

## 5-3B - cflm
image_file_name='5-3best-LA-HR-sampleband-small-narrow-clean.png'

cflm53b = HDTW(image_to_signals(image_file_name, n_signals=16),
               dtw_function_reference=dta_dtw)
cflm53b.execute()
cflm53b.save('HDTW_cflm53b.pkl')

cflm53b=HDTW()
cflm53b.load('HDTW_cflm53b.pkl')
cflm53b.HDTW_report_figure(save_filename='53B_HDTW_report.pdf',fig_size=(25, 31))
sub_annual_age_model_figure(HDTW_object=cflm53b, 
                            age_estimation= 21,
                            age_estimation_deviation= 3,
                            age_scan_range= 0.2, 
                            threshold_value= 8, end_year=2008,
                            threshold_function=find_peaks, 
                            reverse_yesrs=True,
                            age_iterations= 100, figsize=(25, 31), 
                            adjust_best_fit_threshold=None,
                            sample_size_list=[5.4]*16,
                            sample_size_unit='(mm)',
                            save_name='cflm53b_HDTW_sub_annual_age_model.pdf',
                            scale=50)

## 5-3B rev
image_file_name='5-3best-LA-HR-sampleband-small-narrow-clean.png'
rev_signals=[]
for sig in image_to_signals(image_file_name, n_signals=16):
    rev_signals.append(sig[::-1])

cflm53b_rev = HDTW(rev_signals,
               dtw_function_reference=dta_dtw)
cflm53b_rev.execute()
cflm53b_rev.save('HDTW_cflm53b_rev.pkl')
##
cflm53b_rev=HDTW()
cflm53b_rev.load('HDTW_cflm53b_rev.pkl')
cflm53b_rev.HDTW_report_figure(save_filename='cflm53b_rev_HDTW_report.pdf',fig_size=(25, 31))

#eof