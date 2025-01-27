import os
import gc

import cv2
from pathlib import Path

from PIL import Image, ImageSequence
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, argrelextrema
from numpy import unravel_index
import statistics
import pandas as pd
from scipy.interpolate import interp1d

import itertools
import matplotlib.patches as patches
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import ray
from ray import train, tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter,CategoricalHyperparameter

tracked_circle=[]
prev_circles=[] 
ring_index_hough=None
centers=pd.DataFrame(columns=['results'])

# main code ------------------------------------------------------------------------------------
def moving_average(signal, window_size=5):
    return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

def circle_radius_fun(peaks_x,peaks_y,ring_index,ccol,crow):
    # Adjusted peak selection to use closest peaks around ccol
    closest_peak_x = min(peaks_x, key=lambda x: abs(x-ccol))
    closest_peak_y = min(peaks_y, key=lambda y: abs(y-crow))

    # Using peak indices to find distances to closest peaks
    # peak_index_x = peaks_x.tolist().index(closest_peak_x)
    # peak_index_y = peaks_y.tolist().index(closest_peak_y)
    peak_index_x=peaks_x.index(closest_peak_x)
    peak_index_y=peaks_y.index(closest_peak_y)
    # print(f'circle fun {ccol,crow}')

    if peak_index_x > 0:
        if len(peaks_x)==peak_index_x+1:
            peak_index_x-=1
        rl_x = ccol - peaks_x[peak_index_x - ring_index]
        rr_x = peaks_x[peak_index_x + ring_index] - ccol

    if peak_index_y > 0:
        if len(peaks_y)==peak_index_y+1:
            peak_index_y-=1
        ru_y = crow - peaks_y[peak_index_y - ring_index]
        rl_y = peaks_y[peak_index_y + ring_index] - crow


    return round((rl_x + rr_x + ru_y + rl_y) / 4., 5)

def get_center_from_peaks(peaks, current_center):
    # print(f'peaks: {peaks}\ncurrentCenter: {current_center}')
    valid_peaks = [peak for peak in peaks if np.abs(peak - current_center) >= 10]
    left_peaks = np.array([peak for peak in valid_peaks if peak < current_center])
    right_peaks = np.array([peak for peak in valid_peaks if peak > current_center])

    if len(left_peaks) == 0 or len(right_peaks) == 0:
        return current_center, 0

    left_peak = left_peaks[-1] 
    right_peak = right_peaks[0] 

    refined_center = (left_peak + right_peak) / 2

    distance_between_peaks = np.abs(right_peak - left_peak)

    return refined_center, distance_between_peaks

def get_center_from_minimums(minimums, current_center,min_depth=0.25):
    valid_minimums = [minm for minm in minimums if np.abs(minm - current_center) >= 10]
    left_minimums = np.array([minm for minm in valid_minimums if minm < current_center])
    right_minimums = np.array([minm for minm in valid_minimums if minm > current_center])

    if len(left_minimums) == 0 or len(right_minimums) == 0:
        return current_center, 0

    left_minimum = left_minimums[-2] 
    right_minimum = right_minimums[1]  

    refined_center = (left_minimum + right_minimum) // 2

    distance_between_minimums = np.abs(right_minimum - left_minimum)

    return refined_center, distance_between_minimums

def draw_circle(image,ccol,crow,circle_radius):
    cross_length=10

    circle_img = cv2.circle(image.copy(), (ccol, crow), int(circle_radius), (255, 0, 0), 2)
    cv2.line(circle_img,(ccol - cross_length, crow), (ccol + cross_length, crow), (255, 255, 255), 1)
    cv2.line(circle_img, (ccol, crow - cross_length), (ccol, crow + cross_length), (255, 255, 255), 1)
    
    return circle_img

def crop_image(img, box_top_left, box_width, box_height):
        left = int(box_top_left[0])
        top = int(box_top_left[1])
        right = int(left + box_width)
        bottom = int(top + box_height)

        return img[top:bottom, left:right]

def houghCircle(ccol,crow,radius,image,prev_radius,clipLimit,titleGridSize,blur_level,dp,param1,param2):
    global centers
    # ccol -x, crow - y
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=titleGridSize)
    cl1 = clahe.apply(image)
    
    blur = cv2.GaussianBlur(cl1,blur_level,0)
    
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT_ALT, dp=dp, minDist=0.00001,
                            param1=param1, param2=param2, minRadius=0, maxRadius=0)

    results =[]
    frame_no=0
    if circles is not None:
        circles = np.squeeze(circles, axis=0) 
        # # scaling factor
        # circles[:, :3] /= 4
        closest_circle = None
        min_distance = float('inf')

        # print(circles)
        # Iterate through detected circles
        for circle in circles:
            detected_x, detected_y, detected_radius = circle

            center_distance = np.sqrt((detected_x - ccol)**2 + (detected_y - crow)**2)

            radius_difference = abs(detected_radius - radius)
            if(prev_radius==None):
                change_difference=0
            else:
                change_difference = abs(detected_radius - prev_radius)

            # More weight is given to the existing center (is more precise than radius)
            total_metric = 2*center_distance + 1.5*change_difference+radius_difference

            if total_metric < min_distance:
                min_distance = total_metric
                closest_circle = circle

        frame_no+=1

        if closest_circle is not None:
            closest_x, closest_y, closest_radius = closest_circle

        ftimage = fft2(image)
        ftimage = fftshift(ftimage)

        tmp,result_img = filtered_image(ftimage=ftimage,crow=int(closest_y),ccol=int(closest_x),r_in=6,r_out=12)
        # 10
        # print(closest_x,closest_y)
        # print(ccol,crow)
        if(np.sqrt((closest_x - ccol)**2 + (closest_y - crow)**2)<=10):
            return ccol,crow,closest_radius
        else:
            ccol,crow = closest_x,closest_y
            ccol,crow = get_subpixel_center_hough(ccol,crow,tmp)
            radius = get_radius_hough(tmp,ccol,crow,circles)
            return ccol,crow,radius

    else:
        print("No circles detected.")

def get_subpixel_center_hough(ccol,crow,tmp):
    # get peak of ccol

    horizontal_profile = tmp[int(crow),:]
    peaks_horizontal = argrelextrema(horizontal_profile, np.greater)[0]
    distances = np.abs(peaks_horizontal - crow)
    closest_peak_index = peaks_horizontal[np.argmin(distances)]
    ccol = get_subpixel_peak_com([closest_peak_index],horizontal_profile)[0]
    
    # get peak of crow

    vertical_profile = tmp[:,int(ccol)]
    peaks_vertical = argrelextrema(vertical_profile, np.greater)[0]
    distances = np.abs(peaks_vertical - crow)
    closest_peak_index = peaks_vertical[np.argmin(distances)]
    crow = get_subpixel_peak_com([closest_peak_index],vertical_profile)[0]

    return ccol,crow

def get_radius_hough(tmp,ccol,crow,circles):
    # global tracked_circle
    # if not tracked_circle:
    global ring_index_hough

    if ring_index_hough == None:
        tracked_circle=sorted(circles, key=lambda c: c[2])[1]
        radius = tracked_circle[2]
        vertical_profile = tmp[:,int(ccol)]
        
        # Extract horizontal profile (slice along y-axis at crow)
        horizontal_profile = tmp[int(crow),:]
        
        # Find peaks in the vertical intensity profile (edge)
        vertical_peaks = argrelextrema(vertical_profile, np.greater)[0]
        
        # Find peaks in the horizontal intensity profile (edge)
        horizontal_peaks = argrelextrema(horizontal_profile, np.greater)[0]

        left_edge=ccol-tracked_circle[2]
        right_edge=ccol+tracked_circle[2]
        
        top_edge=crow-tracked_circle[2]
        bottom_edge=crow+tracked_circle[2]

        distances = np.abs(horizontal_peaks - left_edge)
        closest_peak = horizontal_peaks[np.argmin(distances)]

        distances = np.abs(horizontal_peaks - ccol)
        closest_center_peak = horizontal_peaks[np.argmin(distances)]

        peaks_between = [peak for peak in horizontal_peaks if closest_peak <= peak < closest_center_peak]

        ring_index_hough=len(peaks_between)

    horizontal_profile = tmp[int(crow),:]
    horizontal_peaks = argrelextrema(horizontal_profile, np.greater)[0]
    vertical_profile = tmp[:,int(ccol)]
    vertical_peaks = argrelextrema(vertical_profile, np.greater)[0]

    radius = circle_radius_fun(horizontal_peaks.tolist(),vertical_peaks.tolist(),ring_index_hough,ccol,crow)

    closest_circle = None
    min_distance = float('inf')

    # Iterate through detected circles
    for circle in circles:
        detected_x, detected_y, detected_radius = circle

        center_distance = np.sqrt((detected_x - ccol)**2 + (detected_y - crow)**2)

        radius_difference = abs(detected_radius - radius)

        # More weight is given to the existing center (is more precise than radius)
        total_metric = 2*center_distance + radius_difference

        if total_metric < min_distance:
            min_distance = total_metric
            closest_circle = circle
    return closest_circle[2]

def draw_circle(image,ccol,crow,circle_radius,result_folder,filename):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    circle_center=(ccol,crow)
    circle = patches.Circle(circle_center, circle_radius, edgecolor='black', facecolor='none', linewidth=1)
    ax.add_patch(circle)

    cross_size = 10
    ax.plot([circle_center[0] - cross_size, circle_center[0] + cross_size], 
                [circle_center[1], circle_center[1]], color='white', linewidth=2) 
    ax.plot([circle_center[0], circle_center[0]], 
            [circle_center[1] - cross_size, circle_center[1] + cross_size], color='white', linewidth=2) 

    fig.savefig(os.path.join(result_folder, f"result_{filename}"))
    plt.cla()
    plt.close(fig)
    plt.close('all')
    del fig,ax

def image_background_brightness(image):
        top_left = image[0:5, 0:5]
        top_right = image[0:5, -5:]
        bottom_left = image[-5:, 0:5]
        bottom_right = image[-5:, -5:]
        brightness = np.concatenate((top_left.flatten(), top_right.flatten(), 
                                  bottom_left.flatten(), bottom_right.flatten()))
        # brightness=top_left+top_right+bottom_left+bottom_right

        return [np.mean(brightness),np.median(brightness),statistics.mode(brightness)]

def tiff_to_png(input_tiff, input_path, output_path):
    try:
        sq = Image.open(os.path.join(input_path, input_tiff))
        for i, img in enumerate(ImageSequence.Iterator(sq)):
            output = os.path.join(output_path, f"frame_{i:06d}.png")
            img.save(output)
    finally:
        sq.close()

def get_first_png(input_tiff, input_path):
    try:
        sq = Image.open(os.path.join(input_path, input_tiff))
        first_frame = next(ImageSequence.Iterator(sq))
        first_frame_np = np.array(first_frame)
        if first_frame_np.ndim == 2:
            return first_frame_np
        else:
            return cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2GRAY)
    finally:
        sq.close()

def filtered_image(ftimage,crow,ccol,r_in,r_out):
    rows, cols = ftimage.shape
    mask=np.zeros((rows,cols),dtype=np.uint8)

    x,y=np.ogrid[:rows,:cols]
    mask_area = np.logical_and(((x - rows//2)**2 + (y - cols//2)**2 >= r_in**2),
                            ((x - rows//2)**2 + (y - cols//2)**2 <= r_out**2))
    mask[mask_area] = 1

    m_app_ftimage = ftimage * mask
    i_ftimage = ifftshift(m_app_ftimage)
    result_img = ifft2(i_ftimage)
    tmp = np.log(np.abs(result_img) + 1)
    
    return tmp , result_img

def get_subpixel_peak_com(peaks, intensity_profile, resolution=0.01):
    refined_peaks = []
    for peak in peaks:
        if 0 < peak < len(intensity_profile) - 1:
            # range around the peak for interpolation
            x_range = np.linspace(peak - 1, peak + 1, int(2 / resolution) + 1)
            
            # Interpolate the intensity profile
            interp_func = interp1d(np.arange(len(intensity_profile)), intensity_profile, kind='cubic')
            y_range = interp_func(x_range)
            
            max_index = np.argmax(y_range)
            refined_peak = x_range[max_index]
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(peak)
    return refined_peaks

def get_subpixel_minimum_com(minimums, intensity_profile):
    refined_peaks = []
    for peak in minimums:
        if 0 < peak < len(intensity_profile) - 1:
            x = np.array([peak - 1, peak, peak + 1], dtype=float)
            y = intensity_profile[x.astype(int)]
            refined_peak = np.sum(x * y) / np.sum(y)
            refined_peaks.append(refined_peak)
        else:
            refined_peaks.append(peak)
    return refined_peaks

def peaks_by_minimums(minimums,peaks):
    refined_peaks=[]
    peaks_and_minimums=zip(peaks,minimums)
    # print(peaks_and_minimums)
    for peak,minimum in peaks_and_minimums:
        if 0<peak<len(peaks)-1:
            refined_peak=(minimum-1+minimum+1)/2
        else:
            refined_peaks.append(peak)
    # print(refined_peaks)
    return refined_peaks            

tracked_circle=[]
prev_circles=[]
ring_index_hough=None
centers=pd.DataFrame(columns=['results'])

def ring_image(image,r_in_center,r_out_center,r_in_radius,r_out_radius,ring_index,output_path,filename,prev_radius,brightness,clipLimit,titleGridSize,blur_level,dp,param1,param2,resolution=0.01):
    # file_path = os.path.join(output_path, filename)
    # image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    imageBright = cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)

    ftimage = fft2(imageBright)
    ftimage = fftshift(ftimage)

    rows, cols = ftimage.shape
    crow, ccol = rows // 2, cols // 2

    # applying filter for center
    tmp_center,result_img = filtered_image(ftimage,crow,ccol,r_in_center,r_out_center)

    # applying filter for radius
    tmp_radius,unused = filtered_image(ftimage,crow,ccol,r_in_radius,r_out_radius)

    # crow, ccol = unravel_index(tmp.argmax(), tmp.shape)
    crow, ccol = unravel_index(tmp_center.argmax(), result_img.shape)
    
    
    # intensity profile for circle center
    central_line_y_center = tmp_center[crow, :]
    central_line_x_center = tmp_center[:, ccol]

    peaks_y_center = get_subpixel_peak_com(argrelextrema(central_line_y_center, np.greater)[0], central_line_y_center,resolution)
    peaks_x_center = get_subpixel_peak_com(argrelextrema(central_line_x_center, np.greater)[0], central_line_x_center,resolution)

    # intensity profile for circle radius
    central_line_y_radius = tmp_radius[crow, :]
    central_line_x_radius = tmp_radius[:, ccol]

    peaks_y_radius = get_subpixel_peak_com(argrelextrema(central_line_y_radius, np.greater)[0], central_line_y_radius,resolution)
    peaks_x_radius = get_subpixel_peak_com(argrelextrema(central_line_x_radius, np.greater)[0], central_line_x_radius,resolution)
    
    minimums_y_radius = get_subpixel_minimum_com(argrelextrema(central_line_y_radius, np.less)[0],central_line_y_radius)
    minimums_x_radius = get_subpixel_minimum_com(argrelextrema(central_line_x_radius, np.less)[0],central_line_x_radius)

    if len(peaks_x_center) > 1 and len(peaks_y_center) > 1:
        # print(f'ccol {ccol}\n crow {crow}')
        prev_ccol,prev_crow=ccol,crow
        ccol, x_distance = get_center_from_peaks(peaks_y_center, ccol)
        crow, y_distance = get_center_from_peaks(peaks_x_center, crow)

        refined_peaks_x_radius=peaks_by_minimums(minimums_x_radius,peaks_x_radius)
        refined_peaks_y_radius=peaks_by_minimums(minimums_y_radius,peaks_y_radius)

        circle_radius = circle_radius_fun(refined_peaks_x_radius,refined_peaks_y_radius,ring_index,ccol,crow)
        
        ccol_hough,crow_hough,circle_radius = houghCircle(ccol,crow,circle_radius,image,prev_radius,clipLimit,titleGridSize,blur_level,dp,param1,param2)

        ccol_hough = round(ccol_hough, 5)
        crow_hough = round(crow_hough, 5)
        circle_radius = round(circle_radius, 5)

        circle_center=(ccol_hough,crow_hough)

        prev_circle_radius=circle_radius

        plt.clf()
        plt.cla()
        plt.close('all')
        del image
        del ftimage, result_img, tmp_center,tmp_radius, unused
        gc.collect()
        
        return circle_radius, pd.DataFrame({
        'center_y(ccol)': [ccol_hough],
        'center_x(crow)': [crow_hough],
        'circle_radius': [circle_radius]
        })
    
        


    return None
# post processing ------------------------------------------------------------------------------------
def interpolation(data,size=50):
    return(np.interp([i for i in range(1,len(data)*size)],xp=[i*size for i in range(0,len(data))],fp=data))

def process_image_low_movement(filename, output_path, preProcData, i,df_list,size=50):
    i=i*size
    file_path = os.path.join(output_path, filename)

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)


    movementNearFramesX=preProcData['diff_center_x'][i:i+100].sum()
    movementNearFramesY=preProcData['diff_center_y'][i:i+100].sum()

    if((movementNearFramesX+movementNearFramesY)>10):
        ccol=preProcData['center_y(ccol)'][i]
        crow=preProcData['center_x(crow)'][i]
    else:
        ccol=preProcData['smooth_center_y(ccol)'][i]
        crow=preProcData['smooth_center_x(crow)'][i]

    circle_radius = preProcData['circle_radius'][i]

    new_data = pd.DataFrame({
        'center_y(ccol)': [ccol],
        'center_x(crow)': [crow],
        'circle_radius': [circle_radius]
    })
    df_list.append(new_data)
    # draw_circle(image, ccol, crow, circle_radius,result_folder,filename)

def process_image_high_movement(filename, output_path, preProcData, i,df_list,size=50):
    i=i*size
    file_path = os.path.join(output_path, filename)
    
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    ccol=preProcData['center_y(ccol)'][i]
    crow=preProcData['center_x(crow)'][i]
    circle_radius = preProcData['circle_radius'][i]

    new_data = pd.DataFrame({
        'center_y(ccol)': [ccol],
        'center_x(crow)': [crow],
        'circle_radius': [circle_radius]
    })
    df_list.append(new_data)

    # draw_circle(image, ccol, crow, circle_radius,result_folder,filename)


def post_processing(output_path,predicted_df,rolling_window,window,movement_thresh, interpolationP=True,size=50):
    movement_threshold=movement_thresh*len(os.listdir(output_path))
    # Path(result_folder).mkdir(parents=True, exist_ok=True)

    preProcData=predicted_df
    if (interpolationP==True):
        center_x_interpolated=interpolation(preProcData['center_x(crow)'],size)
        center_y_interpolated=interpolation(preProcData['center_y(ccol)'],size)
        circle_radius_interpolated=interpolation(preProcData['circle_radius'])
        window=window

        dfList=list(zip(center_x_interpolated,center_y_interpolated,circle_radius_interpolated))
        preProcData=pd.DataFrame(dfList,columns=['center_x(crow)','center_y(ccol)','circle_radius'])
    else:
        window = 3

    preProcData['diff_center_y'] = preProcData['center_y(ccol)'].diff().abs().fillna(0)
    preProcData['diff_center_x'] = preProcData['center_x(crow)'].diff().abs().fillna(0)
    totalMovement=0

    for index, row in preProcData.iterrows():
        totalMovement=totalMovement+row['diff_center_x']+row['diff_center_y']
    if rolling_window == 'mean':
        preProcData['smooth_center_y(ccol)'] = preProcData['center_y(ccol)'].rolling(window=window).mean()
        preProcData['smooth_center_x(crow)'] = preProcData['center_x(crow)'].rolling(window=window).mean()
        preProcData['smooth_radius'] = preProcData['circle_radius'].rolling(window=window).mean()
    else:
        preProcData['smooth_center_y(ccol)'] = preProcData['center_y(ccol)'].rolling(window=window).median()
        preProcData['smooth_center_x(crow)'] = preProcData['center_x(crow)'].rolling(window=window).median()
        preProcData['smooth_radius'] = preProcData['circle_radius'].rolling(window=window).median()

    preProcData.fillna(method='bfill', inplace=True) 
    df = pd.DataFrame(columns=['center_y(ccol)', 'center_x(crow)', 'circle_radius'])

    totalMovement = preProcData['diff_center_x'].sum() + preProcData['diff_center_y'].sum()

    df_list=[]
    if totalMovement < movement_threshold:
        with ThreadPoolExecutor(max_workers=1) as executor:
            files = sorted(os.listdir(output_path))
            
            for i, filename in enumerate(files):
                executor.submit(process_image_low_movement, filename, output_path, preProcData, i,df_list,size)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            files = sorted(os.listdir(output_path))
            for i, filename in enumerate(files):
                executor.submit(process_image_high_movement, filename, output_path, preProcData, i,df_list,size)

    df = pd.concat(df_list, ignore_index=True)
    # print(df)
    # df.to_excel(excel_path)
    return df


# -------------------------------------------------------------------------------------------------------
def crop_image(img, box_top_left, box_width, box_height):
        left = int(box_top_left[0])
        top = int(box_top_left[1])
        right = int(left + box_width)
        bottom = int(top + box_height)

        return img[top:bottom, left:right]
# optimizacija -------------------------------------------------------------------------------------------------------
def calculate_distance(predicted, actual):
    return np.sqrt((predicted[0] - actual[0]) ** 2 + (predicted[1] - actual[1]) ** 2)

def evaluate_excel(file_path,predicted_df, params):
    df = pd.read_excel(file_path,header=1)
    actual_coords = list(zip(df['x'], df['y']))

    predicted_coords=list(zip(predicted_df['center_x(crow)'], predicted_df['center_y(ccol)']))
    distances = [
        calculate_distance(predicted, actual)
        for predicted, actual in zip(predicted_coords, actual_coords)
    ]

    return np.sum(distances)


# metadata=pd.read_excel('005_obj1.xlsx',header=None).iloc[0]

def optimize_params(config):
    global ring_index_hough
    global centers

    # blur_levels = [3, 5, 7, 9, 11, 13, 15]
    rolling_window = 'mean' if config["rolling_window"] == 'mean' else 'median'
    # blur_level = (blur_levels[int(config["blur_level1"])], blur_levels[int(config["blur_level2"])])
    blur_level=(config["blur_level1"],config["blur_level2"])

    r_out_center=config['r_out_center']
    r_out_radius=config['r_out_radius']
    # if r_out_center <= config['r_in_center']:
    #     r_out_center=config['r_in_center']+1
    # if r_out_radius <= config['r_in_radius']:
    #     r_out_radius=config['r_in_radius']+1


    # print(config)
    excel_dir='C:/Users/Tautvydas/Files/ZieduRadimas/Testing_ideas/hpc_excels'
    excel_files = [os.path.join(excel_dir, f) for f in os.listdir(excel_dir) if f.endswith(".xlsx")]
    total_score=0
    for excel_file in excel_files[4:5]:
        # tracked_circle=[]
        # prev_circles=[]
        ring_index_hough=None
        centers=pd.DataFrame(columns=['results'])

        test=excel_file.split('_')[0]
        test=excel_file.split('/')[1]
        metadata=pd.read_excel(excel_file,header=None).iloc[0]
        df_list=[]
        prev_radius=None
        path='C:/Users/Tautvydas/Files/ZieduRadimas/Testing_ideas/HoughCircle/'+metadata[0]
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            image=cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
            image=crop_image(image,[metadata[3],metadata[4]],metadata[1],metadata[2])
            prev_radius,data=ring_image(image,int(config["r_in_center"]),int(r_out_center),int(r_out_radius),int(config["r_out_radius"]),int(config['ring_index']),metadata[0],filename,prev_radius,config['brightness'],config['clipLimit'],(int(config['titleGridSize1']),int(config['titleGridSize2'])),blur_level,config['dp'],config['param1'],config['param2'],config['resolution'])
            df_list.append(data)
        predicted_df = pd.concat(df_list, ignore_index=True)
        post_processed_df=post_processing(output_path=path,predicted_df=predicted_df,rolling_window=rolling_window,window=int(config['window']),movement_thresh=config['movement_thresh'],interpolationP=True,size=int(config['size']))

        score = evaluate_excel(excel_file,post_processed_df, config)
        total_score += score
        avg_score=total_score/len(excel_files)

        train.report({'mean_distance':avg_score,'params':str(config)})
        time.sleep(0.1)
    

# search_space = {
#     "r_in_center": tune.uniform(2, 15), #round later
#     "r_out_center": tune.uniform(2, 15),#round later
#     "r_in_radius": tune.uniform(2, 15),#round later
#     "r_out_radius": tune.uniform(2, 15),#round later
#     "ring_index": tune.uniform(1, 6),#round later
#     'brightness':tune.uniform(0.0,50.0),
#     "clipLimit": tune.uniform(0.1, 10.0),
#     "titleGridSize1": (tune.uniform(4, 32)), # round later, touple
#     "titleGridSize2":(tune.uniform(4, 32)),
#     "blur_level1": tune.uniform(0, len([3, 5, 7, 9, 11, 13, 15]) - 1), #touple
#     "blur_level2":tune.uniform(0, len([3, 5, 7, 9, 11, 13, 15]) - 1),
#     "dp": tune.uniform(0.1, 3.0),
#     "param1": tune.uniform(150,300),
#     "param2": tune.uniform(0.1, 0.9),
#     "resolution": tune.uniform(0.001, 1),
#     "size": tune.uniform(50, 150),
#     "rolling_window": tune.uniform(0, 1),
#     "window": tune.uniform(10, 100),
#     "movement_thresh": tune.uniform(1.0, 2.0)
# }

# optimize_params({"r_in_center":3,"r_out_center":7,"r_in_radius":3,"r_out_radius":8,"ring_index":1,'brightness':0,"clipLimit":5,"titleGridSize1":4,"titleGridSize2":4,
#                  "blur_level1":3,'blur_level2':3,'dp':0.9,'param1':200,'param2':0.9,'resolution':0.01,'size':50,'rolling_window':0.9,
#                  'window':10,'movement_thresh':1.4})

ray.init(include_dashboard=True,dashboard_host='127.0.0.1',dashboard_port=8898)


config_space = ConfigurationSpace()

config_space.add(UniformIntegerHyperparameter("r_in_center", lower=2, upper=15))
config_space.add(UniformIntegerHyperparameter("r_out_center", lower=2, upper=15))
config_space.add(UniformIntegerHyperparameter("r_in_radius", lower=2, upper=15))
config_space.add(UniformIntegerHyperparameter("r_out_radius", lower=2, upper=15))
config_space.add(UniformIntegerHyperparameter("ring_index", lower=1, upper=6))
config_space.add(UniformFloatHyperparameter("brightness", lower=0.0, upper=50.0))
config_space.add(UniformFloatHyperparameter("clipLimit", lower=0.1, upper=10.0))
config_space.add(UniformIntegerHyperparameter("titleGridSize1", lower=4, upper=32))
config_space.add(UniformIntegerHyperparameter("titleGridSize2", lower=4, upper=32))
config_space.add(CategoricalHyperparameter("blur_level1", [3, 5, 7, 9, 11, 13, 15]))
config_space.add(CategoricalHyperparameter("blur_level2", [3, 5, 7, 9, 11, 13, 15])) 
config_space.add(UniformFloatHyperparameter("dp", lower=0.1, upper=3.0))
config_space.add(UniformFloatHyperparameter("param1", lower=150, upper=300))
config_space.add(UniformFloatHyperparameter("param2", lower=0.1, upper=0.9))
config_space.add(UniformFloatHyperparameter("resolution", lower=0.001, upper=0.5))
config_space.add(UniformIntegerHyperparameter("size", lower=50, upper=150))
config_space.add(CategoricalHyperparameter("rolling_window", ['mean','median']))
config_space.add(UniformIntegerHyperparameter("window", lower=10, upper=100))
config_space.add(UniformFloatHyperparameter("movement_thresh", lower=0.5, upper=2.5))

algo = TuneBOHB(config_space,metric="mean_distance", mode="min")
bohb_scheduler = HyperBandForBOHB(time_attr="training_iteration", max_t=100)

tuner = tune.Tuner(
    optimize_params,
    tune_config=tune.TuneConfig(
        metric="mean_distance",
        mode="min",
        search_alg=algo,
        scheduler=bohb_scheduler,
        num_samples=20,
    ),
    run_config=train.RunConfig(
        name="param_optimization",
    ),
    # param_space=search_space,
)
results = tuner.fit()

print('results')
all_results = []
for result in results:
    print(result)
    config = result.config 
    mean_distance = result.metrics.get("mean_distance")
    config["mean_distance"] = mean_distance
    all_results.append(config)


df=pd.DataFrame(all_results)
print(df)
df.to_excel('results.xlsx')