import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt

def url_to_image(url_1,url_2):
    resp_1 = urlopen(url_1)
    resp_2 = urlopen(url_2)
    backg = np.asarray(bytearray(resp_1.read()), dtype="uint8")
    backg = cv2.imdecode(backg, cv2.IMREAD_COLOR)
    obj = np.asarray(bytearray(resp_2.read()), dtype="uint8")
    obj = cv2.imdecode(obj, cv2.IMREAD_COLOR)
    return backg,obj

def mix(backg, obj, size_backg, size_obj):
    size_backg = int(size_backg)
    size_obj = int(size_obj)
    
    hsv = cv2.cvtColor(obj, cv2.COLOR_BGR2HSV)

    backg = cv2.resize(backg, (0,0), fx=0.6, fy=0.6) 
    backg_g = cv2.cvtColor(backg, cv2.COLOR_BGR2GRAY)
    new_mask = np.zeros_like(backg)
    new_mask_2 = np.ones_like(backg_g)

    lower_green = np.array([40,90,90])
    upper_green = np.array([80,255,255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((2,2),np.uint8)
    mask = cv2.dilate(mask, kernel,iterations = 1)

    mask_inv = cv2.bitwise_not(mask)

    obj = cv2.bitwise_and(obj,obj,mask=mask_inv)

    if obj.shape[0] != 354:
        obj = cv2.resize(obj, (0,0), fx=0.2, fy=0.2)
        mask = cv2.resize(mask, (0,0), fx=0.2, fy=0.2)

    row_o, col_o, _ = obj.shape
    row_b, col_b, _ = new_mask.shape

    hrow_o, hcol_o = row_o//2, col_o//2   
    hrow_b, hcol_b = row_b//2, col_b//2

    new_mask[(hrow_b-hrow_o):(hrow_b-hrow_o)+row_o,(hcol_b-hcol_o):(hcol_b-hcol_o)+col_o] = obj

    if size_obj != 0:
        new_mask = cv2.GaussianBlur(new_mask,(size_obj,size_obj),0)

    new_mask_2[(hrow_b-hrow_o):(hrow_b-hrow_o)+row_o,(hcol_b-hcol_o):(hcol_b-hcol_o)+col_o] = mask #inverted mask background size
    
    mix_1 = cv2.bitwise_and(backg,backg,mask=new_mask_2)
    if size_backg != 0:
        mix_1 = cv2.GaussianBlur(mix_1,(size_backg,size_backg),0)
    
    mix_2 = cv2.bitwise_or(new_mask,mix_1)

    return mix_2

def interface():
    print('Select Background Image:')
    print('a - Forest')
    print('b - Pisa')
    print('c - Street')
    
    background_selected = str(input())
    if background_selected != 'a' and background_selected != 'b' and background_selected != 'c':
        print('Invalid input, please try again.')
        x = interface()
        return(x)
    
    print('Select Object Image:')
    print('a - Harry Potter')
    print('b - Standing Guy')
    print('c - Skater')
    
    object_selected = str(input())
    if object_selected != 'a' and object_selected != 'b' and object_selected != 'c':
        print('Invalid input, please try again.')
        x = interface()
        return(x)

    if background_selected == 'a':
        url_1 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/forest.jpg'
    elif background_selected == 'b':
        url_1 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/pisa.jpg'
    elif background_selected == 'c':
        url_1 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/street.jpg'

    if object_selected == 'a':
        url_2 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/green_broom.jpg'
    elif object_selected == 'b':
        url_2 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/green_dude.png'
    elif object_selected == 'c':
        url_2 = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/green_jump.jpg'

    print('Select kernel size for degradation of background (any odd number, 0 for no degradation):')
    size_backg = str(input())

    print('Select kernel size for degradation of object (any odd number, 0 for no degradation):')
    size_obj = str(input())

    return(url_1,url_2,size_backg,size_obj)

def main():
    while True:
        y = interface()
        z = url_to_image(y[0],y[1])
        final = mix(z[0],z[1],y[2],y[3])
        cv2.imshow('Hi',final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('')
        repeat = str(input('Another combination? (Y/N): ')).lower()
        if repeat == 'y': continue
        else: break

main()





    
