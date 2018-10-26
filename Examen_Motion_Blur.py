import cv2
import numpy as np
from urllib.request import urlopen
from matplotlib import pyplot as plt

def url_to_image(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image_Gra = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    image_RGB = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image_Gra, image_RGB

def blur_Horizontal(source, size):
    filter_H = np.zeros((size,size))
    filter_H[int((size-1)/2), :] = np.ones(size)
    filter_H = filter_H / size
    output_H = cv2.filter2D(source, -1, filter_H)
    return(output_H)

def blur_Vertical(source, size):
    filter_V = np.zeros((size,size))
    filter_V[:, int((size-1)/2)] = np.ones(size)
    filter_V = filter_V / size
    output_V = cv2.filter2D(source, -1, filter_V)
    return(output_V)

def blur_Diagonal(source, size):
    filter_D = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if i == j: filter_D[i, j] = 1
    filter_D = filter_D / size
    output_D = cv2.filter2D(source, -1, filter_D)
    return(output_D)

def linear_2_Polar(img, xm, ym):
    value = np.sqrt(((img.shape[0]/1.0)**2)+((img.shape[1]/1.0)**2))
    polar_image = cv2.linearPolar(img,(xm, ym), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    return(polar_image)

def polar_2_Linear(img, polar_img, xm, ym):
    value = np.sqrt(((img.shape[0]/1.0)**2)+((img.shape[1]/1.0)**2))
    polar_rad = cv2.linearPolar(polar_img, (xm, ym), value, cv2.WARP_INVERSE_MAP)
    return(polar_rad)
    

def blur_Radial(source, size, xm, ym, BW):
    if BW: source = source*255.
    filter_V = np.zeros((size,size))
    filter_V[:, int((size-1)/2)] = np.ones(size)
    filter_V = filter_V / size
    polar_image = linear_2_Polar(source, xm, ym)
    polar_rad = cv2.filter2D(polar_image, -1, filter_V)
    polar_rad = polar_2_Linear(source, polar_rad, xm, ym)
    if BW: polar_rad = polar_rad/polar_rad.max()
    return(polar_rad)

def blur_Zoom(source, size, xm, ym, BW):
    if BW: source = source*255.
    filter_H = np.zeros((size,size))
    filter_H[int((size-1)/2), :] = np.ones(size)
    filter_H = filter_H / size
    polar_image = linear_2_Polar(source, xm, ym)
    polar_zoom = cv2.filter2D(polar_image, -1, filter_H)
    polar_zoom = polar_2_Linear(source, polar_zoom, xm, ym)
    if BW: polar_zoom = polar_zoom/polar_zoom.max()
    return(polar_zoom)
    

def _main():
    print('Select Image:')
    print('a - Kanye')
    print('b - Cute Doge')
    print('c - Rad Car')
    print('d - XMAS V SPOOK')
    print('e - Bien Muerta')
    print('f - Dog Sign')
    print('g - Statue')
    print('h - Hard Rock')
    print('i - Generic Wall')
    print('j - Pantages Far')
    print('k - Pantages Close')
    print('l - Tabasco')
    print('m - Indoor Sing')
    print('n - Moo Spook')
    image_selected = str(input())

    print('Select B/W or RGB:')
    print('a - B/W')
    print('b - RGB')
    color_selected = str(input())
    
    print('Select Blur Type:')
    print('a - Horizontal')
    print('b - Vertical')
    print('c - Diagonal')
    print('d - Radial')
    print('e - Zoom')
    blur_selected = str(input())

    print('Select Kernel Size: (10 - 50)')
    size = int(input())

    if ((blur_selected == 'd') or (blur_selected == 'e')):    ##Radial / Zoom##
        print('Select Coordinates: (1 / 512)')
        xm = int(input('X: '))-1
        ym = int(input('Y: '))-1

    if   image_selected == 'a':
        lbl_Source = 'Kanye'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/kanye.png'
    elif image_selected == 'b':
        lbl_Source = 'Cute Doge'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/dog_1.png'
    elif image_selected == 'c':
        lbl_Source = 'Rad Car'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/pink_1.png'
    elif image_selected == 'd':
        lbl_Source = 'XMAS V SPOOK'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/xmas.png'
    elif image_selected == 'e':
        lbl_Source = 'Bien Muerta'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/bienmuerta.png'
    elif image_selected == 'f':
        lbl_Source = 'Dog Sign'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/dog_2.png'
    elif image_selected == 'g':
        lbl_Source = 'Statue'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/elefante.png'
    elif image_selected == 'h':
        lbl_Source = 'Hard Rock'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/hardrock.png'
    elif image_selected == 'i':
        lbl_Source = 'Generic Wall'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/muro.png'
    elif image_selected == 'j':
        lbl_Source = 'Pantages Far'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/pantages_1.png'
    elif image_selected == 'k':
        lbl_Source = 'Pantages Close'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/pantages_2.png'
    elif image_selected == 'l':
        lbl_Source = 'Tabasco'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/tabasco.png'
    elif image_selected == 'm':
        lbl_Source = 'Indoor Sing'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/town.png'
    elif image_selected == 'n':
        lbl_Source = 'Moo Spook'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/vaca_1.png'
        

    source, source_Color = url_to_image(url)
    if color_selected == 'a':    ##BW / RGB##
        source_Selected = source
    elif color_selected == 'b':
        source_Selected = source_Color

    if   blur_selected == 'a':    ##Horizontal##
        lbl_Blur = 'Blur Horizontal'
        Image_Output_Blur = blur_Horizontal(source_Selected, size)
    elif blur_selected == 'b':    ##Vertical##
        lbl_Blur = 'Blur Vertical'
        Image_Output_Blur = blur_Vertical(source_Selected, size)
    elif blur_selected == 'c':    ##Diagonal##
        lbl_Blur = 'Blur Diagonal'
        Image_Output_Blur = blur_Diagonal(source_Selected, size)
    elif blur_selected == 'd':    ##Radial##
        lbl_Blur = 'Blur Radial'
        if (color_selected == 'a') : Image_Output_Blur = blur_Radial(source_Selected, size, xm, ym, True)
        else : Image_Output_Blur = blur_Radial(source_Selected, size, xm, ym, False)
    elif blur_selected == 'e':    ##Zoom##
        lbl_Blur = 'Blur Zoom'
        if (color_selected == 'a') : Image_Output_Blur = blur_Zoom(source_Selected, size, xm, ym, True)
        else : Image_Output_Blur = blur_Zoom(source_Selected, size, xm, ym, False)

    cv2.imshow(lbl_Source, source_Selected)
    cv2.imshow(lbl_Blur, Image_Output_Blur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    while True:
        _main()
        print('')
        One_More_Time = str(input('Another Filter? (Y/N): ')).lower()
        if One_More_Time == 'y': continue
        else: break
main()


##http://chemaguerra.com/circular-radial-blur/
##https://stackoverflow.com/questions/51675940/converting-an-image-from-cartesian-to-polar-limb-darkening
##https://python-forum.io/Thread-Image-conversion-form-cartesian-to-polar-and-back-to-cartesian
