import cv2
from urllib.request import urlopen
import numpy as np

def url_to_image(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image_Gra = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    image_RGB = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image_Gra, image_RGB

##Prewitt##
def Prewitt_Espacial(source, size):
    kernelx = np.ones((size*3, size*3))
    kernelx[0:size-1, :] = np.ones(size*3)
    kernelx[size:(2*size), :] = np.zeros(size*3)
    kernelx[(2*size):(3*size),:] = np.negative(np.ones(size*3))
    
    kernely = np.ones((size*3, size*3))
    kernely[:,0:size-1] = 1.
    kernely[:,size:(2*size)] = 0.
    kernely[:,(2*size):(3*size)] = -1.

    source = source*255.

    outx = cv2.filter2D(source, -1, kernelx)
    outy = cv2.filter2D(source, -1, kernely)

    outz_Prewitt = np.sqrt((outx*outx)+(outy*outy))
    outz_Prewitt = outz_Prewitt/outz_Prewitt.max()
    return(outz_Prewitt)

def Prewitt_Frecuencia(source, size):
    kernelx = np.ones((size*3, size*3))
    kernelx[0:size-1,:] = np.ones(size*3)
    kernelx[size:(2*size),:] = np.zeros(size*3)
    kernelx[(2*size):(3*size),:] = np.negative(np.ones(size*3))
    KERNELX = np.zeros((source.shape[0],source.shape[1]))
    KERNELX[int((source.shape[0]/2)-(size*3/2)):int((source.shape[0]/2)+(size*3/2)),
           int((source.shape[1]/2)-(size*3/2)):int((source.shape[1]/2)+(size*3/2))] = kernelx

    kernely = np.ones((size*3, size*3))
    kernely[:,0:size-1] = 1.
    kernely[:,size:(2*size)] = 0.
    kernely[:,(2*size):(3*size)] = -1.
    KERNELY = np.zeros((source.shape[0],source.shape[1]))
    KERNELY[int((source.shape[0]/2)-(size*3/2)):int((source.shape[0]/2)+(size*3/2)),
           int((source.shape[1]/2)-(size*3/2)):int((source.shape[1]/2)+(size*3/2))] = kernely

    source = source*255.

    source_f = np.fft.fft2(source)
    kernelx_f = np.fft.fft2(KERNELX)
    kernely_f = np.fft.fft2(KERNELY)

    source_kernelx_fshift = kernelx_f * source_f
    source_kernely_fshift = kernely_f * source_f
    
    f_ishiftx = np.fft.ifftshift(source_kernelx_fshift)
    img_backx = np.fft.ifft2(f_ishiftx)
    img_backx = np.abs(img_backx)

    f_ishifty = np.fft.ifftshift(source_kernely_fshift)
    img_backy = np.fft.ifft2(f_ishifty)
    img_backy = np.abs(img_backy)

    img_backz = cv2.add(img_backx, img_backy)
    img_backz = img_backz/img_backz.max()
    
    img_back  = img_backz.copy()
    img_back[0:int((source.shape[0]/2)),:] = img_backz[int((source.shape[0]/2)):source.shape[0],:]
    img_back[int((source.shape[0]/2)):source.shape[0],:] = img_backz[0:int((source.shape[0]/2)),:]
    img_backz = img_back.copy()
    img_back[:,0:int((source.shape[1]/2))] = img_backz[:,int((source.shape[1]/2)):source.shape[1]]
    img_back[:,int((source.shape[1]/2)):source.shape[1]] = img_backz[:,0:int((source.shape[1]/2))]

    return(img_back)

##Sobel##
def Sobel_Espacial(source, size):
    kernelito = np.asarray([[1.,2.,1.],
                            [2.,4.,2.],
                            [1.,2.,1.]])
    
    kernelx = np.ones((size*3, size*3))
    kernelx[0:size-1,:] = np.ones(size*3)
    kernelx[size:(2*size),:] = np.zeros(size*3)
    kernelx[(2*size):(3*size),:] = np.negative(np.ones(size*3))
    kernelx[:,size:(2*size)] = 2*kernelx[:,size:(2*size)]

    kernely = np.ones((size*3, size*3))
    kernely[:,0:size-1] = 1.
    kernely[:,size:(2*size)] = 0
    kernely[:,(2*size):(3*size)] = -1.
    kernely[size:(2*size),:] = 2*kernely[size:(2*size),:]

    source = source*255.

    ##Gausiano para suavizar como el amor de mama##
    img_Gaus = cv2.filter2D(source, -1, kernelito)
    img_Gaus = img_Gaus/img_Gaus.max()
    img_Gaus = img_Gaus*255.

    outx = cv2.filter2D(img_Gaus, -1, kernelx)
    outy = cv2.filter2D(img_Gaus, -1, kernely)

    out_Sobel = np.sqrt((outx*outx)+(outy*outy))
    out_Sobel = out_Sobel/out_Sobel.max()
    
    outx = outx + np.abs(outx.min())
    outx = outx/outx.max()
    outy = outy + np.abs(outy.min())
    outy = outy/outy.max()
    
    out_Sobel_Angle = np.rad2deg(np.arctan(outy/(outx+0.0000001)))
    
    return(out_Sobel)

def Sobel_Frecuencia(source, size):
    kernelito = np.asarray([[1.,2.,1.],
                            [2.,4.,2.],
                            [1.,2.,1.]])
    
    kernelx = np.ones((size*3, size*3))
    kernelx[0:size-1,:] = np.ones(size*3)
    kernelx[size:(2*size),:] = np.zeros(size*3)
    kernelx[(2*size):(3*size),:] = np.negative(np.ones(size*3))
    kernelx[:,size:(2*size)] = 2*kernelx[:,size:(2*size)]
    KERNELX = np.zeros((source.shape[0],source.shape[1]))
    KERNELX[int((source.shape[0]/2)-(size*3/2)):int((source.shape[0]/2)+(size*3/2)),
           int((source.shape[1]/2)-(size*3/2)):int((source.shape[1]/2)+(size*3/2))] = kernelx

    kernely = np.ones((size*3, size*3))
    kernely[:,0:size-1] = 1.
    kernely[:,size:(2*size)] = 0
    kernely[:,(2*size):(3*size)] = -1.
    kernely[size:(2*size),:] = 2*kernely[size:(2*size),:]
    KERNELY = np.zeros((source.shape[0],source.shape[1]))
    KERNELY[int((source.shape[0]/2)-(size*3/2)):int((source.shape[0]/2)+(size*3/2)),
           int((source.shape[1]/2)-(size*3/2)):int((source.shape[1]/2)+(size*3/2))] = kernely

    source = source*255.

    source_f = np.fft.fft2(source)
    kernelx_f = np.fft.fft2(KERNELX)
    kernely_f = np.fft.fft2(KERNELY)

    img_Gaus = cv2.filter2D(source, -1, kernelito)
    img_Gaus = img_Gaus/img_Gaus.max()

    source_kernelx_fshift = kernelx_f * source_f
    source_kernely_fshift = kernely_f * source_f
    
    f_ishiftx = np.fft.ifftshift(source_kernelx_fshift)
    img_backx = np.fft.ifft2(f_ishiftx)
    img_backx = np.abs(img_backx)

    f_ishifty = np.fft.ifftshift(source_kernely_fshift)
    img_backy = np.fft.ifft2(f_ishifty)
    img_backy = np.abs(img_backy)

##    img_backz = cv2.add(img_backx, img_backy)
    img_backz = np.sqrt((img_backx*img_backx)+(img_backy*img_backy))
    img_backz = img_backz/img_backz.max()
    img_backx = img_backx/img_backx.max()
    img_backy = img_backy/img_backy.max()

    out_Sobel_Angle = np.rad2deg(np.arctan(img_backy/(img_backx+0.0000001)))
    out_Sobel_Angle = np.round(out_Sobel_Angle,7)

    img_back  = img_backz.copy()
    img_back[0:int((source.shape[0]/2)),:] = img_backz[int((source.shape[0]/2)):source.shape[0],:]
    img_back[int((source.shape[0]/2)):source.shape[0],:] = img_backz[0:int((source.shape[0]/2)),:]
    img_backz = img_back.copy()
    img_back[:,0:int((source.shape[1]/2))] = img_backz[:,int((source.shape[1]/2)):source.shape[1]]
    img_back[:,int((source.shape[1]/2)):source.shape[1]] = img_backz[:,0:int((source.shape[1]/2))]

    img_back = img_back/img_back.max()
    
    return(img_back)


##Canny##
def Canny(source, upper, lower):
    source = source*255.
    out = source.copy()
    for ym in range(source.shape[0]):
        for xm in range(source.shape[1]):
            if source[ym, xm] < lower:  out[ym, xm] = 0.
    outx = out.copy()
    while True:
        out_base = out.copy()
        for ym in range(1,source.shape[0]-1):
            for xm in range(1,source.shape[1]-1):
                if ((out[ym, xm] > lower) and (out[ym, xm] < upper)):            
                    vec_1 = out[ym-1, xm-1]
                    vec_2 = out[ym-1, xm]
                    vec_3 = out[ym-1, xm+1]
                    vec_4 = out[ym,   xm-1]
                    vec_5 = out[ym,   xm+1]
                    vec_6 = out[ym+1, xm-1]
                    vec_7 = out[ym+1, xm]
                    vec_8 = out[ym+1, xm+1]
                    if ((vec_1 >= upper) or (vec_2 >= upper) or
                        (vec_3 >= upper) or (vec_4 >= upper) or
                        (vec_5 >= upper) or (vec_6 >= upper) or
                        (vec_7 >= upper) or (vec_8 >= upper)):
                        out[ym, xm] = upper
        if out_base.all() == out.all(): break

    for ym in range(source.shape[0]):
        for xm in range(source.shape[1]):
            if out[ym, xm] >= upper :  out[ym, xm] = 255.
            else: out[ym, xm] = 0.
    kernelito = np.ones((2,2))
##    out = cv2.erode(out,kernelito,iterations = 2)
##    outx = outx/outx.max()
    out = out/out.max()
    source = source/source.max()

    return(out)

##Roberts##
def Roberts_Espacial(source, size):    
    kernelx = np.zeros((size*2, size*2))
    kernely = np.zeros((size*2, size*2))
    for ym in range(size*2):
        for xm in range(size*2):
            if ym == xm:
                if ym >= (size):
                    kernelx[ym,xm] = -1.
                    kernely[ym,-xm-1] = -1.
                else:
                    kernelx[ym,xm] = 1.
                    kernely[ym,-xm-1] = 1.
    
    source = source*255.

    outx = cv2.filter2D(source, -1, kernelx)
    outy = cv2.filter2D(source, -1, kernely)

    out_Roberts = np.sqrt((outx*outx)+(outy*outy))
    out_Roberts = out_Roberts/out_Roberts.max()
    
    return(out_Roberts)

def Roberts_Frecuencia(source, size):
    kernelx = np.zeros((size*2, size*2))
    kernely = np.zeros((size*2, size*2))
    for ym in range(size*2):
        for xm in range(size*2):
            if ym == xm:
                if ym >= (size):
                    kernelx[ym,xm] = -1.
                    kernely[ym,-xm-1] = -1.
                else:
                    kernelx[ym,xm] = 1.
                    kernely[ym,-xm-1] = 1.

    KERNELX = np.zeros((source.shape[0],source.shape[1]))
    KERNELX[int((source.shape[0]/2)-(size)):int((source.shape[0]/2)+(size)),
            int((source.shape[1]/2)-(size)):int((source.shape[1]/2)+(size))] = kernelx
    KERNELY = np.zeros((source.shape[0],source.shape[1]))
    KERNELY[int((source.shape[0]/2)-(size)):int((source.shape[0]/2)+(size)),
            int((source.shape[1]/2)-(size)):int((source.shape[1]/2)+(size))] = kernely

    source = source*255.

    source_f = np.fft.fft2(source)
    kernelx_f = np.fft.fft2(KERNELX)
    kernely_f = np.fft.fft2(KERNELY)

    source_kernelx_fshift = kernelx_f * source_f
    source_kernely_fshift = kernely_f * source_f
    
    f_ishiftx = np.fft.ifftshift(source_kernelx_fshift)
    img_backx = np.fft.ifft2(f_ishiftx)
    img_backx = np.abs(img_backx)

    f_ishifty = np.fft.ifftshift(source_kernely_fshift)
    img_backy = np.fft.ifft2(f_ishifty)
    img_backy = np.abs(img_backy)

##    img_backz = cv2.add(img_backx, img_backy)
    img_backz = np.sqrt((img_backx*img_backx)+(img_backy*img_backy))
    img_backz = img_backz/img_backz.max()
    
    img_back  = img_backz.copy()
    img_back[0:int((source.shape[0]/2)),:] = img_backz[int((source.shape[0]/2)):source.shape[0],:]
    img_back[int((source.shape[0]/2)):source.shape[0],:] = img_backz[0:int((source.shape[0]/2)),:]
    img_backz = img_back.copy()
    img_back[:,0:int((source.shape[1]/2))] = img_backz[:,int((source.shape[1]/2)):source.shape[1]]
    img_back[:,int((source.shape[1]/2)):source.shape[1]] = img_backz[:,0:int((source.shape[1]/2))]

    return(img_back)

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
    print('o - Cow')
    image_selected = str(input())
    
    print('Select Filter:')
    print('a - Prewitt')
    print('b - Sobel')
    print('c - Canny')
    print('d - Roberts')
    filter_selected = str(input())

    print('Select Kernel Size: (1 - 6)')
    size = int(input())

    if filter_selected == 'c':    ##Canny##
        print('Select Thresholds: (0/255)')
        threshold_lower = float(input('Lower Threshold: '))
        threshold_upper = float(input('Upper Threshold: '))

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
    elif image_selected == 'o':
        lbl_Source = 'Cow'
        url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/vaca_2.png'
        

    source, source_Color = url_to_image(url)

    if   filter_selected == 'a':    ##Prewitt##
        lbl_Esp, lbl_Fre = 'Prewitt Espacial', 'Prewitt Frecuencia'
        Image_Output_Esp = Prewitt_Espacial(source, size)
        Image_Output_Fre = Prewitt_Frecuencia(source, size)
    elif filter_selected == 'b':    ##Sobel##
        lbl_Esp, lbl_Fre = 'Sobel Espacial', 'Sobel Frecuencia'
        Image_Output_Esp = Sobel_Espacial(source, size)
        Image_Output_Fre = Sobel_Frecuencia(source, size)
    elif filter_selected == 'c':    ##Canny##
        lbl_Esp, lbl_Fre = 'Canny Espacial', 'Canny Frecuencia'
        Image_Output_Esp = Sobel_Espacial(source, size)
        Image_Output_Fre = Sobel_Frecuencia(source, size)
        Image_Output_Esp = Canny(Image_Output_Esp, threshold_upper, threshold_lower)
        Image_Output_Fre = Canny(Image_Output_Fre, threshold_upper, threshold_lower)
    elif filter_selected == 'd':    ##Roberts##
        lbl_Esp, lbl_Fre = 'Roberts Espacial', 'Roberts Frecuencia'
        Image_Output_Esp = Roberts_Espacial(source, size)
        Image_Output_Fre = Roberts_Frecuencia(source, size)

    cv2.imshow(lbl_Source, source_Color)
    cv2.imshow(lbl_Esp, Image_Output_Esp)
    cv2.imshow(lbl_Fre, Image_Output_Fre)
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
##url = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/xmas.png'
##source, source_Color = url_to_image(url)
##cv2.imshow('Roberts_Espacial', Roberts_Espacial(source, 1))
##cv2.imshow('Roberts_Frecuencia', Roberts_Frecuencia(source, 1))
##cv2.waitKey(0)
##cv2.destroyAllWindows()
