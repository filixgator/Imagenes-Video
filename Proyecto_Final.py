import cv2
import requests
import numpy as np
from urllib.request import urlopen

ip = "10.12.5.48"

def url_to_image(url):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image_Gra = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    image_RGB = cv2.imdecode(image, -1)
    return image_Gra, image_RGB

def gradient_img(img, gray, gradi, gray_gradi):
    sobel_horizont = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = 5)
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = 5)
    sobel_out = np.sqrt((sobel_horizont*sobel_horizont)+(sobel_vertical*sobel_vertical))
    sobel_out = 255 * (sobel_out/sobel_out.max())
    mask = cv2.inRange(sobel_out, 0, 40)
    mask_inv = cv2.bitwise_not(mask)
    white_img = np.zeros((480, 640, 3), np.uint8)
    white_img[:] = (255, 255, 255)
    img_out_1 = cv2.bitwise_and(gradi, gradi, mask = mask)
    img_out_2 = cv2.bitwise_and(white_img, white_img, mask = mask_inv)
    img_out = cv2.add(img_out_1, img_out_2)    
    return img_out, mask

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters
 
def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum

def face_img(img, gray, face_cascade, eyes_cascade, smile_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        detected_face_gray = gray[y:y+h, x:x+w]
        detected_face_color = img[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(detected_face_gray, 1.3, 7)
        smiles = smile_cascade.detectMultiScale(detected_face_gray, 3, 20)
        for (xeye, yeye, weye, heye) in eyes:
            cv2.rectangle(detected_face_color, (xeye,yeye), (xeye+weye, yeye+heye), (0,255,0), 2)
        for (xsmile, ysmile, wsmile, hsmile) in smiles:
            cv2.rectangle(detected_face_color, (xsmile,ysmile), (xsmile+wsmile, ysmile+hsmile), (0,0,255), 2)
    return img, gray

def face_img_mask(img, gray, crown_Gris, crown, face_cascade, eyes_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
            crown_resized = cv2.resize(crown, (w, int(h/2)))
            y1, y2 = y + int(h/8), y + int(h/8) + crown_resized.shape[0]
            x1, x2 = x, x + crown_resized.shape[1]
            alpha_s = crown_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (alpha_s * crown_resized[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
            detected_face_gray = gray[y:y+h, x:x+w]
            detected_face_color = img[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(detected_face_gray,1.3, 7)
    return img, gray

def face_img_barba(img, gray, crown_Gris, crown, face_cascade, eyes_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
            crown_resized = cv2.resize(crown, (w, int(h/2)))
            y1, y2 = y + int(h/8), y + int(h/8) + crown_resized.shape[0]
            x1, x2 = x, x + crown_resized.shape[1]
            alpha_s = crown_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                    img[y1:y2, x1:x2, c] = (alpha_s * crown_resized[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
            detected_face_gray = gray[y:y+h, x:x+w]
            detected_face_color = img[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(detected_face_gray,1.3, 7)
    return img, gray
    
def face_img_stars(img, gray, stars, face_cascade, eyes_cascade, smile_cascade):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    mask = np.zeros((480, 640), np.uint8)
    mask[:] = (0)
    for (x, y, w, h) in faces:
            detected_face_gray = gray[y:y+h, x:x+w]
            detected_face_color = img[y:y+h, x:x+w]
            eyes = eyes_cascade.detectMultiScale(detected_face_gray, 1.3, 7)
            smiles = smile_cascade.detectMultiScale(detected_face_gray, 3, 20)
            for (xeye, yeye, weye, heye) in eyes:
                detected_eye_gray = gray[yeye:yeye+heye, xeye:xeye+weye]
                detected_eye_color = img[yeye:yeye+heye, xeye:xeye+weye]
                eye_elipse = 255 * (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(weye,int(heye/2))))
                mask[y+yeye+int(heye/4):y+yeye+int(heye/4)+len(eye_elipse), x+xeye:x+xeye+weye] = eye_elipse
            for (xsmile, ysmile, wsmile, hsmile) in smiles:
                smile_elipse = 255 * (cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(wsmile,int(hsmile/2))))
                mask[y+ysmile+int(hsmile/4):y+ysmile+int(hsmile/4)+len(smile_elipse), x+xsmile:x+xsmile+wsmile] = smile_elipse
    mask_inv = cv2.bitwise_not(mask)
    img_out_1 = cv2.bitwise_and(stars, stars, mask = mask_inv)
    img_out_2 = cv2.bitwise_and(img, img, mask = mask)
    img_out = cv2.add(img_out_1, img_out_2)
    return img_out, mask
    
def main(ip):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade= cv2.CascadeClassifier('haarcascade_smile.xml')
    filter_num = 0

    cap = cv2.VideoCapture(0)
    streaming = False
    shot  = "http://" + ip + ":8080/shot.jpg"
    
    url_purple_mask = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/mask_1.png'
    url_stars = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/space.jpg'
    url_barba = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/barba_1.png'
    url_gradi = 'https://raw.githubusercontent.com/filixgator/Imagenes-Video/master/gradient_colors.jpg'
    gray_mask, mask = url_to_image(url_purple_mask)
    gray_stars, stars = url_to_image(url_stars)
    gray_barba, barba = url_to_image(url_barba)
    gray_gradi, gradi = url_to_image(url_gradi)
###
    filters = build_filters()
###
    while True:
        if streaming:
            try:
                img_resp = requests.get(shot)
                img_arr  = np.array(bytearray(img_resp.content), dtype=np.uint8)
                img = cv2.imdecode(img_arr, -1)
                img = cv2.flip( img, 1 )
            except:
                ret, img = cap.read()
        else:
            ret, img = cap.read()
            
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if   filter_num == 0:
            img, gray_img = face_img(img, gray, face_cascade, eyes_cascade, smile_cascade)
        elif filter_num == 1:
            img, gray_img = face_img_mask(img, gray, gray_mask, mask, face_cascade, eyes_cascade)
        elif filter_num == 2:
            img, gray_img = face_img_barba(img, gray, gray_barba, barba, face_cascade, eyes_cascade)
        elif filter_num == 3:
            img, gray_img = gradient_img(img, gray, gradi, gray_gradi)
        elif filter_num == 4:
            img, gray_img = face_img_stars(img, gray, stars, face_cascade, eyes_cascade, smile_cascade)
        elif filter_num == 5:
            print("5")
            filter_num = 3
        elif filter_num == 6:
            print("6")
            filter_num = 3
        
        img_flipped = img.copy()
        img_flipped = cv2.flip( img, 1 )
        
        cv2.imshow('img', img_flipped)
        cv2.imshow('gray',gray_img)
        k = cv2.waitKey(30)
        
        if k == 27:
##            print("Y: " + str(img_flipped.shape[0]))
##            print("X: " + str(img_flipped.shape[1]))
##            print(gray_img[0,0])
            break
        else:
            if k == -1:
                continue
            if k == 115:
                if streaming:
                    streaming = False
                else:
                    streaming = True
            else:
                filter_num = k -48
            print(k)
            continue
    cap.release()
    cv2.destroyAllWindows()
main(ip)

