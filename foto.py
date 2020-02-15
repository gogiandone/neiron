import dlib
from skimage import io
from scipy.spatial import distance

#подгружаем модели
sp = dlib.shape_predictor('c:/Users/USER/Desktop/neiron/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('c:/Users/USER\Desktop/neiron/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

#загружаем первую фотографию
img = io.imread('C:/Users/USER/Desktop/neiron/1.jpg')

#показываем фотографию средствами dlib
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(img)

dets = detector(img, 1)

for k, d in enumerate(dets):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)
    
#извлечение дискриптора
face_descriptor1 = facerec.compute_face_descriptor(img, shape)
#print(face_descriptor1)

#загружаем вторую фотографию
img2 = io.imread('C:/Users/USER/Desktop/neiron/2.jpg')
win2 = dlib.image_window()
win2.clear_overlay()
win2.set_image(img2)
dets_webcam = detector(img2, 1)
for k, d in enumerate(dets_webcam):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))
    shape = sp(img2, d)
    win2.clear_overlay()
    win2.add_overlay(d)
    win2.add_overlay(shape)
               
face_descriptor2 = facerec.compute_face_descriptor(img2, shape)
               
a = distance.euclidean(face_descriptor1, face_descriptor2)#сравнение дискрипторов если больше 0.6 то не похож
print(a)
