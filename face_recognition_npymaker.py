# make face.npy
import dlib, cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('models/dlib_face_recognition_resnet_model_v1.dat')

#얼굴을 찾는 함수
def find_faces(img):
    dets = detector(img, 1)
    
    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    rects,shapes=[], []
    shapes_np=np.zeros((len(dets), 68, 2), dtype=np.int)

    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()),(d.right(), d.bottom()))
        rects.append(rect)

        shape=sp(img, d)

        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)

    return rects, shapes, shapes_np

#얼굴을 npy로 저장하는 함수
def encode_faces(img, shapes):
    face_descriptors=[]

    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

# img file
img_paths = {
    'YeongHoon': 'img/Face_img.jpg' #영훈 이미지 path
}

descs = {
    'YeongHoon': None
}

#얼굴을 찾아서 npy로 저장
for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    _,img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]

# npy저장
np.save('img\YeongHoon.npy',descs)
print(descs)

