# Importing Libraries
import cv2
import dlib
import numpy as np
from tensorflow.keras.models import load_model


# Initializing Parameters
misses = []
model_dir = "F:/Projects/Beast/Models/Inpainting/Functional_Mask_Trial_3.h5"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("F:/Models/shape_predictor_68_face_landmarks.dat")
model = load_model(model_dir)

# Class Definition

class Local_Predictor():
    def __init__(self,model_dir,dlib_model_dir):
        print("Loading Models")
        self.model = load_model(model_dir)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(dlib_model_dir)
        self.input_shape = (128,128,3)
    
    def extract_landmarks(self,frame):
        faces = self.detector(frame)
        if(len(faces)==0):
            return 0
        for face in faces:
            shape = self.predictor(frame,face)
            left_eye_marks = np.array([[shape.part(z).x,shape.part(z).y] for z in range(36,42)])
            right_eye_marks = np.array([[shape.part(z).x,shape.part(z).y] for z in range(42,48)])
            cv2.fillConvexPoly(frame,left_eye_marks,(255,255,255))
            cv2.fillConvexPoly(frame,right_eye_marks,(255,255,255))
            return [[face.left(),face.right(),face.top(),face.bottom()] ,[[int((shape.part(i).x-face.left())*(128/(face.right()-face.left()))),int((shape.part(i).y-face.top())*(128/(face.bottom()-face.top())))] for i in range(36,48)]]
    
    def predict(self,frame,mask):
        if(frame.shape!=self.input_shape):
            print("Incorrect Input Frame Shape")
            frame = cv2.resize(frame,self.input_shape[:2])
        if(mask.shape[:2]!=self.input_shape[:2]):
            print("Incorrect Mask Shape")
            mask = cv2.resize(mask,self.input_shape[:2])
        return self.model.predict([[frame],[mask.astype(np.float16)]])[0]
            
# # Run
# pred = Local_Predictor(model_dir,"F:/Models/shape_predictor_68_face_landmarks.dat")
# vid = cv2.VideoCapture(0)
# run = True
# while(run):
#     run,img = vid.read()
#     if(run!=True):
#         break
#     input_mask = np.ones((128,128,3))
#     coords = pred.extract_landmarks(img)
#     if(type(coords)!=int): 
#         face_coords,eye_marks = coords
#         if(face_coords[3]>img.shape[0]):
#             face_coords[3] = img.shape[0]
#         if(face_coords[1]>img.shape[1]):
#             face_coords[1] = img.shape[1]
#         input_img = cv2.resize(img[face_coords[2]:face_coords[3],face_coords[0]:face_coords[1]],(128,128))/255
#         cv2.fillConvexPoly(input_mask,np.array(eye_marks[0:6]),(0,0,0))
#         cv2.fillConvexPoly(input_mask,np.array(eye_marks[6:12]),(0,0,0))
#         pred_img = pred.predict(input_img,input_mask)
#         pred_img = cv2.resize(pred_img,(face_coords[1]-face_coords[0],face_coords[3]-face_coords[2]))*255
#         img[face_coords[2]:face_coords[3],face_coords[0]:face_coords[1]] = pred_img
#         cv2.imshow("Image",img)
#     else: #Passthrough
#         cv2.imshow("Image",img)
#     if(cv2.waitKey(1)==27):
#         break
# cv2.destroyAllWindows()
# vid.release()