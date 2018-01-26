import os
from keras import backend as K
from keras.models import load_model
import numpy as np
import scipy
import cv2
import tkinter as tk


_name       = 'Face Generator'
_width      = 360
_height     = 700

_MODEL_PATH         = 'output/100_epochs_tensorflow_FaceGen.YaleFaces.model.d5.adam.h5'
_HAARCASCADE_PATH   = 'C:/opencv/data/haarcascades/haarcascade_frontalface_default.xml'


class MainApplication(tk.Frame):
    global _width, _height, _name

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)

        self._path_model                = _MODEL_PATH
        self._path_output               = 'live_output'
        self._output_counter            = 0

        self._identities                = 28
        self._poses                     = 10
        self._azimuth_low               = -90
        self._azimuth_high              = 90
        self._elevation_low             = -90
        self._elevation_high            = 90

        self.image                      = None
        self.clahe                      = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        self.haarcascade                = cv2.CascadeClassifier(_HAARCASCADE_PATH)

        self.parent                     = parent
        self.widgets_scale_identity_1   = [None] * int((self._identities / 2))
        self.widgets_scale_identity_2   = [None] * int((self._identities / 2))
        self.value_identities_1         = [None] * int(self._identities / 2)
        self.value_identities_2         = [None] * int(self._identities / 2)

        self.color_error                = 'red'
        self.color_warning              = 'orange'
        self.color_ok                   = 'green'
        self.color_validate             = 'orange'
        self.color_default              = 'SystemButtonFace'
        self.color_default_entry        = 'white'

        self.parent.title(_name)

        self.init_layout_settings()
        self.init_frames()
        self.init_variables()
        self.init_widgets()
        self.set_variables()
        self.grid_widgets()

        self.init_deconvfaces()

    def init_deconvfaces(self):
        if not os.path.exists(self._path_output):
            os.makedirs(self._path_output)

        self.model = load_model(self._path_model)

    def init_layout_settings(self):
        self._paddingY      = 5
        self._paddingX      = 2
        self._inlineX       = 15
        self._paddingTop    = 2

    def init_frames(self):
        self.frame1 = tk.Frame(self.parent)
        self.frame2 = tk.Frame(self.parent)
        self.frame3 = tk.Frame(self.parent)
        self.frame4 = tk.Frame(self.parent)

    def init_variables(self):
        # frame1
        for i in range(int(self._identities / 2)):
            self.value_identities_1[i] = tk.IntVar()

        # frame2
        for i in range(int(self._identities / 2)):
            self.value_identities_2[i] = tk.IntVar()

        # frame3
        self.val_frame3_azimuth     = tk.IntVar()
        self.val_frame3_elevation   = tk.IntVar()
        self.val_frame3_checkbox    = tk.IntVar()

    def init_widgets(self):
        # frame1
        self.label_frame_1  = tk.Label(self.frame1, text='IDENTITIES', font='Helvetica 10 bold')
        self.label_current_scale_id  = tk.Label(self.frame1, text='Current ID: 0')

        for i in range(int(self._identities/2)):
            self.widgets_scale_identity_1[i]= tk.Scale(self.frame1, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.value_identities_1[i], state='normal',
                                   command=self.change_current_scale_id_1, activebackground='grey')

        # frame2
        for i in range(int(self._identities/2)):
            self.widgets_scale_identity_2[i]= tk.Scale(self.frame2, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.value_identities_2[i], state='normal',
                                   command=self.change_current_scale_id_2, activebackground='grey')

        # frame3
        self.label_frame_3          = tk.Label(self.frame3, text='POSE', font='Helvetica 10 bold')
        self.listbox_frame3_pose    = tk.Listbox(self.frame3)

        self.label_frame_31         = tk.Label(self.frame3, text='LIGHTING', font='Helvetica 10 bold')
        self.scale_frame3_azimuth   = tk.Scale(self.frame3, from_=self._azimuth_low, to=self._azimuth_high,
                                               orient=tk.HORIZONTAL, variable=self.val_frame3_azimuth, state='normal',
                                               activebackground='grey', label='Azimuth')
        self.scale_frame3_elevation = tk.Scale(self.frame3, from_=self._elevation_low, to=self._elevation_high,
                                                 orient=tk.HORIZONTAL, variable=self.val_frame3_elevation, state='normal',
                                                 activebackground='grey', label='Elevation')
        self.checkbox_frame3_auto_detect = tk.Checkbutton(self.frame3, text='Automatically\ndetect faces',
                                                          variable=self.val_frame3_checkbox, fg='black')
        self.button_frame3_face_detected = tk.Button(self.frame3, text='no face', bg='gray', state='disabled',
                                                     disabledforeground='black', border=0, font='blue')


        # frame5
        self.button_frame4_const = tk.Button(self.frame4, text='constrain', bg='gray', command=self.constrain, width=15)
        self.button_frame4_rand  = tk.Button(self.frame4, text='random', bg='gray', command=self.random, width=15)
        self.button_frame4_reset = tk.Button(self.frame4, text='reset', bg='gray', command=self.reset, width=15)
        self.button_frame4_generate = tk.Button(self.frame4, text='Generate', bg='green', command=self.generate,
                                                width=30)


    def set_variables(self):
        # frame1
        for val in self.value_identities_1:
            val.set(0)

        # frame2
        for val in self.value_identities_2:
            val.set(0)

        # frame3
        for i in range(self._poses):
            self.listbox_frame3_pose.insert(tk.END, i)

        self.val_frame3_azimuth.set(0)
        self.val_frame3_elevation.set(0)
        self.val_frame3_checkbox.set(0)

    def grid_widgets(self):
        # grid frames
        self.frame1.grid(row=0, column=0, sticky=tk.W + tk.N, padx=self._paddingX,
                         pady=(self._paddingTop, self._paddingY))
        self.frame2.grid(row=0, column=1, sticky=tk.E + tk.W + tk.N, padx=self._paddingX,
                         pady=(self._paddingTop+43, self._paddingY))
        self.frame3.grid(row=0, column=2, sticky=tk.E + tk.W + tk.N, padx=self._paddingX*2,
                         pady=(self._paddingTop, self._paddingY))
        self.frame4.grid(row=1, column=0, columnspan=3, padx=self._paddingX, pady=(self._paddingTop, self._paddingY))

        # frame1
        self.label_frame_1.grid(sticky='W')
        self.label_current_scale_id.grid(sticky='W')
        for scale in self.widgets_scale_identity_1:
            scale.grid(sticky='W')

        # frame2
        for scale in self.widgets_scale_identity_2:
            scale.grid(sticky='W')

        # frame3
        self.label_frame_3.grid(sticky='W')
        self.listbox_frame3_pose.grid(sticky='W', pady=45)

        self.label_frame_31.grid(sticky='W')
        self.scale_frame3_azimuth.grid(sticky='W')
        self.scale_frame3_elevation.grid(sticky='W')

        self.checkbox_frame3_auto_detect.grid(sticky='W', pady=25)
        self.button_frame3_face_detected.grid(sticky='W')


        # frame5
        self.button_frame4_const.grid(row=0, column=0,  sticky='W')
        self.button_frame4_rand.grid(row=0, column=1, sticky='W')
        self.button_frame4_reset.grid(row=0, column=2, sticky='W')
        self.button_frame4_generate.grid(row=1, column=0, columnspan=4)

    def change_current_scale_id_1(self, val):
        counter = 0
        for scale in self.widgets_scale_identity_1:
            counter += 1
            if scale.cget('state') == 'active':
                self.label_current_scale_id.config(text='Current ID: ' + str(counter))

    def change_current_scale_id_2(self, val):
        counter = 0
        for scale in self.widgets_scale_identity_2:
            counter += 1
            if scale.cget('state') == 'active':
                self.label_current_scale_id.config(text='Current ID: ' + str(counter + int(self._identities / 2)))

    def reset(self):
        for i in range(len(self.value_identities_1)):
            self.value_identities_1[i].set(0)
        for i in range(len(self.value_identities_2)):
            self.value_identities_2[i].set(0)
        self.val_frame3_azimuth.set(0)
        self.val_frame3_elevation.set(0)
        self.listbox_frame3_pose.select_clear(0, tk.END)
        self.listbox_frame3_pose.select_set(0)

    def constrain(self):
        id_vector = self.get_id_vector()
        id_vector = id_vector / np.linalg.norm(id_vector) * 100
        for i in range(len(self.value_identities_1)):
            self.value_identities_1[i].set(np.around(id_vector[i]))
        for i in range(len(self.value_identities_2)):
            self.value_identities_2[i].set(np.around(id_vector[len(self.value_identities_1) + i]))

    def random(self):
        for i in range(len(self.value_identities_1)):
            self.value_identities_1[i].set(np.around(np.random.rand()*100))
        for i in range(len(self.value_identities_2)):
            self.value_identities_2[i].set(np.around(np.random.rand()*100))

    def set_face_button(self, is_face_detected):
        if is_face_detected:
            self.button_frame3_face_detected.configure(bg='green', text='face detected')
        else:
            self.button_frame3_face_detected.configure(bg='red', text='no face detected')

    def get_id_vector(self):
        id_vector = [0] * self._identities
        for i in range(len(self.value_identities_1)):
            id_vector[i] = self.value_identities_1[i].get() / 100
        for i in range(len(self.value_identities_2)):
            id_vector[len(self.value_identities_1) + i] = self.value_identities_2[i].get() / 100
        return id_vector

    def get_lighting_vector(self):
        azrad = np.deg2rad(self.val_frame3_azimuth.get())
        elrad = np.deg2rad(self.val_frame3_elevation.get())
        return np.array([np.sin(azrad), np.cos(azrad), np.sin(elrad),
                         np.cos(elrad)])

    def get_pose_vector(self):
        pose = [0] * self._poses
        pose[self.listbox_frame3_pose.get(tk.ACTIVE)] = 1
        return pose

    def detect_face(self):
        img = self.clahe.apply(self.image)
        faces = self.haarcascade.detectMultiScale(image=img, scaleFactor=1.3, minNeighbors=5)
        if len(faces) > 0:
            self.set_face_button(True)
        else:
            self.set_face_button(False)



    def generate(self):
        id_vector       = []
        light_vector    = []
        pose_vector     = []
        id_vector.append(self.get_id_vector())
        light_vector.append(self.get_lighting_vector())
        pose_vector.append(self.get_pose_vector())
        id_vector = np.array(id_vector)
        light_vector = np.array(light_vector)
        pose_vector = np.array(pose_vector)

        batch = {
            'identity': id_vector,
            'pose': pose_vector,
            'lighting': light_vector,
        }

        gen = self.model.predict_on_batch(batch)

        for i in range(0, gen.shape[0]):
            if K.image_dim_ordering() == 'th':
                image[:, :] = gen[i, 0, :, :]
            else:
                image = gen[i, :, :, 0]
            self.image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
            file_path = os.path.join(self._path_output, '{:05}.{}'.format(self._output_counter, 'jpg'))
            scipy.misc.imsave(file_path, image)
            cv2.imshow('Generated image', image)
            self._output_counter += 1
            if self.val_frame3_checkbox.get():
                self.detect_face()


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root).grid()
    root.wm_geometry('%dx%d'%(_width, _height))
    root.mainloop()
