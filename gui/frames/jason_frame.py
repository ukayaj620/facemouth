from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
import cv2

from gui.config import *
from facial_extractor.face_extractor import FaceExtractor


class JasonFrame(tk.Frame):
    def __init__(self, **kwargs):
        self.image_utama = None
        self.image_utama_rgb = None
        self.gambar_sebelum = None
        self.gambar_sesudah = None

        super().__init__(**kwargs)
        self._init_components()

    def _init_components(self):
        root_frame = self

        self.btn_load_image = tk.Button(
            root_frame, text="Load Image", command=self._on_btn_load_image_pressed)
        self.btn_load_image.grid(row=0, column=0, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.btn_process = tk.Button(
            root_frame, text="Process", command=self._on_btn_process_pressed)
        self.btn_process.grid(row=0, column=1, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.lbl_before = tk.Label(
            root_frame, text="Before", font=(DEFAULT_FONT, FONT_SIZE_H1))
        self.lbl_before.grid(row=1, column=0, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.canvas_before = tk.Canvas(
            root_frame, width=DEFAULT_CANVAS_WIDTH, height=DEFAULT_CANVAS_HEIGHT)
        self.canvas_before.create_rectangle(
            0, 0, DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT, fill='white')
        self.canvas_before.grid(row=2, column=0, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.lbl_after = tk.Label(
            root_frame, text="After", font=(DEFAULT_FONT, FONT_SIZE_H1))
        self.lbl_after.grid(row=1, column=1, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.canvas_after = tk.Canvas(
            root_frame, width=DEFAULT_CANVAS_WIDTH, height=DEFAULT_CANVAS_HEIGHT)
        self.canvas_after.create_rectangle(
            0, 0, DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT, fill='white')
        self.canvas_after.grid(row=2, column=1, padx=(
            DEFAULT_PAD_X, DEFAULT_PAD_X), pady=(DEFAULT_PAD_Y, DEFAULT_PAD_Y))

        self.pack()

    def _on_btn_load_image_pressed(self):
        self.filename = filedialog.askopenfilename(
            title="Open an image file", filetypes=IMAGE_FILE_TYPES)
        if self.filename != "":
            image_utama = cv2.cvtColor(cv2.imread(
                self.filename), cv2.COLOR_BGR2RGB)

            x = image_utama.shape[1]
            y = image_utama.shape[0]
            if y/x > 1:
                dim = (int(x / y * DEFAULT_CANVAS_WIDTH),
                       int(DEFAULT_CANVAS_HEIGHT))
            else:
                dim = (int(DEFAULT_CANVAS_WIDTH), int(
                    y / x * DEFAULT_CANVAS_HEIGHT))

            image_utama = cv2.resize(image_utama, dim)
            self.gambar_sebelum = ImageTk.PhotoImage(
                Image.fromarray(image_utama))
            self.canvas_before.create_image(
                0, 0, anchor=tk.NW, image=self.gambar_sebelum)

    def _on_btn_process_pressed(self):
        if self.filename != "":
            face_extractor = FaceExtractor()
            face_extractor.load_image(self.filename)
            image = face_extractor.get_face()

            x = image.shape[1]
            y = image.shape[0]
            if y / x > 1:
                dim = (int(x / y * DEFAULT_CANVAS_WIDTH),
                       int(DEFAULT_CANVAS_HEIGHT))
            else:
                dim = (int(DEFAULT_CANVAS_WIDTH), int(
                    y / x * DEFAULT_CANVAS_HEIGHT))

            hasil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hasil = cv2.resize(hasil, dim)
            self.gambar_sesudah = ImageTk.PhotoImage(Image.fromarray(hasil))
            self.canvas_after.create_image(
                0, 0, anchor=tk.NW, image=self.gambar_sesudah)
        else:
            messagebox.showerror("Tidak ada gambar!",
                                 "Tolong Masukan Gambar terlebih dahulu!")
