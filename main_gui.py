import tkinter.font as tk_font
import tkinter as tk
import tkinter.ttk as ttk

from gui.config import *
from gui.frames.ferdy_frame import FerdyFrame
from gui.frames.jason_frame import JasonFrame
from gui.frames.jayaku_frame import JayakuFrame
from gui.frames.kevin_frame import KevinFrame
from gui.frames.martien_frame import MartienFrame


class FaceMouthGUI:
    def __init__(self, root):
        self.root = root
        self._init_components()
        self.update_default_font()

    def _init_components(self):
        tab_parent = ttk.Notebook(self.root)

        frames = []
        frames += [("Ferdy", FerdyFrame(master=self.root))]
        frames += [("Jason", JasonFrame(master=self.root))]
        frames += [("Jayaku", JayakuFrame(master=self.root))]
        frames += [("Kevin", KevinFrame(master=self.root))]
        frames += [("Martien", MartienFrame(master=self.root))]

        padding = (DEFAULT_PAD_X, DEFAULT_PAD_Y, DEFAULT_PAD_X, DEFAULT_PAD_Y)
        for frame_title, frame in frames:
            tab_parent.add(frame, text=frame_title, padding=padding)

        tab_parent.pack()


    def update_default_font(self):
        default_font = tk_font.nametofont("TkDefaultFont")
        default_font.config(family=DEFAULT_FONT)
        default_font.config(size=FONT_SIZE_NORMAL)


if __name__ == '__main__':
    root = tk.Tk()
    root.title("FaceMouth - Face and Mouth Analysis System")
    FaceMouthGUI(root)
    root.mainloop()
