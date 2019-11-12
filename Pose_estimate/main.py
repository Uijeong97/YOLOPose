import numpy
from pygame import mixer
import time
import cv2
from tkinter import*
import tkinter.messagebox
from Pose_estimates_webcam import estimatePose

root = Tk()
root.geometry('512x512')
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('YOLOPose')
frame.config(background='black')
label = Label(frame, text="YOLOPose", fg="yellow", bg="black",font=('Times 35 bold'))
label.pack(side=TOP)

def exitt():
   exit()

but1=Button(frame,padx=5,pady=5,width=45,bg='white',fg='black',text='Open Cam',command=estimatePose,font=('helvetica 15 bold'))
but1.place(x=20,y=104)

but2=Button(frame,padx=5,pady=5,width=45,bg='white',fg='black',text='Check Score',command=exitt,font=('helvetica 15 bold'))
but2.place(x=20,y=176)

root.mainloop()