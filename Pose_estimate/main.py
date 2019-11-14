from pygame import mixer
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from Pose_estimates_server import estimatePose
import tensorflow as tf
import pandas as pd

root = tk.Tk()
root.geometry('512x512')
frame = tk.Frame(root, relief='ridge', borderwidth=2)
frame.pack(fill='both', expand=1)
root.title('YOLOPose')
frame.config(background='black')
label = tk.Label(frame, text="YOLOPose", fg="yellow", bg="black",font=('Times 35 bold'))
label.pack(side='top')
lunge_image = tk.PhotoImage(file="./data/demo/Lunge.png")

def exitt():
    exit()

def checkScore():
    score = pd.read_csv('./data/score/result.csv', header=None)
    window = tk.Toplevel(root, width=400, height=400)
    window.title("daily record")
    
    excelview=ttk.Treeview(window, columns=["0","1","2","3","4","5"])
    excelview.pack()
    excelview.column("#1", width=100)
    excelview.heading("#1",text="Date")
    excelview.column("#2", width=100)
    excelview.heading("#2",text="Sport")
    excelview.column("#3", width=100)
    excelview.heading("#3",text="Knee")
    excelview.column("#4", width=100)
    excelview.heading("#4",text="Wrist")
    excelview.column("#5", width=100)
    excelview.heading("#5",text="Speed")
    excelview.column("#6", width=0)
    excelview['show'] = 'headings'
    
    
    dates = score[0].drop_duplicates()
    for d in dates:
        excelview.insert('','end',values=[d,"Lunge",'','',''], iid=d)
        for sc in score[score[0]==d].values:
             excelview.insert('','end',values=['',sc[1],sc[2],sc[3],sc[4]])
        
#         excelview.insert('','end',values=[lunge_image,set_t[1],]


#     label = tk.Label(window, width=300, height=300, relief='ridge', borderwidth=2)
#     label.pack(side="top", fill="both", padx=10, pady=10)


but1=tk.Button(frame,padx=5,pady=5,width=45,bg='white',fg='black',text='Open Cam',command=estimatePose,font=('helvetica 15 bold'))
but1.place(x=20,y=104)

but2=tk.Button(frame,padx=5,pady=5,width=45,bg='white',fg='black',text='Check Score',command=checkScore,font=('helvetica 15 bold'))
but2.place(x=20,y=176)

root.mainloop()