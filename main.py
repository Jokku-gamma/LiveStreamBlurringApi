import gradio as gr
import cv2
import numpy as np

face_net=cv2.dnn.readNetFromCaffe(
    'deploy.prototxt',
    'res10_300x300_ssd_iter_140000.caffemodel'
)
plate_cascade=cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
def apply_blur(frame,face_blur,plate_blur,blur_intensity):
    if frame is None:
        return np.zeros((480,640,3),dtype=np.uint8)
    k=blur_intensity//2 * 2 +1
    proc_frame=frame.copy()
    h,w=proc_frame.shape[:2]
    if face_blur:
        blob=cv2.dnn.blobFromImage(
            cv2.resize(proc_frame,(300,300)),
            1.0,
            (300,300),
            (104.0,177.0,123.0)
        )
        face_net.setInput(blob)
        dets=face_net.forward()
        for i in range(dets.shape[2]):
            conf=dets[0,0,i,2]
            if conf>0.5:
                box=dets[0,0,i,3:7] * np.array([w,h,w,h])
                (startx,starty,endx,endy)=box.astype("int")

                startx,starty=max(0,startx),max(0,starty)
                endx,endy=min(w,endx),min(h,endy)

                if endy>starty and endx>startx:
                    face_roi=proc_frame[starty:endy,startx:endx]
                    proc_frame[starty:endy,startx:endx]=cv2.GaussianBlur(
                        face_roi,(k,k),0
                    )
    if plate_blur:
        gray=cv2.cvtColor(proc_frame,cv2.COLOR_BGR2GRAY)
        plates=plate_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)
        )
        for (x,y,plate_w,plate_h) in plates:
            plate_roi=proc_frame[y:y+plate_h,x:x+plate_w]
            proc_frame[y:y+plate_h,x:x+plate_w]
    return proc_frame

iface=gr.Interface(
    fn=apply_blur,
    inputs=[
        gr.Image(type="numpy",label="Webcam Feed (or upload image)",streaming=True),
        gr.Checkbox(value=True,label="Enable Face Blurring"),
        gr.Checkbox(value=True,label="Enable License plate blurring"),
        gr.Slider(minimum=1,maximum=99,step=2,value=99,label="Blur Intensity",interactive=True),

    ],
    outputs=gr.Image(type="numpy",label="Blurred output"),
    live=True,
    title="Real time Face and License plate blurring",
    description="This application can blur faces and license plates detected on the webcam. Say Cheese....",
    css="footer {display:none !important;}",
)
import os
port=int(os.environ.get("PORT",7860))
if __name__=="__main__":
    iface.launch(server_name="0.0.0.0",server_port=port)