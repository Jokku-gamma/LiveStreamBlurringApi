# Real-time Face and License Plate Blurring

This project is a web application built with Gradio that can detect and blur faces and license plates in a live webcam feed. It's designed for privacy protection in video streams.

## Features

* **Real-time Blurring:** Processes video from your webcam live.
* **Face Detection & Blurring:** Utilizes a deep learning model to find and obscure faces.
* **License Plate Detection & Blurring:** Uses Haar Cascades to find and obscure vehicle license plates.
* **Adjustable Blur Intensity:** Control how much blur is applied with a simple slider.
* **Web-based Interface:** Easy to use directly in your browser.

## How it Works

The application captures video frames from your webcam. For each frame, it uses:
* OpenCV's DNN module with a Caffe model (`deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`) to detect human faces.
* OpenCV's Haar Cascade classifier (`haarcascade_russian_plate_number.xml`) to detect license plates.
Once detected, the regions of interest (faces or plates) are blurred using Gaussian blur with an intensity you can control.

## Setup and Installation

1.  **Clone the Repository (or save the code):**
    If you have a Git repository, clone it:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```
    Otherwise, save the Python script (e.g., `app.py`) and the model files in the same directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install gradio opencv-python numpy
    ```

4.  **Download Model Files:**
    You need to download the pre-trained model files and place them in the **same directory** as your Python script:
    * **For Face Detection:**
        * `deploy.prototxt`
        * `res10_300x300_ssd_iter_140000.caffemodel`
        You can often find these by searching for "Caffe SSD Face Detection Model" or on places like the OpenCV GitHub repository's `opencv_extra` or `opencv_zoo` (though direct links vary over time).
    * **For License Plate Detection:**
        * `haarcascade_russian_plate_number.xml`
        This is typically found in the OpenCV data repository. Search for "haarcascades GitHub".

## Usage (Running the App)

1.  **Navigate to your project directory** in your terminal or command prompt (where your `app.py` and model files are).
2.  **Ensure your virtual environment is active** (if you created one).
3.  **Run the application:**
    ```bash
    python app.py
    ```
    (Replace `app.py` with your script's filename if it's different).

4.  **Open in Browser:** Gradio will provide a local URL (e.g., `http://127.0.0.1:7860`). Open this URL in your web browser.

### Using the Interface:

* **Start Blurring:** Click the **'Record'**  button directly below the webcam feed to begin streaming and blurring.
* **Adjust Settings:** Use the checkboxes to enable/disable face or plate blurring, and drag the slider to change the blur intensity in real-time.
