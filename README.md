🎭 Multi-Modal Emotion Detection System
AI-Based Emotion Recognition Using Face, Voice & Text

🚀 Features
.Real-time emotion detection from face using webcam
.Emotion detection from voice input
.Emotion detection from text input
.User-friendly GUI application
.Multi-modal approach for improved accuracy

🛠️ Technologies Used
.Python
.TensorFlow / Keras
.OpenCV
.Librosa
.SpeechRecognition
.Scikit-learn
.CustomTkinter

📂 Project Structure
Emotion_project/
│
├── UI_app.py              # Main GUI application
├── requirements.txt
│
├── facial/                # Facial emotion detection
├── voice/                 # Voice emotion detection
└── text/                  # Text emotion detection

⚙️ Installation & Running the Project
1. Clone the repository
    git clone https://github.com/Deepika-19i/emotion
detection-project.git
   
2.Navigate to the project folder
    cd Emotion_project

3.Create a virtual environment (optional but 
recommended)
    python -m venv venv
Activate it

.Mac/Linux:
source venv/bin/activate

.Windows
venv\Scripts\activate

4. Install dependencies
   pip install -r requirements.txt

5. Run the application
   python UI_app.py

💻 Output
The application will open as a desktop GUI window, where you can:
.Detect emotion from face
.Detect emotion from voice
.Detect emotion from text

📌 Note
.This project currently runs as a desktop application.
.Model files (.h5 / .keras) are not uploaded due to size limitations.
.To run it in a browser, it can be extended using Flask / Django or Streamlit.

💡 Future Scope
.Improve accuracy using advanced deep learning models
.Combine all modalities into a single prediction system
.Deploy as a web application
.Add real-time analytics dashboard

👩‍💻 Author
.Deepika M
.🐙 GitHub: https://github.com/Deepika-19i
.💼 LinkedIn: https://www.linkedin.com/in/deepika-m-819742357
