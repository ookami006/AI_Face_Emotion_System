from flask import Flask, render_template, redirect
import subprocess
import os
import sys

app = Flask(
    __name__,
    template_folder="../../frontend",
    static_folder="../../frontend/static"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def run_script(script_name):
    script_path = os.path.join(BASE_DIR, "backend", "app", script_name)
    subprocess.Popen([sys.executable, script_path])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run-detection")
def run_detection():
    run_script("face_detection.py")
    return redirect("/")

@app.route("/run-recognition")
def run_recognition():
    run_script("face_recognition.py")
    return redirect("/")

@app.route("/run-emotion")
def run_emotion():
    run_script("emotion_detection.py")
    return redirect("/")

@app.route("/run-combined")
def run_combined():
    run_script("combined_system.py")
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)