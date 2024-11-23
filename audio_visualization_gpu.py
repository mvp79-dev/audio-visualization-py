import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, VideoClip
from pathlib import Path
from tqdm import tqdm
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QSpinBox, QLabel, QLineEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
import sys
import cupy as cp  # Import CuPy for GPU acceleration


class AnimatedAudioVisualizer:
    def __init__(self, visualization_type='waveform', fps=30, video_size=(1280, 720)):
        self.visualization_type = visualization_type
        self.fps = fps
        self.video_size = video_size

    def create_gradient(self, width, height):
        """Create a vertical gradient from purple to blue on the GPU."""
        gradient = cp.linspace(0, 1, height)
        gradient = cp.tile(gradient, (width, 1)).T
        start_color = cp.array([0.2, 0.2, 0.2])
        end_color = cp.array([0.1328, 0.1328, 0.1328])
        gradient_color = cp.zeros((height, width, 3))

        for i in range(3):
            gradient_color[:, :, i] = start_color[i] * (1 - gradient) + end_color[i] * gradient

        return gradient_color

    def make_frame(self, t, y, sr, total_duration):
        """Generate the frame for time `t` with GPU acceleration."""
        fig, ax = plt.subplots(figsize=(16, 9))

        frame_width, frame_height = self.video_size
        ax.set_position([0, 0, 1, 1])
        gradient = self.create_gradient(frame_width, frame_height)
        ax.imshow(gradient.get(), aspect='auto', extent=[0, len(y), -1, 1], origin='lower')  # Use .get() to transfer data from GPU to CPU

        samples_per_frame = int(sr / self.fps)
        current_sample = int(t * sr)
        start = max(0, current_sample - samples_per_frame // 2)
        end = min(len(y), current_sample + samples_per_frame // 2)

        displacement = cp.sin(cp.linspace(0, 2 * cp.pi, len(y))) * 0.1  # GPU-accelerated displacement
        y_displaced = y + displacement
        window_size = 100
        y_smooth = cp.convolve(y_displaced, cp.ones(window_size)/window_size, mode='valid')

        start_smooth = start - window_size + 1
        end_smooth = end - window_size + 1
        if start_smooth < 0: start_smooth = 0
        if end_smooth > len(y_smooth): end_smooth = len(y_smooth)
        x_smooth = cp.arange(start_smooth, end_smooth)
        y_smooth_segment = y_smooth[start_smooth:end_smooth]

        ax.plot(x_smooth.get(), y_smooth_segment.get() - 0.01, color='white', alpha=0.2, lw=2)
        ax.plot(x_smooth.get(), y_smooth_segment.get(), color='white', alpha=0.5, lw=3)
        ax.plot(x_smooth.get(), y_smooth_segment.get() + 0.01, color='white', alpha=0.2, lw=2)

        ax.set_ylim(-1, 1)
        ax.set_xlim(start, end)
        ax.axis('off')

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return frame

    def create_visualization(self, audio_path, output_path):
        """Create animated audio visualization video."""
        try:
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = librosa.get_duration(y=y, sr=sr)

            clip = VideoClip(lambda t: self.make_frame(t, cp.array(y), sr, total_duration), duration=total_duration)
            clip = clip.set_fps(self.fps)

            audio_clip = AudioFileClip(audio_path)
            video = clip.set_audio(audio_clip)

            video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise

    def process_folder(self, input_folder, output_folder, progress_callback):
        """Process all MP3 files in the input folder with GPU acceleration."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        mp3_files = list(input_path.rglob("*.mp3"))

        if not mp3_files:
            logging.warning(f"No MP3 files found in {input_folder}")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        for index, mp3_file in tqdm(enumerate(mp3_files), desc="Processing audio files", total=len(mp3_files)):
            relative_path = mp3_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.mp4')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.create_visualization(str(mp3_file), str(output_file))

            # Update the progress bar
            progress_callback.emit(int((index + 1) / len(mp3_files) * 100))


class WorkerThread(QThread):
    progress_updated = pyqtSignal(int)

    def __init__(self, input_folder, output_folder, visualization_type, fps, video_size):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.visualization_type = visualization_type
        self.fps = fps
        self.video_size = video_size

    def run(self):
        visualizer = AnimatedAudioVisualizer(
            visualization_type=self.visualization_type,
            fps=self.fps,
            video_size=self.video_size
        )
        visualizer.process_folder(self.input_folder, self.output_folder, self.progress_updated)


class AudioVisualizerUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Audio Visualizer")
        self.setGeometry(300, 200, 400, 250)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Input folder selection
        self.input_folder_label = QLabel("Input Folder:")
        self.input_folder_button = QPushButton("Select Input Folder")
        self.input_folder_button.clicked.connect(self.select_input_folder)

        self.input_folder_line_edit = QLineEdit()
        self.input_folder_line_edit.setReadOnly(True)

        # Output folder selection
        self.output_folder_label = QLabel("Output Folder:")
        self.output_folder_button = QPushButton("Select Output Folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)

        self.output_folder_line_edit = QLineEdit()
        self.output_folder_line_edit.setReadOnly(True)

        # Visualization Type ComboBox
        self.visualization_type_label = QLabel("Visualization Type:")
        self.visualization_combo = QComboBox()
        self.visualization_combo.addItems(["waveform", "spectrogram"])

        # FPS setting
        self.fps_label = QLabel("Frames Per Second (FPS):")
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(30)

        # Video resolution (width, height)
        self.resolution_label = QLabel("Video Resolution (Width x Height):")
        self.resolution_width_spinbox = QSpinBox()
        self.resolution_width_spinbox.setRange(320, 1920)
        self.resolution_width_spinbox.setValue(1280)

        self.resolution_height_spinbox = QSpinBox()
        self.resolution_height_spinbox.setRange(240, 1080)
        self.resolution_height_spinbox.setValue(720)

        # Progress Bar
        self.progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        # Process button
        self.process_button = QPushButton("Start Processing")
        self.process_button.clicked.connect(self.start_processing)

        # Add all widgets to the layout
        layout.addWidget(self.input_folder_label)
        layout.addWidget(self.input_folder_button)
        layout.addWidget(self.input_folder_line_edit)

        layout.addWidget(self.output_folder_label)
        layout.addWidget(self.output_folder_button)
        layout.addWidget(self.output_folder_line_edit)

        layout.addWidget(self.visualization_type_label)
        layout.addWidget(self.visualization_combo)

        layout.addWidget(self.fps_label)
        layout.addWidget(self.fps_spinbox)

        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_width_spinbox)
        layout.addWidget(self.resolution_height_spinbox)

        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        layout.addWidget(self.process_button)

        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_line_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_line_edit.setText(folder)

    def start_processing(self):
        input_folder = self.input_folder_line_edit.text()
        output_folder = self.output_folder_line_edit.text()
        visualization_type = self.visualization_combo.currentText()
        fps = self.fps_spinbox.value()
        video_size = (self.resolution_width_spinbox.value(), self.resolution_height_spinbox.value())

        self.worker = WorkerThread(input_folder, output_folder, visualization_type, fps, video_size)
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.start()

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioVisualizerUI()
    window.show()
    sys.exit(app.exec_())
