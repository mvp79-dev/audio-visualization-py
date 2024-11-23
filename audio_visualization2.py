import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import librosa
import numpy as np  # Importing NumPy
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from moviepy.editor import ImageClip, AudioFileClip
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AudioVisualizer:
    def __init__(self, visualization_type='both'):
        self.visualization_type = visualization_type

    def create_waveform(self, y, sr, ax):
        """Create a waveform plot."""
        ax.plot(y, color='blue')
        ax.set_title('Waveform')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude')

    def create_spectrogram(self, y, sr, ax):
        """Create a spectrogram plot."""
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = ax.imshow(D, aspect='auto', origin='lower', 
                        extent=[0, len(y) / sr, 0, sr / 2], 
                        cmap='coolwarm')  # Corrected here
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')

    def create_visualization(self, audio_path, output_path):
        """Create audio visualization video."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            # Create figure based on visualization type
            if self.visualization_type == 'both':
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
                self.create_waveform(y, sr, ax1)
                self.create_spectrogram(y, sr, ax2)
            else:
                fig, ax = plt.subplots(figsize=(12, 8))
                if self.visualization_type == 'waveform':
                    self.create_waveform(y, sr, ax)
                else:  # spectrogram
                    self.create_spectrogram(y, sr, ax)

            # Save temporary image
            temp_image = 'temp_image.png'
            plt.savefig(temp_image, bbox_inches='tight', dpi=300)
            plt.close()

            # Create video
            duration = librosa.get_duration(y=y, sr=sr)
            image_clip = ImageClip(temp_image).set_duration(duration)
            audio_clip = AudioFileClip(audio_path)
            video = image_clip.set_audio(audio_clip)

            # Write video
            video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise

    def process_folder(self, input_folder, output_folder, max_workers=4):
        """Process all MP3 files in input folder."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        # Get all MP3 files
        mp3_files = list(input_path.rglob("*.mp3"))

        if not mp3_files:
            logging.warning(f"No MP3 files found in {input_folder}")
            return

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Create a list to hold the file paths for processing
        files_to_process = []

        def collect_files(mp3_file):
            relative_path = mp3_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.mp4')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            files_to_process.append((str(mp3_file), str(output_file)))

        # Collect files using threading
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(collect_files, mp3_files),
                total=len(mp3_files),
                desc="Collecting audio files"
            ))

        # Now process the files in the main thread
        for audio_file, output_file in tqdm(files_to_process, desc="Processing audio files"):
            try:
                self.create_visualization(audio_file, output_file)
            except Exception as e:
                logging.error(f"Error processing {audio_file}: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert MP3 files to visualized MP4 videos')
    parser.add_argument('input_folder', help='Input folder containing MP3 files')
    parser.add_argument('output_folder', help='Output folder for MP4 files')
    parser.add_argument('--type', choices=['spectrogram', 'waveform', 'both'], default='both',
                        help='Type of visualization to create (default: both)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of concurrent workers (default: 4)')

    args = parser.parse_args()

    visualizer = AudioVisualizer(visualization_type=args.type)
    visualizer.process_folder(args.input_folder, args.output_folder, max_workers=args.workers)
