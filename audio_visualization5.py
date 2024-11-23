import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, VideoClip
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnimatedAudioVisualizer:
    def __init__(self, visualization_type='waveform', fps=30):
        self.visualization_type = visualization_type
        self.fps = fps

    def make_frame(self, t, y, sr, total_duration):
        """Generate the frame for time `t`."""
        fig, (ax_waveform, ax_spectrogram) = plt.subplots(1, 2, figsize=(18, 5))

         # Set background to black
        fig.patch.set_facecolor('black')  # Black background for the figure
        ax_waveform.set_facecolor('black')  # Black background for the waveform axis
        ax_spectrogram.set_facecolor('black')  # Black background for the spectrogram axis


        # Determine the segment of the audio to plot based on time `t`
        samples_per_frame = int(sr / self.fps)
        current_sample = int(t * sr)
        start = max(0, current_sample - samples_per_frame // 2)
        end = min(len(y), current_sample + samples_per_frame // 2)

        if self.visualization_type in ['waveform', 'both']:
            # Plot the waveform segment
            ax_waveform.plot(np.arange(start, end), y[start:end], color='blue')
            ax_waveform.set_ylim(-1, 1)  # Set consistent y-axis limits
            ax_waveform.set_xlim(start, end)
            ax_waveform.set_title("Waveform")
            ax_waveform.axis('off')  # Hide axes for a clean visualization

        if self.visualization_type in ['spectrogram', 'both']:
            # Compute the spectrogram using librosa
            hop_length = 512  # Controls the overlap in STFT
            n_fft = 2048  # FFT window size
            D = librosa.amplitude_to_db(librosa.stft(y[start:end], n_fft=n_fft, hop_length=hop_length), ref=np.max)
            
            # Plot the spectrogram as an image
            img = ax_spectrogram.imshow(D, aspect='auto', origin='lower', cmap='inferno', extent=[start, end, 0, sr // 2])
            ax_spectrogram.set_title("Spectrogram")
            ax_spectrogram.axis('off')  # Hide axes for a clean visualization

        # Save the frame to a numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return frame

    def create_visualization(self, audio_path, output_path):
        """Create animated audio visualization video."""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            total_duration = librosa.get_duration(y=y, sr=sr)

            # Create VideoClip with the make_frame function
            clip = VideoClip(lambda t: self.make_frame(t, y, sr, total_duration), duration=total_duration)
            clip = clip.set_fps(self.fps)

            # Attach audio
            audio_clip = AudioFileClip(audio_path)
            video = clip.set_audio(audio_clip)

            # Write to file
            video.write_videofile(str(output_path), codec='libx264', audio_codec='aac')

        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise

    def process_folder(self, input_folder, output_folder):
        """Process all MP3 files in the input folder."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        # Get all MP3 files
        mp3_files = list(input_path.rglob("*.mp3"))

        if not mp3_files:
            logging.warning(f"No MP3 files found in {input_folder}")
            return

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        for mp3_file in tqdm(mp3_files, desc="Processing audio files"):
            relative_path = mp3_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.mp4')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.create_visualization(str(mp3_file), str(output_file))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert MP3 files to animated visualized MP4 videos')
    parser.add_argument('input_folder', help='Input folder containing MP3 files')
    parser.add_argument('output_folder', help='Output folder for MP4 files')
    parser.add_argument('--type', choices=['spectrogram', 'waveform', 'both'], default='waveform',
                        help='Type of visualization to create (default: waveform)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video (default: 30)')

    args = parser.parse_args()

    visualizer = AnimatedAudioVisualizer(visualization_type=args.type, fps=args.fps)
    visualizer.process_folder(args.input_folder, args.output_folder)
