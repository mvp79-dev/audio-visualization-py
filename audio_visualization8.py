import os
import logging
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import AudioFileClip, VideoClip
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AnimatedAudioVisualizer:
    def __init__(self, visualization_type='waveform', fps=30, video_size=(1280, 720)):
        self.visualization_type = visualization_type
        self.fps = fps
        self.video_size = video_size  # Default video size (width, height)

    # def add_blur_effect(frame):
    #     pil_image = Image.fromarray(frame)
    #     blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=5))  # Adjust the radius for more blur
    #     return np.array(blurred_image)

    def create_gradient(self, width, height):
        """Create a vertical gradient from purple to blue."""
        # Create a gradient array from 0 to 1 (vertical)
        gradient = np.linspace(0, 1, height)  # Gradient values from 0 to 1
        gradient = np.tile(gradient, (width, 1)).T  # Tile to cover the width of the image
        
        # Define color transitions (purple to blue)
        start_color = np.array([0.2, 0.2, 0.2])  # Purple (RGB)
        end_color = np.array([0.1328, 0.1328, 0.1328])  # Blue (RGB)

        # Create an empty array for the gradient colors, with the shape (height, width, 3)
        gradient_color = np.zeros((height, width, 3))

        # Interpolate for each RGB channel separately
        for i in range(3):  # For each color channel (R, G, B)
            gradient_color[:, :, i] = start_color[i] * (1 - gradient) + end_color[i] * gradient

        return gradient_color


    def make_frame(self, t, y, sr, total_duration):
        """Generate the frame for time `t`."""
        fig, ax = plt.subplots(figsize=(16, 9))

        frame_width, frame_height = fig.get_size_inches() * fig.dpi
        ax.set_position([0, 0, 1, 1])

         # Get the size of the video frame
        frame_width, frame_height = self.video_size  # Video size set earlier

        # gradient = self.create_gradient(fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1])
         # Create the gradient for the current video size
        gradient = self.create_gradient(frame_width, frame_height)
        ax.imshow(gradient, aspect='auto', extent=[0, len(y), -1, 1], origin='lower')
        # fig.patch.set_facecolor('none')  # Make sure fig background is transparent
        # fig.figimage(gradient, 0, 0, origin='lower', alpha=1)

        # Determine the segment of the audio to plot based on time `t`
        samples_per_frame = int(sr / self.fps)
        current_sample = int(t * sr)
        start = max(0, current_sample - samples_per_frame // 2)
        end = min(len(y), current_sample + samples_per_frame // 2)

        # Plot the waveform segment
        # alpha = np.abs(np.sin(t * np.pi / total_duration))
        # ax.plot(np.arange(start, end), y[start:end], color=(1, 1, 1, alpha))

        # ax.plot(np.arange(start, end), y[start:end], color='white', alpha=0.5)  # Base waveform
        # ax.plot(np.arange(start, end), y[start:end] + 0.01, color='white', alpha=0.3)

        displacement = np.sin(np.linspace(0, 2 * np.pi, len(y))) * 0.1  # Sinusoidal displacement
        y_displaced = y + displacement  # Apply displacement

        window_size = 50
        y_smooth = np.convolve(y_displaced, np.ones(window_size)/window_size, mode='valid')

        ax.plot(np.arange(start, end), y_smooth[start:end], color='white', alpha=0.5, lw=3)
        ax.plot(np.arange(start, end), y_smooth[start:end] + 0.01, color='white', alpha=0.1, lw=2)

        ax.set_ylim(-1, 1)  # Set consistent y-axis limits
        ax.set_xlim(start, end)
        ax.axis('off')  # Hide axes for a clean visualization

        # Save the frame to a numpy array
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        # frame = self.add_blur_effect(frame)
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
    parser.add_argument('--type', choices=['spectrogram', 'waveform'], default='waveform',
                        help='Type of visualization to create (default: waveform)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video (default: 30)')
    parser.add_argument('--video_size', type=int, nargs=2, default=(1280, 720),
                        help='Video resolution (width height)')

    args = parser.parse_args()

    visualizer = AnimatedAudioVisualizer(visualization_type=args.type, fps=args.fps, video_size=tuple(args.video_size))
    visualizer.process_folder(args.input_folder, args.output_folder)
