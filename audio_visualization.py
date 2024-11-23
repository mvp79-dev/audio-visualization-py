import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import *
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

class AudioVisualizer:
    def __init__(self, visualization_type='spectrogram'):
        """
        Initialize the audio visualizer with specified visualization type.
        
        Args:
            visualization_type (str): Type of visualization ('spectrogram', 'waveform', or 'both')
        """
        self.visualization_type = visualization_type
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('audio_processing.log'),
                logging.StreamHandler()
            ]
        )
    
    def create_waveform(self, y, sr, ax):
        """Create waveform visualization"""
        times = np.linspace(0, len(y)/sr, len(y))
        ax.plot(times, y, color='b', alpha=0.7)
        ax.set_title('Waveform')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        
    def create_spectrogram(self, y, sr, ax):
        """Create spectrogram visualization"""
        D = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram')
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        
    def create_visualization(self, audio_path, output_path):
        """
        Create audio visualization video
        
        Args:
            audio_path (str): Path to input audio file
            output_path (str): Path to output video file
        """
        try:
            # Load audio with warning suppression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                y, sr = librosa.load(audio_path)
            
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
            temp_dir = Path('temp_visualizations')
            temp_dir.mkdir(exist_ok=True)
            temp_image = temp_dir / f'temp_{os.path.basename(audio_path)}.png'
            plt.savefig(temp_image, bbox_inches='tight', dpi=300)
            plt.close()
            
            # Create video with progress bar support
            duration = librosa.get_duration(y=y, sr=sr)
            image_clip = ImageClip(str(temp_image)).set_duration(duration)
            audio_clip = AudioFileClip(audio_path)
            video = image_clip.set_audio(audio_clip)
            
            # Write video with proper codec settings
            video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                logger=None  # Disable moviepy's printing
            )
            
            # Cleanup
            temp_image.unlink()
            logging.info(f"Successfully processed: {audio_path}")
            
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise
            
    def process_folder(self, input_folder, output_folder, max_workers=4):
        """
        Process all MP3 files in input folder
        
        Args:
            input_folder (str): Input folder path
            output_folder (str): Output folder path
            max_workers (int): Maximum number of concurrent workers
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        # Get all MP3 files
        mp3_files = list(input_path.rglob("*.mp3"))
        
        if not mp3_files:
            logging.warning(f"No MP3 files found in {input_folder}")
            return
            
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        def process_file(mp3_file):
            relative_path = mp3_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.mp4')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            self.create_visualization(str(mp3_file), str(output_file))
            
        # Process files with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(
                executor.map(process_file, mp3_files),
                total=len(mp3_files),
                desc="Processing audio files"
            ))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MP3 files to visualized MP4 videos')
    parser.add_argument('input_folder', help='Input folder containing MP3 files')
    parser.add_argument('output_folder', help='Output folder for MP4 files')
    parser.add_argument('--type', choices=['spectrogram', 'waveform', 'both'], 
                        default='spectrogram', help='Type of visualization')
    parser.add_argument('--workers', type=int, default=4, 
                        help='Number of concurrent workers')
    
    args = parser.parse_args()
    
    visualizer = AudioVisualizer(args.type)
    visualizer.process_folder(args.input_folder, args.output_folder, args.workers)