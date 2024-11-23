import cloudconvert
import os
import requests

class CloudConvertAudioVisualizer:
    def __init__(self, api_key):
        # Initialize the CloudConvert client with your API key
        self.api_key = api_key
        self.client = cloudconvert.Client(api_key=self.api_key)

    def convert_audio_to_video(self, input_audio_path, output_video_path):
        """
        Convert audio file to a video using CloudConvert API.
        The video is created with basic properties (MP4, 1280x720, 30 FPS).
        """
        try:
            # Upload the audio file to CloudConvert
            print(f"Uploading {input_audio_path}...")
            upload_task = self.client.upload(file=input_audio_path)
            input_file_id = upload_task['id']
            print("Upload successful.")
            
            # Create the conversion job with CloudConvert API
            print(f"Converting {input_audio_path} to video...")
            job = self.client.jobs.create(payload={
                'tasks': {
                    'import-my-audio': {
                        'operation': 'import/upload',
                        'file': input_file_id
                    },
                    'convert-to-video': {
                        'operation': 'convert',
                        'input': 'import-my-audio',
                        'output_format': 'mp4',
                        'audio_codec': 'aac',
                        'video_codec': 'libx264',
                        'fps': 30,
                        'video_width': 1280,
                        'video_height': 720
                    },
                    'export-my-video': {
                        'operation': 'export/url',
                        'input': 'convert-to-video',
                        'output_format': 'mp4'
                    }
                }
            })
            print("Conversion started.")

            # Wait for the job to complete (job status is checked)
            job_id = job['id']
            job = self.client.jobs.wait(job_id)
            print(f"Conversion completed. Downloading video...")

            # Get the export task result, which contains the video download URL
            export_task = job['tasks'][-1]
            video_url = export_task['result']['files'][0]['url']
            
            # Download the converted video to the local system
            self.download_video(video_url, output_video_path)

            print(f"Video saved to {output_video_path}")

        except Exception as e:
            print(f"An error occurred: {e}")

    def download_video(self, video_url, output_video_path):
        """
        Download the converted video from CloudConvert's export URL.
        """
        try:
            # Send an HTTP request to download the video file
            response = requests.get(video_url, stream=True)
            response.raise_for_status()  # Check if the request was successful

            # Write the content of the response (video) to a local file
            with open(output_video_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Video saved to {output_video_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading video: {e}")

if __name__ == "__main__":
    # Step 1: Set your CloudConvert API key
    api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiIxIiwianRpIjoiZDYyMzk0YTUzOGE0ZTM4NjAyMGNmNmQzNzM3NDNjZTk4NGRkYzA2NzYzODMyNzA1ZTgyNTI2ZWY1YjgxNDEyZDgxNjg3OTcwYWIxMTJjMjAiLCJpYXQiOjE3MzEwOTM2NjAuNTU2MjIsIm5iZiI6MTczMTA5MzY2MC41NTYyMjEsImV4cCI6NDg4Njc2NzI2MC41NTA0MDMsInN1YiI6IjI4MTk5NzMiLCJzY29wZXMiOlsidGFzay5yZWFkIiwidGFzay53cml0ZSIsInByZXNldC5yZWFkIl19.g5h9b-gohROOX7TKGnZWw6ovhAjs_2R2JCF4q8cjihfuLAVuOCaktamSHgVhpPCk_2H8WfKIbvQpthNvlpITYLQYJsYHfiueUVCUN1l_qDk3tp-yl-11ph-yrBYJCzYk7m090JnVDHM4COU4WUVWjYtrQraEgfJqT9Zn8CyjHaAMD8rKZQ6CNcpWnu1rK3xezGxtxG-pQeny8BQKQEXNmSGLT_YkC3Fcze7l6kJyGvVCEL7EBhrj4lX6gw3pX6lCwwvschMZ-viThWctGPXyb2BEXwYzoUFVqEE2iKw7PmrToW5hZHyJKdrrFjHVVKwLTh7vfX1mMA9jjULjxB14DGXQfTU1RP6boMPm3_bCezOQ9kMNw6Wod9GYhwpi2wqVj5rfrqACxr5J_PuhZWSe9YM7tvyHypRbEl_shpOkbZ79rZUdsR7bGJwdh9hqHMKYi1croehxUybRbgwY-XpDYquksPpBkwr0T2YChFpGHh4ptz2E8VbQGecBNUF8mirNath5aAmoZJYfzuRsp74axwzJIRocOBGquzuz_niiQgRavuUIXV5rptUzL6WpMD90XUhgSZtrhK8PowZwArybo1yuV5YWi2bATnb0ZewXm8DGDxoPnm8Aq9h22v82BwUc7JZoDAOvhC7kzOW90_9AcIM47-kTIcJ3Ur17416ZXRo"  # Replace with your CloudConvert API key

    # Step 2: Define the paths for the input and output files
    input_audio_path = "E:/USB_vue_US/converter/input/123.mp3"  # Replace with the path to your audio file
    output_video_path = "E:/USB_vue_US/converter/output/video.mp4"  # Replace with the desired output video path

    # Step 3: Create an instance of CloudConvertAudioVisualizer and start the conversion process
    cloudconvert_visualizer = CloudConvertAudioVisualizer(api_key)
    cloudconvert_visualizer.convert_audio_to_video(input_audio_path, output_video_path)
