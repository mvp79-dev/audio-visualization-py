import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
CHUNK = 1024  # Number of audio samples per frame
RATE = 44100  # Sampling rate in Hz

# Create a PyAudio object
p = pyaudio.PyAudio()

# Open an audio stream
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the figure and axis for plotting
fig, ax = plt.subplots()
x = np.arange(0, 2 * CHUNK, 2)  # X-axis for the samples
line, = ax.plot(x, np.random.rand(CHUNK), color='blue')
ax.set_ylim(-30000, 30000)  # Set y-axis limits for waveform
ax.set_xlim(0, CHUNK)  # Set x-axis limits
plt.title('Real-Time Audio Waveform')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

def update(frame):
    """Update the waveform plot."""
    data = stream.read(CHUNK)
    data_int16 = np.frombuffer(data, dtype=np.int16)  # Convert byte data to numpy array
    line.set_ydata(data_int16)  # Update line data with new audio samples
    return line,

# Create an animation that calls the update function every frame
ani = animation.FuncAnimation(fig, update, blit=True)

# Show the plot
plt.show()

# Clean up on exit
stream.stop_stream()
stream.close()
p.terminate()