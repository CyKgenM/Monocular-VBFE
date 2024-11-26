import gi
import csv
import serial
import time

gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# Initialize GStreamer
Gst.init(None)

# Define serial port settings for reading force data
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 115200
serial_conn = serial.Serial(SERIAL_PORT, BAUD_RATE)

# GStreamer pipeline
pipeline_str = """
decklinkvideosrc mode=pal device-number=0 !
videorate ! video/x-raw,framerate=30/1 !
videoconvert ! jpegenc quality=100 ! appsink name=sink
"""
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")
appsink.set_property("emit-signals", True)

# Open CSV file for writing force and frame index data
with open('labels.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Frame counter
    frame_index = 0
    latest_force_data = None
    last_force_timestamp = None
    frame_log = []
    start_time = time.time()

    def on_new_sample(sink):
        global frame_index, latest_force_data, last_force_timestamp

        # Pull sample (frame) from appsink
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        # Capture frame timestamp
        frame_timestamp = time.time() - start_time

        # Create the path using frame_index
        img_path = f"/home/ros2/ECM_frames/left/frame_{frame_index}.jpg"

        # Save image to file
        buffer = sample.get_buffer()
        with open(img_path, "wb") as img_file:
            img_file.write(buffer.extract_dup(0, buffer.get_size()))

        # Read force data if available, else use the latest reading
        if serial_conn.in_waiting > 0:
            latest_force_data = serial_conn.readline().decode().strip()
            last_force_timestamp = frame_timestamp  # Update force timestamp

        # Append frame, timestamp, and force data to the log
        frame_log.append([f"frame_{frame_index}", frame_timestamp, latest_force_data])
        print(f"Saved frame_{frame_index} at {frame_timestamp} with force data: {latest_force_data}")

        # Increment the frame index
        frame_index += 1

        return Gst.FlowReturn.OK

    # Connect the on_new_sample function to appsink
    appsink.connect("new-sample", on_new_sample)

    # Start pipeline
    pipeline.set_state(Gst.State.PLAYING)

    # Run the loop
    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up resources
        pipeline.set_state(Gst.State.NULL)
        serial_conn.close()

    # After loop ends, write the collected frame data to CSV
    for frame, timestamp, force in frame_log:
        writer.writerow([frame, timestamp, force])
