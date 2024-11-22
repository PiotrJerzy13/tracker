import os
import sys
import cv2
import subprocess

# Function to list and select video files
def select_video_file():
    # List all MP4 files in the current directory
    files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    
    if not files:
        print("No MP4 files found in the current directory.")
        sys.exit()

    # Display options to the user
    print("Available MP4 files:")
    for i, file in enumerate(files, 1):
        print(f"{i}: {file}")

    # Prompt user to select a file
    while True:
        try:
            choice = int(input(f"Select a file to use (1-{len(files)}): "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Create results folder if it doesn't exist
results_folder = "results"
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Tracker type
tracker_type = "CSRT"  # Choose your tracker type

# Function to create a tracker
def create_tracker(tracker_type):
    if tracker_type == "BOOSTING":
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == "MIL":
        return cv2.legacy.TrackerMIL_create()
    elif tracker_type == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    elif tracker_type == "TLD":
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    elif tracker_type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    else:
        raise ValueError(f"Unsupported tracker type: {tracker_type}")

# Select the video file
video_input_file_name = select_video_file()
print(f"Using video file: {video_input_file_name}")

# Read video
video = cv2.VideoCapture(video_input_file_name)

if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read the first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# MultiTracker
multi_tracker = cv2.legacy.MultiTracker()

# Initial selection of ROIs
bboxes = cv2.selectROIs("MultiTracker", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

if len(bboxes) > 0:
    print(f"{len(bboxes)} initial tracker(s) selected.")
else:
    print("No initial trackers selected. Exiting.")
    sys.exit()

# Initialize trackers for selected ROIs
for bbox in bboxes:
    tracker = create_tracker(tracker_type)
    multi_tracker.add(tracker, frame, tuple(bbox))

# Output video properties
tracked_cars_file = os.path.join(results_folder, "tracked_cars.mp4")
video_out = cv2.VideoWriter(tracked_cars_file, cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame.shape[1], frame.shape[0]))

# Playback and tracking loop
while True:
    ok, frame = video.read()
    if not ok:
        break

    # Update all trackers
    ok, boxes = multi_tracker.update(frame)

    # Draw bounding boxes
    for box in boxes:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

    # Display the frame
    cv2.imshow("MultiTracker", frame)

    # Handle user input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        print("Exiting program.")
        break
    elif key == ord("p"):  # Pause and add new tracker
        print("Paused. Select a new object to track and press ENTER or ESC.")
        new_bbox = cv2.selectROI("MultiTracker", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        if new_bbox[2] > 0 and new_bbox[3] > 0:  # Ensure a valid selection
            new_tracker = create_tracker(tracker_type)
            multi_tracker.add(new_tracker, frame, tuple(new_bbox))
            print("New tracker added successfully.")
        else:
            print("No tracker was selected. Resuming playback.")

    # Write to output video
    video_out.write(frame)

# Release resources
video.release()
video_out.release()
cv2.destroyAllWindows()

# Convert video to h264 and save in results folder
tracked_cars_h264_file = os.path.join(results_folder, "tracked_cars_h264.mp4")
try:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", tracked_cars_file,
            "-c:v", "libx264",
            tracked_cars_h264_file,
            "-hide_banner",
            "-loglevel", "error"
        ],
        check=True
    )
    print(f"Video conversion successful: {tracked_cars_h264_file}")
except FileNotFoundError:
    print("FFmpeg not found. Please install FFmpeg and try again.")
except subprocess.CalledProcessError as e:
    print("Error during video conversion:", e)

