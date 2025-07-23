import cv2
import mediapipe as mp
import numpy as np
import pygame
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import librosa
import soundfile as sf
import tempfile
import os
import time
import threading

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100)

# Global variables
music_loaded = False
music_playing = False
music_paused = False
original_audio = None
original_sr = None
current_speed = 1.0
current_pitch = 0
processing_lock = threading.Lock()
audio_processing_active = False
current_audio_file = None
next_audio_file = None
audio_files = []
current_position = 0
last_processed_speed = 1.0
last_processed_pitch = 0
processing_needed = False

def cleanup_temp_files():
    """Clean up temporary audio files"""
    global audio_files
    for file in audio_files:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            print(f"Error removing temp file {file}: {e}")
    audio_files = []

def choose_music():
    """Select and load a music file"""
    global music_loaded, music_paused, music_playing, original_audio, original_sr
    global current_speed, current_pitch, current_audio_file, processing_needed
    
    Tk().withdraw()
    filename = askopenfilename(filetypes=[("Audio Files", "*.mp3 *.wav")])
    if filename:
        try:
            # Clean up any existing temp files
            cleanup_temp_files()
            
            # Load the audio file with librosa for processing
            print(f"Loading audio file: {filename}")
            original_audio, original_sr = librosa.load(filename, sr=None)
            print(f"Audio loaded: {len(original_audio)} samples, {original_sr}Hz")
            
            # Create initial processed file with default settings
            temp_file = process_audio_file(original_audio, original_sr, 1.0, 0)
            current_audio_file = temp_file
            
            # Load the processed file
            pygame.mixer.music.load(current_audio_file)
            music_loaded = True
            music_paused = False
            music_playing = False
            
            # Reset controls
            current_speed = 1.0
            current_pitch = 0
            last_processed_speed = 1.0
            last_processed_pitch = 0
            processing_needed = False
            
            print(f"Music ready to play from {current_audio_file}")
        except Exception as e:
            print(f"Error loading music file: {e}")

def process_audio_file(audio_data, sr, speed_factor, pitch_shift):
    """Process audio and save to a temporary file"""
    global audio_files
    
    try:
        # Create a temporary file with a unique name
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"music_processed_{int(time.time() * 1000)}.wav")
        
        # Apply speed change (time stretching)
        if speed_factor != 1.0:
            audio_data = librosa.effects.time_stretch(audio_data, rate=speed_factor)
        
        # Apply pitch shift
        if pitch_shift != 0:
            audio_data = librosa.effects.pitch_shift(audio_data, sr=sr, n_steps=pitch_shift)
        
        # Save to temporary file
        sf.write(temp_file, audio_data, sr)
        audio_files.append(temp_file)
        
        print(f"Processed audio: speed={speed_factor}, pitch={pitch_shift}, saved to {temp_file}")
        return temp_file
    
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def audio_processor_thread():
    """Background thread for audio processing"""
    global processing_needed, current_speed, current_pitch, original_audio, original_sr
    global next_audio_file, current_audio_file, last_processed_speed, last_processed_pitch
    global audio_processing_active
    
    audio_processing_active = True
    
    while audio_processing_active:
        if processing_needed and original_audio is not None:
            with processing_lock:
                # Only process if there's a significant change
                if (abs(current_speed - last_processed_speed) > 0.05 or 
                    abs(current_pitch - last_processed_pitch) > 0.2):
                    
                    print(f"Processing audio with speed={current_speed}, pitch={current_pitch}")
                    
                    # Process the audio
                    temp_file = process_audio_file(
                        original_audio, original_sr, current_speed, current_pitch
                    )
                    
                    if temp_file:
                        next_audio_file = temp_file
                        last_processed_speed = current_speed
                        last_processed_pitch = current_pitch
                    
                    processing_needed = False
        
        # Sleep to avoid high CPU usage
        time.sleep(0.1)

def update_playback():
    """Update playback with newly processed audio"""
    global next_audio_file, current_audio_file, current_position, music_playing
    
    if next_audio_file and next_audio_file != current_audio_file:
        try:
            # Get current position
            if music_playing:
                current_position = pygame.mixer.music.get_pos() / 1000.0
            
            # Load new audio file
            pygame.mixer.music.load(next_audio_file)
            current_audio_file = next_audio_file
            next_audio_file = None
            
            # Resume playback from current position
            if music_playing:
                # Adjust position based on speed factor
                adjusted_position = current_position * current_speed
                pygame.mixer.music.play(start=adjusted_position)
                print(f"Resumed playback at position {adjusted_position}s")
        
        except Exception as e:
            print(f"Error updating playback: {e}")

def hand_type(hand_landmarks, results, index):
    """Determine if the hand is left or right"""
    if results.multi_handedness[index].classification[0].label == 'Left':
        return 'Right'  # Flipped because camera is mirrored
    else:
        return 'Left'  # Flipped because camera is mirrored

# Start audio processor thread
processor_thread = threading.Thread(target=audio_processor_thread)
processor_thread.daemon = True
processor_thread.start()

# Main application loop
with mp_hands.Hands(
    model_complexity=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7, 
    max_num_hands=2
) as hands:
    
    wave_history = []
    wave_update_counter = 0
    base_wave_height = 24
    text_pos = None
    
    # For tracking finger positions
    right_hand_fingers = None
    left_hand_fingers = None
    
    # For throttling parameter updates
    last_update_time = 0
    update_interval = 200  # ms
    
    # Create window
    cv2.namedWindow('Hand Wave Music Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Wave Music Control', 2560, 1600)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        result = hands.process(rgb_frame)
        
        # Initialize variables for this frame
        hands_detected = 0
        points = []
        h, w, _ = frame.shape
        
        # Reset hand finger positions
        right_hand_fingers = None
        left_hand_fingers = None
        
        # Check for hand landmarks
        if result.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                # Get hand type
                hand = hand_type(hand_landmarks, result, i)
                
                # Get index and thumb tip positions
                index_tip = hand_landmarks.landmark[8]  # Index finger tip
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # Store finger positions based on hand type
                if hand == 'Right':
                    right_hand_fingers = {
                        'index': (ix, iy),
                        'thumb': (tx, ty)
                    }
                    # Draw right hand controls in blue
                    cv2.line(frame, (ix, iy), (tx, ty), (255, 0, 0), 2)
                    cv2.circle(frame, (ix, iy), 8, (255, 0, 0), 2)
                    cv2.circle(frame, (tx, ty), 8, (255, 0, 0), 2)
                    
                    # Calculate distance for visualization
                    distance = np.linalg.norm([ix - tx, iy - ty])
                    cv2.putText(frame, f"Frequency Control: {int(distance)}px", 
                              (tx, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    left_hand_fingers = {
                        'index': (ix, iy),
                        'thumb': (tx, ty)
                    }
                    # Draw left hand controls in green
                    cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                    cv2.circle(frame, (ix, iy), 8, (0, 255, 0), 2)
                    cv2.circle(frame, (tx, ty), 8, (0, 255, 0), 2)
                    
                    # Calculate distance for visualization
                    distance = np.linalg.norm([ix - tx, iy - ty])
                    cv2.putText(frame, f"Speed Control: {int(distance)}px", 
                              (tx, ty - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                points.append(((ix, iy), (tx, ty)))
                hands_detected += 1
        
        # Update playback with newly processed audio if available
        update_playback()
        
        # Process right hand for frequency control
        if right_hand_fingers and music_loaded:
            # Calculate distance between index and thumb
            index_thumb_dist = np.linalg.norm(
                [right_hand_fingers['index'][0] - right_hand_fingers['thumb'][0], 
                 right_hand_fingers['index'][1] - right_hand_fingers['thumb'][1]]
            )
            
            # Map distance to pitch shift range (-6 to 6 semitones)
            # Using a narrower range for better control
            new_pitch = np.interp(index_thumb_dist, [30, 250], [-6, 6])
            new_pitch = round(new_pitch * 2) / 2  # Round to nearest 0.5
            
            # Display pitch value
            cv2.putText(frame, f"Pitch: {new_pitch:+.1f} semitones", 
                      (right_hand_fingers['index'][0], right_hand_fingers['index'][1] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Update pitch if changed significantly
            current_time = pygame.time.get_ticks()
            if abs(new_pitch - current_pitch) > 0.2 and (current_time - last_update_time > update_interval):
                current_pitch = new_pitch
                with processing_lock:
                    processing_needed = True
                last_update_time = current_time
        
        # Process left hand for speed control
        if left_hand_fingers and music_loaded:
            # Calculate distance between index and thumb
            index_thumb_dist = np.linalg.norm(
                [left_hand_fingers['index'][0] - left_hand_fingers['thumb'][0], 
                 left_hand_fingers['index'][1] - left_hand_fingers['thumb'][1]]
            )
            
            # Map distance to speed range (0.5 to 1.5)
            # Using a narrower range for better control
            new_speed = np.interp(index_thumb_dist, [30, 250], [0.5, 1.5])
            new_speed = round(new_speed * 10) / 10  # Round to nearest 0.1
            
            # Display speed value
            cv2.putText(frame, f"Speed: {new_speed:.1f}x", 
                      (left_hand_fingers['index'][0], left_hand_fingers['index'][1] - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update speed if changed significantly
            current_time = pygame.time.get_ticks()
            if abs(new_speed - current_speed) > 0.05 and (current_time - last_update_time > update_interval):
                current_speed = new_speed
                with processing_lock:
                    processing_needed = True
                last_update_time = current_time
        
        # Original volume control with two hands
        if hands_detected == 2 and len(points) >= 2:
            mid_x1 = (points[0][0][0] + points[0][1][0]) // 2
            mid_y1 = (points[0][0][1] + points[0][1][1]) // 2
            mid_x2 = (points[1][0][0] + points[1][1][0]) // 2
            mid_y2 = (points[1][0][1] + points[1][1][1]) // 2
            
            distance = np.linalg.norm([mid_x1 - mid_x2, mid_y1 - mid_y2])
            
            wave_update_counter += 1
            if wave_update_counter >= 1.6:
                new_wave = []
                for i in range(18):  # 18 waves
                    if 7 <= i <= 9:  # 3 central waves (indices 7,8,9)
                        wave_value = np.sin(pygame.mixer.music.get_pos() / 150 + i / 1.5) * (base_wave_height * 1.5)
                    else:
                        wave_value = np.sin(pygame.mixer.music.get_pos() / 80 + i / 3.0) * (base_wave_height * 0.8)
                    
                    new_wave.append(wave_value)
                
                wave_history = [new_wave] * 2
                base_wave_height = min(base_wave_height * 1.05, 60)
                wave_update_counter = 0
            
            # Draw waves with different characteristics
            for delay, wave in enumerate(wave_history):
                for i, offset in enumerate(np.linspace(1, 0, len(wave))):
                    x = int(mid_x1 * (1 - offset) + mid_x2 * offset)
                    y = int(mid_y1 * (1 - offset) + mid_y2 * offset)
                    
                    if i == 0 or i == len(wave) - 1:
                        cv2.line(frame, (x, y), (x, y), (255, 255, 255), 2)
                    else:
                        line_thickness = 3 if 7 <= i <= 9 else 2
                        cv2.line(frame, (x, y - int(wave[i])), (x, y + int(wave[i])), 
                                (255, 255, 255), line_thickness)
            
            text_x = (mid_x1 + mid_x2) // 2
            text_y = (mid_y1 + mid_y2) // 2 - 50
            text_pos = (text_x, text_y)
            
            if music_loaded and not music_playing:
                pygame.mixer.music.play()
                music_playing = True
                music_paused = False
            
            volume = np.interp(distance, [20, w // 2], [0, 100])
            volume = np.clip(volume, 0, 100)
            pygame.mixer.music.set_volume(volume / 100)
            
            if text_pos:
                text_size = cv2.getTextSize(f'{int(volume)}%', cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                text_x = text_pos[0] - text_size[0] // 2
                text_y = text_pos[1]
                cv2.putText(frame, f'{int(volume)}%', (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, f'{int(volume)}%', (text_x, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        else:
            if music_playing and hands_detected == 0:
                pygame.mixer.music.pause()
                music_paused = True
                music_playing = False
            base_wave_height = 24
        
        # Display instructions
        cv2.putText(frame, "Press 'M' to select music file", (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Right hand (thumb & index): Control frequency", (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Left hand (thumb & index): Control speed", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Both hands: Control volume", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current settings
        cv2.putText(frame, f"Current Speed: {current_speed:.1f}x", (w - 300, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Current Pitch: {current_pitch:+.1f}", (w - 300, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Display processing status
        if processing_needed:
            cv2.putText(frame, "Processing audio...", (w - 300, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        # Show the frame
        cv2.imshow('Hand Wave Music Control', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
        elif key in [ord('M'), ord('m')]:
            choose_music()
        elif cv2.getWindowProperty('Hand Wave Music Control', cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Clean up
    audio_processing_active = False
    if processor_thread.is_alive():
        processor_thread.join(timeout=1.0)
    
    cleanup_temp_files()

cap.release()
cv2.destroyAllWindows()