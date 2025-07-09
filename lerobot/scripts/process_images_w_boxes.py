#!/usr/bin/env python3

import os
import sys
import argparse
import cv2
import numpy as np
import supervision as sv
from supervision.draw.color import ColorPalette
from inference import get_model
import torch
from pathlib import Path
from tqdm import tqdm
import subprocess
import tempfile
import json

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b", 
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

class VideoProcessor:
    def __init__(self, confidence_threshold=0.4):
        """Initialize the video processor with ethernet cable detection model"""
        print("Loading ethernet cable detection model...")
        self.model_ethernet = get_model(model_id="ethernet-cable-detection/1")
        print("Model loaded successfully!")
        
        # Initialize annotators
        self.box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        self.label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
        
        # Check FFmpeg availability
        self.check_ffmpeg_availability()
        
    def check_ffmpeg_availability(self):
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ FFmpeg is available")
            else:
                print("Error: FFmpeg is not working properly")
                sys.exit(1)
        except FileNotFoundError:
            print("=" * 60)
            print("ERROR: FFmpeg not found!")
            print("=" * 60)
            print("FFmpeg is required for video processing.")
            print()
            print("INSTALLATION:")
            print("Ubuntu/Debian: sudo apt install ffmpeg")
            print("macOS: brew install ffmpeg")
            print("Windows: Download from https://ffmpeg.org/")
            print("=" * 60)
            sys.exit(1)
    
    def get_video_info(self, video_path):
        """Get video information using FFprobe"""
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
            str(video_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in info['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Get bitrate (try stream first, then format)
            bitrate = None
            if 'bit_rate' in video_stream:
                bitrate = int(video_stream['bit_rate'])
            elif 'bit_rate' in info['format']:
                bitrate = int(info['format']['bit_rate'])
            
            return {
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': eval(video_stream['r_frame_rate']),  # Convert fraction to float
                'codec': video_stream['codec_name'],
                'duration': float(video_stream.get('duration', 0)),
                'frames': int(video_stream.get('nb_frames', 0)),
                'bitrate': bitrate
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to get video info: {e}")
    
    def get_encoder_for_codec(self, codec_name):
        """Map codec names to their corresponding FFmpeg encoder names"""
        codec_map = {
            'av1': 'libsvtav1',   # Use SVT-AV1 encoder for AV1 (faster than libaom-av1)
            'h264': 'libx264',    # Use libx264 encoder for H.264
            'h265': 'libx265',    # Use libx265 encoder for H.265
            'hevc': 'libx265',    # HEVC is H.265
            'vp9': 'libvpx-vp9',  # VP9 encoder
            'vp8': 'libvpx',      # VP8 encoder
        }
        
        # Return mapped encoder or original codec name
        return codec_map.get(codec_name.lower(), codec_name)
    
    def process_frame(self, frame):
        """Process a single frame with ethernet cable detection"""
        # Run inference
        results = self.model_ethernet.infer(frame)[0]
        detections = sv.Detections.from_inference(results)
        
        # Filter detections by confidence
        detections = detections[detections.confidence > self.confidence_threshold]

        # Annotate frame
        annotated_frame = frame.copy()
        if len(detections) > 0:
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
        
        return annotated_frame
    
    def process_video(self, input_path, output_path):
        """Process a single MP4 video file using FFmpeg for I/O"""
        print(f"Processing MP4: {input_path}")
        
        try:
            # Get video information
            video_info = self.get_video_info(input_path)
            bitrate_info = f", bitrate: {video_info['bitrate']} bps" if video_info['bitrate'] else ", bitrate: unknown"
            print(f"Video info: {video_info['width']}x{video_info['height']}, "
                  f"{video_info['fps']:.2f} FPS, codec: {video_info['codec']}{bitrate_info}")
            
            # Get the appropriate encoder
            encoder = self.get_encoder_for_codec(video_info['codec'])
            print(f"Using encoder: {encoder}")
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary directory for frames
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                
                # Extract frames using FFmpeg
                print("Extracting frames...")
                extract_cmd = [
                    'ffmpeg', '-i', str(input_path),
                    '-f', 'image2',
                    '-vcodec', 'png',
                    str(temp_dir_path / 'frame_%06d.png')
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error extracting frames: {result.stderr}")
                    return False
                
                # Get list of extracted frames
                frame_files = sorted(temp_dir_path.glob('frame_*.png'))
                if not frame_files:
                    print("No frames extracted")
                    return False
                
                print(f"Extracted {len(frame_files)} frames")
                
                # Verify frame count matches expected
                if video_info['frames'] > 0 and len(frame_files) != video_info['frames']:
                    print(f"Warning: Expected {video_info['frames']} frames but extracted {len(frame_files)}")
                
                # Process each frame
                print("Processing frames...")
                processed_count = 0
                
                for frame_file in tqdm(frame_files, desc="Processing frames"):
                    try:
                        # Read frame
                        frame = cv2.imread(str(frame_file))
                        if frame is None:
                            print(f"Warning: Could not read frame {frame_file}")
                            continue
                        
                        # Process frame
                        processed_frame = self.process_frame(frame)
                        
                        # Save processed frame
                        if cv2.imwrite(str(frame_file), processed_frame):
                            processed_count += 1
                        else:
                            print(f"Warning: Could not save processed frame {frame_file}")
                            
                    except Exception as e:
                        print(f"Warning: Error processing frame {frame_file}: {e}")
                        continue
                
                print(f"Successfully processed {processed_count}/{len(frame_files)} frames")
                
                if processed_count == 0:
                    print("No frames were processed successfully")
                    return False
                
                # Reassemble video using FFmpeg with mapped encoder
                print("Reassembling video...")
                reassemble_cmd = [
                    'ffmpeg', '-y',  # Overwrite output
                    '-framerate', str(video_info['fps']),
                    '-i', str(temp_dir_path / 'frame_%06d.png'),
                    '-c:v', encoder,  # Use mapped encoder
                    '-pix_fmt', 'yuv420p',  # Standard pixel format
                ]
                
                # Use bitrate-based encoding to preserve file size
                if video_info['bitrate']:
                    print(f"Original bitrate: {video_info['bitrate']} bps")
                    
                    # Add codec-specific parameters
                    if encoder == 'libsvtav1':
                        # SVT-AV1 uses CRF mode only (no bitrate)
                        reassemble_cmd.extend([
                            '-g', '2',  # GOP size
                            '-crf', '30',  # Constant Rate Factor
                        ])
                        print("Using CRF mode for SVT-AV1 (bitrate not supported with CRF)")
                    elif encoder == 'libx264':
                        # Use original bitrate for H.264
                        reassemble_cmd.extend(['-b:v', str(video_info['bitrate'])])
                        reassemble_cmd.extend([
                            '-preset', 'ultrafast',  # Fastest encoding preset
                            '-threads', '0',         # Use all available CPU cores
                        ])
                        print(f"Using original bitrate: {video_info['bitrate']} bps")
                    elif encoder == 'libx265':
                        # Use original bitrate for H.265
                        reassemble_cmd.extend(['-b:v', str(video_info['bitrate'])])
                        reassemble_cmd.extend([
                            '-preset', 'ultrafast',  # Fastest encoding preset
                            '-x265-params', 'pools=+',  # Enable all CPU cores
                        ])
                        print(f"Using original bitrate: {video_info['bitrate']} bps")
                else:
                    # Fallback to CRF if no bitrate available
                    print("No bitrate information available, using CRF mode")
                    if encoder == 'libsvtav1':
                        reassemble_cmd.extend([
                            '-g', '2',  # GOP size
                            '-crf', '30',  # Constant Rate Factor
                        ])
                    elif encoder == 'libx264':
                        reassemble_cmd.extend([
                            '-crf', '23',  # Reasonable quality
                            '-preset', 'ultrafast',
                            '-threads', '0',
                        ])
                    elif encoder == 'libx265':
                        reassemble_cmd.extend([
                            '-crf', '25',  # Reasonable quality
                            '-preset', 'ultrafast',
                            '-x265-params', 'pools=+',
                        ])
                
                # Add output file
                reassemble_cmd.append(str(output_path))
                
                print(f"Running: {' '.join(reassemble_cmd)}")
                
                result = subprocess.run(reassemble_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error reassembling video:")
                    print(f"Command: {' '.join(reassemble_cmd)}")
                    print(f"Error output: {result.stderr}")
                    return False
                
                print(f"Successfully created output video: {output_path}")
                
                # Verify output video frame count
                try:
                    output_info = self.get_video_info(output_path)
                    print(f"Output video: {len(frame_files)} frames processed → {output_info['frames']} frames in output")
                    if output_info['frames'] != len(frame_files):
                        print(f"Warning: Frame count mismatch! Expected {len(frame_files)}, got {output_info['frames']}")
                except Exception as e:
                    print(f"Could not verify output frame count: {e}")
                
                return True
                
        except Exception as e:
            print(f"Error processing video {input_path}: {e}")
            return False
    
    def process_folder(self, input_folder, output_folder):
        """Process all MP4 files in a folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            print(f"Error: Input folder {input_folder} does not exist")
            return
        
        # Find all MP4 files
        mp4_files = list(input_path.glob("*.mp4")) + list(input_path.glob("*.MP4"))
        
        if not mp4_files:
            print(f"No MP4 files found in {input_folder}")
            return
        
        print(f"Found {len(mp4_files)} MP4 files to process")
        
        # Process each video
        successful = 0
        for video_file in mp4_files:
            # Create output filename
            output_file = output_path / f"{video_file.stem}.mp4"
            
            try:
                if self.process_video(video_file, output_file):
                    successful += 1
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                print("-" * 50)
        
        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful}/{len(mp4_files)} videos")
        print(f"Output folder: {output_folder}")


def main():
    parser = argparse.ArgumentParser(description="Process MP4 videos with ethernet cable detection using FFmpeg")
    parser.add_argument("--input_folder", required=True, help="Path to folder containing MP4 files")
    parser.add_argument("--output_folder", required=True, help="Path to folder for processed MP4 videos")
    parser.add_argument("--confidence", type=float, default=0.4, 
                       help="Confidence threshold for detections (default: 0.4)")
    
    args = parser.parse_args()
    
    # Validate input folder
    if not os.path.exists(args.input_folder):
        print(f"Error: Input folder '{args.input_folder}' does not exist")
        sys.exit(1)
    
    print("MP4 Video Processor with Ethernet Cable Detection (FFmpeg-based)")
    print("=" * 60)
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Confidence threshold: {args.confidence}")
    print("Video I/O: FFmpeg")
    print("Image processing: OpenCV")
    print("=" * 60)
    
    # Create processor and run
    processor = VideoProcessor(confidence_threshold=args.confidence)
    processor.process_folder(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
    

