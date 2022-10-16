#!/usr/bin/env python3
import argparse
import sys
import os

from pydub import AudioSegment
import uuid

# Setup Vars
__version__ = "0.0.1"

__cwdpath__ = os.path.normpath(os.getcwd())
sys.path.append(__cwdpath__)
import common as common
__basepath__ = common.getOneLevelUp(__cwdpath__)
__binpath__ = os.path.join(__basepath__, 'bin')
__inppath__ = os.path.join(__basepath__, 'audio', 'input')
__bgnpath__ = os.path.join(__basepath__, 'audio', 'noise')
__outpath__ = os.path.join(__basepath__, 'audio', 'train', 'wakewordTesting')
__trnpath__ = os.path.join(__basepath__, 'audio', 'train')
sys.path.append(__basepath__)
sys.path.append(__binpath__)



common.ensureFolder(__outpath__)

# Settings
BGN_SOFTENERS = [15, 17, 20]
BAS_LOUDER = [0, 5, 15, 20]
SPL_SPEED = [0.8, 0.9, 1.0, 1.1, 1.2]

# Add bin to syspath, for access to FFMPEG
sys.path.append(os.path.normpath(__binpath__))


# --------------------------------------
# main
# --------------------------------------
#
#
# --------------------------------------
def main(args):
    # Passed Args
    inputs = get_inputs()
    process_inputs(inputs)


# --------------------------------------
# process_inputs
# --------------------------------------
#
#
# --------------------------------------
def process_inputs(input_files):
    print()
    print('-#-#-GENERATING-#-#')
    # Overlay with Noise
    for i in input_files:
        overlay_noise(i)


# --------------------------------------
# overlay_noise
# --------------------------------------
#
#
# --------------------------------------
def overlay_noise(base):

    base_sounds = []
    base_sound = AudioSegment.from_wav(base)

    for x in SPL_SPEED:
        for l in BAS_LOUDER:
                louder = base_sound + l
                base_sounds.append(speed_change(louder, x))

    for s in base_sounds:
        base_outfilename = f"{uuid.uuid1()}.wav"
        base_outfilepath = os.path.join(__outpath__, base_outfilename)
        s.export(base_outfilepath, format="wav", bitrate="1411", parameters=["-ac", "1", "-ar", "16000"])
        base_duration = s.duration_seconds

        noise_sounds = []
        noise_files = get_noise_files()
        for x in noise_files:

            # Load to Sound
            nsound = AudioSegment.from_wav(x)

            # Cut it down to input length
            nsound = nsound[:base_duration * 1000]

            # Lower the DB and save to list
            for d in BGN_SOFTENERS:
                softer = nsound - d
                noise_sounds.append(softer)
                # outfilename = f"BGS-{uuid.uuid1()}.wav"
                # outfilepath = os.path.join(__outpath__, 'BSG', outfilename)
                # common.ensureFolder(outfilepath)
                # softer.export(outfilepath)

        for noise in noise_sounds:
            # Overlay the sounds
            new_sound = s.overlay(noise)
            # Output the result
            outfilename = f"{uuid.uuid1()}.wav"
            outfilepath = os.path.join(__outpath__, outfilename)
            new_sound.export(outfilepath, format="wav", bitrate="1411", parameters=["-ac", "1", "-ar", "16000"])


# --------------------------------------
# get_noise_files
# --------------------------------------
#
#
# --------------------------------------
def get_noise_files():
    noise_files = []
    for filename in os.listdir(__bgnpath__):
        f = os.path.join(__bgnpath__, filename)
        if os.path.isfile(f):
            noise_files.append(f)
    return noise_files


# --------------------------------------
# get_inputs
# --------------------------------------
#
#
# --------------------------------------
def get_inputs():
    input_files = []
    print()
    print('-#-#-INPUT-#-#')
    for filename in os.listdir(__inppath__):
        f = os.path.join(__inppath__, filename)
        if os.path.isfile(f):
            print(f"{filename}")
            input_files.append(f)
    return input_files


# --------------------------------------
# speed_change
# --------------------------------------
#
#
# --------------------------------------
def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
     # convert the sound with altered frame rate to a standard frame rate
     # so that regular playback programs will work right. They often only
     # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)


# --------------------------------------
# __name__
# --------------------------------------
#
#
# --------------------------------------
if __name__ == "__main__":
    main(sys.argv)