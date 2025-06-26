import os
import json
import random
import requests
import zipfile
import io
import time
import math
import numpy as np
from scipy.interpolate import CubicSpline
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, List, Dict
from struct import pack
import logging
import tkinter as tk
from tkinter import messagebox, ttk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# BSOR encoding constants
MAGIC_HEX = 0x442d3d69
MAX_SUPPORTED_VERSION = 1
NOTE_EVENT_GOOD = 0
NOTE_EVENT_BAD = 1
NOTE_EVENT_MISS = 2
NOTE_EVENT_BOMB = 3
SABER_LEFT = 0  # Red saber
SABER_RIGHT = 1  # Blue saber
NOTE_SCORE_TYPE_NORMAL = 0
NOTE_SCORE_TYPE_SLIDERHEAD = 1
NOTE_SCORE_TYPE_SLIDERTAIL = 2
NOTE_SCORE_TYPE_BURSTSLIDERHEAD = 3
NOTE_SCORE_TYPE_BURSTSLIDERELEMENT = 4

# Beat Saber constants
GAME_VERSION = "1.29.0"
MOD_VERSION = "1.0.0"
PLATFORM = "oculus"
TRACKING_SYSTEM = "Oculus"
HMD = "Oculus Quest 2"
CONTROLLER = "Oculus Touch"
PLAYER_NAME = "Grok"
PLAYER_ID = "76561197960265728"
BOT_HEIGHT = 1.6
MISS_CHANCE = 0.05
WALL_MISS_CHANCE = 0.15
BOMB_MISS_CHANCE = 0.12
CONFUSION_CHANCE = 0.08
REACTION_DELAY = 0.1
SHAKE_MAGNITUDE = 0.0005
CUT_SHAKE_MAGNITUDE = 0.0005
AFK_TIMEOUT = 3.0
IDLE_SABER_LEFT_POS = [-0.25, 0.7, 0.0]
IDLE_SABER_RIGHT_POS = [0.25, 0.7, 0.0]
IDLE_TRANSITION_TIME = 2.0
MAX_SABER_SPEED = 10.0

# Path for saving replays
BASE_REPLAY_PATH = Path("BeatSaberDemos")

# Global variable for error mode
ERROR_MODE = False

class BSException(Exception):
    pass

def encode_int(f: BinaryIO, value: int):
    f.write(pack('<i', value))

def encode_byte(f: BinaryIO, value: int):
    f.write(pack('<B', value))

def encode_float(f: BinaryIO, value: float):
    value = float(value)
    if not math.isfinite(value):
        logging.warning(f"Invalid float value: {value}, replaced with 0.0")
        value = 0.0
    f.write(pack('<f', value))

def encode_bool(f: BinaryIO, value: bool):
    f.write(pack('<?', value))

def encode_long(f: BinaryIO, value: int):
    f.write(pack('<q', value))

def encode_string(f: BinaryIO, value: str):
    encoded = value.encode('utf-8')
    encode_int(f, len(encoded))
    f.write(encoded)

class Writable:
    def write(self, f: BinaryIO):
        pass

class VRObject(Writable):
    def __init__(self, x=0.0, y=0.0, z=0.0, x_rot=0.0, y_rot=0.0, z_rot=0.0, w_rot=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.x_rot = float(x_rot)
        self.y_rot = float(y_rot)
        self.z_rot = float(z_rot)
        self.w_rot = float(w_rot)

    def write(self, f: BinaryIO):
        for v in [self.x, self.y, self.z, self.x_rot, self.y_rot, self.z_rot, self.w_rot]:
            encode_float(f, v)

def quaternion_from_direction(direction: List[float]) -> tuple:
    norm = math.sqrt(sum(d * d for d in direction))
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    direction = [d / norm for d in direction]
    yaw = -math.atan2(direction[0], direction[2])
    pitch = math.radians(15)
    qx = math.sin(pitch / 2) * math.cos(yaw / 2)
    qy = math.sin(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(pitch / 2) * math.sin(yaw / 2)
    qw = math.cos(pitch / 2) * math.cos(yaw / 2)
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm < 1e-6:
        return (0.0, 0.0, 0.0, 1.0)
    return (qx/norm, qy/norm, qz/norm, qw/norm)

def analyze_section_difficulty(notes: List[Dict], current_time: float, window: float = 1.0) -> float:
    section_notes = [n for n in notes if abs(n['_time'] - current_time) <= window]
    if not section_notes:
        return 0.0
    note_density = len(section_notes) / (window * 2)
    pattern_complexity = 0.0
    for i in range(len(section_notes) - 1):
        if section_notes[i]['_type'] != section_notes[i+1]['_type']:
            pattern_complexity += 0.3
        if section_notes[i]['_cutDirection'] in [4, 5, 6, 7]:
            pattern_complexity += 0.2
    return min(note_density + pattern_complexity, 2.0)

def saber_swing_quaternion(cut_direction: int, swing_magnitude: float, pre_swing: bool = False) -> tuple:
    swing_directions = {
        0: ([0, -1.0, 0], [0, 1.0, 1.0]),    # Up -> Down
        1: ([0, 1.0, 0], [0, -1.0, 1.0]),     # Down -> Up
        2: ([-1.0, 0, 0], [1.0, 0, 1.0]),     # Left
        3: ([1.0, 0, 0], [-1.0, 0, 1.0]),     # Right
        4: ([-1.0, 1.0, 0], [1.0, -1.0, 1.0]),  # Up-left
        5: ([1.0, 1.0, 0], [-1.0, -1.0, 1.0]),  # Up-right
        6: ([-1.0, -1.0, 0], [1.0, 1.0, 1.0]),  # Down-left
        7: ([1.0, -1.0, 0], [-1.0, 1.0, 1.0]),  # Down-right
        8: ([0, 0, 0], [0, 0, 1.0])          # Dot
    }
    windup_dir, swing_dir = swing_directions.get(cut_direction, ([0, 0, 0], [0, 0, 1.0]))
    direction = windup_dir if pre_swing else swing_dir
    norm = math.sqrt(sum(d * d for d in direction))
    if norm < 1e-6:
        direction = [0,0, 1.0]
    else:
        direction = [d / norm for d in direction]
    forward = [0, 0, 1.0]
    dot = sum(a * b for a, b in zip(direction, forward))
    if abs(dot) > 0.99999:
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0 if dot > 0 else -1.0
    else:
        cross = [
            direction[1] * forward[2] - direction[2] * forward[1],
            direction[2] * forward[0] - direction[0] * forward[2],
            direction[0] * forward[1] - direction[1] * forward[0]
        ]
        qx, qy, qz = cross
        qw = math.sqrt(sum(x*x for x in direction) * sum(x*x for x in forward)) + dot
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm < 1e-6:
            qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
        else:
            qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    swing_scale = min(swing_magnitude / 0.3, 1.5)
    if ERROR_MODE:
        qx += random.uniform(-0.02, 0.02) * swing_scale
        qy += random.uniform(-0.02, 0.02) * swing_scale
        qz += random.uniform(-0.02, 0.02) * swing_scale
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        if norm < 1e-6:
            return (0.0, 0.0, 0.0, 1.0)
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    return (-qx, -qy, -qz, qw)

def afk_saber_spin(time: float, saber_type: int) -> VRObject:
    radius = 0.2
    angular_speed = 2.0
    phase = math.pi if saber_type == SABER_LEFT else 0
    x = math.sin(time * angular_speed + phase) * radius
    y = BOT_HEIGHT - 0.6 + math.cos(time * angular_speed + phase) * radius
    z = 0.0
    qx, qy, qz, qw = saber_swing_quaternion(8, 0.0)
    return VRObject(x=x, y=y, z=z, x_rot=qx, y_rot=qy, z_rot=qz, w_rot=qw)

class Info(Writable):
    def __init__(self, song_info: Dict, score: int, modifiers: str, map_hash: str):
        if '_songName' not in song_info:
            raise BSException("Map data does not contain '_songName' field")
        self.version = MOD_VERSION
        self.gameVersion = GAME_VERSION
        self.timestamp = str(int(time.time()))
        self.playerId = PLAYER_ID
        self.playerName = PLAYER_NAME
        self.platform = PLATFORM
        self.trackingSystem = TRACKING_SYSTEM
        self.hmd = HMD
        self.controller = CONTROLLER
        self.songHash = map_hash
        self.songName = song_info['_songName']
        self.mapper = song_info.get('metadata', {}).get('levelAuthorName', 'Unknown')
        self.difficulty = song_info['difficulty']
        self.score = score
        self.mode = "Standard"
        self.environment = "DefaultEnvironment"
        self.modifiers = modifiers
        self.jumpDistance = song_info.get('jumpDistance', 2.0)
        self.leftHanded = False
        self.height = BOT_HEIGHT
        self.startTime = 0.0
        self.failTime = 0.0
        self.speed = 1.0

    def write(self, f: BinaryIO):
        encode_byte(f, 0)
        data = [
            self.version, self.gameVersion, self.timestamp, self.playerId, self.playerName, self.platform,
            self.trackingSystem, self.hmd, self.controller, self.songHash, self.songName, self.mapper,
            self.difficulty, self.score, self.mode, self.environment, self.modifiers, self.jumpDistance,
            self.leftHanded, self.height, self.startTime, self.failTime, self.speed
        ]
        types = [str, str, str, str, str, str, str, str, str, str, str, str, str, int, str, str, str, float, bool, float, float, float, float]
        for d, t in zip(data, types):
            if t == str:
                encode_string(f, d)
            elif t == int:
                encode_int(f, d)
            elif t == float:
                encode_float(f, d)
            elif t == bool:
                encode_bool(f, d)

class Frame(Writable):
    def __init__(self, time: float, fps: int, head: VRObject, left_hand: VRObject, right_hand: VRObject):
        self.time = time
        self.fps = fps
        self.head = head
        self.left_hand = left_hand
        self.right_hand = right_hand

    def write(self, f: BinaryIO):
        encode_float(f, self.time)
        encode_int(f, self.fps)
        self.head.write(f)
        self.left_hand.write(f)
        self.right_hand.write(f)

class Cut(Writable):
    def __init__(self, saber_type: int, cut_direction: int, swing_magnitude: float, note_pos: List[float], note_type: int = 0, is_bad: bool = False):
        self.speedOK = not is_bad
        self.directionOK = not is_bad
        self.saberTypeOK = (saber_type == note_type) and not is_bad
        self.wasCutTooSoon = False
        self.saberSpeed = 7.0 if not is_bad else random.uniform(2.0, 5.0)
        angle = math.radians(cut_direction * 45)
        saber_dir = [math.sin(angle), math.cos(angle), 0.0]
        norm = math.sqrt(sum(d * d for d in saber_dir))
        self.saberDirection = [d / norm if norm > 1e-6 else 0.0 for d in saber_dir]
        if is_bad:
            self.saberDirection = [d + random.uniform(-0.1, 0.1) for d in self.saberDirection]
            norm = math.sqrt(sum(d * d for d in self.saberDirection))
            self.saberDirection = [d / norm if norm > 1e-6 else 0.0 for d in self.saberDirection]
        self.saberType = saber_type
        self.timeDeviation = 0.0 if not ERROR_MODE else random.uniform(-0.02, 0.02)
        self.cutDeviation = 0.0 if not ERROR_MODE else random.uniform(-0.05, 0.05)
        shake_offset = [0.0, 0.0, 0.0] if not ERROR_MODE else [random.uniform(-CUT_SHAKE_MAGNITUDE, CUT_SHAKE_MAGNITUDE) for _ in range(3)]
        self.cutPoint = [note_pos[i] + shake_offset[i] for i in range(3)]
        self.cutNormal = [0.0, 0.0, 1.0]
        self.cutDistanceToCenter = 0.0 if not is_bad else random.uniform(0.1, 0.3)
        self.cutAngle = 100.0 if not is_bad else random.uniform(50.0, 80.0)
        base_rating = min(max(swing_magnitude / 0.3, 0.7), 1.0)
        if ERROR_MODE:
            variation = random.uniform(-0.2, 0.2)
            self.beforeCutRating = max(0.5, min(1.0, base_rating + variation)) if not is_bad else random.uniform(0.3, 0.7)
            self.afterCutRating = max(0.5, min(1.0, base_rating + random.uniform(-0.15, 0.15))) if not is_bad else random.uniform(0.3, 0.7)
        else:
            self.beforeCutRating = base_rating
            self.afterCutRating = base_rating
        logging.debug(f"Cut: saberType={saber_type}, direction={self.saberDirection}, cutPoint={self.cutPoint}, swing_magnitude={swing_magnitude}, is_bad={is_bad}")

    def write(self, f: BinaryIO):
        encode_bool(f, self.speedOK)
        encode_bool(f, self.directionOK)
        encode_bool(f, self.saberTypeOK)
        encode_bool(f, self.wasCutTooSoon)
        encode_float(f, self.saberSpeed)
        for v in self.saberDirection:
            encode_float(f, v)
        encode_int(f, self.saberType)
        encode_float(f, self.timeDeviation)
        encode_float(f, self.cutDeviation)
        for v in self.cutPoint:
            encode_float(f, v)
        for v in self.cutNormal:
            encode_float(f, v)
        encode_float(f, self.cutDistanceToCenter)
        encode_float(f, self.cutAngle)
        encode_float(f, self.beforeCutRating)
        encode_float(f, self.afterCutRating)

class Note(Writable):
    def __init__(self, note_id: int, event_time: float, spawn_time: float, event_type: int, cut: Cut = None, scoring_type: int = 0, line_index: int = 0, note_line_layer: int = 0, color_type: int = 0, cut_direction: int = 0):
        self.note_id = note_id
        self.event_time = event_time
        self.spawn_time = spawn_time
        self.event_type = event_type
        self.cut = cut
        self.scoringType = scoring_type
        self.lineIndex = line_index
        self.noteLineLayer = note_line_layer
        self.colorType = color_type
        self.cutDirection = cut_direction

    def write(self, f: BinaryIO):
        encode_int(f, self.note_id)
        encode_float(f, self.event_time)
        encode_float(f, self.spawn_time)
        encode_int(f, self.event_type)
        if self.cut and self.event_type in [NOTE_EVENT_GOOD, NOTE_EVENT_BAD]:
            self.cut.write(f)

class Bsor(Writable):
    def __init__(self, info: Info, frames: List[Frame], notes: List[Note]):
        self.magic_number = MAGIC_HEX
        self.file_version = 1
        self.info = info
        self.frames = frames
        self.notes = notes
        self.walls = []
        self.heights = []
        self.pauses = []
        self.controller_offsets = None
        self.user_data = []

    def write(self, f: BinaryIO):
        encode_int(f, self.magic_number)
        encode_byte(f, self.file_version)
        self.info.write(f)
        encode_byte(f, 1)
        encode_int(f, len(self.frames))
        for frame in self.frames:
            frame.write(f)
        encode_byte(f, 2)
        encode_int(f, len(self.notes))
        for note in self.notes:
            note.write(f)
        for magic, data in [(3, self.walls), (4, self.heights), (5, self.pauses)]:
            encode_byte(f, magic)
            encode_int(f, len(data))

def fetch_beat_saver_map(map_id: str) -> Dict:
    url = f"https://api.beatsaver.com/maps/id/{map_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise BSException(f"Failed to fetch map {map_id}: {response.status_code} - {response.text}")
    return response.json()

def download_and_extract_map(map_id: str) -> Dict:
    try:
        map_data = fetch_beat_saver_map(map_id)
        if 'versions' not in map_data or not map_data['versions']:
            raise BSException("Map data does not contain 'versions' field")
        zip_url = map_data['versions'][0]['downloadURL']
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise BSException(f"Failed to download map {map_id}: {response.status_code}")
        zip_file = io.BytesIO(response.content)
        with zipfile.ZipFile(zip_file) as z:
            info_file = [f for f in z.namelist() if f.lower() == 'info.dat']
            if not info_file:
                raise BSException("'info.dat' file not found in map archive")
            with z.open(info_file[0]) as f:
                info = json.load(f)
            difficulties = []
            for diff_set in info.get('_difficultyBeatmapSets', []):  # Fixed: Replaced 'eterna' with '[]'
                if diff_set.get('_beatmapCharacteristicName') == 'Standard':
                    for diff in diff_set.get('_difficultyBeatmaps', []):
                        difficulties.append({
                            'difficulty': diff['_difficulty'],
                            'file': diff['_beatmapFilename'],
                            'bpm': info.get('_beatsPerMinute', 120.0),
                            'njs': diff.get('_noteJumpMovementSpeed', 10.0),
                            'offset': diff.get('_noteJumpStartBeatOffset', 0.0)
                        })
            if not difficulties:
                raise BSException("No Standard difficulties found")
            return {
                'info': info,
                'difficulties': difficulties,
                'map_hash': map_data['versions'][0]['hash'],
                'versions': map_data['versions']  # Ensure versions is included for later use
            }
    except Exception as e:
        raise BSException(f"Error processing map {map_id}: {str(e)}")

def get_bpm_at_beat(bpm_events: List[Dict], beat: float, base_bpm: float) -> float:
    current_bpm = base_bpm
    for event in sorted(bpm_events, key=lambda x: x['_time']):
        if event['_time'] <= beat:
            current_bpm = event['_bpm']
        else:
            break
    return current_bpm

def beats_to_seconds(beats: float, bpm_events: List[Dict], base_bpm: float) -> float:
    if not bpm_events:
        return (beats * 60.0) / base_bpm
    seconds = 0.0
    current_beat = 0.0
    current_bpm = base_bpm
    sorted_events = sorted(bpm_events, key=lambda x: x['_time'])
    for event in sorted_events + [{'_time': beats, '_bpm': current_bpm}]:
        event_beat = min(event['_time'], beats)
        if event_beat > current_beat:
            seconds += ((event_beat - current_beat) * 60.0) / current_bpm
            current_beat = event_beat
            current_bpm = event.get('_bpm', current_bpm)
        if current_beat >= beats:
            break
    return seconds

def calculate_spawn_time(event_time: float, njs: float, offset: float, bpm: float) -> float:
    half_jump_duration = (60.0 / bpm) * (4.0 * (10.0 / njs))
    offset_time = beats_to_seconds(offset, [], bpm)
    return max(0.0, event_time - half_jump_duration - offset_time)

def interpolate_movement(start_pos: List[float], end_pos: List[float], start_time: float, end_time: float, times: np.ndarray, swing_arc: bool = False, pre_swing: bool = False) -> List[VRObject]:
    if end_time <= start_time:
        return [VRObject(*start_pos) for _ in times]
    num_points = max(20, int((end_time - start_time) * 90 * 4))
    if swing_arc:
        windup_time = start_time + (end_time - start_time) * 0.1
        pre_time = start_time + (end_time - start_time) * 0.25
        mid_time = (start_time + end_time) / 2
        follow_time = start_time + (end_time - start_time) * 0.9
        follow_end_time = end_time + (end_time - start_time) * 0.2
        windup_pos = [
            start_pos[0] - (end_pos[0] - start_pos[0]) * 0.3,
            start_pos[1] + 0.3,
            start_pos[2] - 0.15
        ]
        pre_pos = [
            start_pos[0] - (end_pos[0] - start_pos[0]) * 0.8,
            start_pos[1] + 0.7,
            start_pos[2] - 0.5
        ]
        mid_pos = [
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[1] + end_pos[1]) / 2 + 0.6,
            (start_pos[2] + end_pos[2]) / 2
        ]
        follow_pos = [
            end_pos[0] + (end_pos[0] - start_pos[0]) * 0.4,
            end_pos[1] - 0.4,
            end_pos[2]
        ]
        follow_end_pos = [
            end_pos[0] + (end_pos[0] - start_pos[0]) * 0.6,
            end_pos[1] - 0.6,
            end_pos[2]
        ]
        times_key = [start_time, windup_time, pre_time, mid_time, follow_time, follow_end_time] if pre_swing else [start_time, mid_time, end_time, follow_end_time]
        pos_x = [start_pos[0], windup_pos[0], pre_pos[0], mid_pos[0], follow_pos[0], follow_end_pos[0]] if pre_swing else [start_pos[0], mid_pos[0], end_pos[0], follow_end_pos[0]]
        pos_y = [start_pos[1], windup_pos[1], pre_pos[1], mid_pos[1], follow_pos[1], follow_end_pos[1]] if pre_swing else [start_pos[1], mid_pos[1], end_pos[1], follow_end_pos[1]]
        pos_z = [start_pos[2], windup_pos[2], pre_pos[2], mid_pos[2], follow_pos[2], follow_end_pos[2]] if pre_swing else [start_pos[2], mid_pos[2], end_pos[2], follow_end_pos[2]]
        times_key = np.array(times_key)
    else:
        times_key = np.linspace(start_time, end_time, num_points)
        pos_x = np.linspace(start_pos[0], end_pos[0], num_points)
        pos_y = np.linspace(start_pos[1], end_pos[1], num_points)
        pos_z = np.linspace(start_pos[2], end_pos[2], num_points)
    cs_x = CubicSpline(times_key, pos_x, bc_type='natural')
    cs_y = CubicSpline(times_key, pos_y, bc_type='natural')
    cs_z = CubicSpline(times_key, pos_z, bc_type='natural')
    positions = []
    prev_pos = None
    dt = 1/90
    for t in times:
        pos = [float(cs_x(t)), float(cs_y(t)), float(cs_z(t))]
        if prev_pos:
            dist = math.sqrt(sum((p - q) ** 2 for p, q in zip(pos, prev_pos)))
            speed = dist / dt
            if speed > MAX_SABER_SPEED:
                scale = MAX_SABER_SPEED / speed
                pos = [prev_pos[i] + (pos[i] - prev_pos[i]) * scale for i in range(3)]
        positions.append(VRObject(x=pos[0], y=pos[1], z=pos[2], w_rot=1.0))
        prev_pos = pos
    return positions

def generate_rhythm_sway(time: float, bpm: float, saber_type: int) -> VRObject:
    beat_interval = 60.0 / bpm
    beat_time = time / beat_interval
    phase = math.pi if saber_type == SABER_LEFT else 0
    x = math.sin(beat_time * math.pi) * 0.15
    y = BOT_HEIGHT - 0.6 + math.cos(beat_time * math.pi) * 0.08
    z = 0.0
    if ERROR_MODE:
        x += random.uniform(-0.02, 0.02)
        y += random.uniform(-0.02, 0.02)
    return VRObject(x=x, y=y, z=z, w_rot=1.0)

def generate_dance_movement(time: float) -> List[float]:
    x = math.sin(time * 0.8) * 0.04
    y = BOT_HEIGHT + math.cos(time * 0.7) * 0.02
    z = math.sin(time * 0.75) * 0.04
    if ERROR_MODE:
        x += random.uniform(-0.01, 0.01)
        y += random.uniform(-0.01, 0.01)
        z += random.uniform(-0.01, 0.01)
    return [x, y, z]

def calculate_multiplier(combo: int) -> int:
    if combo >= 16:
        return 8
    elif combo >= 8:
        return 4
    elif combo >= 2:
        return 2
    return 1

def generate_replay(map_data: Dict, selected_difficulty: Dict) -> Bsor:
    song_info = map_data['info']
    map_hash = map_data['map_hash']
    difficulty = selected_difficulty['difficulty']
    diff_file = selected_difficulty['file']
    base_bpm = selected_difficulty['bpm']
    njs = selected_difficulty['njs']
    offset = selected_difficulty['offset']
    
    with zipfile.ZipFile(io.BytesIO(requests.get(map_data['versions'][0]['downloadURL']).content)) as z:
        if diff_file not in z.namelist():
            raise BSException(f"Difficulty file {diff_file} not found in map archive")
        with z.open(diff_file) as f:
            diff_data = json.load(f)
    
    bpm_events = diff_data.get('_bpmEvents', [])
    for event in bpm_events:
        logging.info(f"BPM event at beat {event['_time']:.2f}: {event['_bpm']} BPM")
    notes = diff_data.get('_notes', [])
    if not notes:
        raise BSException("No notes in the map")
    song_duration = beats_to_seconds(max([n['_time'] for n in notes] + [0]), bpm_events, base_bpm)
    fps = 90
    frame_times = np.arange(0, song_duration + 1, 1/fps)
    walls = []
    for obstacle in diff_data.get('_obstacles', []):
        start_time = beats_to_seconds(obstacle['_time'], bpm_events, base_bpm)
        duration = beats_to_seconds(obstacle['_duration'], bpm_events, base_bpm)
        dodge_duration = 0.1 if ERROR_MODE else 0.0
        walls.append({
            'time': start_time,
            'end_time': start_time + duration + dodge_duration,
            'line_index': obstacle['_lineIndex'],
            'type': obstacle['_type'],
            'width': obstacle['_width']
        })
        logging.info(f"Wall from {start_time:.3f}s to {start_time + duration + dodge_duration:.3f}s, index={obstacle['_lineIndex']}, width={obstacle['_width']}, type={obstacle['_type']}")
    frames = []
    replay_notes = []
    score = 0
    combo = 0
    multiplier = 1
    valid_note_ids = {}
    note_positions = {0: [], 1: []}
    note_swings = {}
    processed_note_times = set()

    for idx, note in enumerate(notes):
        if note['_type'] in [0, 1]:
            saber_type = SABER_LEFT if note['_type'] == 0 else SABER_RIGHT
            event_time = beats_to_seconds(note['_time'], bpm_events, base_bpm)
            if (event_time, note['_type'], note['_lineIndex'], note['_lineLayer']) in processed_note_times:
                logging.warning(f"Duplicate note at {event_time:.3f}s, type={note['_type']}, skipping")
                continue
            processed_note_times.add((event_time, note['_type'], note['_lineIndex'], note['_lineLayer']))
            line_index = note['_lineIndex']
            line_layer = note['_lineLayer']
            cut_direction = note['_cutDirection']
            color_type = 0 if note['_type'] == 0 else 1
            scoring_type = NOTE_SCORE_TYPE_NORMAL
            if 'customData' in note and '_noteType' in note['customData']:
                note_type = note['customData']['_noteType']
                if 'SliderHead' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_SLIDERHEAD
                elif 'SliderTail' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_SLIDERTAIL
                elif 'BurstSliderHead' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERHEAD
                elif 'BurstSliderElement' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERELEMENT
            note_id = scoring_type * 10000 + line_index * 1000 + line_layer * 100 + color_type * 10 + cut_direction
            valid_note_ids[idx] = (line_index, line_layer, note['_type'], cut_direction, scoring_type)
            x = line_index * 0.5 - 0.75
            y = line_layer * 0.3 + 0.7
            note_positions[saber_type].append((event_time, [x, y, 0.0], cut_direction, note_id, idx))
            note_swings[idx] = 0.0
            logging.info(f"Note at {event_time:.3f}s: saber={saber_type}, position=({x:.2f}, {y:.2f}, 0.0), id={note_id}, scoringType={scoring_type}, cutDirection={cut_direction}")
    bomb_positions = []
    for note in notes:
        if note['_type'] == 3:
            event_time = beats_to_seconds(note['_time'], bpm_events, base_bpm)
            line_index = note['_lineIndex']
            line_layer = note['_lineLayer']
            x = line_index * 0.5 - 0.75
            y = line_layer * 0.3 + 0.7
            bomb_positions.append((event_time, [x, y, 0.0]))
            logging.info(f"Bomb at {event_time:.3f}s: position=({x:.2f}, {y:.2f}, 0.0)")
    last_left_pos = [-0.3, BOT_HEIGHT - 0.6, 0.0]
    last_right_pos = [0.3, BOT_HEIGHT - 0.6, 0.0]
    dodge_end_times = []
    idle_start_time = None
    last_note_time = 0.0
    afk_mode = False
    current_notes = [n for n in notes if n['_type'] in [0, 1]]

    for i, t in enumerate(frame_times):
        section_difficulty = analyze_section_difficulty(current_notes, t)
        head_pos = generate_dance_movement(t)
        head_rot = [0.0, 0.0, 0.0, 1.0]
        left_hand = VRObject(x=-0.3, y=BOT_HEIGHT - 0.6, z=0.0, w_rot=1.0)
        right_hand = VRObject(x=0.3, y=BOT_HEIGHT - 0.6, z=0.0, w_rot=1.0)
        nearby_notes = {s: [(nt, pos, cd, nid, idx) for nt, pos, cd, nid, idx in pos_list if abs(nt - t) <= 2.0]
                       for s, pos_list in note_positions.items()}
        next_notes = {s: [(nt, pos, cd, nid, idx) for nt, pos, cd, nid, idx in pos_list if nt > t and nt <= t + 2.0]
                      for s, pos_list in note_positions.items()}
        hit_notes = {s: [(nt, pos, cd, nid, idx) for nt, pos, cd, nid, idx in pos_list if abs(nt - t) < 0.03]
                     for s, pos_list in note_positions.items()}
        if any(nearby_notes.values()):
            last_note_time = max(nt for s, notes in nearby_notes.items() for nt, _, _, _, _ in notes)
            afk_mode = False
            idle_start_time = None
        elif t - last_note_time >= AFK_TIMEOUT:
            afk_mode = True
            idle_start_time = None
        active_walls = [w for w in walls if w['time'] <= t <= w['end_time']]
        dodge_pos = None
        for wall in active_walls:
            if not ERROR_MODE or random.random() > WALL_MISS_CHANCE:
                if wall['type'] == 1:
                    dodge_delay = REACTION_DELAY
                    if t >= wall['time'] + dodge_delay:
                        head_pos[1] = max(BOT_HEIGHT - 0.7, head_pos[1] - 0.5)
                        dodge_pos = head_pos.copy()
                else:
                    dodge_delay = REACTION_DELAY
                    if t >= wall['time'] + dodge_delay:
                        center_x = wall['line_index'] * 0.5 - 0.75 + (wall['width'] - 1) * 0.25
                        if abs(head_pos[0] - center_x) < wall['width'] * 0.5:
                            dodge_direction = -1.0 if head_pos[0] < center_x else 1.0
                            if ERROR_MODE and random.random() < 0.1:
                                dodge_direction *= -1
                            head_pos[0] = center_x + dodge_direction
                            dodge_pos = head_pos.copy()
            else:
                logging.info(f"Bot missed wall at {t:.3f}s!")
                head_pos[0] += random.uniform(-0.1, 0.1)
                head_pos[1] += random.uniform(-0.05, 0.05)
        if dodge_pos:
            dodge_end_times.append((t + (0.1 if not ERROR_MODE else random.uniform(0.08, 0.2)), dodge_pos))
        for end_time, pos in dodge_end_times[:]:
            if t <= end_time:
                head_pos = pos
            else:
                dodge_end_times.remove((end_time, pos))
        nearby_bombs = [b for b in bomb_positions if abs(b[0] - t) < 0.5]
        for b_time, bomb_pos in nearby_bombs:
            if not ERROR_MODE or random.random() > BOMB_MISS_CHANCE:
                for saber, last_pos in [(left_hand, last_left_pos), (right_hand, last_right_pos)]:
                    dist = math.sqrt((saber.x - bomb_pos[0])**2 + (saber.y - bomb_pos[1])**2)
                    if dist < 0.5:
                        reaction_delay = REACTION_DELAY
                        if t >= b_time - 0.2 + reaction_delay:
                            dodge_x = -0.5 if saber.x < bomb_pos[0] else 0.5
                            dodge_y = 0.5 if saber.y < bomb_pos[1] else -0.25
                            if ERROR_MODE and random.random() < 0.2:
                                dodge_x *= random.uniform(0.5, 1.5)
                                dodge_y *= random.uniform(0.5, 1.5)
                            saber.x += dodge_x
                            saber.y += dodge_y
            else:
                logging.info(f"Bot hit a bomb at {t:.3f}s!")
                left_hand.x += random.uniform(-0.3, 0.3)
                left_hand.y += random.uniform(-0.3, 0.3)
                right_hand.x += random.uniform(-0.3, 0.3)
                right_hand.y += random.uniform(-0.3, 0.3)
        if afk_mode:
            left_hand = afk_saber_spin(t, SABER_LEFT)
            right_hand = afk_saber_spin(t, SABER_RIGHT)
            last_left_pos = [left_hand.x, left_hand.y, left_hand.z]
            last_right_pos = [right_hand.x, right_hand.y, right_hand.z]
            frames.append(Frame(
                time=t,
                fps=fps,
                head=VRObject(
                    x=head_pos[0],
                    y=head_pos[1],
                    z=head_pos[2],
                    x_rot=head_rot[0],
                    y_rot=head_rot[1],
                    z_rot=head_rot[2],
                    w_rot=head_rot[3]
                ),
                left_hand=left_hand,
                right_hand=right_hand
            ))
            continue
        if ERROR_MODE:
            left_hand.x += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
            left_hand.y += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
            left_hand.z += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
            right_hand.x += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
            right_hand.y += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
            right_hand.z += random.uniform(-SHAKE_MAGNITUDE, SHAKE_MAGNITUDE)
        swing_magnitude = 0.0
        if any(next_notes.values()):
            for saber_type, saber_notes in next_notes.items():
                if saber_notes:
                    next_time, next_pos, cut_direction, note_id, _ = saber_notes[0]
                    last_pos = last_left_pos if saber_type == SABER_LEFT else last_right_pos
                    direction = [next_pos[0] - head_pos[0], next_pos[1] - head_pos[1], 10.0 - head_pos[2]]
                    head_rot = quaternion_from_direction(direction)
                    swing_magnitude = math.sqrt(sum((next_pos[i] - last_pos[i])**2 for i in range(3))) + 0.5
                    note_swings[saber_notes[0][4]] = swing_magnitude
                    break
        if swing_magnitude > 0.5:
            head_rot = (
                head_rot[0],
                head_rot[1] + math.sin(swing_magnitude * 0.5) * 0.1,
                head_rot[2],
                math.cos(math.acos(max(min(head_rot[3], 1.0), -1.0)) + swing_magnitude * 0.05)
            )
        for saber_type, notes_at_t in hit_notes.items():
            if notes_at_t:
                note_time, note_pos, cut_direction, note_id, idx = notes_at_t[0]
                note_positions[saber_type] = [(nt, p, cd, nid, i) for nt, p, cd, nid, i in note_positions[saber_type] if i != idx]
                swing_magnitude = note_swings.get(idx, 0.5)
                swing_rot = saber_swing_quaternion(cut_direction, swing_magnitude)
                shake_offset = [0.0, 0.0, 0.0] if not ERROR_MODE else [random.uniform(-CUT_SHAKE_MAGNITUDE, CUT_SHAKE_MAGNITUDE) for _ in range(3)]
                hit_pos = [note_pos[i] + shake_offset[i] for i in range(3)]
                if saber_type == SABER_LEFT:
                    left_hand = VRObject(
                        x=hit_pos[0], y=hit_pos[1], z=hit_pos[2],
                        x_rot=swing_rot[0], y_rot=swing_rot[1], z_rot=swing_rot[2], w_rot=swing_rot[3]
                    )
                    last_left_pos = hit_pos
                else:
                    right_hand = VRObject(
                        x=hit_pos[0], y=hit_pos[1], z=hit_pos[2],
                        x_rot=swing_rot[0], y_rot=swing_rot[1], z_rot=swing_rot[2], w_rot=swing_rot[3]
                    )
                    last_right_pos = hit_pos
                logging.info(f"Hit at {t:.3f}s: saber={saber_type}, position=({hit_pos[0]:.2f}, {hit_pos[1]:.2f}, {hit_pos[2]:.2f}), id={note_id}, swing_magnitude={swing_magnitude:.2f}")
                continue
        if not any(next_notes.values()) and not afk_mode:
            if idle_start_time is None:
                idle_start_time = t
            transition_progress = min((t - idle_start_time) / IDLE_TRANSITION_TIME, 1.0)
            interp_times = np.array([t])
            left_positions = interpolate_movement(
                last_left_pos, IDLE_SABER_LEFT_POS, idle_start_time, idle_start_time + IDLE_TRANSITION_TIME, interp_times
            )
            right_positions = interpolate_movement(
                last_right_pos, IDLE_SABER_RIGHT_POS, idle_start_time, idle_start_time + IDLE_TRANSITION_TIME, interp_times
            )
            left_hand = left_positions[0]
            right_hand = right_positions[0]
            swing_rot = saber_swing_quaternion(8, 0.0)
            left_hand.x_rot, left_hand.y_rot, left_hand.z_rot, left_hand.w_rot = swing_rot
            right_hand.x_rot, right_hand.y_rot, right_hand.z_rot, right_hand.w_rot = swing_rot
            last_left_pos = [left_hand.x, left_hand.y, left_hand.z]
            last_right_pos = [right_hand.x, right_hand.y, right_hand.z]
        else:
            idle_start_time = None
            for saber_type, saber_notes in next_notes.items():
                if saber_notes:
                    next_time, next_pos, cut_direction, _, _ = saber_notes[0]
                    current_pos = last_left_pos if saber_type == SABER_LEFT else last_right_pos
                    interp_duration = (next_time - t) * 0.9
                    interp_times = np.linspace(t, t + interp_duration, max(20, int(interp_duration * fps * 4)))
                    positions = interpolate_movement(current_pos, next_pos, t, t + interp_duration, interp_times, swing_arc=True, pre_swing=True)
                    idx = min(len(positions) - 1, int((t - interp_times[0]) * fps * 4))
                    swing_rot = saber_swing_quaternion(cut_direction, swing_magnitude, pre_swing=(t < next_time - 0.15))
                    if saber_type == SABER_LEFT:
                        left_hand = positions[idx]
                        left_hand.x_rot, left_hand.y_rot, left_hand.z_rot, left_hand.w_rot = swing_rot
                        last_left_pos = [left_hand.x, left_hand.y, left_hand.z]
                    else:
                        right_hand = positions[idx]
                        right_hand.x_rot, right_hand.y_rot, right_hand.z_rot, right_hand.w_rot = swing_rot
                        last_right_pos = [right_hand.x, right_hand.y, right_hand.z]
        frames.append(Frame(
            time=t,
            fps=fps,
            head=VRObject(
                x=head_pos[0],
                y=head_pos[1],
                z=head_pos[2],
                x_rot=head_rot[0],
                y_rot=head_rot[1],
                z_rot=head_rot[2],
                w_rot=head_rot[3]
            ),
            left_hand=left_hand,
            right_hand=right_hand
        ))

    processed_note_indices = set()
    for idx, note in enumerate(notes):
        event_time = beats_to_seconds(note['_time'], bpm_events, base_bpm)
        spawn_time = calculate_spawn_time(event_time, njs, offset, base_bpm)
        if note['_type'] in [0, 1]:
            saber_type = SABER_LEFT if note['_type'] == 0 else SABER_RIGHT
            line_index = note['_lineIndex']
            line_layer = note['_lineLayer']
            cut_direction = note['_cutDirection']
            color_type = 0 if note['_type'] == 0 else 1
            scoring_type = NOTE_SCORE_TYPE_NORMAL
            if 'customData' in note and '_noteType' in note['customData']:
                note_type = note['customData']['_noteType']
                if 'SliderHead' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_SLIDERHEAD
                elif 'SliderTail' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_SLIDERTAIL
                elif 'BurstSliderHead' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERHEAD
                elif 'BurstSliderElement' in note_type:
                    scoring_type = NOTE_SCORE_TYPE_BURSTSLIDERELEMENT
            note_id = scoring_type * 10000 + line_index * 1000 + line_layer * 100 + color_type * 10 + cut_direction
            if idx in processed_note_indices:
                logging.warning(f'Skipping duplicate note at {event_time:.3f}s, idx={idx}')
                continue
            processed_note_indices.add(idx)
            x = line_index * 0.5 - 0.75
            y = line_layer * 0.3 + 0.7
            note_pos = [x, y, 0.0]
            swing_magnitude = note_swings.get(idx, 0.6)
            is_bad = False
            event_type = NOTE_EVENT_GOOD
            if ERROR_MODE and random.random() < MISS_CHANCE:
                event_type = NOTE_EVENT_MISS
                combo = 0
                multiplier = calculate_multiplier(combo)
                logging.info(f"Miss at {event_time:.3f}s, id={note_id}, combo={combo}, multiplier={multiplier}")
            elif ERROR_MODE and random.random() < CONFUSION_CHANCE:
                event_type = NOTE_EVENT_BAD
                is_bad = True
                combo = 0
                multiplier = calculate_multiplier(combo)
                logging.info(f"Bad cut at {event_time:.3f}s, id={note_id}, combo={combo}, multiplier={multiplier}")
            else:
                combo += 1
                multiplier = calculate_multiplier(combo)
                logging.info(f"Good cut at {event_time:.3f}s, id={note_id}, combo={combo}, multiplier={multiplier}")
            cut = None
            if event_type in [NOTE_EVENT_GOOD, NOTE_EVENT_BAD]:
                cut = Cut(saber_type=saber_type, cut_direction=cut_direction, swing_magnitude=swing_magnitude, note_pos=note_pos, note_type=note['_type'], is_bad=is_bad)
            replay_notes.append(Note(
                note_id=note_id,
                event_time=event_time,
                spawn_time=spawn_time,
                event_type=event_type,
                cut=cut,
                scoring_type=scoring_type,
                line_index=line_index,
                note_line_layer=line_layer,
                color_type=color_type,
                cut_direction=cut_direction
            ))
            if event_type == NOTE_EVENT_GOOD:
                note_score = int(15 * (1 - min(max(cut.cutDistanceToCenter / 0.3, 0), 1)) + 70 * cut.beforeCutRating + 30 * cut.afterCutRating)
                score += note_score * multiplier
                logging.info(f"Scored note hit at {event_time:.3f}s, id={note_id}, swing_magnitude={swing_magnitude:.2f}, score={note_score * multiplier}, cutPoint={cut.cutPoint}, combo={combo}, multiplier={multiplier}")
            elif event_type == NOTE_EVENT_BAD:
                score += 10
                logging.info(f"Bad cut at {event_time:.3f}s, id={note_id}, score=10, cutPoint={cut.cutPoint}, combo={combo}, multiplier={multiplier}")
        elif note['_type'] == 3:
            note_id = -1
            replay_notes.append(Note(
                note_id=note_id,
                event_time=event_time,
                spawn_time=spawn_time,
                event_type=NOTE_EVENT_BOMB,
                scoring_type=0,
                line_index=0,
                note_line_layer=0,
                color_type=0,
                cut_direction=0
            ))
            combo = 0
            multiplier = calculate_multiplier(combo)
            logging.info(f"Bomb at {event_time:.3f}s, id={note_id}, combo={combo}, multiplier={multiplier}")
    song_info['difficulty'] = difficulty
    song_info['jumpDistance'] = njs / 1.0
    info = Info(song_info, score, "", map_hash)
    return Bsor(info, frames, replay_notes)

def save_replay(bsor: Bsor, song_name: str, difficulty: str):
    invalid_chars = ['<>:"/\\|?*']
    for char in invalid_chars:
        song_name = song_name.replace(char, '')
    song_name = song_name.strip() or 'UnknownSong'
    song_dir = BASE_REPLAY_PATH / song_name
    song_dir.mkdir(parents=True, exist_ok=True)
    replay_path = song_dir / f"{difficulty}.bsor"
    with open(replay_path, 'wb') as f:
        bsor.write(f)
    return replay_path

class BeatSaberBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Beat Saber Replay Generator")
        self.root.geometry("400x300")
        
        self.map_data = None
        self.difficulties = []
        
        # Map ID Entry
        tk.Label(root, text="Beat Saver Map ID:").pack(pady=5)
        self.map_id_entry = tk.Entry(root, width=30)
        self.map_id_entry.pack(pady=5)
        
        # Error Mode Checkbox
        self.error_mode_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Allow Human Errors", variable=self.error_mode_var).pack(pady=5)
        
        # Difficulty Selection
        tk.Label(root, text="Difficulty:").pack(pady=5)
        self.difficulty_combo = ttk.Combobox(root, state="disabled", width=27)
        self.difficulty_combo.pack(pady=5)
        
        # Buttons
        tk.Button(root, text="Fetch Map", command=self.fetch_map).pack(pady=10)
        self.generate_button = tk.Button(root, text="Generate Replay", command=self.generate_replay, state="disabled")
        self.generate_button.pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(root, text="", wraplength=350)
        self.status_label.pack(pady=10)
        
    def fetch_map(self):
        map_id = self.map_id_entry.get().strip()
        if not map_id:
            messagebox.showerror("Error", "Please enter a valid Beat Saver Map ID")
            return
        
        self.status_label.config(text="Fetching map data...")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            self.map_data = download_and_extract_map(map_id)
            self.difficulties = self.map_data['difficulties']
            self.difficulty_combo['values'] = [
                f"{d['difficulty']} (BPM: {d['bpm']}, NJS: {d['njs']}, Offset: {d['offset']})"
                for d in self.difficulties
            ]
            self.difficulty_combo['state'] = 'readonly'
            self.difficulty_combo.current(0)
            self.status_label.config(text="Map fetched successfully!")
            self.generate_button.config(state="normal")
        except BSException as e:
            messagebox.showerror("Error", f"Failed to fetch map: {e}")
            self.status_label.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            self.status_label.config(text="")
        finally:
            self.root.config(cursor="")
    
    def generate_replay(self):
        if not self.map_data or not self.difficulties:
            messagebox.showerror("Error", "No map data available. Please fetch a map first.")
            return
        
        global ERROR_MODE
        ERROR_MODE = self.error_mode_var.get()
        
        selected_index = self.difficulty_combo.current()
        if selected_index < 0:
            messagebox.showerror("Error", "Please select a difficulty")
            return
        
        selected_difficulty = self.difficulties[selected_index]
        self.status_label.config(text="Generating replay...")
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            bsor = generate_replay(self.map_data, selected_difficulty)
            replay_path = save_replay(bsor, self.map_data['info']['_songName'], selected_difficulty['difficulty'])
            self.status_label.config(text=f"Replay saved to {replay_path}")
            messagebox.showinfo("Success", f"Replay generated and saved to {replay_path}")
        except BSException as e:
            messagebox.showerror("Error", f"Failed to generate replay: {e}")
            self.status_label.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            self.status_label.config(text="")
        finally:
            self.root.config(cursor="")

def main():
    root = tk.Tk()
    app = BeatSaberBotGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()