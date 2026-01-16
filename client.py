import socket
import threading
import json
import struct
import time
import os
import io
import random
import base64
import math
import sys
import ctypes
import ctypes.wintypes
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
import queue as Queue

# ===== Optional deps =====
try:
    import mss
    MSS_AVAILABLE = True
except:
    MSS_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except:
    PYAUDIO_AVAILABLE = False

# ===== Constants / Config =====
SERVER_TCP_PORT = 9000
VIDEO_UDP_PORT = 10000
AUDIO_UDP_PORT = 11000
SCREEN_TCP_PORT = 9001
FILE_TCP_PORT = 9002

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 15
VIDEO_CHUNK = 1100
JPEG_QUALITY = 78  # slightly reduced for bandwidth

AUDIO_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None
AUDIO_CHUNK = 256          # playback chunk
AUDIO_INPUT_CHUNK = 256       # capture chunk
AUDIO_JITTER_MAX = 30         # LAG FIX: Reduced from 90 to 30 (~0.5s buffer)
AUDIO_JITTER_MIN = 5          # min buffered packets to start playback

MAX_UDP_SIZE = 65507
SCREEN_TARGET_W = 1280
SCREEN_TARGET_H = 720
SCREEN_FPS = 10
SCREEN_QUALITY = 55  # JPEG quality

REACTION_DURATION_SECONDS = 2.6

MAX_USERS = 12

DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads", "ConferenceFiles")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

IS_WINDOWS = (os.name == "nt")

# ===== Safe UDP bind helper (avoid 10048) =====
def _find_free_port(preferred, host="0.0.0.0", max_tries=15):
    """Try preferred, else scan ephemeral ports."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, preferred))
        p = s.getsockname()[1]
        s.close()
        return p
    except OSError:
        pass
    # try random ephemeral
    for _ in range(max_tries):
        try_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            try_sock.bind((host, 0))
            p = try_sock.getsockname()[1]
            try_sock.close()
            return p
        except:
            try_sock.close()
            continue
    # fallback
    return preferred

# dynamic (prevent address-in-use)
LOCAL_VIDEO_LISTEN_PORT = _find_free_port(10001)
LOCAL_AUDIO_LISTEN_PORT = _find_free_port(11001)

# ====== Framing Utilities ======
def send_json_packet(sock, obj):
    try:
        data = json.dumps(obj).encode('utf-8')
        length = struct.pack('!I', len(data))
        sock.sendall(length + data)
        return True
    except:
        return False

def recv_json_packet(sock, timeout=15.0):
    try:
        sock.settimeout(timeout)
        length_data = sock.recv(4)
        if not length_data or len(length_data) < 4:
            return None
        length = struct.unpack('!I', length_data)[0]
        if length > 50 * 1024 * 1024:
            return None
        data = b''
        while len(data) < length:
            chunk = sock.recv(min(16384, length - len(data)))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode('utf-8'))
    except socket.timeout:
        return None
    except Exception:
        return None
    finally:
        try:
            sock.settimeout(None)
        except:
            pass

def create_control_packet(obj):
    return (json.dumps(obj) + "\n").encode()

# ====== Networking Sockets (global UDP) ======
video_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
video_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    video_recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
except:
    pass
video_recv_sock.bind(("0.0.0.0", LOCAL_VIDEO_LISTEN_PORT))

audio_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
audio_recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    audio_recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
except:
    pass
audio_recv_sock.bind(("0.0.0.0", LOCAL_AUDIO_LISTEN_PORT))

# ====== Windows Foreground Window helpers (no extra deps) ======
def _get_foreground_window_rect():
    """Returns (left, top, right, bottom) of foreground window on Windows."""
    if not IS_WINDOWS:
        return None
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    rect = ctypes.wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    # check iconic/minimized
    is_iconic = user32.IsIconic(hwnd)
    if is_iconic:
        return None
    # ensure rect has area
    if rect.right - rect.left < 40 or rect.bottom - rect.top < 40:
        return None
    return (rect.left, rect.top, rect.right, rect.bottom)

# ======================================================================
# ===== BACKEND: STATE MANAGER
# ======================================================================
class StateManager:
    def __init__(self):
        self.username = None
        self.my_addr = None
        self.server_ip = None
        self.connected = False
        self.sending_video = False
        self.sending_audio = False
        self.sharing_screen = False
        self.viewing_screen = False
        self.active_users_dict = {}
        self.video_states = {}
        self.screen_share_lock = threading.Lock()

    def reset(self):
        self.username = None
        self.my_addr = None
        self.server_ip = None
        self.connected = False
        self.sending_video = False
        self.sending_audio = False
        self.sharing_screen = False
        self.viewing_screen = False
        self.active_users_dict.clear()
        self.video_states.clear()

    def set_user_list(self, users, my_username):
        my_info = next((u for u in users if u.get("name") == my_username), None)
        is_first_init = self.my_addr is None
        if my_info:
            self.my_addr = my_info.get("addr")
        # Limit list to MAX_USERS-1 others for performance
        others = [u for u in users if u.get("name") != my_username]
        self.active_users_dict = {u['addr']: u for u in others[:MAX_USERS-1]}
        for u in users:
            self.video_states.setdefault(u.get('addr'), True)
        return is_first_init and self.my_addr

    def remove_user(self, addr):
        self.video_states.pop(addr, None)
        self.active_users_dict.pop(addr, None)

    def set_video_state(self, addr, is_on):
        self.video_states[addr] = is_on

# ======================================================================
# ===== BACKEND: COMMAND CONNECTION
# ======================================================================
class CommandConnection:
    def __init__(self, controller, state):
        self.controller = controller
        self.state = state
        self.main_socket = None
        self.running = True

        self.command_handlers = {
            "chat": self._handle_chat,
            "reaction": self._handle_reaction,
            "private_chat": self._handle_private_chat,
            "private_chat_sent": self._handle_private_chat_sent,
            "user_list": self._handle_user_list,
            "join": self._handle_join,
            "leave": self._handle_leave,
            "video_start": self._handle_video_state_change,
            "video_stop": self._handle_video_state_change,
            "file_offer": self._handle_file_offer,
            "error": self._handle_error,
            "present_start": self._handle_present_start,
        }

    def _apply_tcp_optimizations(self, sock):
        try: sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except: pass
        try: sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except: pass

    def connect(self, server_ip, username):
        if self.state.connected:
            return
        self.state.server_ip = server_ip
        self.state.username = username
        try:
            self.main_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._apply_tcp_optimizations(self.main_socket)
            self.main_socket.settimeout(15.0)
            self.main_socket.connect((self.state.server_ip, SERVER_TCP_PORT))
            self.main_socket.settimeout(None)

            hello = {
                "type": "hello",
                "name": self.state.username,
                "video_port": LOCAL_VIDEO_LISTEN_PORT,
                "audio_port": LOCAL_AUDIO_LISTEN_PORT
            }
            self.main_socket.sendall(create_control_packet(hello))
            self.state.connected = True
            self.controller.post_log(f"✓ Connected as {username}", "success")
            self.controller.post_task(lambda: self.controller.view.on_connection_state_changed("connected", self.state.username))

            threading.Thread(target=self._listen_loop, daemon=True, name="TCP-Control-Listen").start()
            return True
        except Exception as e:
            self.controller.post_task(lambda: self.controller.view.on_error("Connection failed", str(e)))
            self.state.reset()
            self.controller.post_task(lambda: self.controller.view.on_connection_state_changed("disconnected"))
            return False

    def disconnect(self):
        if not self.state.connected: return
        try:
            if self.main_socket:
                self.main_socket.sendall(create_control_packet({"type": "bye"}))
        except: pass
        try:
            if self.main_socket:
                try: self.main_socket.shutdown(socket.SHUT_RDWR)
                except: pass
                self.main_socket.close()
        except: pass
        self.main_socket = None
        self.state.reset()
        self.controller.post_log("✓ Left call", "system")
        self.controller.post_task(lambda: self.controller.view.on_connection_state_changed("disconnected"))

    def stop(self):
        self.running = False
        self.disconnect()

    def send_command(self, packet):
        try:
            if not self.state.connected or not self.main_socket:
                raise RuntimeError("Not connected")
            self.main_socket.sendall(packet)
            return True
        except Exception as e:
            self.controller.post_log(f"✗ Command send failed: {e}", "error")
            self.disconnect()
            return False

    def _listen_loop(self):
        buf = b""
        while self.running and self.state.connected:
            try:
                data = self.main_socket.recv(4096)
                if not data:
                    self.controller.post_log("✗ Disconnected from server", "error")
                    self.controller.disconnect()
                    break
                buf += data
                parts = buf.split(b"\n")
                buf = parts[-1]
                for part in parts[:-1]:
                    if not part: continue
                    try:
                        msg = json.loads(part.decode())
                        self._dispatch_command(msg)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                if self.state.connected:
                    self.controller.post_log(f"✗ TCP error: {e}", "error")
                    self.controller.disconnect()
                break

    def _dispatch_command(self, msg):
        mtype = msg.get("type")
        handler = self.command_handlers.get(mtype)
        if handler:
            handler(msg)
        else:
            self.controller.post_log(f"Unknown command type: {mtype}", "system")

    # --- Handlers ---
    def _handle_chat(self, msg):
        self.controller.post_task(lambda: self.controller.view.on_chat_received(f"{msg.get('from')}: {msg.get('message')}"))

    def _handle_reaction(self, msg):
        if msg.get("addr") and msg.get("emoji"):
            self.controller.post_task(lambda: self.controller.view.on_reaction_received(msg.get("addr"), msg.get("emoji")))

    def _handle_private_chat(self, msg):
        self.controller.post_task(lambda: self.controller.view.on_chat_received(f"🔒 {msg.get('from')} (private): {msg.get('message')}"))

    def _handle_private_chat_sent(self, msg):
        self.controller.post_task(lambda: self.controller.view.on_chat_received(f"🔒 You to {msg.get('to')}: {msg.get('message')}"))

    def _handle_user_list(self, msg):
        users = msg.get("users", [])
        just_initialized = self.state.set_user_list(users, self.state.username)
        if just_initialized:
            self.controller.post_log("✓ Ready to stream.", "success")
            self.controller.post_task(lambda: self.controller.view.on_ready_to_stream())
        self.controller.post_task(lambda: self.controller.view.on_users_update(list(self.state.active_users_dict.values()), self.state.my_addr))

    def _handle_join(self, msg):
        self.controller.post_log(f"→ {msg.get('name')} joined", "system")
        self.state.set_video_state(msg.get('addr'), True)

    def _handle_leave(self, msg):
        addr = msg.get('addr')
        self.controller.post_log(f"← {msg.get('name')} left", "system")
        self.state.remove_user(addr)
        self.controller.post_task(lambda: self.controller.view.on_user_leave(addr))

    def _handle_video_state_change(self, msg):
        addr = msg.get("addr")
        is_on = (msg.get("type") == "video_start")
        self.state.set_video_state(addr, is_on)
        log_msg = "started" if is_on else "stopped"
        self.controller.post_log(f"📷 {msg.get('from')} {log_msg} video", "system")
        self.controller.post_task(lambda: self.controller.view.on_video_state_change(addr, is_on))

    def _handle_file_offer(self, msg):
        self.controller.post_task(lambda: self.controller.view.on_file_offer(msg.get("from"), msg.get("filename"), msg.get("size")))

    def _handle_error(self, msg):
        error_msg = msg.get('message', '')
        self.controller.post_log(f"⚠ {error_msg}", "error")
        if "Username already taken" in error_msg:
            self.controller.disconnect()
            self.controller.post_task(lambda: self.controller.view.on_error("Error", "Username is already taken."))

    def _handle_present_start(self, msg):
        presenter_name = msg.get('from')

        # FIX: Check if the presenter is ourself. If so, do nothing.
        if presenter_name == self.state.username:
            return 
        
        self.controller.post_log(f"🖥️ {presenter_name} started presenting", "system")
        
        # This logic is now safe, as it will only run for other clients
        if not self.state.viewing_screen and not self.state.sharing_screen:
            self.controller.post_log("👁 Auto-starting screen view...", "system")
            self.controller.start_screen_view()

# ======================================================================
# ===== BACKEND: MEDIA MANAGER
# ======================================================================
class MediaManager:
    def __init__(self, controller, state):
        self.controller = controller
        self.state = state
        self.running = True

        self.pa = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.audio_play_stream = None
        self.audio_capture_stream = None
        self.audio_buffer = Queue.Queue(maxsize=AUDIO_JITTER_MAX)
        self.packet_reassembler = {}
        
        # LAG FIX: Add queue for offloading video decoding
        self.video_decode_queue = Queue.Queue(maxsize=10)

        self.screen_frame = None
        self.current_presenter_addr = None
        self.sharing_screen_sock = None
        self.viewing_screen_sock = None

        self._rx_frames = 0
        self._fps_last = time.time()
        self._fps_last_val = 0.0

        # thread flags & refs
        self._threads = {}
        self._stop_flags = {}

    # ---------- Thread helpers ----------
    def _start_thread(self, name, target):
        flag = threading.Event()
        self._stop_flags[name] = flag
        t = threading.Thread(target=target, args=(flag,), daemon=True, name=name)
        self._threads[name] = t
        t.start()

    def _stop_thread(self, name, timeout=1.5):
        flag = self._stop_flags.get(name)
        t = self._threads.get(name)
        if flag:
            flag.set()
        if t:
            t.join(timeout=timeout)
        self._stop_flags.pop(name, None)
        self._threads.pop(name, None)

    def start_media_loops(self):
        self._start_thread("UDP-Video-Recv", self._video_receive_loop)
        self._start_thread("UDP-Audio-Recv", self._audio_receive_loop)
        
        # LAG FIX: Start the dedicated decoder thread
        self._start_thread("Video-Decode", self._video_decode_loop) 
        
        self._start_thread("Audio-Playback", self._audio_playback_loop)
        self._start_thread("Stats", self._fps_stats_loop)
        if PYAUDIO_AVAILABLE:
            self._start_audio_output()

    def stop_all_streams(self):
        self.running = False
        # Reset flags
        self.state.sending_video = False
        self.state.sending_audio = False
        # Stop specific features
        self._stop_audio_capture()
        self.stop_screen_sharing()
        self.stop_screen_view()

        # kill background loops
        for name in list(self._threads.keys()):
            self._stop_thread(name)

        try:
            if self.audio_play_stream:
                self.audio_play_stream.stop_stream()
                self.audio_play_stream.close()
        except:
            pass
        if self.pa:
            try: self.pa.terminate()
            except: pass

    # ---------- Video ----------
    def toggle_video_stream(self):
        new_state = not self.state.sending_video
        # Make sure screen share is not active (so they don't fight camera draw)
        if new_state and self.state.sharing_screen:
            self.controller.post_log("Stopping screen share before starting camera...", "system")
            self.stop_screen_sharing()

        self.state.sending_video = new_state
        self.controller.post_task(lambda: self.controller.view.on_media_state_change("video", self.state.sending_video))
        if self.state.sending_video:
            self._start_thread("Cam-Send", self._video_stream_loop)
            self.controller.post_log("📷 Video started", "success")
            self.controller.connection.send_command(create_control_packet({"type": "video_start"}))
        else:
            self._stop_thread("Cam-Send")
            self.controller.post_log("📷 Video stopped", "system")
            self.controller.connection.send_command(create_control_packet({"type": "video_stop"}))
            self.controller.post_task(lambda: self.controller.view.on_video_frame(self.state.my_addr, None))

    def _video_stream_loop(self, stop_flag: threading.Event):
        cap = None
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if IS_WINDOWS else 0)
            if not cap or not cap.isOpened():
                self.controller.post_log("✗ Cannot open camera. Is it in use?", "error")
                self.state.sending_video = False
                self.controller.post_task(lambda: self.controller.view.on_media_state_change("video", False))
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            frame_id, failure_count, max_failures = 0, 0, 30
            frame_interval = 1.0 / VIDEO_FPS
            last_preview = 0.0
            preview_interval = 1.0 / 15  # update UI at max 15fps for smoothness

            while not stop_flag.is_set() and self.state.sending_video and cap.isOpened():
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    failure_count += 1
                    if failure_count > max_failures:
                        self.controller.post_log(f"✗ Camera failed {max_failures} times.", "error")
                        break
                    time.sleep(0.03)
                    continue
                failure_count = 0
                frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

                # Throttled local preview
                now = time.time()
                if now - last_preview >= preview_interval:
                    last_preview = now
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(frame_rgb)
                        self.controller.post_task(lambda p=pil: self.controller.view.on_video_frame(self.state.my_addr, p))
                    except Exception:
                        pass

                ret2, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not ret2:
                    continue
                data = jpg.tobytes()
                parts = (len(data) + VIDEO_CHUNK - 1) // VIDEO_CHUNK
                for i in range(parts):
                    chunk = data[i*VIDEO_CHUNK:(i+1)*VIDEO_CHUNK]
                    header = struct.pack('!IHH', frame_id, parts, i)
                    try:
                        video_send_sock.sendto(header + chunk, (self.state.server_ip, VIDEO_UDP_PORT))
                    except:
                        pass
                frame_id = (frame_id + 1) & 0xFFFFFFFF
                sleep_time = max(0, frame_interval - (time.time() - start))
                if sleep_time > 0:
                    # small busy wait for consistent pacing
                    end_t = time.time() + sleep_time
                    while time.time() < end_t:
                        time.sleep(0.001)

        except Exception as e:
            self.controller.post_log(f"✗ Video stream error: {e}", "error")
        finally:
            try:
                if cap: cap.release()
            except:
                pass
            self.state.sending_video = False
            self.controller.post_task(lambda: self.controller.view.on_media_state_change("video", False))
            self.controller.post_task(lambda: self.controller.view.on_video_frame(self.state.my_addr, None))

    def _video_receive_loop(self, stop_flag: threading.Event):
        while not stop_flag.is_set():
            try:
                if not self.state.connected or not self.state.my_addr:
                    time.sleep(0.05)
                    continue

                pkt, addr = video_recv_sock.recvfrom(MAX_UDP_SIZE)
                
                if not pkt or len(pkt) < 13:
                    continue
                
                try:
                    src_ip = socket.inet_ntoa(pkt[:4])
                except Exception:
                    continue 

                if src_ip == self.state.my_addr:
                    continue

                if not self.state.video_states.get(src_ip, True):
                    continue

                frame_id, total_parts, part_idx = struct.unpack("!IHH", pkt[4:12])
                payload = pkt[12:]
                key = (src_ip, frame_id)

                # --- START OF FIX ---
                now = time.time()

                # If this is the first packet of a frame, create a new entry with a timestamp
                if part_idx == 0:
                    self.packet_reassembler[key] = (now, [None] * total_parts)
                
                # Get the entry for this frame
                entry = self.packet_reassembler.get(key)

                # If we don't have an entry (e.g., joined mid-stream), drop the packet
                if entry is None:
                    continue
                
                timestamp, arr = entry
                
                # If the entry is stale (e.g., > 1 sec old), discard it
                if now - timestamp > 1.0:
                    self.packet_reassembler.pop(key, None)
                    continue
                
                # If packet metadata is corrupt or doesn't match, discard
                if part_idx >= total_parts or len(arr) != total_parts:
                    self.packet_reassembler.pop(key, None)
                    continue
                
                # Store the packet payload
                arr[part_idx] = payload

                # --- END OF FIX ---

                # If all parts are here, send to decode queue
                if all(p is not None for p in arr):
                    jpg = b"".join(self.packet_reassembler.pop(key)[1]) # Pop and get array [1]
                    try:
                        self.video_decode_queue.put((src_ip, jpg), block=False)
                    except Queue.Full:
                        pass
                        
            except Exception:
                time.sleep(0.001)

    # LAG FIX: Add the new dedicated decoder loop
    def _video_decode_loop(self, stop_flag: threading.Event):
        """
        This dedicated thread decodes JPEGs from the queue,
        preventing the network thread from blocking.
        """
        while not stop_flag.is_set():
            try:
                # Wait for a new (src_ip, jpg_bytes) tuple
                (src_ip, jpg) = self.video_decode_queue.get(timeout=0.1)

                # Perform the slow, CPU-bound decoding
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame)
                
                self._rx_frames += 1
                
                # Now, send the final ready-to-display image to the UI
                self.controller.post_task(lambda s=src_ip, p=pil: self.controller.view.on_video_frame(s, p))

            except Queue.Empty:
                continue
            except Exception as e:
                # self.controller.post_log(f"Video decode error: {e}", "error")
                time.sleep(0.01)

    def _fps_stats_loop(self, stop_flag: threading.Event):
        last_count = 0
        last_time = time.time()
        
        while not stop_flag.is_set():
            time.sleep(1.0)
            now = time.time()
            
            # Calculate FPS
            rx = self._rx_frames
            fps = max(0.0, rx - last_count) / max(0.001, (now - last_time))
            last_time = now
            last_count = rx
            self.controller.post_task(lambda f=fps: self.controller.view.on_network_stats(f))

            # --- START OF FIX ---
            # Also use this loop to clean up stale, incomplete frames
            try:
                stale_keys = [k for k, (ts, arr) in self.packet_reassembler.items() if now - ts > 1.5]
                for k in stale_keys:
                    self.packet_reassembler.pop(k, None)
            except Exception:
                pass # Ignore errors during cleanup
            # --- END OF FIX ---

    # ---------- Audio ----------
    def toggle_audio_stream(self):
        if not PYAUDIO_AVAILABLE:
            self.controller.post_task(lambda: self.controller.view.on_error("Audio library missing", "pyaudio not installed"))
            return

        new_state = not self.state.sending_audio
        self.state.sending_audio = new_state
        self.controller.post_task(lambda: self.controller.view.on_media_state_change("audio", self.state.sending_audio))
        if self.state.sending_audio:
            self._start_thread("Mic-Capture", self._audio_capture_loop)
            self.controller.post_log("🎙️ Mic ON", "success")
        else:
            self._stop_audio_capture()
            self.controller.post_log("🎙️ Mic OFF", "system")

    def _audio_capture_loop(self, stop_flag: threading.Event):
        try:
            self.audio_capture_stream = self.pa.open(
                format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE,
                input=True, frames_per_buffer=AUDIO_INPUT_CHUNK
            )
            while not stop_flag.is_set() and self.state.sending_audio:
                try:
                    data = self.audio_capture_stream.read(AUDIO_INPUT_CHUNK, exception_on_overflow=False)
                    # quick RMS for speaking ring
                    if data:
                        s = np.frombuffer(data, dtype=np.int16)
                        rms = float(np.sqrt(np.mean(s.astype(np.float32)**2)) / 32768.0)
                        self.controller.post_task(lambda r=rms: self.controller.view.on_self_speaking_energy(r))
                    if len(data) > 0 and self.state.connected:
                        audio_send_sock.sendto(data, (self.state.server_ip, AUDIO_UDP_PORT))
                except IOError as e:
                    # overflow/underrun — keep going
                    time.sleep(0.005)
                except Exception:
                    time.sleep(0.005)
        except Exception as e:
            self.controller.post_log(f"Audio capture error: {str(e)}", "error")
        finally:
            try:
                if self.audio_capture_stream:
                    self.audio_capture_stream.stop_stream()
                    self.audio_capture_stream.close()
                    self.audio_capture_stream = None
            except:
                pass
            self.state.sending_audio = False
            self.controller.post_task(lambda: self.controller.view.on_media_state_change("audio", False))
            self.controller.post_task(lambda: self.controller.view.on_self_speaking_energy(0.0))

    def _stop_audio_capture(self):
        self.state.sending_audio = False
        self._stop_thread("Mic-Capture")

    def _start_audio_output(self):
        try:
            if self.audio_play_stream:
                self.audio_play_stream.stop_stream()
                self.audio_play_stream.close()
            self.audio_play_stream = self.pa.open(
                format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE,
                output=True, frames_per_buffer=AUDIO_CHUNK
            )
            self.controller.post_log("🔊 Audio playback active", "success")
        except Exception as e:
            self.controller.post_log(f"Audio playback error: {str(e)}", "error")

    def _audio_receive_loop(self, stop_flag: threading.Event):
        # receives raw PCM packets mixed by server, push to jitter buffer
        while not stop_flag.is_set():
            try:
                # FIX: Make sure we are connected AND have our own address info
                if self.state.connected and self.state.my_addr:
                    try:
                        audio_recv_sock.settimeout(0.05)
                        pkt, addr = audio_recv_sock.recvfrom(8192) # Server's addr

                        # FIX: Packet must be > 4 bytes (IP) + audio data
                        if not pkt or len(pkt) <= 4:
                            continue
                        
                        # FIX: Extract the original sender's IP (prepended by server)
                        src_ip = socket.inet_ntoa(pkt[:4])
                        
                        # FIX: Don't play our own audio back to ourselves (no echo)
                        if src_ip == self.state.my_addr:
                            continue
                        
                        # FIX: Extract the raw audio payload
                        payload = pkt[4:]

                        try:
                            # FIX: Put only the payload into the buffer
                            self.audio_buffer.put(payload, block=False)
                        except Queue.Full:
                            # drop oldest
                            try:
                                _ = self.audio_buffer.get_nowait()
                                self.audio_buffer.put(payload, block=False)
                            except:
                                pass
                    except socket.timeout:
                        pass
                else:
                    # Not connected or ready, just sleep
                    time.sleep(0.08)
            except Exception:
                time.sleep(0.01)

    def _audio_playback_loop(self, stop_flag: threading.Event):
        primed = False
        while not stop_flag.is_set():
            try:
                if self.audio_play_stream and self.state.connected:
                    # Wait for some packets to reduce underflow
                    if not primed:
                        if self.audio_buffer.qsize() >= AUDIO_JITTER_MIN:
                            primed = True
                        else:
                            time.sleep(0.02)
                            continue
                    try:
                        pkt = self.audio_buffer.get(timeout=0.100)
                        self.audio_play_stream.write(pkt, exception_on_underflow=False)
                        # avoid runaway latency
                        if self.audio_buffer.qsize() > AUDIO_JITTER_MAX * 0.8:
                            # drop tail excess
                            drops = int(self.audio_buffer.qsize() - AUDIO_JITTER_MAX * 0.5)
                            for _ in range(max(1, drops)):
                                try: self.audio_buffer.get_nowait()
                                except: break
                    except Queue.Empty:
                        primed = False
                    except IOError:
                        primed = False
                else:
                    time.sleep(0.12)
            except Exception:
                time.sleep(0.02)

    # ---------- Screen Sharing (Window Only) ----------
    def toggle_screen_sharing(self):
        if not MSS_AVAILABLE:
            self.controller.post_task(lambda: self.controller.view.on_error("MSS not available", "Install mss library"))
            return
        # if camera is running, stop it for smooth switch
        if self.state.sending_video:
            self.controller.post_log("Stopping camera before screen share...", "system")
            self.toggle_video_stream()  # will stop cam

        with self.state.screen_share_lock:
            is_sharing = self.state.sharing_screen
        if not is_sharing:
            self.controller.post_log("🖥️ Starting window share...", "system")
            self.controller.post_task(lambda: self.controller.view.on_media_state_change("screen", "loading"))
            self._start_thread("Screen-Share", self._screen_share_run)
        else:
            self.controller.post_log("🖥️ Stopping screen share...", "system")
            self.stop_screen_sharing()

    def stop_screen_sharing(self):
        with self.state.screen_share_lock:
            self.state.sharing_screen = False
            self.screen_frame = None
            self.current_presenter_addr = None
        self.controller.post_task(lambda: self.controller.view.on_screen_frame(None, None, False))
        # Close socket async
        if threading.current_thread().name != "Screen-Share":
              self._stop_thread("Screen-Share")
        self._cleanup_share_socket()

    def _cleanup_share_socket(self):
        with self.state.screen_share_lock:
            if self.sharing_screen_sock:
                try: self.sharing_screen_sock.shutdown(socket.SHUT_RDWR)
                except: pass
                try: self.sharing_screen_sock.close()
                except: pass
                self.sharing_screen_sock = None
        self.controller.post_task(lambda: self.controller.view.on_media_state_change("screen", False))

    def _screen_share_run(self, stop_flag: threading.Event):
        # Connect as presenter
        self._cleanup_share_socket()
        time.sleep(0.1)
        try:
            self.sharing_screen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sharing_screen_sock.settimeout(10.0)
            self.sharing_screen_sock.connect((self.state.server_ip, SCREEN_TCP_PORT))

            if not send_json_packet(self.sharing_screen_sock, {"role": "presenter", "name": self.state.username}):
                raise ConnectionError("Failed to send presenter role")

            response = recv_json_packet(self.sharing_screen_sock, timeout=10.0)
            if response is None: raise ConnectionError("Server did not respond")
            if response.get("status") != "ok":
                raise ConnectionError(response.get("reason", "Server rejected request"))

            with self.state.screen_share_lock:
                self.state.sharing_screen = True

            self.current_presenter_addr = self.state.my_addr
            self.sharing_screen_sock.settimeout(None)
            self.controller.post_task(lambda: self.controller.view.on_media_state_change("screen", True))
            self.controller.post_log("✓ Window sharing started!", "success")

            # start capture loop: foreground window only
            frame_interval = 1.0 / SCREEN_FPS
            with mss.mss() as sct:
                while not stop_flag.is_set():
                    with self.state.screen_share_lock:
                        if not self.state.sharing_screen:
                            break

                    rect = _get_foreground_window_rect() if IS_WINDOWS else None
                    if rect is None:
                        # Fallback: capture entire monitor[1]
                        monitor = sct.monitors[1]
                        bbox = {
                            "left": monitor["left"],
                            "top": monitor["top"],
                            "width": monitor["width"],
                            "height": monitor["height"]
                        }
                    else:
                        l, t, r, b = rect
                        width = max(1, r - l)
                        height = max(1, b - t)
                        bbox = {"left": int(l), "top": int(t), "width": int(width), "height": int(height)}

                    start = time.time()
                    try:
                        shot = sct.grab(bbox)
                        img = Image.frombytes('RGB', shot.size, shot.rgb)
                        img = img.resize((SCREEN_TARGET_W, SCREEN_TARGET_H), Image.Resampling.LANCZOS)
                        self.screen_frame = img
                        self.controller.post_task(lambda p=img: self.controller.view.on_screen_frame(self.state.my_addr, p, True))

                        buf = io.BytesIO()
                        img.save(buf, format='JPEG', quality=SCREEN_QUALITY, optimize=True)
                        frame_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

                        if not send_json_packet(self.sharing_screen_sock, {"type": "frame", "data": frame_b64}):
                            self.controller.post_log("✗ Screen send failed", "error")
                            break
                    except Exception as e:
                        self.controller.post_log(f"✗ Screen frame error: {e}", "error")
                        break

                    elapsed = time.time() - start
                    sleep_time = max(0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

        except Exception as e:
            self.controller.post_log(f"✗ Screen share error: {e}", "error")
            self.controller.post_task(lambda e=e: self.controller.view.on_error("Screen Share", f"Connection failed: {e}"))
        finally:
    # Do NOT stop from inside the same thread — only clean socket & flag
            with self.state.screen_share_lock:
                        self.state.sharing_screen = False
            self._cleanup_share_socket()
            self.controller.post_task(lambda: self.controller.view.on_media_state_change("screen", False))
            self.controller.post_task(lambda: self.controller.view.on_screen_frame(None, None, False))

    
    # ---------- Screen Viewing ----------
    def start_screen_view(self):
        self._start_thread("Screen-View", self._screen_view_connect)

    def _screen_view_connect(self, stop_flag: threading.Event):
        self._cleanup_view_socket()
        time.sleep(0.1)
        try:
            self.viewing_screen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.viewing_screen_sock.settimeout(10.0)
            self.viewing_screen_sock.connect((self.state.server_ip, SCREEN_TCP_PORT))

            if not send_json_packet(self.viewing_screen_sock, {"role": "viewer", "name": self.state.username}):
                raise ConnectionError("Failed to send role")

            response = recv_json_packet(self.viewing_screen_sock, timeout=10.0)
            if response is None: raise ConnectionError("No response from server")
            if response.get("status") != "ok":
                raise ConnectionError(response.get("reason", "Unknown error"))

            with self.state.screen_share_lock:
                self.state.viewing_screen = True

            self.controller.post_log("👁 Viewing screen.", "success")
            self.viewing_screen_sock.settimeout(None)

            # receive loop
            while not stop_flag.is_set():
                with self.state.screen_share_lock:
                    if not self.state.viewing_screen:
                        break
                msg = recv_json_packet(self.viewing_screen_sock, timeout=5.0)
                if not msg:
                    # timeout or closed - connection is dead, stop viewing
                    self.controller.post_log("🖥️ Presenter connection lost.", "system")
                    break # <--- THIS IS THE FIX. It must be 'break'.
                
                mtype = msg.get("type")
                if mtype == "frame":
                    
                    self.current_presenter_addr = msg.get("addr")
                    try:
                        img = Image.open(io.BytesIO(base64.b64decode(msg["data"])))
                        self.screen_frame = img
                        self.controller.post_task(lambda p=img, a=self.current_presenter_addr: self.controller.view.on_screen_frame(a, p, True))
                    except Exception as e:
                        self.controller.post_log(f"Screen frame decode error: {e}", "error")
                elif mtype == "present_stop":
                    self.controller.post_log(f"🖥️ Presenter stopped", "system")
                    
                    # FIX: Don't check the address. Just stop the presentation.
                    self.screen_frame = None
                    self.current_presenter_addr = None
                    self.controller.post_task(lambda: self.controller.view.on_screen_frame(None, None, False))
                    
                    # FIX 2: Break out of this viewing loop. Its job is done.
                    break

        except Exception as e:
            if not stop_flag.is_set():
                self.controller.post_log(f"✗ View failed: {e}", "error")
                self.controller.post_task(lambda e=e: self.controller.view.on_error("View Screen", f"Failed to connect: {e}"))
        finally:
            self._cleanup_view_socket()

    def stop_screen_view(self):
        self._stop_thread("Screen-View")
        self._cleanup_view_socket()

    def _cleanup_view_socket(self):
        with self.state.screen_share_lock:
            self.state.viewing_screen = False
            self.screen_frame = None
            self.current_presenter_addr = None
            if self.viewing_screen_sock:
                try: self.viewing_screen_sock.shutdown(socket.SHUT_RDWR)
                except: pass
                try: self.viewing_screen_sock.close()
                except: pass
                self.viewing_screen_sock = None
        self.controller.post_task(lambda: self.controller.view.on_screen_frame(None, None, False))

# ======================================================================
# ===== BACKEND: MAIN CONTROLLER
# ======================================================================
class MeetingController:
    def __init__(self, view_instance):
        self.view = view_instance
        self.state = StateManager()
        self.connection = CommandConnection(self, self.state)
        self.media = MediaManager(self, self.state)
        self.media.start_media_loops()

    def post_task(self, func):
        self.view.gui_queue.put(func)

    def post_log(self, message, level):
        self.view.gui_queue.put(lambda: self.view.on_log(message, level))

    # Public API used by UI
    def connect(self, server_ip, username):
        if self.connection.connect(server_ip, username):
            if PYAUDIO_AVAILABLE:
                self.media._start_audio_output()

    def disconnect(self):
        self.connection.disconnect()
        self.media.stop_all_streams()
        self.media = MediaManager(self, self.state)
        self.media.start_media_loops()

    def stop_application(self):
        self.media.stop_all_streams()
        self.connection.stop()

    def send_chat_message(self, text, target):
        if not text: return False
        
        # FIX: Implement private chat logic
        if target != "Everyone":
            addr = next((addr for addr, u in self.state.active_users_dict.items() if u.get("name") == target), None)
            if addr:
                return self.connection.send_command(create_control_packet({"type": "private_chat", "to_addr": addr, "to_name": target, "message": text}))
            else:
                self.post_log(f"Could not find user {target}", "error")
                return False
        
        # Fallback to public chat
        return self.connection.send_command(create_control_packet({"type": "chat", "message": text}))

    def send_reaction(self, emoji):
        if not self.state.connected or not self.state.my_addr: return
        self.post_task(lambda: self.view.on_reaction_received(self.state.my_addr, emoji))
        self.connection.send_command(create_control_packet({
            "type": "reaction", "emoji": emoji, "addr": self.state.my_addr
        }))

    def upload_file(self, path):
        threading.Thread(target=self._upload_file_thread, args=(path,), daemon=True, name="File-Upload").start()

    def _upload_file_thread(self, path):
        try:
            name = os.path.basename(path); size = os.path.getsize(path)
            self.post_log(f"📤 Uploading {name}...", "system")
            file_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); file_sock.settimeout(60.0)
            file_sock.connect((self.state.server_ip, FILE_TCP_PORT))
            req = json.dumps({"type": "file_upload", "filename": name, "size": size, "from": self.state.username}).encode()
            file_sock.sendall(req)
            if file_sock.recv(10) != b"READY": raise Exception("Server not ready")
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk: break
                    file_sock.sendall(chunk)
            _ = file_sock.recv(10)
            try: file_sock.shutdown(socket.SHUT_RDWR)
            except: pass
            file_sock.close()
            self.post_log(f"✓ Upload complete: {name}", "success")
        except Exception as e:
            self.post_log(f"✗ Upload failed: {e}", "error")

    def download_file(self, filename):
        threading.Thread(target=self._download_file_thread, args=(filename,), daemon=True, name="File-Download").start()

    def _download_file_thread(self, filename):
        try:
            file_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            file_sock.settimeout(30.0)
            file_sock.connect((self.state.server_ip, FILE_TCP_PORT))
            req = json.dumps({"type": "file_download", "filename": filename}).encode()
            file_sock.sendall(req)
            info_data = file_sock.recv(4096)
            if info_data == b"ERROR": raise Exception(f"File not found: {filename}")
            info = json.loads(info_data.decode().strip())
            file_sock.sendall(b"READY")
            dest = os.path.join(DOWNLOADS_DIR, filename)
            remaining = info["size"]
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = file_sock.recv(min(65536, remaining))
                    if not chunk: break
                    f.write(chunk); remaining -= len(chunk)
            try: file_sock.shutdown(socket.SHUT_RDWR)
            except: pass
            file_sock.close()
            self.post_log(f"✓ Downloaded: {filename}", "success")
        except Exception as e:
            self.post_log(f"✗ Download error: {e}", "error")

    def toggle_video(self):
        self.media.toggle_video_stream()

    def toggle_audio(self):
        self.media.toggle_audio_stream()

    def toggle_screen_sharing(self):
        self.media.toggle_screen_sharing()

    def start_screen_view(self):
        self.media.start_screen_view()

# ======================================================================
# ===== UI: MAIN APPLICATION VIEW (Tk/CTk)
# ======================================================================
class MainAppView:
    def __init__(self, master):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.master = master
        self.master.title("📽️ ConnectSphere")
        self.master.geometry("1400x900")

        # UI State
        self.username = None
        self.my_addr = None
        self.video_feed_frames = {}
        self.screen_frame = None
        self.screen_maximized = False
        self.current_presenter_addr = None
        self.active_users_dict = {}
        self.video_states = {}
        self.active_reactions = []
        self.dark_mode = True
        self.sidebar_visible = False
        self.sidebar_is_animating = False
        self.sending_video = False
        self.sending_audio = False

        # STABILITY FIX: Add render flag and start new render loop
        self.needs_render = False
        self.render_loop() 

        self.layout_mode = "grid"   # grid | speaker | focus
        self.pinned_addr = None
        self.self_rms = 0.0

        self.gui_queue = Queue.Queue()
        self.controller = MeetingController(self)

        self.setup_ui()
        self.on_log(f"📂 Files save to: {DOWNLOADS_DIR}", "system")

        self.update_gui_queue()
        self.update_reaction_animations()

        self.master.bind("<Configure>", self.handle_resize)
        self.master.protocol("WM_DELETE_WINDOW", self.handle_close)

        # Keyboard shortcuts
        self.master.bind("m", lambda e: self.toggle_mic())
        self.master.bind("M", lambda e: self.toggle_mic())
        self.master.bind("v", lambda e: self.toggle_camera())
        self.master.bind("V", lambda e: self.toggle_camera())
        self.master.bind("s", lambda e: self.toggle_screen_sharing_ui())
        self.master.bind("S", lambda e: self.toggle_screen_sharing_ui())
        self.master.bind("c", lambda e: self.toggle_panel("💬 Messages"))
        self.master.bind("u", lambda e: self.toggle_panel("👥 Attendees"))
        self.master.bind("p", lambda e: self.toggle_pin_hotkey())
        self.master.bind("<Escape>", lambda e: self.exit_meeting())

        self.connection_overlay_frame.place(relx=0.5, rely=0.5, anchor="center")

    # ====== Build UI ======
    def setup_ui(self):
        self.main_container = ctk.CTkFrame(self.master, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)

        self.main_container.grid_rowconfigure(0, weight=0)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=0)
        self.main_container.grid_columnconfigure(1, weight=1)

        self._build_navbar()
        self._build_control_bar()
        self._build_main_content()
        self._build_sidebar()
        self._build_connection_overlay()

    def _build_navbar(self):
        self.nav_bar = ctk.CTkFrame(self.main_container, height=60, corner_radius=0, fg_color="gray17")
        self.nav_bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.nav_bar.grid_columnconfigure(0, weight=1)
        self.nav_bar.grid_columnconfigure(1, weight=0)
        self.nav_bar.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(self.nav_bar, text="ConnectSphere", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=10, sticky="w")

        status_frame = ctk.CTkFrame(self.nav_bar, fg_color="transparent")
        status_frame.grid(row=0, column=1, padx=20, pady=10, sticky="ew")
        self.status_indicator = ctk.CTkLabel(status_frame, text="● LIVE", font=ctk.CTkFont(size=14, weight="bold"), text_color="#ef4444")
        self.status_indicator.pack(side="left", padx=5)
        self.status_label = ctk.CTkLabel(status_frame, text="Disconnected", font=ctk.CTkFont(size=13))
        self.status_label.pack(side="left", padx=5)
        
        # FONT FIX: Add Emoji font
        self.users_label = ctk.CTkLabel(status_frame, text="👤 0 users", font=ctk.CTkFont(family="Segoe UI Emoji", size=13))
        self.users_label.pack(side="left", padx=15)
        self.net_label = ctk.CTkLabel(status_frame, text="🟡 Net: -- fps", font=ctk.CTkFont(family="Segoe UI Emoji", size=13))
        self.net_label.pack(side="left", padx=10)

        nav_right_frame = ctk.CTkFrame(self.nav_bar, fg_color="transparent")
        nav_right_frame.grid(row=0, column=2, padx=20, pady=10, sticky="e")

        # FONT FIX: Add Emoji font
        self.layout_grid_btn = ctk.CTkButton(nav_right_frame, text="▦", width=36, height=36, corner_radius=18, command=lambda: self.set_layout("grid"), font=ctk.CTkFont(family="Segoe UI Emoji", size=18))
        self.layout_speaker_btn = ctk.CTkButton(nav_right_frame, text="▭", width=36, height=36, corner_radius=18, command=lambda: self.set_layout("speaker"), font=ctk.CTkFont(family="Segoe UI Emoji", size=18))
        self.layout_focus_btn = ctk.CTkButton(nav_right_frame, text="◎", width=36, height=36, corner_radius=18, command=lambda: self.set_layout("focus"), font=ctk.CTkFont(family="Segoe UI Emoji", size=18))
        self.layout_grid_btn.pack(side="left", padx=6)
        self.layout_speaker_btn.pack(side="left", padx=6)
        self.layout_focus_btn.pack(side="left", padx=6)

        self.settings_btn = ctk.CTkButton(nav_right_frame, text="⚙", width=40, height=40, corner_radius=20, command=self.open_settings, font=ctk.CTkFont(family="Segoe UI Emoji", size=20))
        self.settings_btn.pack(side="left", padx=10)
        self.theme_btn = ctk.CTkButton(nav_right_frame, text="🌚", width=40, height=40, corner_radius=20, command=self.switch_theme, font=ctk.CTkFont(family="Segoe UI Emoji", size=20))
        self.theme_btn.pack(side="left", padx=10)

    def _build_control_bar(self):
        self.vertical_control_bar = ctk.CTkFrame(self.main_container, width=80, fg_color="gray17", corner_radius=0)
        self.vertical_control_bar.grid(row=1, column=0, sticky="ns")
        ctk.CTkFrame(self.vertical_control_bar, height=20, fg_color="transparent").pack(side="top")

        # FONT FIX: Add Emoji font
        font_awesome = ctk.CTkFont(family="Segoe UI Emoji", size=24)
        self.audio_btn = ctk.CTkButton(self.vertical_control_bar, text="🎙️", width=50, height=50, corner_radius=25, font=font_awesome, command=self.toggle_mic, fg_color="#374151", hover_color="#4b5563", state="disabled")
        self.audio_btn.pack(side="top", pady=6)
        self.video_btn = ctk.CTkButton(self.vertical_control_bar, text="📷", width=50, height=50, corner_radius=25, font=font_awesome, command=self.toggle_camera, fg_color="#374151", hover_color="#4b5563", state="disabled")
        self.video_btn.pack(side="top", pady=6)
        self.screen_btn = ctk.CTkButton(self.vertical_control_bar, text="🖥️", width=50, height=50, corner_radius=25, font=font_awesome, command=self.toggle_screen_sharing_ui, fg_color="#374151", hover_color="#4b5563", state="disabled")
        self.screen_btn.pack(side="top", pady=6)
        self.chat_toggle_btn = ctk.CTkButton(self.vertical_control_bar, text="💬", width=50, height=50, corner_radius=25, font=font_awesome, command=lambda: self.toggle_panel("💬 Messages"), fg_color="#374151", hover_color="#4b5563", state="disabled")
        self.chat_toggle_btn.pack(side="top", pady=6)
        self.users_toggle_btn = ctk.CTkButton(self.vertical_control_bar, text="👤", width=50, height=50, corner_radius=25, font=font_awesome, command=lambda: self.toggle_panel("👥 Attendees"), fg_color="#374151", hover_color="#4b5563", state="disabled")
        self.users_toggle_btn.pack(side="top", pady=6)
        
        ctk.CTkFrame(self.vertical_control_bar, fg_color="transparent").pack(side="top", fill="y", expand=True)
        self.leave_btn = ctk.CTkButton(self.vertical_control_bar, text="Exit", width=60, height=50, corner_radius=25, font=ctk.CTkFont(size=16, weight="bold"), command=self.exit_meeting, fg_color="#ef4444", hover_color="#dc2626", state="disabled")
        self.leave_btn.pack(side="bottom", pady=20)

    def _build_main_content(self):
        # FIX 1: Create the content_frame and place it in the main grid
        self.content_frame = ctk.CTkFrame(self.main_container, fg_color="transparent", corner_radius=0)
        self.content_frame.grid(row=1, column=1, sticky="nsew")

        # FIX 3: Define the initial background color for the canvas
        canvas_bg_color = "#0a0a0a" # Default dark mode background

        # This line will now work
        self.video_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.video_frame.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(self.video_frame, bg=canvas_bg_color, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.splitter = tk.Frame(self.content_frame, cursor="sb_h_double_arrow", bg="#444444", width=6)
        self.splitter.pack(side="left", fill="y")
        self.splitter.bind("<Button-1>", self._start_resize)
        self.splitter.bind("<B1-Motion>", self._perform_resize)
        
        # FIX 2: REMOVED self.sidebar_frame.pack(side="left", fill="y") from here

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self._hover_tile = None

    def _build_sidebar(self):
        # This line will now work because content_frame was created in the previous step
        self.sidebar_frame = ctk.CTkFrame(self.content_frame, width=0, corner_radius=0, fg_color=("gray90", "gray17"))
        
        # FIX 2: Pack the sidebar frame right after it's created
        self.sidebar_frame.pack(side="left", fill="y") 

        self.sidebar_target_width = 350

        # FONT FIX: Remove bad font argument from TabView
        self.tabview = ctk.CTkTabview(self.sidebar_frame, corner_radius=0, fg_color=("gray90", "gray17"), segmented_button_fg_color=("gray90", "gray17"), segmented_button_selected_color="#3b82f6", segmented_button_unselected_color=("gray80", "gray20"))
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)

        # Chat Tab
        chat_tab = self.tabview.add("💬 Messages")
        chat_tab.configure(fg_color=("gray85", "gray14"))
        target_frame = ctk.CTkFrame(chat_tab, fg_color="transparent")
        target_frame.pack(fill="x", pady=(10, 10), padx=10)
        ctk.CTkLabel(target_frame, text="To:", font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        self.chat_target = ctk.CTkComboBox(target_frame, values=["Everyone"], width=150)
        self.chat_target.pack(side="left", fill="x", expand=True, padx=5)
        self.chat_target.set("Everyone")
        chat_display_frame = ctk.CTkFrame(chat_tab, corner_radius=0, fg_color=("gray80", "gray20"))
        chat_display_frame.pack(fill="both", expand=True, pady=0, padx=10)
        self.chat_text = ctk.CTkTextbox(chat_display_frame, wrap="word", font=ctk.CTkFont(size=12), fg_color="transparent")
        self.chat_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # FONT FIX: Add Emoji font
        emoji_frame = ctk.CTkFrame(chat_tab, fg_color="transparent")
        emoji_frame.pack(fill="x", pady=10, padx=10)
        emoji_font = ctk.CTkFont(family="Segoe UI Emoji", size=18)
        for i, emoji in enumerate(["😄","🌟","💖","😢","🙌","👍","😊","👌","🎉","💯"]):
            ctk.CTkButton(emoji_frame, text=emoji, width=35, height=35, corner_radius=17, font=emoji_font, command=lambda e=emoji: self.send_reaction(e), fg_color="transparent", hover_color="#374151").grid(row=i // 5, column=i % 5, padx=3, pady=3, sticky="ew")
        emoji_frame.grid_columnconfigure(list(range(5)), weight=1)
        
        input_frame = ctk.CTkFrame(chat_tab, fg_color="transparent")
        input_frame.pack(fill="x", pady=10, padx=10)
        self.msg_entry = ctk.CTkEntry(input_frame, placeholder_text="Type a message...", height=40, corner_radius=20)
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.msg_entry.bind('<Return>', lambda e: self.send_message())
        ctk.CTkButton(input_frame, text="➤", width=70, height=40, corner_radius=20, command=self.send_message, fg_color="#10b981", hover_color="#059669").pack(side="right")

        # Users Tab
        users_tab = self.tabview.add("👥 Attendees")
        users_tab.configure(fg_color=("gray85", "gray14"))
        users_display_frame = ctk.CTkFrame(users_tab, corner_radius=0, fg_color=("gray80", "gray20"))
        users_display_frame.pack(fill="both", expand=True, pady=10, padx=10)
        self.users_textbox = ctk.CTkTextbox(users_display_frame, wrap="word", font=ctk.CTkFont(size=13), fg_color="transparent")
        self.users_textbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Files/Log Tab
        files_tab = self.tabview.add("📎 Files")
        files_tab.configure(fg_color=("gray85", "gray14"))
        files_content_frame = ctk.CTkFrame(files_tab, fg_color="transparent")
        files_content_frame.pack(fill="both", expand=True, pady=10, padx=10)
        
        # FONT FIX: Add Emoji font
        button_font = ctk.CTkFont(family="Segoe UI Emoji", size=14, weight="bold")
        self.upload_btn = ctk.CTkButton(files_content_frame, text="📤 Upload File", height=50, corner_radius=10, command=self.prompt_upload_file, fg_color="#3b82f6", hover_color="#2563eb", font=button_font, state="disabled")
        self.upload_btn.pack(fill="x", pady=10)
        ctk.CTkButton(files_content_frame, text="📂 View Downloads", height=50, corner_radius=10, command=self.open_downloads_folder, fg_color="#8b5cf6", hover_color="#7c3aed", font=button_font).pack(fill="x", pady=10)
        
        log_frame = ctk.CTkFrame(files_content_frame, corner_radius=10, fg_color=("gray80", "gray20"))
        log_frame.pack(fill="both", expand=True, pady=(10, 0))
        ctk.CTkLabel(log_frame, text="📋 Event Log", font=ctk.CTkFont(family="Segoe UI Emoji", size=14, weight="bold")).pack(pady=5)
        self.log_textbox = ctk.CTkTextbox(log_frame, wrap="word", font=ctk.CTkFont(size=11), fg_color="transparent")
        self.log_textbox.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_connection_overlay(self):
        self.connection_overlay_frame = ctk.CTkFrame(self.master, corner_radius=20, fg_color="gray17")
        ctk.CTkLabel(self.connection_overlay_frame, text="Join Call", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20, padx=50)
        self.ip_entry = ctk.CTkEntry(self.connection_overlay_frame, placeholder_text="Server IP", width=250, height=40, corner_radius=10)
        self.ip_entry.pack(pady=10, padx=30)
        self.ip_entry.insert(0, "127.0.0.1")
        try:
            default_name = os.getlogin()
        except:
            default_name = "user"
        self.name_entry = ctk.CTkEntry(self.connection_overlay_frame, placeholder_text="Username", width=250, height=40, corner_radius=10)
        self.name_entry.pack(pady=10, padx=30)
        self.name_entry.insert(0, default_name)
        self.connect_btn = ctk.CTkButton(self.connection_overlay_frame, text="Join", width=250, height=40, command=self.join_meeting, corner_radius=20, fg_color="#10b981", hover_color="#059669", font=ctk.CTkFont(size=14, weight="bold"))
        self.connect_btn.pack(pady=20, padx=30)

    # ====== Settings Dialog ======
    def open_settings(self):
        win = ctk.CTkToplevel(self.master)
        win.title("Device Settings")
        win.geometry("460x420")
        ctk.CTkLabel(win, text="Devices", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)

        cam_frame = ctk.CTkFrame(win); cam_frame.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(cam_frame, text="Camera Index:").pack(side="left", padx=6, pady=10)
        self.cam_index_entry = ctk.CTkEntry(cam_frame, width=60)
        self.cam_index_entry.insert(0, "0")
        self.cam_index_entry.pack(side="left")

        mic_var = tk.StringVar(value="Default")
        out_var = tk.StringVar(value="Default")

        if PYAUDIO_AVAILABLE:
            try:
                pa = pyaudio.PyAudio()
                inputs, outputs = [], []
                for i in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(i)
                    name = f"{i}: {info.get('name', 'Device')}"
                    if info.get('maxInputChannels', 0) > 0: inputs.append(name)
                    if info.get('maxOutputChannels', 0) > 0: outputs.append(name)
            except:
                inputs, outputs = [], []
            finally:
                try: pa.terminate()
                except: pass
        else:
            inputs, outputs = [], []

        mic_frame = ctk.CTkFrame(win); mic_frame.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(mic_frame, text="Microphone:").pack(side="left", padx=6, pady=10)
        mic_list = ["Default"] + inputs
        self.mic_combo = ctk.CTkComboBox(mic_frame, values=mic_list, variable=mic_var, width=280)
        self.mic_combo.pack(side="left", padx=6)

        spk_frame = ctk.CTkFrame(win); spk_frame.pack(fill="x", padx=12, pady=6)
        ctk.CTkLabel(spk_frame, text="Speaker:").pack(side="left", padx=6, pady=10)
        spk_list = ["Default"] + outputs
        self.spk_combo = ctk.CTkComboBox(spk_frame, values=spk_list, variable=out_var, width=280)
        self.spk_combo.pack(side="left", padx=6)

        tips = ("Tips:\n"
                "• If camera fails, try another index (0..3).\n"
                "• Mic/Speaker lists are informational; pipeline uses system defaults.\n"
                "• Shortcuts: M (mic), V (video), S (screen), C (chat), U (users), P (pin).")
        ctk.CTkLabel(win, text=tips, justify="left").pack(padx=12, pady=10)

        ctk.CTkButton(win, text="Close", command=win.destroy).pack(pady=10)

    # ====== UI Event Handlers ======
    def join_meeting(self):
        self.connect_btn.configure(text="Joining...", state="disabled")
        ip = self.ip_entry.get().strip()
        name = self.name_entry.get().strip()
        if not ip or not name:
            messagebox.showerror("Error", "IP and Username are required.")
            self.connect_btn.configure(text="Join", state="normal")
            return
        threading.Thread(target=self.controller.connect, args=(ip, name), daemon=True, name="Join-Thread").start()

    def exit_meeting(self):
        if messagebox.askyesno("Leave Call", "Are you sure you want to leave?"):
            # FIX: Don't block the UI. Run disconnect in a new thread.
            threading.Thread(target=self.controller.disconnect, daemon=True, name="Disconnect-Thread").start()

    def handle_close(self):
        # FIX: Run the heavy stop_application in a thread
        # and then schedule the final window close from that thread.
        def _stop_and_destroy():
            self.controller.stop_application()
            # We can't call destroy() from a background thread,
            # so we post it back to the (now-free) GUI queue.
            self.gui_queue.put(self.master.destroy)

        # Start the shutdown thread
        threading.Thread(target=_stop_and_destroy, daemon=True, name="Stop-App-Thread").start()

    def send_message(self):
        text = self.msg_entry.get().strip()
        if not text: return
        target = self.chat_target.get()
        if self.controller.send_chat_message(text, target):
            self.msg_entry.delete(0, "end")

    def send_reaction(self, emoji):
        self.controller.send_reaction(emoji)

    def prompt_upload_file(self):
        path = filedialog.askopenfilename(title="Select File")
        if not path: return
        self.controller.upload_file(path)

    def toggle_camera(self):
        self.controller.toggle_video()

    def toggle_mic(self):
        self.controller.toggle_audio()

    def toggle_screen_sharing_ui(self):
        self.controller.toggle_screen_sharing()

    # ====== Controller Hooks ======
    def on_log(self, text, msg_type="default"):
        try:
            timestamp = time.strftime("%H:%M:%S")
            self.log_textbox.insert("end", f"[{timestamp}] {text}\n")
            self.log_textbox.see("end")
        except:
            pass

    def on_error(self, title, message):
        messagebox.showerror(title, message)

    def on_connection_state_changed(self, state, username=None):
        if state == "connected":
            self.username = username
            self.status_label.configure(text="Connected")
            self.status_indicator.configure(text_color="#10b981")
            self.leave_btn.configure(state="normal")
            self.connection_overlay_frame.place_forget()
            self.connect_btn.configure(text="Join", state="normal")
        elif state == "disconnected":
            self.disconnect_cleanup_ui()

    def on_ready_to_stream(self):
        self.video_btn.configure(state="normal")
        self.audio_btn.configure(state="normal")
        self.screen_btn.configure(state="normal")
        self.chat_toggle_btn.configure(state="normal")
        self.users_toggle_btn.configure(state="normal")
        self.upload_btn.configure(state="normal")

    def on_users_update(self, users_list, my_addr):
        self.my_addr = my_addr
        self.active_users_dict = {u['addr']: u for u in users_list}
        for u in users_list:
            self.video_states.setdefault(u.get('addr'), True)
        active_users_names = [u.get("name") for u in users_list]
        self.chat_target.configure(values=['Everyone'] + active_users_names)
        if self.chat_target.get() not in (['Everyone'] + active_users_names):
            self.chat_target.set("Everyone")
        self.update_attendees_display()
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def on_user_leave(self, addr):
        self.video_states.pop(addr, None)
        self.video_feed_frames.pop(addr, None)
        self.active_users_dict.pop(addr, None)
        self.active_reactions = [r for r in self.active_reactions if r.get("addr") != addr]
        if self.pinned_addr == addr: self.pinned_addr = None
        self.update_attendees_display()
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def on_chat_received(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.chat_text.insert("end", f"[{timestamp}] {text}\n")
        self.chat_text.see("end")

    def on_reaction_received(self, addr, emoji):
        try:
            canvas_width = self.canvas.winfo_width()
            self.active_reactions.append({
                "emoji": emoji, "addr": addr, "start_time": time.time(),
                "x_pos": random.randint(int(canvas_width*0.1), int(canvas_width*0.9))
            })
        except:
            pass

    def on_file_offer(self, from_user, filename, size):
        size_mb = size / (1024 * 1024)
        msg = f"{from_user} wants to share:\n\n{filename}\nSize: {size_mb:.2f} MB\n\nDownload?"
        if messagebox.askyesno("File Offer", msg):
            self.on_log(f"📥 Downloading {filename}...", "system")
            self.controller.download_file(filename)
        else:
            self.on_log(f"✗ Declined {filename}", "system")

    def on_video_frame(self, addr, pil_image):
        if pil_image is None:
            self.video_feed_frames.pop(addr, None)
        else:
            self.video_feed_frames[addr] = pil_image
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def on_screen_frame(self, addr, pil_image, is_maximized):
        self.current_presenter_addr = addr
        self.screen_frame = pil_image
        self.screen_maximized = is_maximized
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def on_video_state_change(self, addr, is_on):
        self.video_states[addr] = is_on
        if not is_on:
            self.video_feed_frames.pop(addr, None)
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def on_media_state_change(self, media_type, state):
        if media_type == "video":
            btn, on_color, off_color = self.video_btn, "#dc2626", "#374151"
            self.sending_video = state
        elif media_type == "audio":
            btn, on_color, off_color = self.audio_btn, "#dc2626", "#374151"
            self.sending_audio = state
        elif media_type == "screen":
            btn, on_color, off_color = self.screen_btn, "#dc2626", "#374151"
            if state == "loading":
                btn.configure(text="⏳", fg_color="#d97706", state="disabled"); return
            else:
                btn.configure(text="🖥️", state="normal")
        else:
            return
        if state: btn.configure(fg_color=on_color, hover_color="#b91c1c")
        else: btn.configure(fg_color=off_color, hover_color="#4b5563")

    def on_self_speaking_energy(self, rms):
        self.self_rms = float(rms)
        if rms > 0.02:
            # STABILITY FIX: Request render, don't demand it
            self.gui_queue.put(self.set_render_flag)

    def on_network_stats(self, fps):
        q = "🟢" if fps >= 10 else ("🟡" if fps >= 5 else "🔴")
        self.net_label.configure(text=f"{q} Net: {fps:.1f} fps")

    # ====== Core UI Loops & Renderers ======
    def update_gui_queue(self):
        # FIX: Set a max number of tasks to process per 20ms tick.
        # This prevents UI starvation and the "Not Responding" error.
        max_tasks_per_tick = 20 
        tasks_processed = 0

        try:
            # FIX: Loop only up to the max_tasks limit, not "while True"
            while tasks_processed < max_tasks_per_tick:
                callback = self.gui_queue.get_nowait()
                if callable(callback): 
                    callback()
                tasks_processed += 1
        except Queue.Empty:
            # This is normal, just means the queue is empty
            pass
        finally:
            # Always reschedule the next check
            self.master.after(20, self.update_gui_queue)

    def update_reaction_animations(self):
        now = time.time()
        self.active_reactions = [r for r in self.active_reactions if (now - r["start_time"]) < REACTION_DURATION_SECONDS]
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)
        self.master.after(60, self.update_reaction_animations)

    # -----------------------------------------------------------------
    # ===== STABILITY FIX: ADD THESE TWO NEW METHODS =====
    # -----------------------------------------------------------------
    
    def set_render_flag(self):
        """
        A very fast, cheap function called by backend threads.
        It just sets a flag that the canvas needs a redraw.
        """
        self.needs_render = True

    def render_loop(self):
        """
        This is the new master render loop.
        It runs at a fixed rate (~30 FPS) and only redraws the canvas
        if the 'needs_render' flag has been set.
        This prevents the GUI queue from being spammed with thousands
        of expensive render calls.
        """
        try:
            if self.needs_render:
                self.needs_render = False # Reset the flag
                self.render_canvas()      # Perform the single, expensive render
        finally:
            # Schedule the next check in 33ms (approx 30 FPS)
            self.master.after(33, self.render_loop) 
            
    # -----------------------------------------------------------------
    # ================= END OF NEW METHODS ==================
    # -----------------------------------------------------------------

    def render_canvas(self):
        try:
            self.canvas.delete("all")
        except:
            return
        if not hasattr(self.canvas, '_imgs'):
            self.canvas._imgs = []
        self.canvas._imgs.clear()
        # reset hitboxes
        self._tile_hitboxes = []

        if self.screen_maximized and self.screen_frame:
            self._render_presenter_view()
        else:
            if self.layout_mode == "grid":
                self._render_grid_view()
            elif self.layout_mode == "speaker":
                self._render_speaker_view()
            elif self.layout_mode == "focus":
                self._render_focus_view()
            else:
                self._render_grid_view()

        self._draw_floating_reactions()

        # hover overlay controls
        if self._hover_tile:
            self._draw_tile_overlay(*self._hover_tile)

    # ==== Layouts ====
    def _participants_list(self):
        users = []
        if self.my_addr:
            users.append({"addr": self.my_addr, "name": f"{self.username} (You)", "is_self": True})
        users.extend(self.active_users_dict.values())
        return users[:MAX_USERS]

    def _render_grid_view(self):
        users_to_draw = self._participants_list()
        n = len(users_to_draw)
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if n == 0:
            if not self.connection_overlay_frame.winfo_viewable():
                color = "#666" if self.dark_mode else "#999"
                self.canvas.create_text(w/2, h/2, text="No one else is here.", fill=color, font=('Segoe UI', 16))
            return
        if n == 1: cols, rows = 1, 1
        elif n == 2: cols, rows = 2, 1
        elif n <= 4: cols, rows = 2, 2
        elif n <= 6: cols, rows = 3, 2
        elif n <= 9: cols, rows = 3, 3
        else:
            cols = 4; rows = int(np.ceil(n / cols))

        cw = max(1, w // cols); ch = max(1, h // rows)
        for idx, user in enumerate(users_to_draw):
            r, c = divmod(idx, cols)
            x, y = c * cw, r * ch
            self._render_video_tile(user, x, y, cw, ch)

    def _render_speaker_view(self):
        users = self._participants_list()
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if not users: return
        main_h = int(h * 0.75)
        if self.pinned_addr:
            main_user = next((u for u in users if u.get("addr")==self.pinned_addr), users[0])
        else:
            main_user = users[0]
        self._render_video_tile(main_user, 0, 0, w, main_h)

        others = [u for u in users if u is not main_user]
        if not others: return
        strip_h = h - main_h
        tile_w = max(1, w // max(1, len(others)))
        x = 0
        for u in others:
            self._render_video_tile(u, x, main_h, tile_w, strip_h)
            x += tile_w

    def _render_focus_view(self):
        users = self._participants_list()
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        if not users: return
        if self.pinned_addr:
            focus = next((u for u in users if u.get("addr")==self.pinned_addr), users[0])
        else:
            focus = users[0]
        pad = int(min(w, h) * 0.05)
        self._render_video_tile(focus, pad, pad, w - 2*pad, h - 2*pad)

    def _render_presenter_view(self):
        users = self._participants_list()
        w = self.canvas.winfo_width(); h = self.canvas.winfo_height()
        rail_w = int(w * 0.20)
        screen_w = w - rail_w; screen_h = h

        try:
            img = self.screen_frame.resize((screen_w, screen_h), Image.Resampling.LANCZOS)
            tkimg = ImageTk.PhotoImage(img)
            self.canvas.create_image(screen_w/2, screen_h/2, image=tkimg)
            self.canvas._imgs.append(tkimg)
        except:
            pass

        self.canvas.create_rectangle(screen_w, 0, w, h, fill=("#1a1a1a" if self.dark_mode else "#e5e5e5"), outline="")
        if len(users) > 0:
            tile_w = rail_w - 10
            tile_h = int(tile_w * (VIDEO_HEIGHT / VIDEO_WIDTH))
            y = 5
            for u in users:
                self._render_video_tile(u, screen_w + 5, y, tile_w, tile_h)
                y += tile_h + 5
                if y > h: break

    # ==== Reactions ====
    def _draw_floating_reactions(self):
        try:
            now = time.time()
            H = self.canvas.winfo_height()
            for r in self.active_reactions:
                age = now - r["start_time"]; T = REACTION_DURATION_SECONDS
                t = max(0.0, min(1.0, age / T))
                ease = 1 - (1 - t)**3
                y = H * (1.0 - 0.75*ease - 0.05)
                x = r["x_pos"]
                size = int(32 - 14*ease)
                self.canvas.create_text(x, y, text=r["emoji"], font=('Segoe UI Emoji', size, 'bold'), fill="#FFD700", anchor=tk.CENTER)
        except Exception:
            pass

    # ==== Tiles ====
    def _render_video_tile(self, user, x, y, cw, ch):
        addr = user.get("addr"); name = user.get("name")
        is_self = user.get("is_self", False)
        video_on = self.video_states.get(addr, True)
        if is_self: video_on = self.sending_video
        pil = self.video_feed_frames.get(addr)

        border = "#333333" if self.dark_mode else "#cccccc"
        bg = "#101214" if self.dark_mode else "#eaeaea"
        text_col = "white"

        r = 16
        self._rounded_rect(x+4, y+4, x+cw-4, y+ch-4, r, fill=bg, outline=border)

        if pil and video_on:
            try:
                iw, ih = pil.size
                scale = min(max(0.01, (cw - 12) / iw), max(0.01, (ch - 50) / ih))
                nw, nh = max(1, int(iw*scale)), max(1, int(ih*scale))
                img = pil.resize((nw, nh), Image.Resampling.BILINEAR)
                tkimg = ImageTk.PhotoImage(img)
                self.canvas.create_image(x + cw/2, y + ch/2 - 12, image=tkimg)
                self.canvas._imgs.append(tkimg)
            except:
                pass
        else:
            placeholder = f"{name}\n(Video Off)"
            self.canvas.create_text(x+cw/2, y+ch/2 - 10, text=placeholder, fill="#8b8b8b", font=('Segoe UI', 12, 'bold'), justify=tk.CENTER)

        badge_y = y + ch - 26
        label = name
        self.canvas.create_text(x + 14, badge_y, text=label, fill=text_col, anchor=tk.SW, font=('Segoe UI', 11, 'bold'))

        bx = x + cw - 60
        by = badge_y - 4
        if is_self:
            mic_icon = "🎤" if self.sending_audio else "🔇"
            cam_icon = "📷" if self.sending_video else "🚫"
            
            # FONT FIX: Use Emoji font
            self.canvas.create_text(bx, by, text=mic_icon, fill="#ddd", font=('Segoe UI Emoji', 12, 'bold'))
            self.canvas.create_text(bx+24, by, text=cam_icon, fill="#ddd", font=('Segoe UI Emoji', 12, 'bold'))

        if is_self and self.self_rms > 0.035:
            glow = "#22c55e"
            self._rounded_rect(x+2, y+2, x+cw-2, y+ch-2, r+2, outline=glow, width=3)

        self._tile_hitboxes.append((x, y, cw, ch, addr, name))

    def _rounded_rect(self, x1, y1, x2, y2, r, fill=None, outline=None, width=2):
        # draw rounded rectangle using basic shapes (fast)
        self.canvas.create_arc(x1, y1, x1+2*r, y1+2*r, start=90, extent=90, fill=fill, outline=outline, width=width)
        self.canvas.create_arc(x2-2*r, y1, x2, y1+2*r, start=0, extent=90, fill=fill, outline=outline, width=width)
        self.canvas.create_arc(x1, y2-2*r, x1+2*r, y2, start=180, extent=90, fill=fill, outline=outline, width=width)
        
        # SYNTAX FIX: Was 9VCS, changed to 90
        self.canvas.create_arc(x2-2*r, y2-2*r, x2, y2, start=270, extent=90, fill=fill, outline=outline, width=width)
        
        self.canvas.create_rectangle(x1+r, y1, x2-r, y2, fill=fill, outline=outline, width=width)
        self.canvas.create_rectangle(x1, y1+r, x2, y2-r, fill=fill, outline=outline, width=width)

    # ==== Hover overlay + Pin (fixed color alpha issue) ====
    def _on_canvas_motion(self, event):
        self._hover_tile = None
        for (x, y, cw, ch, addr, name) in getattr(self, "_tile_hitboxes", []):
            if x <= event.x <= x+cw and y <= event.y <= y+ch:
                self._hover_tile = (x, y, cw, ch, addr, name)
                break
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def _on_canvas_click(self, event):
        for (x, y, cw, ch, addr, name) in getattr(self, "_tile_hitboxes", []):
            if x <= event.x <= x+cw and y <= event.y <= y+ch:
                if self.pinned_addr == addr: self.pinned_addr = None
                else: self.pinned_addr = addr
                if self.pinned_addr and self.layout_mode != "focus":
                    self.layout_mode = "focus"
                
                # STABILITY FIX: Request render, don't demand it
                self.gui_queue.put(self.set_render_flag)
                break

    # -----------------------------------------------------------------
    # ===== ADD THESE TWO NEW METHODS FOR THE SPLITTER =====
    # -----------------------------------------------------------------
    def _start_resize(self, event):
        """Stores the starting X position of the splitter drag."""
        self.splitter_start_x = event.x

    def _perform_resize(self, event):
        """Called continuously as the splitter is dragged."""
        # Calculate the change in x from the last known position
        try:
            delta_x = event.x - self.splitter_start_x
        except:
            return # Event fired before start

        # Get the sidebar's current width
        current_width = self.sidebar_frame.winfo_width()
        
        # Calculate the new width. 
        # We subtract the delta: dragging left (negative delta) makes the sidebar wider.
        new_width = current_width - delta_x

        # Enforce min/max widths
        min_width = 250
        max_width = 700
        if new_width < min_width:
            new_width = min_width
        if new_width > max_width:
            new_width = max_width

        # Apply the new width to the sidebar
        self.sidebar_frame.configure(width=new_width)
        
        # Store this as the new "target" width for the open/close animation
        self.sidebar_target_width = new_width
        
        # Update the "start" position to the current position for the next motion event
        self.splitter_start_x = event.x
    # -----------------------------------------------------------------
    # ================= END OF NEW METHODS ==================
    # -----------------------------------------------------------------

    def _draw_tile_overlay(self, x, y, cw, ch, addr, name):
        # Tk doesn't accept hex alpha like #00000080; use stipple for translucency
        overlay_h = 34
        self.canvas.create_rectangle(
            x+8, y+8, x+cw-8, y+8+overlay_h,
            fill="#000000", stipple="gray25", outline=""
        )
        
        # FONT FIX: Use Emoji font
        label = f"📌 {'Unpin' if self.pinned_addr == addr else 'Pin'} {name}"
        self.canvas.create_text(x+cw-16, y+8+overlay_h/2, text=label, fill="white", anchor=tk.E, font=('Segoe UI Emoji', 11, 'bold'))

    def toggle_pin_hotkey(self):
        if self._hover_tile:
            _,_,_,_,addr,_ = self._hover_tile
            if self.pinned_addr == addr: self.pinned_addr = None
            else: self.pinned_addr = addr
            if self.pinned_addr and self.layout_mode != "focus":
                self.layout_mode = "focus"
            
            # STABILITY FIX: Request render, don't demand it
            self.gui_queue.put(self.set_render_flag)

    # ====== Pure UI & Helpers ======
    def handle_resize(self, event):
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def update_attendees_display(self):
        self.users_textbox.delete("1.0", "end")
        if self.username: self.users_textbox.insert("end", f"● {self.username} (You)\n")
        if not self.active_users_dict:
            if not self.username: self.users_textbox.insert("end", "No users connected\n")
        else:
            for addr, user in self.active_users_dict.items():
                name = user.get("name", "Unknown")
                self.users_textbox.insert("end", f"● {name} ({addr})\n")
        total = len(self.active_users_dict) + (1 if self.username else 0)
        
        # FONT FIX: Use Emoji font
        self.users_label.configure(text=f"👤 {total} user{'s' if total != 1 else ''}")

    def open_downloads_folder(self):
        import platform, subprocess
        try:
            if platform.system() == "Windows": os.startfile(DOWNLOADS_DIR)
            elif platform.system() == "Darwin": subprocess.Popen(["open", DOWNLOADS_DIR])
            else: subprocess.Popen(["xdg-open", DOWNLOADS_DIR])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open folder: {e}")

    def disconnect_cleanup_ui(self):
        self.status_label.configure(text="Disconnected")
        self.status_indicator.configure(text_color="#ef4444")
        self.leave_btn.configure(state="disabled")
        self.connection_overlay_frame.place(relx=0.5, rely=0.5, anchor="center")

        btn_color = "#374151"; btn_hover = "#4b5563"
        self.video_btn.configure(fg_color=btn_color, hover_color=btn_hover, state="disabled")
        self.audio_btn.configure(fg_color=btn_color, hover_color=btn_hover, state="disabled")
        self.screen_btn.configure(fg_color=btn_color, hover_color=btn_hover, state="disabled", text="🖥️")
        self.chat_toggle_btn.configure(state="disabled"); self.users_toggle_btn.configure(state="disabled")
        self.upload_btn.configure(state="disabled")

        if self.sidebar_visible: self.toggle_panel()

        self.video_feed_frames.clear()
        self.screen_frame = None
        self.screen_maximized = False
        self.current_presenter_addr = None
        self.active_users_dict.clear()
        self.video_states.clear()
        self.active_reactions.clear()
        self.my_addr = None; self.username = None
        self.pinned_addr = None; self.self_rms = 0.0

        self.sending_video = False; self.sending_audio = False

        self.chat_target.configure(values=["Everyone"]); self.chat_target.set("Everyone")
        self.update_attendees_display()
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    def set_layout(self, mode):
        self.layout_mode = mode
        
        # STABILITY FIX: Request render, don't demand it
        self.gui_queue.put(self.set_render_flag)

    # Animated sidebar open/close
    def toggle_panel(self, tab_to_select=None):
        # If an animation is already running, ignore this click.
        if self.sidebar_is_animating:
            return

        # --- NEW, MORE ROBUST LOGIC ---

        # Case 1: Panel is open, we need to close or switch
        if self.sidebar_visible:
            current_tab = self.tabview.get()

            # Case 1a: Clicked same tab (or no tab). We must CLOSE the panel.
            if not tab_to_select or tab_to_select == current_tab:
                
                # Set the lock and the final state *before* animating
                self.sidebar_is_animating = True
                self.sidebar_visible = False 
                
                # FIX: Read the width *ONCE* before starting
                start_width = self.sidebar_frame.winfo_width()

                # FIX: Create a recursive function that passes its state
                def step_out(current_width):
                    """Recursive function to animate closing."""
                    new_w = max(0, current_width - 70) # Animate 70px per step
                    self.sidebar_frame.configure(width=new_w)
                    
                    if new_w <= 0:
                        # Animation finished
                        try: self.sidebar_frame.pack_forget()
                        except: pass
                        
                        # STABILITY FIX: Request render
                        self.gui_queue.put(self.set_render_flag)
                        self.sidebar_is_animating = False # Release lock
                    else:
                        # Schedule next step, passing the new (smaller) width
                        self.master.after(10, lambda: step_out(new_w))
                
                # FIX: Start the closing animation from the width we just read
                step_out(start_width)

            # Case 1b: Clicked different tab. Just SWITCH tabs.
            else:
                self.tabview.set(tab_to_select)

        # Case 2: Panel is closed. We must OPEN it.
        else:
            # Set the lock and the final state *before* animating
            self.sidebar_is_animating = True
            self.sidebar_visible = True
            
            # Pack and set tab *before* animating
            self.sidebar_frame.pack(side="right", fill="y", padx=0, pady=0)
            if tab_to_select: 
                self.tabview.set(tab_to_select)

            def step_in(current_width=0):
                """Recursive function to animate opening."""
                target = self.sidebar_target_width
                new_w = min(target, current_width + 70) # Animate 70px per step
                self.sidebar_frame.configure(width=new_w)

                if new_w >= target:
                    # Animation finished
                    self.sidebar_frame.configure(width=target) # Ensure exact width
                    
                    # STABILITY FIX: Request render
                    self.gui_queue.put(self.set_render_flag)
                    self.sidebar_is_animating = False # Release lock
                else:
                    # Schedule next step, passing the new (larger) width
                    self.master.after(10, lambda: step_in(new_w))
            
            # Start the opening animation from width 0
            step_in(0)
        
        # STABILITY FIX: Removed redundant render call from here

    def switch_theme(self):
        self.dark_mode = not self.dark_mode
        new_mode = "dark" if self.dark_mode else "light"
        new_emoji = "🌚" if self.dark_mode else "☀️"
        new_bg = "#0a0a0a" if self.dark_mode else "#f5f5f5"
        ctk.set_appearance_mode(new_mode)
        self.theme_btn.configure(text=new_emoji)
        self.canvas.configure(bg=new_bg)
        
        # STABILITY FIX: Request render
        self.master.after(150, lambda: self.gui_queue.put(self.set_render_flag))


# ======================================================================
# ===== Main
# ======================================================================
if __name__ == "__main__":
    try:
        if IS_WINDOWS:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = ctk.CTk()
    app = MainAppView(root)
    root.mainloop()