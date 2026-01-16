"""
Modular Conference Server
- Object-oriented design with encapsulated state.
- Message dispatcher pattern for handling client commands.
- Dedicated managers for audio and screen-sharing logic.
"""
import socket
import threading
import json
import struct
import os
import time
import logging
from collections import defaultdict, deque
import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Network/Framing Utilities =====

class NetUtils:
    """
    Utility class for handling socket message framing.
    
    This server uses two types of framing:
    1.  JSON-prefixed (write_json_msg/read_json_msg): [4-byte length][JSON data]
        - Used for binary channels (like screen-share) where a JSON handshake is needed.
    2.  Line-delimited (pack_control_msg): [JSON data]\n
        - Used for the main TCP control channel.
    """
    @staticmethod
    def write_json_msg(sock, obj):
        """
        Packs and sends a JSON object with a 4-byte length prefix.
        Protocol: [4-byte big-endian length][UTF-8 JSON string]
        """
        try:
            # Encode the JSON object to bytes
            data = json.dumps(obj).encode('utf-8')
            # Pack the length as a 4-byte big-endian unsigned integer
            length = struct.pack('!I', len(data))
            # Send the length prefix followed by the data
            sock.sendall(length + data)
            return True
        except Exception as e:
            logger.debug(f"NetUtils: Failed to write msg: {e}")
            return False

    @staticmethod
    def read_json_msg(sock):
        """
        Reads a 4-byte length prefix, then a JSON object.
        Protocol: [4-byte big-endian length][UTF-8 JSON string]
        """
        try:
            # Read the 4-byte length prefix
            length_data = sock.recv(4)
            if not length_data or len(length_data) < 4:
                return None  # Connection closed or truncated message
            
            # Unpack the length
            length = struct.unpack('!I', length_data)[0]
            
            # Sanity check to prevent memory exhaustion from a malicious packet
            if length > 50 * 1024 * 1024: # 50MB sanity limit
                logger.warning(f"NetUtils: Received excessive length: {length}")
                return None

            # Read exactly 'length' bytes of data
            data = b''
            while len(data) < length:
                # Read in chunks, handling fragmented TCP packets
                chunk = sock.recv(min(16384, length - len(data)))
                if not chunk:
                    return None  # Connection closed prematurely
                data += chunk
                
            # Decode and parse the JSON
            return json.loads(data.decode('utf-8'))
        except socket.timeout:
            logger.warning("NetUtils: Socket read timed out")
            return None
        except Exception as e:
            logger.error(f"NetUtils: Error reading msg: {e}")
            return None

    @staticmethod
    def pack_control_msg(obj):
        """
        Packs a JSON object for the line-based control protocol.
        Protocol: [UTF-8 JSON string]\n
        """
        return (json.dumps(obj) + "\n").encode()

# ===== Audio Mixing Logic =====

class AudioMixer:
    """
    Encapsulates all logic for receiving and mixing audio streams.
    
    This class runs two threads:
    1.  _run_audio_receiver: Listens on a UDP socket for incoming audio packets
        from all clients.
    2.  _run_audio_mixer: Runs a high-precision ticker. Every audio chunk
        duration, it mixes all received audio and sends a custom mix
        to each client (excluding their own audio).
    """
    def __init__(self, server_instance):
        self.server = server_instance  # Reference to the main server for state
        
        # Stores incoming audio packets: { (ip, port): deque([pkt1, pkt2, ...]) }
        self.audio_queues = defaultdict(lambda: deque(maxlen=AUDIO_BUFFER_SIZE))
        
        # Packet Loss Concealment (PLC): { (ip, port): last_good_packet }
        # If a queue is empty during a mix cycle, we'll use the last good packet.
        self.last_good_audio = {}
        
        # Active speaker detection state
        self.last_speaker_addr = None
        self.last_speaker_broadcast_time = 0
        
        # The single UDP socket for all audio I/O
        self.audio_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Increase the OS receive buffer size for this UDP socket
            self.audio_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 2 * 1024 * 1024)
        except:
            pass # Not critical if it fails

    def bind(self, host, port):
        """Binds the audio UDP socket to the server's host and port."""
        self.audio_sock.bind((host, port))
        logger.info(f"[AUDIO] Receiver listening on UDP {port}")

    def start_threads(self):
        """Starts the receiver and mixer threads."""
        threading.Thread(target=self._run_audio_receiver, daemon=True, name="AUDIO-Receiver").start()
        threading.Thread(target=self._run_audio_mixer, daemon=True, name="AUDIO-Mixer").start()

    def _run_audio_receiver(self):
        """
        Thread target: Listens for UDP audio packets and queues them.
        This is the "ingestion" loop.
        """
        while True:
            try:
                data, addr = self.audio_sock.recvfrom(8192) # Read one audio packet
                if data:
                    # Add the packet to the queue for the sender's address
                    self.audio_queues[addr].append(data)
                    
                    # --- Active Speaker Notification ---
                    # To avoid flooding the control channel, we only send an
                    # "speaker_active" message if the speaker changes or
                    # if the same speaker has been talking for a set interval.
                    now = time.time()
                    if (addr != self.last_speaker_addr or
                        now - self.last_speaker_broadcast_time > SPEAKER_BROADCAST_INTERVAL):
                        
                        self.last_speaker_addr = addr
                        self.last_speaker_broadcast_time = now
                        # Broadcast on the main control channel
                        self.server._broadcast({"type": "speaker_active", "addr": addr[0]})
                        
            except Exception as e:
                logger.error(f"[AUDIO] Receiver error: {e}")

    def _run_audio_mixer(self):
        """
        Thread target: Mixes and distributes audio packets.
        This runs on a high-precision ticker to maintain a smooth audio interval.
        """
        logger.info("[AUDIO] Mixer started - High-Precision Ticker & PLC enabled")
        while True:
            tick_start = time.time()  # Mark the start of the mix cycle
            try:
                frames, sources = [], []
                # Get a snapshot of currently connected client IPs from the main server
                known_ips = self.server._get_all_client_ips()

                # --- 1. Gather frames (with Packet Loss Concealment) ---
                for addr, q in list(self.audio_queues.items()):
                    # Prune audio queues from clients who have disconnected
                    if addr[0] not in known_ips:
                        self.last_good_audio.pop(addr, None)
                        self.audio_queues.pop(addr, None)
                        continue
                        
                    if len(q) > 0:
                        # Queue has data: pop one packet
                        try:
                            pkt = q.popleft()
                            frames.append(pkt)
                            sources.append(addr)
                            self.last_good_audio[addr] = pkt # Save as last good packet
                        except IndexError:
                            pass # deque was emptied by another thread (rare, but safe)
                    elif addr in self.last_good_audio:
                        # --- PLC ---
                        # Queue is empty: use the last good packet to avoid silence
                        frames.append(self.last_good_audio[addr])
                        sources.append(addr)

                # --- 2. Clean up PLC buffer for disconnected clients ---
                for addr in list(self.last_good_audio.keys()):
                    if addr[0] not in known_ips:
                        self.last_good_audio.pop(addr, None)

                # --- 3. Mix and Send ---
                if frames:
                    # Convert raw byte frames to numpy arrays of 16-bit integers
                    arrays = [np.frombuffer(f, dtype=np.int16) for f in frames if len(f) > 0 and len(f) % 2 == 0]
                    
                    if arrays:
                        # Ensure all arrays are the same length by truncating to the shortest
                        minlen = min(a.shape[0] for a in arrays)
                        arrays = [a[:minlen] for a in arrays]

                        # Get all clients who should receive audio
                        all_targets = self.server._get_audio_targets()
                        
                        for tgt_addr_tuple, (_tgt_conn, _tgt_name) in all_targets:
                            # --- Per-Client Mix ---
                            # Create a mix for this target that *excludes* their own audio
                            tgt_arrays = [arrays[i] for i, src_addr in enumerate(sources) if src_addr[0] != tgt_addr_tuple[0]]
                            
                            if not tgt_arrays:
                                continue # Nothing to send to this target

                            # Stack arrays vertically (one row per source)
                            stacked = np.vstack(tgt_arrays)
                            
                            # Mix:
                            # 1. Convert to float32 to prevent overflow during summation.
                            # 2. Sum all sources (axis=0).
                            mixed_float = np.sum(stacked.astype(np.float32), axis=0)
                            
                            # 3. Clip the result to the valid int16 range (-32768 to 32767)
                            #    This is "hard clipping" and prevents audio distortion.
                            mixed = np.clip(mixed_float, -32768, 32767).astype(np.int16)
                            
                            # 4. Convert back to raw bytes
                            pkt = mixed.tobytes()
                            
                            try:
                                # Send the custom-mixed packet via UDP
                                self.audio_sock.sendto(pkt, tgt_addr_tuple)
                            except Exception:
                                # Don't prune here; let the main server's disconnect
                                # logic handle target cleanup.
                                continue

            except Exception as e:
                logger.error(f"[AUDIO] Mixer error: {e}")

            # --- High-Precision Ticker Logic ---
            elapsed = time.time() - tick_start
            sleep_time = AUDIO_CHUNK_DURATION - elapsed
            
            # Sleep for *most* of the remaining time
            if sleep_time > 0.002: # 2ms threshold
                time.sleep(sleep_time - 0.001) # Sleep 1ms less
            
            # "Busy-wait" (spin) for the last moment to ensure precise timing
            while time.time() < tick_start + AUDIO_CHUNK_DURATION:
                pass  # micro-wait for precision

# ===== Screen Share Logic (BINARY TRANSPORT) =====

class ScreenShareManager:
    """
    Manages screen sharing, which uses a dedicated TCP port.
    
    This manager allows only ONE presenter at a time.
    Protocol:
    1.  Client connects to the SCREEN_TCP_PORT.
    2.  Client sends a JSON handshake: `{"role": "presenter"}` or `{"role": "viewer"}`.
    3.  Server responds with JSON: `{"status": "ok"}` or `{"status": "denied"}`.
    4.  If "presenter":
        -   Presenter streams binary frames: [4-byte length][JPEG bytes]
        -   Server forwards these frames to all viewers.
        -   A frame with [4-byte length] == 0 signals "stop presenting".
    5.  If "viewer":
        -   Server streams binary frames: [4-byte length][JPEG bytes]
        -   A frame with [4-byte length] == 0 signals "presenter stopped".
    """
    def __init__(self, server_instance):
        self.server = server_instance
        self.presenter_sock = None  # Holds the *single* presenter's socket
        self.presenter_addr = None
        self.viewers = {}  # { conn: addr } of all connected viewers
        self.lock = threading.RLock() # Protects presenter_sock and viewers

    # --- binary helpers ---
    
    def _recv_exact(self, sock, n):
        """Helper to read *exactly* n bytes from a TCP socket."""
        buf = b''
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                return None  # Connection closed
            buf += chunk
        return buf

    def _send_frame(self, sock, payload: bytes) -> bool:
        """Helper to send a binary frame with the 4-byte length prefix."""
        try:
            # Send [4-byte length][payload]
            sock.sendall(struct.pack('!I', len(payload)) + payload)
            return True
        except OSError:
            # Broken pipe or connection reset
            return False

    def handle_connection(self, conn, addr):
        """
        Entry point for a new connection on the screen share port.
        Handles the initial JSON handshake to determine the role.
        """
        role = None
        try:
            conn.settimeout(10.0)
            # 1. Read the JSON handshake
            role_msg = NetUtils.read_json_msg(conn)
            if not role_msg:
                return
                
            role = role_msg.get("role")
            
            # 2. Route to the correct handler based on role
            if role == "presenter":
                self._handle_presenter(conn, addr)
            elif role == "viewer":
                self._handle_viewer(conn, addr)
                
        except Exception as e:
            logger.error(f"[SCREEN] Connection error: {e}")
        finally:
            # 3. Clean up the connection regardless of what happened
            self._cleanup_connection(conn, role)

    def _handle_presenter(self, conn, addr):
        """Manages the lifecycle of a presenter connection."""
        with self.lock:
            # --- 1. Try to acquire the presenter "lock" ---
            if self.presenter_sock is not None:
                # Deny if a presenter is already active
                NetUtils.write_json_msg(conn, {"status": "denied", "reason": "Presenter already active"})
                conn.close()
                return
                
            # --- 2. Accept the presenter ---
            self.presenter_sock = conn
            self.presenter_addr = addr
            NetUtils.write_json_msg(conn, {"status": "ok"})
            logger.info(f"[SCREEN] Presenter connected: {addr}")
            
            # --- 3. Notify all clients on the *control* channel ---
            try:
                name = self.server._name_for_ip(addr[0]) or addr[0]
                self.server._broadcast({"type": "present_start", "from": name})
            except Exception:
                pass

        # --- 4. Binary streaming loop ---
        # (Lock is released here; only this thread writes frames)
        try:
            conn.settimeout(None) # Remove timeout for long-lived stream
            while True:
                # Read the 4-byte length prefix
                hdr = self._recv_exact(conn, 4)
                if not hdr:
                    break # Presenter disconnected
                    
                (length,) = struct.unpack('!I', hdr)
                
                if length == 0:
                    # Presenter gracefully stopped (sent a 0-length frame)
                    logger.info(f"[SCREEN] Presenter {addr} stopped gracefully.")
                    break
                    
                # Read the payload (JPEG data)
                payload = self._recv_exact(conn, length)
                if payload is None:
                    break # Presenter disconnected
                    
                # --- 5. Broadcast the frame to all viewers ---
                self._broadcast_frame_binary(payload)
                
        except Exception as e:
            logger.debug(f"[SCREEN] presenter loop ended: {e}")

    def _handle_viewer(self, conn, addr):
        """Manages the lifecycle of a viewer connection."""
        with self.lock:
            # --- 1. Add viewer to the broadcast list ---
            self.viewers[conn] = addr
            # Send 'ok', informing them if a presenter is currently active
            NetUtils.write_json_msg(conn, {"status": "ok", "reason": "Presenter active" if self.presenter_sock else "No presenter"})
            
        logger.info(f"[SCREEN] Viewer connected: {addr}")
        
        # --- 2. Keep the connection alive ---
        # This thread just blocks. Frames are *pushed* to this socket
        # from the _handle_presenter thread (via _broadcast_frame_binary).
        try:
            conn.settimeout(None)
            while True:
                # Keep socket open. A recv() or sleep() is fine.
                time.sleep(60) # Wake up every 60s
                # A more robust way would be to recv(1), but sleep is simpler.
                
        except:
            # Connection will be cleaned up in the 'finally' block
            pass

    def _broadcast_frame_binary(self, payload: bytes):
        """Pushes a single binary frame to all active viewers."""
        with self.lock:
            dead = []
            # Iterate a *copy* of the keys to allow modification
            for viewer_sock in list(self.viewers.keys()):
                ok = self._send_frame(viewer_sock, payload)
                if not ok:
                    # send_frame failed (broken pipe), mark viewer for removal
                    dead.append(viewer_sock)
                    
            # Prune dead viewers
            for d in dead:
                self.viewers.pop(d, None)
                try:
                    d.close()
                except:
                    pass

    def _cleanup_connection(self, conn, role):
        """Cleans up state when a presenter or viewer disconnects."""
        with self.lock:
            if role == "presenter" and self.presenter_sock == conn:
                # --- Presenter disconnected ---
                logger.info(f"[SCREEN] Presenter disconnected. Releasing lock.")
                self.presenter_sock = None
                self.presenter_addr = None
                
                # --- Notify all clients ---
                try:
                    # 1. Send a 0-length "stop" frame to all binary viewers
                    stop_frame = struct.pack('!I', 0)
                    for v in list(self.viewers.keys()):
                        try:
                            v.sendall(stop_frame)
                        except:
                            pass
                            
                    # 2. Send a "present_stop" message on the control channel
                    self.server._broadcast({"type": "present_stop", "from": "presenter"})
                except Exception:
                    pass
                    
                # Prune any closed viewers while we're at it
                self.viewers = {v: a for v, a in self.viewers.items() if v.fileno() != -1}
                
            elif role == "viewer":
                # --- Viewer disconnected ---
                self.viewers.pop(conn, None)
                logger.info(f"[SCREEN] Viewer disconnected.")
                
        try:
            conn.close()
        except:
            pass

# ===== Main Server Class =====

class ConferenceServer:
    """
    Main server class that orchestrates all subsystems.
    
    -   Manages the central participant state.
    -   Runs the main TCP control channel (line-delimited JSON).
    -   Runs the UDP video relay (dumb forwarder).
    -   Runs the TCP file transfer listener.
    -   Owns and starts the AudioMixer and ScreenShareManager.
    """
    def __init__(self, host='0.0.0.0'):
        self.host = host
        
        # --- Port Definitions ---
        self.tcp_port = TCP_PORT            # Main control channel
        self.video_udp_port = VIDEO_UDP_PORT  # Video relay
        self.audio_udp_port = AUDIO_UDP_PORT  # Audio mixer
        self.screen_tcp_port = SCREEN_TCP_PORT # Screen share
        self.file_tcp_port = FILE_TCP_PORT   # File transfers

        # --- Central State ---
        self.state_lock = threading.Lock() # Protects all state below
        
        # The "source of truth" for participant info
        # { conn_obj: {name, addr, vport, aport} }
        self.participants = {}
        
        # Reverse lookup for private messages: { name: conn_obj }
        self.participant_names = {}
        
        # Targets for media relays
        self.udp_video_targets = set()     # { (ip, port) }
        self.udp_audio_targets = {}     # { (ip, port): (conn, name) }

        # --- File Storage ---
        self.file_dir = "server_files"
        os.makedirs(self.file_dir, exist_ok=True)

        # --- Subsystems ---
        self.audio_mixer = AudioMixer(self)
        self.screen_manager = ScreenShareManager(self)
        
        # The single UDP socket for all video I/O
        self.video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Increase OS receive buffer
            self.video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except:
            pass

        # --- Message Dispatcher Pattern ---
        # Maps "type" from a control message to a handler function.
        self.message_handlers = {
            "hello": self._on_hello,
            "chat": self._on_chat,
            "reaction": self._on_reaction,
            "hand_raise": self._on_hand_raise,
            "video_start": self._on_video_start,
            "video_stop": self._on_video_stop,
            "private_chat": self._on_private_chat,
            "present_start": self._on_present_start, # Client *requests* to start
            "present_stop": self._on_present_stop,   # Client *requests* to stop
            "bye": self._on_bye
        }

    # --- helpers used by subsystems (thread-safe) ---
    
    def _name_for_ip(self, ip):
        """Thread-safe lookup of a user's name from their IP."""
        with self.state_lock:
            for info in self.participants.values():
                if info["addr"][0] == ip:
                    return info.get("name")
        return None

    def _get_all_client_ips(self):
        """Thread-safe fetch of all connected client IPs."""
        with self.state_lock:
            return {info["addr"][0] for info in self.participants.values()}

    def _get_audio_targets(self):
        """Thread-safe fetch of all audio targets for the mixer."""
        with self.state_lock:
            # Return a copy to avoid lock contention
            return list(self.udp_audio_targets.items())

    # --- startup ---
    
    def start(self):
        """Binds all sockets and starts all listener threads."""
        # Bind UDP sockets
        self.video_sock.bind((self.host, self.video_udp_port))
        self.audio_mixer.bind(self.host, self.audio_udp_port)

        # Start media threads
        threading.Thread(target=self._run_video_relay, daemon=True, name="VIDEO-Forwarder").start()
        self.audio_mixer.start_threads()

        # Start TCP acceptor threads for subsystems
        threading.Thread(target=self._run_acceptor,
                         args=(self.screen_tcp_port, self.screen_manager.handle_connection, "SCREEN"),
                         daemon=True, name="SCREEN-Acceptor").start()

        threading.Thread(target=self._run_acceptor,
                         args=(self.file_tcp_port, self._handle_file_transfer, "FILE"),
                         daemon=True, name="FILE-Acceptor").start()

        # Start the main control channel acceptor (BLOCKING)
        # This will run in the main thread.
        self._run_acceptor(self.tcp_port, self._handle_control_client, "TCP")

    def _run_acceptor(self, port, handler, name):
        """
        A generic TCP socket acceptor loop.
        Listens on 'port' and spawns a new 'handler' thread for each connection.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((self.host, port))
            s.listen(50)
            logger.info(f"[{name}] Server listening on {self.host}:{port}")
            while True:
                conn, addr = s.accept()
                # Spawn a new thread to handle this client
                threading.Thread(target=handler, args=(conn, addr), daemon=True, name=f"{name}-Client").start()
        except Exception as e:
            logger.error(f"[{name}] Acceptor error: {e}")
        finally:
            s.close()

    # --- control plane (line-delimited JSON) ---
    
    def _send_to_client(self, conn, obj):
        """Sends a control message (JSON + \n) to a single client."""
        try:
            raw = NetUtils.pack_control_msg(obj)
            conn.sendall(raw)
        except:
            # Client likely disconnected, will be cleaned up by _remove_client
            pass

    def _broadcast(self, obj, exclude_conn=None):
        """Sends a control message to all participants, optionally excluding one."""
        with self.state_lock:
            # Iterate a copy of the list to be thread-safe
            all_conns = list(self.participants.keys()) 
            
        for conn in all_conns:
            if conn is exclude_conn:
                continue
            self._send_to_client(conn, obj)

    def _get_user_list(self):
        """Generates the current user list (thread-safe)."""
        with self.state_lock:
            return [{"name": info["name"], "addr": f"{info['addr'][0]}"} for info in self.participants.values()]

    def _remove_client(self, conn, name_from_info=None):
        """
        The central cleanup function for a disconnected client.
        This is called by the 'finally' block of _handle_control_client.
        """
        info = None
        name = name_from_info
        client_ip = None

        # --- 1. Remove from central state (under lock) ---
        with self.state_lock:
            info = self.participants.pop(conn, None)
            if info:
                # This was a fully registered client
                name = info.get("name", "unknown")
                client_ip = info.get("addr", ["unknown"])[0]
                self.participant_names.pop(name, None)
                
                # Remove from media relays
                try:
                    if info.get("video_port"):
                        self.udp_video_targets.discard((info["addr"][0], info["video_port"]))
                    if info.get("audio_port"):
                        self.udp_audio_targets.pop((info["addr"][0], info["audio_port"]), None)
                except:
                    pass
            
        # --- 2. Redundant cleanup (robustness) ---
        # Purge any *other* lingering UDP targets for this IP, just in case.
        if client_ip:
            with self.state_lock:
                self.udp_video_targets = {x for x in self.udp_video_targets if x[0] != client_ip}
                self.udp_audio_targets = {k: v for k, v in self.udp_audio_targets.items() if k[0] != client_ip}

        # --- 3. Broadcast the departure (lock released) ---
        if info:
            logger.info(f"[LEFT] {name} @ {info.get('addr')}")
            # Get the *new* user list
            user_list = self._get_user_list()
            # Broadcast the new list and the leave event
            self._broadcast({"type": "user_list", "users": user_list})
            self._broadcast({"type": "leave", "name": name, "addr": client_ip})
        elif name_from_info:
            # This client disconnected *before* sending a valid "hello"
            logger.info(f"[LEFT] {name_from_info} (redundant cleanup)")

        # --- 4. Close the socket ---
        try:
            conn.close()
        except:
            pass

    def _handle_control_client(self, conn, addr):
        """
        Thread target for a single client on the main control channel.
        This loop reads line-delimited JSON messages and dispatches them.
        """
        # client_info holds state *for this thread*
        client_info = {"conn": conn, "addr": addr, "name": None}
        try:
            buf = b""
            while True:
                # Read from the socket
                data = conn.recv(4096)
                if not data:
                    break # Connection closed
                    
                buf += data
                
                # Process all complete lines (messages) in the buffer
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                        
                    # --- Parse and Dispatch ---
                    try:
                        msg = json.loads(line.decode())
                    except Exception as e:
                        logger.error(f"Bad JSON from {addr}: {e}")
                        continue

                    mtype = msg.get("type")
                    # Use the dispatcher to find the right handler
                    handler = self.message_handlers.get(mtype)
                    
                    if handler:
                        handler(client_info, msg)
                    
                    # --- State/Exit Checks ---
                    if mtype == "bye":
                        raise ConnectionAbortedError("Client signaled 'bye'")
                    
                    # If 'hello' failed (e.g., name taken), client_info["name"]
                    # will still be None. We kick them.
                    if mtype == "hello" and client_info["name"] is None:
                        raise ConnectionAbortedError("Client 'hello' failed")

                    # Client *must* send "hello" first.
                    if client_info["name"] is None and mtype != "hello":
                        raise ConnectionAbortedError("Client sent data before 'hello'")

        except Exception as e:
            # Log expected disconnects (AbortedError) at a lower level
            if isinstance(e, ConnectionAbortedError):
                logger.info(f"Control handler for {addr} exiting: {e}")
            else:
                logger.debug(f"Control handler for {addr} exiting: {e}")
        finally:
            # --- CRITICAL ---
            # This ensures cleanup no matter how the loop exits
            self._remove_client(conn, client_info.get("name"))

    # --- control handlers (called by _handle_control_client) ---
    
    def _on_hello(self, client_info, msg):
        """Handles a client's initial join message."""
        conn = client_info["conn"]
        addr = client_info["addr"]
        name = msg.get("name", "anonymous")
        vport = int(msg.get("video_port", 0) or 0) # Client's UDP video port
        aport = int(msg.get("audio_port", 0) or 0) # Client's UDP audio port

        with self.state_lock:
            # --- 1. Check for name collision ---
            if name in self.participant_names:
                self._send_to_client(conn, {"type": "error", "message": "Username already taken"})
                client_info["name"] = None # Signal to handler to disconnect
                return

            # --- 2. Register the new client ---
            info = {"name": name, "addr": addr, "video_port": vport, "audio_port": aport, "last_seen": time.time()}
            self.participants[conn] = info
            self.participant_names[name] = conn
            
            # --- 3. Add to media relays ---
            if vport:
                self.udp_video_targets.add((addr[0], vport))
            if aport:
                self.udp_audio_targets[(addr[0], aport)] = (conn, name)

        # --- 4. Update this client's thread state ---
        client_info["name"] = name
        logger.info(f"[JOIN] {name} @ {addr} vport={vport} aport={aport}")

        # --- 5. Send/Broadcast updates ---
        user_list = self._get_user_list()
        # Send the *full* list to the new user
        self._send_to_client(conn, {"type": "user_list", "users": user_list})
        # Tell *everyone else* about the new user
        self._broadcast({"type": "join", "name": name, "addr": addr[0]}, exclude_conn=conn)
        # Send the *updated* list to everyone else
        self._broadcast({"type": "user_list", "users": user_list}, exclude_conn=conn)

    def _on_chat(self, client_info, msg):
        """Handles a public chat message."""
        self._broadcast({"type": "chat", "from": client_info["name"], "message": msg.get("message", "")})

    def _on_reaction(self, client_info, msg):
        """Handles an emoji reaction."""
        self._broadcast({"type": "reaction", "from": client_info["name"], "addr": client_info["addr"][0], "emoji": msg.get("emoji")})

    def _on_hand_raise(self, client_info, msg):
        """Handles a hand raise toggle."""
        self._broadcast({"type": "hand_raise", "from": client_info["name"], "addr": client_info["addr"][0], "state": msg.get("state")})

    def _on_video_start(self, client_info, msg):
        """Handles a "video started" notification."""
        self._broadcast({"type": "video_start", "from": client_info["name"], "addr": client_info["addr"][0]}, exclude_conn=client_info["conn"])

    def _on_video_stop(self, client_info, msg):
        """Handles a "video stopped" notification."""
        self._broadcast({"type": "video_stop", "from": client_info["name"], "addr": client_info["addr"][0]}, exclude_conn=client_info["conn"])

    def _on_private_chat(self, client_info, msg):
        """Handles a private chat message (DM)."""
        target_name = msg.get("to")
        message = msg.get("message", "")
        target_conn = None
        
        with self.state_lock:
            # Look up the target's connection object by their name
            target_conn = self.participant_names.get(target_name)
            
        if target_conn:
            # Send to the target
            self._send_to_client(target_conn, {"type": "private_chat", "from": client_info["name"], "message": message})
            # Send confirmation to the sender
            self._send_to_client(client_info["conn"], {"type": "private_chat_sent", "to": target_name, "message": message})
        else:
            # Target not found
            self._send_to_client(client_info["conn"], {"type": "error", "message": f"User {target_name} not found"})

    def _on_present_start(self, client_info, msg):
        """Relays a 'present_start' *request*."""
        # Note: This just *relays* the message. The *actual* start is
        # handled by the ScreenShareManager when the client connects
        # to the screen share port. This message is mostly for the UI.
        self._broadcast({"type": "present_start", "from": client_info["name"]}, exclude_conn=client_info["conn"])

    def _on_present_stop(self, client_info, msg):
        """Relays a 'present_stop' *request*."""
        self._broadcast({"type": "present_stop", "from": client_info["name"]}, exclude_conn=client_info["conn"])

    def _on_bye(self, client_info, msg):
        """Handles a graceful 'bye' message."""
        # No action needed here. The _handle_control_client loop
        # will see the "bye" type and exit, triggering the
        # 'finally' block which calls _remove_client.
        pass

    # --- file transfer (dedicated TCP port) ---
    
    def _handle_file_transfer(self, conn, addr):
        """
        Entry point for a new connection on the file transfer port.
        Reads a single JSON handshake to route to upload/download.
        """
        try:
            # Read the one-and-only handshake message
            data = conn.recv(4096)
            if not data:
                return
            msg = json.loads(data.decode())
            
            if msg.get("type") == "file_upload":
                self._handle_file_upload(conn, msg)
            elif msg.get("type") == "file_download":
                self._handle_file_download(conn, msg)
                
        except Exception as e:
            logger.error(f"[FILE] Transfer error: {e}")
        finally:
            try: conn.close()
            except: pass

    def _handle_file_upload(self, conn, msg):
        """Handles a client uploading a file."""
        filename = msg.get("filename")
        size = msg.get("size")
        sender_name = msg.get("from")
        
        # Sanitize filename to prevent directory traversal
        safe = os.path.basename(filename)
        dest = os.path.join(self.file_dir, safe)

        # --- 1. Send "READY" to client ---
        conn.sendall(b"READY")
        
        # --- 2. Receive file data ---
        remaining = size
        with open(dest, "wb") as f:
            while remaining > 0:
                chunk = conn.recv(min(65536, remaining))
                if not chunk:
                    break # Connection dropped
                f.write(chunk)
                remaining -= len(chunk)

        # --- 3. Notify all clients (on control channel) ---
        logger.info(f"[FILE] Received {safe} ({size} bytes) from {sender_name}")
        self._broadcast({"type": "file_offer", "from": sender_name, "filename": safe, "size": size})
        
        # --- 4. Send "DONE" to uploader ---
        conn.sendall(b"DONE")

    def _handle_file_download(self, conn, msg):
        """Handles a client downloading a file."""
        filename = msg.get("filename")
        # Sanitize path
        path = os.path.join(self.file_dir, os.path.basename(filename))
        
        if not os.path.exists(path):
            conn.sendall(b"ERROR")
            return
            
        # --- 1. Send file metadata (size) ---
        size = os.path.getsize(path)
        info = json.dumps({"size": size}).encode()
        conn.sendall(info + b"\n")
        
        # --- 2. Wait for "READY" from client ---
        conn.recv(10) # Simple ACK
        
        # --- 3. Stream file data ---
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break # End of file
                conn.sendall(chunk)
        logger.info(f"[FILE] Sent {filename} to client")

    # --- Video Relay (UDP) ---
    
    def _run_video_relay(self):
        """
        Thread target: Listens for and forwards UDP video packets.
        This is a "dumb" forwarder. It does not mix or transcode.
        
        Protocol:
        -   Client sends: [Video Payload]
        -   Server receives: [Video Payload] from (src_ip, src_port)
        -   Server sends: [4-byte src_ip][Video Payload] to all other clients.
        
        The 4-byte IP prefix allows clients to know *who* the video is from.
        """
        logger.info(f"[VIDEO] Forwarder listening on UDP {self.video_udp_port}")
        while True:
            try:
                data, addr = self.video_sock.recvfrom(MAX_UDP_SIZE)
                if len(data) < 8: # Ignore tiny packets
                    continue
                    
                # --- 1. Pack the sender's IP ---
                try:
                    src_ip_packed = socket.inet_aton(addr[0])
                except:
                    src_ip_packed = b'\x00\x00\x00\x00' # Placeholder for invalid IP
                    
                # --- 2. Prepend the IP to the packet ---
                outpkt = src_ip_packed + data
                
                # --- 3. Get a snapshot of targets ---
                with self.state_lock:
                    targets = list(self.udp_video_targets)
                    
                # --- 4. Forward to all *other* clients ---
                for tgt_addr, tgt_port in targets:
                    if tgt_addr == addr[0]:
                        continue # Don't send video back to the sender
                    try:
                        self.video_sock.sendto(outpkt, (tgt_addr, tgt_port))
                    except Exception:
                        # Ignore; disconnect handler will clean up
                        continue
                        
            except Exception as e:
                logger.error(f"[VIDEO] Forwarder error: {e}")

# ===== Configuration =====
TCP_PORT = 9000
VIDEO_UDP_PORT = 10000
AUDIO_UDP_PORT = 11000
SCREEN_TCP_PORT = 9001
FILE_TCP_PORT = 9002

MAX_UDP_SIZE = 65507       # Max theoretical UDP packet size
VIDEO_CHUNK_DATA = 1100    # (Constant from client, not used in server)
AUDIO_BUFFER_SIZE = 10       # Max audio packets to buffer per client
AUDIO_CHUNK_DURATION = 0.016 # 16ms. (e.g., 256 samples / 16000 Hz)
SPEAKER_BROADCAST_INTERVAL = 1.0 # 1 second

# ===== Server IP Configuration =====
# network interfaces (e.g., WiFi, Ethernet, localhost).
# Using a specific IP will *only* work if clients are on that exact network.

# ===== Automatically Detect Server IP =====
try:
    SERVER_HOST = socket.gethostbyname(socket.gethostname())
    if SERVER_HOST.startswith("127."):
        # Handle localhost/loopback issue by determining external-facing IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            SERVER_HOST = s.getsockname()[0]
        finally:
            s.close()
except Exception:
    # Fallback to listening on all interfaces if detection fails
    SERVER_HOST = "0.0.0.0"

# Log the detected IP
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(f"[INFO] Server running on: {SERVER_HOST}")

# ===== Start Server =====
if __name__ == "__main__":
    try:
        server = ConferenceServer(host=SERVER_HOST)
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server.")