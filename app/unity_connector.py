import enum
import socket
import signal
import struct
import threading

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional
from pydantic import BaseModel

from .lmx_generator import LMXGenerator
from .music_converter import MusicConverter


class ServerMessages(enum.Enum):
    """Defines message types for client-server communication."""

    SEND_MUSICXML = 1


class Request(BaseModel):
    """Client request format."""

    requestType: str
    parameters: Optional[dict[str, Any]] = None


class Response(BaseModel):
    """Standardized server response format."""

    statusCode: int
    content: str


class UnityServerSocket:
    """TCP server for real-time music generation communication with Unity clients.

    Features:
    - Threaded connection handling
    - Graceful shutdown capabilities
    - Concurrent request processing
    - LMX-to-MusicXML conversion pipeline
    """

    def __init__(self, lmx_generator: LMXGenerator, host: str, port: int) -> None:
        """Initialize server with generation capabilities.

        Args:
            lmx_generator (LMXGenerator): Pre-trained music generation model_name
            host (str): Server binding address
            port (int): Communication port
        """
        self.lmx_generator = lmx_generator
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.running = False
        self.executor = None
        self.active_connections = set()
        self.connections_lock = threading.Lock()

    def signal_handler(self, sig: int, frame: Any) -> None:
        """Handle OS termination signals for clean shutdown.

        Args:
            sig (int): Signal number
            frame: Current stack frame
        """
        print("\nReceived shutdown signal. Shutting down server...")
        self.shutdown()

    def shutdown(self) -> None:
        """Terminate server operations and clean up resources."""
        if not self.running:
            return

        self.running = False

        # Close all active connections
        with self.connections_lock:
            for conn in self.active_connections:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                    conn.close()
                    print(f"Connection closed for {conn}")
                except:
                    pass
            self.active_connections.clear()

        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
                self.server_socket.close()
            except:
                pass

        print("Server shutdown complete.")

    def create_server(self) -> None:
        """Main server loop with connection handling and error management."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.running = True
        self.executor = (
            ThreadPoolExecutor()
        )  # Thread pool for asynchronous message handling

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Allow socket address reuse
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.settimeout(
                1.0
            )  # Short timeout to check for shutdown regularly
            print(f"Listening on {self.host}:{self.port}")
            self.server_socket.listen()

            while self.running:
                try:
                    conn, addr = self.server_socket.accept()
                    print(f"Connected by {addr}")
                    with self.connections_lock:
                        self.active_connections.add(conn)
                    # Handle the connection in a new thread
                    self.executor.submit(self.handle_connection, conn, addr)
                except socket.timeout:
                    # Timeout to check if server is still running
                    continue
                except Exception as e:
                    if (
                        self.running
                    ):  # Only log errors if server is still supposed to be running
                        print(f"Error in server accept: {e}")

            print("Server loop exited.")

        except Exception as e:
            print(f"Server error: {e}")
        finally:
            self.shutdown()
            if self.executor:
                self.executor.shutdown(wait=False)

    def handle_request(self, data: bytes, conn: socket.socket) -> None:
        """Process incoming client requests and generate responses.

        Args:
            data (bytes): Raw request data
            conn (socket.socket): Client connection object
        """
        try:
            decoded_message = data.decode()
            request = Request.model_validate_json(decoded_message)
            print(f"Received {request}")
            match request.requestType:
                case ServerMessages.SEND_MUSICXML.name:
                    response_data = self.get_music_data(request.parameters or {})
                case _:
                    response_data = Response(statusCode=404, content="Invalid request")
        except Exception as e:
            response_data = Response(statusCode=500, content=str(e))

        try:
            encoded_data = response_data.model_dump_json().encode()
            conn.sendall(struct.pack(">I", len(encoded_data)))
            conn.sendall(encoded_data)
        except Exception as e:
            print(f"Error sending response: {e}")

    def handle_connection(self, conn, addr):
        """Manage individual client connection lifecycle.

        Args:
            conn (socket.socket): Client socket
            addr (tuple): Client address info
        """
        try:
            while self.running:
                try:
                    # Set a timeout on the connection socket to regularly check if server is still running
                    conn.settimeout(1.0)
                    data = conn.recv(1024)
                    if not data:
                        print(f"Disconnected from {addr}")
                        break
                    self.handle_request(data, conn)
                except socket.timeout:
                    # Just a timeout to check if server is still running
                    continue
                except Exception as e:
                    if self.running:
                        print(f"Error receiving data from {addr}: {e}")
                    break
        except Exception as e:
            print(f"Error handling connection {addr}: {e}")
        finally:
            # Remove connection from active connections
            with self.connections_lock:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
            try:
                conn.close()
                print(f"Connection closed for {conn}")
            except:
                pass

    def get_music_data(self, params: dict[str, Any]) -> Response:
        """Generate music data based on client parameters.

        Args:
            params (dict): Generation parameters including:
                - fifths (int): Key signature
                - grand_staff (bool): Staff configuration

        Returns:
            Response: Formatted server response with MusicXML
        """
        try:
            fifths = params.get("fifths", 0)
            generate_grand_staff = params.get("grand_staff", False)
            seed = f"measure key:fifths:{fifths}"
            if generate_grand_staff:
                seed += " clef:G2 staff:1 clef:F4 staff:2"

            music_data = self.lmx_generator.generate(seed)
            music_xml = MusicConverter.lmx_to_musicxml(music_data, normalize=True)
            if not music_xml:
                return Response(statusCode=500, content="Music normalization failed")
            print("Music filepath sent.")
            return Response(statusCode=200, content=music_xml)
        except Exception as e:
            print(f"Error sending music data: {e}")
            return Response(statusCode=500, content=str(e))
