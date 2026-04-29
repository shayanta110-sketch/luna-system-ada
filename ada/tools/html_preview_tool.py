import os
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Optional


class PreviewServer:
    """Manage a local HTTP server for HTML previews."""

    def __init__(self, directory: str, port: int = 8000):
        self.directory = os.path.abspath(directory)
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start the HTTP server in a background thread."""
        if self.server:
            return False

        os.chdir(self.directory)
        handler = SimpleHTTPRequestHandler
        self.server = HTTPServer(("", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return True

    def stop(self) -> bool:
        """Stop the HTTP server."""
        if not self.server:
            return False
        self.server.shutdown()
        self.server.server_close()
        self.server = None
        self.thread = None
        return True

    def is_running(self) -> bool:
        return self.server is not None

    def url(self, path: str = "") -> str:
        """Get the full URL for a given path."""
        base = f"http://localhost:{self.port}"
        if path.startswith("/"):
            return base + path
        return base + "/" + path


# Global server instances by directory
_servers: dict[str, PreviewServer] = {}


def serve_html_directory(directory: str, port: int = 8000, open_browser: bool = True) -> str:
    """
    Start a local HTTP server for a directory containing HTML files.

    Args:
        directory: Path to the directory to serve.
        port: Port number for the server (default 8000).
        open_browser: Whether to open the browser automatically.

    Returns:
        URL where the server is accessible.
    """
    abs_dir = os.path.abspath(directory)

    if abs_dir in _servers and _servers[abs_dir].is_running():
        # Server already running
        server = _servers[abs_dir]
    else:
        # Create and start new server
        server = PreviewServer(abs_dir, port)
        if not server.start():
            raise RuntimeError(f"Failed to start server on port {port}")
        _servers[abs_dir] = server

    url = server.url()
    if open_browser:
        webbrowser.open(url)
    return url


def stop_server(directory: Optional[str] = None) -> bool:
    """
    Stop the HTTP server for a specific directory or all servers.

    Args:
        directory: Path to the directory whose server to stop. If None, stop all.

    Returns:
        True if at least one server was stopped, False otherwise.
    """
    if directory is None:
        # Stop all servers
        if not _servers:
            return False
        for server in list(_servers.values()):
            server.stop()
        _servers.clear()
        return True

    abs_dir = os.path.abspath(directory)
    if abs_dir not in _servers:
        return False

    server = _servers[abs_dir]
    result = server.stop()
    if result:
        del _servers[abs_dir]
    return result


def get_server_url(directory: str) -> Optional[str]:
    """Get the URL of a running server for the given directory."""
    abs_dir = os.path.abspath(directory)
    if abs_dir in _servers and _servers[abs_dir].is_running():
        return _servers[abs_dir].url()
    return None
