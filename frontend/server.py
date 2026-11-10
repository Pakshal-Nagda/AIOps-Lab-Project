import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

class CustomHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Enable CORS and disable aggressive caching for dev iteration
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

def run(port=8501):
    directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(directory)
    
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, CustomHandler)
    print(f"Project Medhā Frontend Web App running at http://0.0.0.0:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down frontend server.")
        httpd.server_close()

if __name__ == '__main__':
    port = int(os.getenv("PORT", "8501"))
    run(port)
