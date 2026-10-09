
from http.server import BaseHTTPRequestHandler
import json
import os

from backend import build_retriever, demo_conversation, demo_memory


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({
                "status": "ok",
                "message": "CareerPath AI API is running"
            }).encode()
        )

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            question = body.get("question", "").strip()

            if not question:
                self.send_error(400, "Please enter a question.")
                return

            retriever = build_retriever()
            memory = demo_memory()

            answer, _ = demo_conversation(
                input_text=question,
                memory=memory,
                retriever=retriever,
            )

            response = json.dumps({"answer": answer}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

        except Exception:
            self.send_error(500, "Unable to answer right now. Check server logs.")