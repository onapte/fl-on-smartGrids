from flask import Flask
from flwr.server import start_server

app = Flask(__name__)

@app.route('/')
def index():
    return "Server is running!"

if __name__ == "__main__":
    app.run(port=5000)
    start_server(server_address="localhost:8080", config={"num_rounds": 7})
