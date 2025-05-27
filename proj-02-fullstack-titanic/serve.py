from flask import Flask, send_from_directory

app = Flask(__name__, static_folder='templates')

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

if __name__ == '__main__':
    app.run(debug=True, port=3000) 