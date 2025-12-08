# app.py
from flask import Flask
from ui.routes import bp as ui_bp

app = Flask(__name__)
app.register_blueprint(ui_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
