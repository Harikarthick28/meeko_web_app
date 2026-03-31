"""
Meeko — Your Empathetic Music Companion
Run this file to start the web application.
"""

from app import app

if __name__ == '__main__':
    print("Starting Meeko — Empathetic Music Companion")
    print("Open your browser: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)