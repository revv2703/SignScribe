from flask import Flask, send_file
# from composer import main
# import main
import composer.composer_main as composer_main

app = Flask(__name__)

@app.route('/get_gif')
def get_gif():
    # Call the function from main.py to generate the GIF
    print("function called")
    composer_main.func(composer_main.hear_voice())
    return send_file('out.gif', mimetype='image/gif')

if __name__ == "__main__":
    # Run the Flask app
    print("server started")
    app.run(host='0.0.0.0', port=5000)
