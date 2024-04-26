import requests
import cv2

def main():
    while True:
        try:
            # Fetch the GIF from the Flask server
            response = requests.get('http://localhost:5000/get_gif')
            print("requested")
            with open('out.gif', 'wb') as f:
                f.write(response.content)

            # Display the GIF using OpenCV
            gif = cv2.VideoCapture('out.gif')
            while True:
                ret, frame = gif.read()
                if not ret:
                    break
                cv2.imshow('GIF', frame)
                if cv2.waitKey(10) == ord('q'):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
