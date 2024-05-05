from composer.server import app
from interpreter.checking_output import real_time_asl



def main():
    print("Welcome to Two Way Sign Language Translator!")
    while True:
        print("\nChoose an option:")
        print("1. Sign to Voice")
        print("2. Voice to Sign")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            real_time_asl()
        
        elif choice == '2':
            print("Starting the server...")
            print("server started")
            app.run(host='0.0.0.0', port=5000)
            
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
