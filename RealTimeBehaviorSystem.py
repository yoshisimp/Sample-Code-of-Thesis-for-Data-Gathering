import os

def launch_main():
    os.system("python main.py")

def register_face():
    os.system("python face_module/face_register_gui.py")

def view_logs():
    print("\n[📄 Log Files]")
    print("1. CSV: logs/behavior_log.csv")
    print("2. SQLite: logs/behavior_log.db")
    print("Open them manually using Excel or DB Browser for SQLite.")

def menu():
    while True:
        print("\n🔹 Real-Time Behavior System 🔹")
        print("1. Run Full Analysis")
        print("2. Register New Face")
        print("3. View Log File Info")
        print("4. Exit")
        choice = input("Select an option: ")

        if choice == '1':
            launch_main()
        elif choice == '2':
            register_face()
        elif choice == '3':
            view_logs()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    menu()
