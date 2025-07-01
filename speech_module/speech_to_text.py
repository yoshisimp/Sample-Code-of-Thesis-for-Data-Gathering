import speech_recognition as sr

def transcribe_speech(timeout=3, phrase_time_limit=5):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[🎤 Listening...]")
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = recognizer.recognize_google(audio)
            print(f"[📝 Recognized] {text}")
            return text
        except sr.WaitTimeoutError:
            print("[⚠️ Timeout] No speech detected.")
        except sr.UnknownValueError:
            print("[⚠️ Error] Could not understand audio.")
        except sr.RequestError:
            print("[❌ Error] Could not connect to Google Speech API.")
    
    return ""

