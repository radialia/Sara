from responsive_voice import ResponsiveVoice
import speech_recognition as sr




class Azleen:
    def __init__(self) -> None:
        self.name = "Azleen"
        self.speechEngine = ResponsiveVoice()

    def SpeechDriver(self, sentence, rate=0.43, gender=ResponsiveVoice.FEMALE):
        """Speech engine for Azleen"""
        print(f"\nAzleen: {sentence}\n")
        self.speechEngine.say(
            sentence, gender=gender, rate=rate)

    def SpeechRecognizer(self):
        """Speech Recognizer for Azleen"""
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source,0,2)

        try:
            print("Recognizing...")
            transcript = r.recognize_google(audio, language="en-IN")
            print(f"User: {transcript}")

        except:
            return None

        transcript = str(transcript)
        return transcript.lower()



if __name__ == '__main__':
    Assistant = Azleen()

    query = Assistant.SpeechRecognizer()
    Assistant.SpeechDriver(query)
