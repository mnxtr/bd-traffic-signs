# BD Traffic Signs Android App

Android application for Bangladesh traffic signs with Bengali Text-to-Speech (TTS) support.

## Features

- Display Bangladesh traffic signs
- Bengali Text-to-Speech (TTS) for sign names
- Simple and intuitive UI

## Requirements

- Android Studio Arctic Fox or later
- Android SDK 24 or higher
- Bengali TTS engine (Google Text-to-Speech recommended)

## Setup

1. Open the project in Android Studio
2. Sync Gradle files
3. Ensure Bengali language is available in your device's TTS settings
4. Run the app on an emulator or physical device

## Bengali TTS Setup

To enable Bengali TTS on your Android device:
1. Go to Settings → System → Languages & Input → Text-to-Speech
2. Install Google Text-to-Speech if not already installed
3. Download Bengali language data
4. Set Bengali (বাংলা) as an available language

## Project Structure

```
android-app/
├── app/
│   └── src/
│       └── main/
│           ├── java/com/trafficapp/
│           │   └── MainActivity.java
│           ├── res/
│           │   ├── layout/
│           │   │   └── activity_main.xml
│           │   └── values/
│           │       └── strings.xml
│           └── AndroidManifest.xml
└── build.gradle
```

## Usage

1. Launch the app
2. View the traffic sign displayed
3. Tap "বাংলায় বলুন" button to hear the sign name in Bengali
4. The app will speak the traffic sign name using Bengali TTS

## License

MIT License
