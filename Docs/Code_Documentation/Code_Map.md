# Code Map

### Basic Transcription--->Summarization--->Ingestion Loop
```mermaid
%%{ init : { "theme" : "forest", "flowchart" : { "curve" : "stepBefore" }}}%%
graph TD
    A1[Get YouTube URL with no options checked] --> B1[Attempt to download m4a audio stream]
    A1[Get YouTube URL with summarize option checked] --> B2[Attempt to download m4a audio stream]
    A2[Get YouTube URL with Audio download option checked] --> B3[Attempt to download m4a audio stream]]
    A3[Get YouTube URL with Video download option checked] --> B4[Attempt to download m4a audio+video stream] ---> FIXME
    A4[Get YouTube URL with both options for Audio and Video] --> B5[Attempt to download m4a audio+video stream] ---> FIXME
    A5[Get YouTube URL with rolling summary and detail set options checked] --> B6[Attempt to download m4a audio stream]]

    B1 -->|Success| C1[Attempt to download audio stream]
    B2 -->|Success| C2[Attempt to download audio stream]
    B3 -->|Success| C2[Attempt to download audio stream]
    B4 -->|Success| C3[Attempt to download audio+video stream]
	B5 -->|Success| C4[Attempt to download audio+video stream]
    B6 -->|Success| C5[Attempt to download audiostream]

    C1 -->|Success| D1[Process Audio stream file]
    C1 -->|Fail| F[Abort process]
    C2 -->|Success| D2[Process Audio stream file]
    C2 -->|Fail| F[Abort process]
    C3 -->|Success| D3[Process Audio stream file]
    C3 -->|Fail| F[Abort process]
    C4 -->|Success| D4[Process audio stream file and video file]
    C4 -->|Fail| F[Abort process]
    C5 -->|Success| D5[Process audio stream file and video file]
    C5 -->|Fail| F[Abort process]
    C6 -->|Success| D6[Process Audio stream file]
    C6 -->|Fail| F[Abort process]

    D1 --> E1[Convert audio file to `.wav` format using ffmpeg]
    D2 --> E2[Convert audio file to `.wav` format using ffmpeg]
    D5 --> E5[Convert audio file to `.wav` format using ffmpeg]
    D3 --> E3[Convert audio file to `.wav` format using ffmpeg, and also combine the audio and video files into one]
    D4 --> E4[Convert audio file to `.wav` format using ffmpeg, and also combine the audio and video files into one]
    D6 --> E6[Convert audio file to `.wav` format using ffmpeg]

    E1 --> F1[Use faster_whisper library to transcribe the spoken words from the `.wav` file]
    E2 --> F2[Use faster_whisper library to transcribe the spoken words from the `.wav` file]
    E2 --> F3[Use faster_whisper library to transcribe the spoken words from the `.wav` file]
	E3 --> F4[Use faster_whisper library to transcribe the spoken words from the `.wav` file, and if in the GUI, present to the user the option to download a video file]
	E4 --> F5[Use faster_whisper library to transcribe the spoken words from the `.wav` file, and if in the GUI, present to the user the option to download an audio & Video file]
	E5 --> F6[Use faster_whisper library to transcribe the spoken words from the `.wav` file, and if in the GUI, present to the user the option to download an audio & Video file]


    F1 --> G1[Save the generated transcript as a JSON object to a local file]
    F2 --> G2[Save the generated transcript as a JSON object to a local file]
    F3 --> G3[Save the generated transcript as a JSON object to a local file]
    F4 --> G4[Save the generated transcript as a JSON object to a local file]
    F5 --> G5[Save the generated transcript as a JSON object to a local file]
    F6 --> G6[Save the generated transcript as a JSON object to a local file]


    G1 --> END/LOOP[If in the CLI, do nothing at this point and end the script, or if there are more entries in the input list, continue looping.\n If in the GUI, then the transcript is displayed to the user]
    G2 --> H2[Identify if any 'custom_prompt' value was passed during initial processing, and if so, combine that with the transcript and send it off to a chat API, if none was provided, it uses the default instead]
    G3 --> END/LOOP[If in the CLI, do nothing at this point and end the script, or if there are more entries in the input list, continue looping.\n If in the GUI, then the transcript is displayed to the user]
    G4 --> END/LOOP[If in the CLI, do nothing at this point and end the script, or if there are more entries in the input list, continue looping.\n If in the GUI, then the transcript is displayed to the user]
    G5 --> END/LOOP[If in the CLI, do nothing at this point and end the script, or if there are more entries in the input list, continue looping.\n If in the GUI, then the transcript is displayed to the user]
    G6 --> H5/H6[Split prompt into N chunks, each chunk being a Y amount of tokens, and where n is an amount determined by the detail slider. The chunking implementation used is determined by what is selected in the UI/passed in the CLI args]

    H2 --> I2[Submit the transcript + custom_prompt variable contents to the selected LLM API Endpoint]
    H5 --> I5[Submit each chunk + custom_prompt variable to the selected LLM API Endpoint, append each output to each other and process each chunk individually until all chunks are done]
    H6 --> I6[If recursive summarization was selected, each chunk + custom_prompt variable is submitted to the selected LLM API Endpoint, with each output appended to the prior and processed until all chunks are done]

    I2 --> K[Receive the generated summary, save it to a file, and then display it to the user, along with ingesting it into the database]
	I5 --> K[Receive the generated summary, save it to a file, and then display it to the user, along with ingesting it into the database]
	I6 --> K[Receive the generated summary, save it to a file, and then display it to the user, along with ingesting it into the database]

    K --> L[END or LOOP if more URLs/files in the input list]
```