import AudioRecorder from "@/components/audio-recorder"

export default function Home() {
  return (
    <main className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-2">Audio Transcription & Analysis</h1>
        <p className="text-muted-foreground mb-8">
          Capture audio from your browser tab and microphone simultaneously for transcription and analysis.
        </p>
        <AudioRecorder />
      </div>
    </main>
  )
}

