"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { AlertCircle, Mic, Pause, Play, Send, Share2, StopCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import AudioVisualizer from "./audio-visualizer"
import { sendAudioChunkForProcessing, sendAudioForProcessing } from "@/lib/audio-service"

export default function AudioRecorder() {
  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioData, setAudioData] = useState<Blob | null>(null)
  const [transcription, setTranscription] = useState("")
  const [partialTranscription, setPartialTranscription] = useState("")
  const [analysis, setAnalysis] = useState<any>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isStreamingTranscription, setIsStreamingTranscription] = useState(false)
  const [audioSource, setAudioSource] = useState<"both" | "mic" | "tab">("mic") // Default to mic only
  const [error, setError] = useState<string | null>(null)
  const [tabAudioAvailable, setTabAudioAvailable] = useState(true) // Assume available until proven otherwise
  const [chunkProcessingError, setChunkProcessingError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const micStreamRef = useRef<MediaStream | null>(null)
  const tabStreamRef = useRef<MediaStream | null>(null)
  const combinedStreamRef = useRef<MediaStream | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const transcriptionTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const processingChunkRef = useRef<boolean>(false)

  // Check if tab audio is available
  useEffect(() => {
    // Feature detection for getDisplayMedia
    if (!navigator.mediaDevices || !("getDisplayMedia" in navigator.mediaDevices)) {
      setTabAudioAvailable(false)
      // If tab audio was selected but isn't available, default to mic
      if (audioSource === "tab" || audioSource === "both") {
        setAudioSource("mic")
      }
    }
  }, [audioSource])

  // Clean up function to stop all streams and recording
  const cleanupStreams = () => {
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((track) => track.stop())
      micStreamRef.current = null
    }

    if (tabStreamRef.current) {
      tabStreamRef.current.getTracks().forEach((track) => track.stop())
      tabStreamRef.current = null
    }

    if (combinedStreamRef.current) {
      combinedStreamRef.current.getTracks().forEach((track) => track.stop())
      combinedStreamRef.current = null
    }

    if (mediaRecorderRef.current) {
      if (mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop()
      }
      mediaRecorderRef.current = null
    }

    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }

    if (transcriptionTimeoutRef.current) {
      clearTimeout(transcriptionTimeoutRef.current)
      transcriptionTimeoutRef.current = null
    }

    audioChunksRef.current = []
  }

  // Handle recording timer
  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1)
      }, 1000)
    } else if (timerRef.current) {
      clearInterval(timerRef.current)
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [isRecording, isPaused])

  // Format time for display
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  // Process audio chunk
  const processAudioChunk = async (chunk: Blob) => {
    if (!isStreamingTranscription || isPaused || processingChunkRef.current) return

    try {
      processingChunkRef.current = true
      setChunkProcessingError(null)

      // Only process chunks that are at least 1KB in size
      if (chunk.size < 1024) {
        processingChunkRef.current = false
        return
      }

      // Create a new chunk with proper MIME type
      const properChunk = new Blob([chunk], { type: "audio/webm" })

      // Process the chunk
      const result = await sendAudioChunkForProcessing(properChunk)

      if (result.transcription) {
        setPartialTranscription((prev) => {
          // If the new transcription is significantly different, replace it
          // Otherwise append to existing transcription
          if (result.transcription.length > 10 && !prev.includes(result.transcription.substring(0, 10))) {
            return prev + " " + result.transcription
          }
          return result.transcription
        })
      }

      if (result.analysis) {
        setAnalysis(result.analysis)
      }
    } catch (err) {
      console.error("Error processing audio chunk:", err)
      setChunkProcessingError("Failed to process audio chunk. Live transcription may be unavailable.")
    } finally {
      processingChunkRef.current = false
    }
  }

  // Start recording function
  const startRecording = async () => {
    try {
      setError(null)
      setChunkProcessingError(null)
      audioChunksRef.current = []
      setRecordingTime(0)
      setPartialTranscription("")
      setTranscription("")
      setAnalysis(null)

      // Get microphone stream if needed
      if (audioSource === "mic" || audioSource === "both") {
        try {
          micStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true })
        } catch (err) {
          throw new Error("Microphone access denied. Please allow microphone access and try again.")
        }
      }

      // Get tab audio stream if needed
      if ((audioSource === "tab" || audioSource === "both") && tabAudioAvailable) {
        try {
          // @ts-ignore - TypeScript doesn't recognize getDisplayMedia options
          tabStreamRef.current = await navigator.mediaDevices.getDisplayMedia({
            video: true,
            audio: true,
          })

          // Check if audio track was actually obtained
          const hasAudioTrack = tabStreamRef.current.getAudioTracks().length > 0
          if (!hasAudioTrack && audioSource === "tab") {
            throw new Error(
              "No audio track available from the selected tab. Please make sure the tab is playing audio and try again.",
            )
          }
        } catch (err) {
          // If tab audio fails but we're in "both" mode, continue with just mic
          if (audioSource === "both" && micStreamRef.current) {
            console.warn("Tab audio capture failed, continuing with microphone only")
            setError("Tab audio capture failed, continuing with microphone only")
          } else {
            // If we're in tab-only mode or both mic and tab failed, throw error
            setTabAudioAvailable(false)
            throw new Error(
              "Screen sharing access denied or not supported in this environment. Try using microphone only.",
            )
          }
        }
      }

      // Combine streams if both are selected and available
      if (audioSource === "both" && micStreamRef.current && tabStreamRef.current) {
        try {
          const audioContext = new AudioContext()

          // Create sources for each stream
          const micSource = audioContext.createMediaStreamSource(micStreamRef.current)
          const tabSource = audioContext.createMediaStreamSource(tabStreamRef.current)

          // Create a destination for the combined audio
          const destination = audioContext.createMediaStreamDestination()

          // Connect both sources to the destination
          micSource.connect(destination)
          tabSource.connect(destination)

          combinedStreamRef.current = destination.stream
        } catch (err) {
          console.error("Error combining audio streams:", err)
          // If combining fails, fall back to just microphone
          if (micStreamRef.current) {
            setError("Failed to combine audio streams, using microphone only")
          } else {
            throw new Error("Failed to set up audio recording")
          }
        }
      }

      // Determine which stream to use for recording
      let streamToRecord: MediaStream | null = null

      if (audioSource === "both" && combinedStreamRef.current) {
        streamToRecord = combinedStreamRef.current
      } else if (audioSource === "both" && micStreamRef.current && !combinedStreamRef.current) {
        // Fallback to mic only if combined stream failed
        streamToRecord = micStreamRef.current
      } else if (audioSource === "mic" && micStreamRef.current) {
        streamToRecord = micStreamRef.current
      } else if (audioSource === "tab" && tabStreamRef.current) {
        streamToRecord = tabStreamRef.current
      }

      if (!streamToRecord) {
        throw new Error("Failed to create audio stream")
      }

      // Create and configure MediaRecorder with supported MIME type
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/mp4"

      mediaRecorderRef.current = new MediaRecorder(streamToRecord, {
        mimeType,
        audioBitsPerSecond: 128000,
      })

      // Collect chunks and process them if streaming is enabled
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)

          // Process this chunk if streaming is enabled
          if (isStreamingTranscription && !processingChunkRef.current) {
            // Use a timeout to avoid processing every tiny chunk
            if (transcriptionTimeoutRef.current) {
              clearTimeout(transcriptionTimeoutRef.current)
            }

            transcriptionTimeoutRef.current = setTimeout(() => {
              processAudioChunk(event.data)
            }, 1000) // Process chunks with a slight delay
          }
        }
      }

      mediaRecorderRef.current.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType })
        setAudioData(audioBlob)

        // Save the partial transcription as the final transcription when recording stops
        if (partialTranscription) {
          setTranscription(partialTranscription)
        }
      }

      // Start recording with smaller timeslice for more frequent chunks (1000ms)
      // Using a larger timeslice to reduce the number of API calls
      mediaRecorderRef.current.start(1000)
      setIsRecording(true)
      setIsPaused(false)
    } catch (err) {
      console.error("Error starting recording:", err)
      setError(err instanceof Error ? err.message : "Failed to start recording")
      cleanupStreams()
    }
  }

  // Stop recording function
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
    }

    setIsRecording(false)
    setIsPaused(false)
    setIsStreamingTranscription(false)

    // Stop all tracks and clean up
    cleanupStreams()
  }

  // Pause/resume recording
  const togglePause = () => {
    if (!mediaRecorderRef.current) return

    if (isPaused) {
      // Resume recording
      mediaRecorderRef.current.resume()
      setIsPaused(false)
    } else {
      // Pause recording
      mediaRecorderRef.current.pause()
      setIsPaused(true)
    }
  }

  // Toggle streaming transcription
  const toggleStreamingTranscription = () => {
    setIsStreamingTranscription((prev) => !prev)
  }

  // Send audio for processing
  const processAudio = async () => {
    if (!audioData) return

    try {
      setIsProcessing(true)
      setError(null)

      const result = await sendAudioForProcessing(audioData)

      setTranscription(result.transcription)
      setAnalysis(result.analysis)
    } catch (err) {
      console.error("Error processing audio:", err)
      setError(err instanceof Error ? err.message : "Failed to process audio")
    } finally {
      setIsProcessing(false)
    }
  }

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      cleanupStreams()
    }
  }, [])

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between flex-wrap gap-4">
            <span>Audio Recorder</span>
            <div className="flex items-center">
              <div className="flex rounded-md border">
                <Button
                  type="button"
                  variant="ghost"
                  className={`rounded-r-none ${audioSource === "both" ? "bg-primary text-primary-foreground" : ""}`}
                  onClick={() => setAudioSource("both")}
                  disabled={isRecording || !tabAudioAvailable}
                >
                  Both
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className={`rounded-none border-x ${audioSource === "mic" ? "bg-primary text-primary-foreground" : ""}`}
                  onClick={() => setAudioSource("mic")}
                  disabled={isRecording}
                >
                  <Mic className="h-4 w-4 mr-1" /> Mic Only
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className={`rounded-l-none ${audioSource === "tab" ? "bg-primary text-primary-foreground" : ""}`}
                  onClick={() => setAudioSource("tab")}
                  disabled={isRecording || !tabAudioAvailable}
                >
                  <Share2 className="h-4 w-4 mr-1" /> Tab Only
                </Button>
              </div>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center space-y-4">
            {!tabAudioAvailable && (audioSource === "tab" || audioSource === "both") && (
              <Alert variant="warning" className="mb-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Tab audio not available</AlertTitle>
                <AlertDescription>
                  Screen/tab audio capture is not available in this environment. This feature requires a secure context
                  and may not work in preview modes or iframes. Try using microphone only.
                </AlertDescription>
              </Alert>
            )}

            <AudioVisualizer isRecording={isRecording} isPaused={isPaused} />

            {error && <div className="w-full p-3 bg-red-50 text-red-600 rounded-md border border-red-200">{error}</div>}

            {chunkProcessingError && (
              <div className="w-full p-3 bg-yellow-50 text-yellow-600 rounded-md border border-yellow-200">
                {chunkProcessingError}
              </div>
            )}

            <div className="flex items-center gap-2">
              <div className="text-2xl font-mono">{formatTime(recordingTime)}</div>
              {isStreamingTranscription && isRecording && (
                <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                  Live Transcription
                </Badge>
              )}
            </div>

            <div className="flex items-center space-x-2 w-full justify-center">
              <Switch
                id="live-transcription"
                checked={isStreamingTranscription}
                onCheckedChange={toggleStreamingTranscription}
                disabled={isRecording}
              />
              <Label htmlFor="live-transcription">Live Transcription</Label>
            </div>
          </div>
        </CardContent>
        <CardFooter className="flex justify-center space-x-4">
          {!isRecording ? (
            <Button onClick={startRecording} disabled={isProcessing}>
              <Play className="h-4 w-4 mr-2" /> Start Recording
            </Button>
          ) : (
            <>
              <Button variant="outline" onClick={togglePause}>
                {isPaused ? (
                  <>
                    <Play className="h-4 w-4 mr-2" /> Resume
                  </>
                ) : (
                  <>
                    <Pause className="h-4 w-4 mr-2" /> Pause
                  </>
                )}
              </Button>
              <Button variant="destructive" onClick={stopRecording}>
                <StopCircle className="h-4 w-4 mr-2" /> Stop Recording
              </Button>
            </>
          )}
        </CardFooter>
      </Card>

      {/* Live transcription display */}
      {isStreamingTranscription && isRecording && partialTranscription && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Live Transcription
              <span className="animate-pulse">
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 mr-1"></span>
                <span className="inline-block h-2 w-2 rounded-full bg-green-500 mr-1"></span>
                <span className="inline-block h-2 w-2 rounded-full bg-green-500"></span>
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="whitespace-pre-wrap bg-muted p-4 rounded-md">{partialTranscription}</div>
          </CardContent>
        </Card>
      )}

      {audioData && !isRecording && (
        <Card>
          <CardHeader>
            <CardTitle>Recording Results</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <audio controls className="w-full">
                <source src={URL.createObjectURL(audioData)} type="audio/webm" />
                Your browser does not support the audio element.
              </audio>

              {!transcription && (
                <div className="flex justify-end">
                  <Button onClick={processAudio} disabled={isProcessing}>
                    <Send className="h-4 w-4 mr-2" />
                    {isProcessing ? "Processing..." : "Send for Transcription & Analysis"}
                  </Button>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {(transcription || analysis) && (
        <Tabs defaultValue="transcription">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="transcription">Transcription</TabsTrigger>
            <TabsTrigger value="analysis">Analysis</TabsTrigger>
          </TabsList>
          <TabsContent value="transcription">
            <Card>
              <CardHeader>
                <CardTitle>Transcription</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="whitespace-pre-wrap bg-muted p-4 rounded-md">
                  {transcription || "No transcription available."}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="analysis">
            <Card>
              <CardHeader>
                <CardTitle>Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                {analysis ? (
                  <div className="space-y-4">
                    {Object.entries(analysis).map(([key, value]) => (
                      <div key={key} className="border-b pb-2">
                        <h3 className="font-medium capitalize">{key.replace(/_/g, " ")}</h3>
                        <p>{String(value)}</p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-muted p-4 rounded-md">No analysis available.</div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  )
}

