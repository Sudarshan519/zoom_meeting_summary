import { type NextRequest, NextResponse } from "next/server"
import { OpenAI } from "openai"

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,dangerouslyAllowBrowser: true
})

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const audioFile = formData.get("audio") as File

    if (!audioFile) {
      return NextResponse.json({ error: "No audio file provided" }, { status: 400 })
    }

    // Send to OpenAI's Whisper API for transcription
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: audioFile,
      model: "whisper-1",
    })

    const transcription = transcriptionResponse.text

    // Analyze the transcription using OpenAI
    const analysisResponse = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content:
            "You are an audio analysis assistant. Analyze the following transcription and provide insights about sentiment, key topics, estimated word count, speaking pace, and language detected. Return your analysis as a JSON object with these fields: sentiment, key_topics (array), word_count (number), speaking_pace, confidence_score (number between 0-1), language_detected.",
        },
        {
          role: "user",
          content: `Analyze this transcription: "${transcription}"`,
        },
      ],
      response_format: { type: "json_object" },
    })

    // Parse the analysis JSON
    const analysis = JSON.parse(analysisResponse.choices[0].message.content)

    // Return the results
    return NextResponse.json({
      transcription,
      analysis,
    })
  } catch (error) {
    console.error("Error in transcribe API:", error)
    return NextResponse.json({ error: "Failed to process audio" }, { status: 500 })
  }
}

