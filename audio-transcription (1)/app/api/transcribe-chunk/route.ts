import { type NextRequest, NextResponse } from "next/server"
import { OpenAI } from "openai"

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const audioChunk = formData.get("audio") as File

    if (!audioChunk) {
      return NextResponse.json({ error: "No audio chunk provided" }, { status: 400 })
    }

    // Send to OpenAI's Whisper API for transcription
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: audioChunk,
      model: "whisper-1",
    })

    const transcription = transcriptionResponse.text

    // For chunks, we'll do a simpler analysis or skip it to improve performance
    // Only do analysis if there's meaningful transcription
    let analysis = null
    if (transcription && transcription.split(" ").length > 3) {
      try {
        const analysisResponse = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content:
                "You are an audio analysis assistant. Analyze the following transcription and provide a brief sentiment analysis. Return your analysis as a JSON object with these fields: sentiment (string), key_topics (array of strings, max 3).",
            },
            {
              role: "user",
              content: `Analyze this transcription: "${transcription}"`,
            },
          ],
          response_format: { type: "json_object" },
          max_tokens: 150, // Limit token usage for faster response
        })

        analysis = JSON.parse(analysisResponse.choices[0].message.content)
      } catch (error) {
        console.warn("Skipping analysis for this chunk due to error:", error)
      }
    }

    // Return the results
    return NextResponse.json({
      transcription,
      analysis,
    })
  } catch (error) {
    console.error("Error in transcribe chunk API:", error)
    return NextResponse.json({
      transcription: "",
      analysis: null,
    })
  }
}

