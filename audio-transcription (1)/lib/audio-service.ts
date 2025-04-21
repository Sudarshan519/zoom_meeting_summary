"use server"

import { OpenAI } from "openai"
import { writeFile } from "fs/promises"
import { join } from "path"
import { tmpdir } from "os"
const fs = require('fs')
// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
})

/**
 * Sends an audio chunk for real-time processing
 */
export async function sendAudioChunkForProcessing(audioChunk: Blob) {
  try {
    // Convert blob to array buffer for server processing
    const arrayBuffer = await audioChunk.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Create a temporary file path
    const tempFilePath = join(tmpdir(), `audio-chunk-${Date.now()}.webm`)

    // Write the buffer to a temporary file
    await writeFile(tempFilePath, buffer)

    // Send to OpenAI's Whisper API for transcription using the file path
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempFilePath),
      model: "whisper-1",
    });

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

        analysis = JSON.parse(analysisResponse.choices[0].message.content??'')
      } catch (error) {
        console.warn("Skipping analysis for this chunk due to error:", error)
      }
    }

    // Return the results
    return {
      transcription,
      analysis,
    }
  } catch (error) {
    console.error("Error in sendAudioChunkForProcessing:", error)
    // Return empty results instead of throwing to avoid disrupting the recording
    return {
      transcription: "",
      analysis: null,
    }
  }
}

/**
 * Sends complete audio data to the server for transcription and analysis
 */
export async function sendAudioForProcessing(audioBlob: Blob) {
  try {
    // Convert blob to array buffer for server processing
    const arrayBuffer = await audioBlob.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)

    // Create a temporary file path
    const tempFilePath = join(tmpdir(), `audio-${Date.now()}.webm`)

    // Write the buffer to a temporary file
    await writeFile(tempFilePath, buffer)

    // Send to OpenAI's Whisper API for transcription using the file path
    const transcriptionResponse = await openai.audio.transcriptions.create({
      file: fs.createReadStream(tempFilePath),
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
    const analysis = JSON.parse(analysisResponse.choices[0].message.content??'')

    // Return the results
    return {
      transcription,
      analysis,
    }
  } catch (error) {
    console.error("Error in sendAudioForProcessing:", error)
    throw new Error("Failed to process audio")
  }
}

