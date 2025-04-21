"use client"

import { useRef, useEffect } from "react"

interface AudioVisualizerProps {
  isRecording: boolean
  isPaused: boolean
}

export default function AudioVisualizer({ isRecording, isPaused }: AudioVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Set canvas dimensions
    const setCanvasDimensions = () => {
      const dpr = window.devicePixelRatio || 1
      canvas.width = canvas.offsetWidth * dpr
      canvas.height = canvas.offsetHeight * dpr
      ctx.scale(dpr, dpr)
    }

    setCanvasDimensions()
    window.addEventListener("resize", setCanvasDimensions)

    // Animation function
    const animate = () => {
      if (!ctx) return

      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (isRecording && !isPaused) {
        // Draw active visualization
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const maxRadius = Math.min(centerX, centerY) * 0.8

        // Draw circles
        const time = Date.now() / 1000
        const numCircles = 3

        for (let i = 0; i < numCircles; i++) {
          const phase = (i / numCircles) * Math.PI * 2
          const pulseFactor = 0.2 * Math.sin(time * 3 + phase) + 0.8
          const radius = maxRadius * (0.3 + (0.7 * (i + 1)) / numCircles) * pulseFactor

          ctx.beginPath()
          ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
          ctx.strokeStyle = `rgba(79, 70, 229, ${0.8 - i * 0.2})`
          ctx.lineWidth = 2
          ctx.stroke()
        }

        // Draw waveform
        ctx.beginPath()
        ctx.moveTo(0, centerY)

        const waveAmplitude = canvas.height * 0.1
        const waveFrequency = 0.02

        for (let x = 0; x < canvas.width; x++) {
          const y = centerY + Math.sin(x * waveFrequency + time * 5) * waveAmplitude * (1 + 0.5 * Math.sin(time * 2))
          ctx.lineTo(x, y)
        }

        ctx.strokeStyle = "rgba(79, 70, 229, 0.5)"
        ctx.lineWidth = 2
        ctx.stroke()
      } else if (isPaused) {
        // Draw paused visualization
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const size = Math.min(centerX, centerY) * 0.3

        // Draw pause symbol
        ctx.fillStyle = "rgba(79, 70, 229, 0.7)"
        ctx.fillRect(centerX - size - 10, centerY - size, size, size * 2)
        ctx.fillRect(centerX + 10, centerY - size, size, size * 2)
      } else {
        // Draw inactive visualization
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = Math.min(centerX, centerY) * 0.5

        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
        ctx.strokeStyle = "rgba(100, 100, 100, 0.3)"
        ctx.lineWidth = 3
        ctx.stroke()

        // Draw microphone icon
        const micWidth = radius * 0.5
        const micHeight = radius * 0.8

        ctx.beginPath()
        ctx.roundRect(centerX - micWidth / 2, centerY - micHeight / 2, micWidth, micHeight, 5)
        ctx.fillStyle = "rgba(100, 100, 100, 0.3)"
        ctx.fill()

        // Mic stand
        ctx.beginPath()
        ctx.moveTo(centerX, centerY + micHeight / 2)
        ctx.lineTo(centerX, centerY + micHeight / 2 + radius * 0.2)
        ctx.strokeStyle = "rgba(100, 100, 100, 0.3)"
        ctx.lineWidth = micWidth / 4
        ctx.stroke()

        // Mic base
        ctx.beginPath()
        ctx.moveTo(centerX - micWidth / 2, centerY + micHeight / 2 + radius * 0.2)
        ctx.lineTo(centerX + micWidth / 2, centerY + micHeight / 2 + radius * 0.2)
        ctx.stroke()
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animate()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      window.removeEventListener("resize", setCanvasDimensions)
    }
  }, [isRecording, isPaused])

  return (
    <div className="w-full h-40 bg-muted/30 rounded-lg overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" style={{ display: "block" }} />
    </div>
  )
}

