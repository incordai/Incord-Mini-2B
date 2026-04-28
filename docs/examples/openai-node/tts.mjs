// QORA TTS via the openai Node SDK (/v1/audio/speech).
//
// Install: npm install openai
// Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com node tts.mjs
//
// Output is saved to output.wav.

import OpenAI from "openai";
import { writeFileSync } from "fs";

const client = new OpenAI({
    apiKey: process.env.QORA_KEY,
    baseURL: (process.env.QORA_URL ?? "https://api.example.com") + "/v1",
});

const response = await client.audio.speech.create({
    model: "qora-tts",
    voice: "en-female-1",
    input: "Hello, this is QORA text-to-speech.",
    response_format: "wav",
});

const buffer = Buffer.from(await response.arrayBuffer());
writeFileSync("output.wav", buffer);
console.log("Audio saved to output.wav");
