// Incord TTS via the openai Node SDK (/v1/audio/speech).
//
// Install: npm install openai
// Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com node tts.mjs
//
// Output is saved to output.wav.

import OpenAI from "openai";
import { writeFileSync } from "fs";

const client = new OpenAI({
    apiKey: process.env.INCORD_KEY,
    baseURL: (process.env.INCORD_URL ?? "https://api.example.com") + "/v1",
});

const response = await client.audio.speech.create({
    model: "incord-tts",
    voice: "en-female-1",
    input: "Hello, this is Incord text-to-speech.",
    response_format: "wav",
});

const buffer = Buffer.from(await response.arrayBuffer());
writeFileSync("output.wav", buffer);
console.log("Audio saved to output.wav");
