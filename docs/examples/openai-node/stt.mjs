// Incord STT via the openai Node SDK (/v1/audio/transcriptions).
//
// Install: npm install openai
// Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com node stt.mjs recording.wav

import OpenAI from "openai";
import { createReadStream } from "fs";

const client = new OpenAI({
    apiKey: process.env.INCORD_KEY,
    baseURL: (process.env.INCORD_URL ?? "https://api.example.com") + "/v1",
});

const audioFile = process.argv[2] ?? "recording.wav";

const transcript = await client.audio.transcriptions.create({
    model: "incord-stt",
    file: createReadStream(audioFile),
    language: "en",
});

console.log(transcript.text);
