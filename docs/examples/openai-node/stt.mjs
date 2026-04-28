// QORA STT via the openai Node SDK (/v1/audio/transcriptions).
//
// Install: npm install openai
// Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com node stt.mjs recording.wav

import OpenAI from "openai";
import { createReadStream } from "fs";

const client = new OpenAI({
    apiKey: process.env.QORA_KEY,
    baseURL: (process.env.QORA_URL ?? "https://api.example.com") + "/v1",
});

const audioFile = process.argv[2] ?? "recording.wav";

const transcript = await client.audio.transcriptions.create({
    model: "qora-stt",
    file: createReadStream(audioFile),
    language: "en",
});

console.log(transcript.text);
