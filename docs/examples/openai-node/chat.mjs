// QORA chat completion via the openai Node SDK.
//
// Install: npm install openai
// Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com node chat.mjs

import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.QORA_KEY,
    baseURL: (process.env.QORA_URL ?? "https://api.example.com") + "/v1",
});

const resp = await client.chat.completions.create({
    model: "incord-mini-2b",
    messages: [{ role: "user", content: "Hello!" }],
    max_tokens: 200,
});
console.log(resp.choices[0].message.content);
