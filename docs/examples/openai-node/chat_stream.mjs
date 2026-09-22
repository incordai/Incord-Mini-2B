// Streaming chat completion via the openai Node SDK.

import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.INCORD_KEY,
    baseURL: (process.env.INCORD_URL ?? "https://api.example.com") + "/v1",
});

const stream = await client.chat.completions.create({
    model: "incord-mini-2b",
    messages: [{ role: "user", content: "Hello!" }],
    max_tokens: 200,
    stream: true,
});

for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0].delta.content ?? "");
}
process.stdout.write("\n");
