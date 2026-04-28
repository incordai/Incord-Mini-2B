// QORA embeddings via the openai Node SDK (/v1/embeddings).
//
// Install: npm install openai
// Run:     QORA_KEY=qora_xxx QORA_URL=https://api.example.com node embed.mjs

import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.QORA_KEY,
    baseURL: (process.env.QORA_URL ?? "https://api.example.com") + "/v1",
});

const response = await client.embeddings.create({
    model: "qora-embed",
    input: ["Hello world", "QORA embedding example"],
});

for (const item of response.data) {
    console.log(`index=${item.index}  dims=${item.embedding.length}  first5=${item.embedding.slice(0, 5)}`);
}
