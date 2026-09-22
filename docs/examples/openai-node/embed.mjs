// Incord embeddings via the openai Node SDK (/v1/embeddings).
//
// Install: npm install openai
// Run:     INCORD_KEY=incord_xxx INCORD_URL=https://api.example.com node embed.mjs

import OpenAI from "openai";

const client = new OpenAI({
    apiKey: process.env.INCORD_KEY,
    baseURL: (process.env.INCORD_URL ?? "https://api.example.com") + "/v1",
});

const response = await client.embeddings.create({
    model: "incord-embed",
    input: ["Hello world", "Incord embedding example"],
});

for (const item of response.data) {
    console.log(`index=${item.index}  dims=${item.embedding.length}  first5=${item.embedding.slice(0, 5)}`);
}
