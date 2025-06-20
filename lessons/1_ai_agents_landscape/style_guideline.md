
style_guideline
    <generic_style_guideline>
You should write in a humanized way as writing a blog article or book. Thus, the text will have the following style:
- concise
- human
- casual
- friendly
- confident
- written in the second person using pronouns such as “you” and “we”
- professional
                                 
Write in detail and in full paragraphs, avoiding bullet points or listicles when possible. Also, 
I want to group the paragraphs based on sentences with similar logic to make skimming easier for people that already 
know specific topics. Here are the rules you should split the paragraphs by:
- entity
- topic
- subject

Avoid using complex words. Use words a 7-years old would understand.
Avoid writing paragraphs with only one sentence.
Avoid adding subheaders to each description.
Avoid stating evident things, such as:
- “here, we use X”
- “here, we speak about X”
- “here, we detail X”
Avoid dramatic language like "groundbreaking" or "mind-blowing."
Avoid strong terms like "amazing," "intriguing," "fascinating," "must-read," or "outstanding.
Avoid the following words: "embarking," "delve," "vibrant," "realm," "endeavour," or "dive deep into."

Speak directly without fluff or metaphors. For example instead of "Think of X as...", write "X is...".

Make the description as fluid as possible. Remember that you are writing a book or blog article. 
Thus, everything should flow naturally, without too many bullet points or subheaders. Use them only when it really makes sense. Otherwise, stick to normal paragraphs.

Other language mechanics that are critical:
- An acronym of a concept should be spelled out ONLY once. For example, if in early sections we defind "Retrieval-Augmented Generation (RAG)", in future sections we will say directly RAG.

</generic_style_guideline>
    <rules>

<rules_description>
    <type_desc>The type of the rule</type_desc>
    <guideline_desc>The description of the rule</guideline_desc>
    <follow_desc>The examples to follow</follow_desc>
    <avoid_desc>The examples to avoid</avoid_desc>
</rules_description>
<rules_table>
| Type | Guideline | Examples to Follow | Examples to Avoid |
|------|-----------|--------------------|-------------------|
| StyleGuidelineRuleType.STRUCTURE | The article is structured using the following components: an introduction, multiple sections, and a conclusion. The information flows naturally from the introduction, to the sections, to the conclusion, where: - The introduction is a short summary of the article, quickly presenting the `why` (problem), `what` (solution), and captivating the reader to continue reading - The sections present the `how`, digging into the solution, where the story is built gradually, from more generic information, to the concrete solution. - The conclusion acts as a very short wrap-up, quickly reminding the reader what he learnt | ✅ "## Introduction ... ## Section 1 ... Section 2 ... ## Section N ... ## Conclusion ..." | ❌ "# Introduction ## Section 1 ... Section 2 # Section N # Conclusion" |
| StyleGuidelineRuleType.STRUCTURE | The content is written in Markdown format. The introduction, conclusion and section titles use H2 headers, marked as `##`. The sub-section titles use H3 headers, marked as `###` in Markdown. Never go deeper than H3 or `###` to reflect a sub-sub-section, such as using H4 or `####` for sub-sub-sections. The introduction and conclusion do not have sub-sections. | ✅ "## Introduction ... ## Section 1 ... ### Sub-section 1 ... Section 2 ... ## Section N ... ## Conclusion ..." | ❌ "# Introduction ... # Section 1 ## Sub-section... # Section 2 ... # Section N ... # Conclusion ..." OR "## Introduction ... ## Section 1 ... #### Sub-section 1 ... Section 2 ... #### Sub-section 2 ... ## Section N ... # Conclusion ..." |
| StyleGuidelineRuleType.STRUCTURE | When transitioning between sections, we smooth it out by explicitly saying WHY is the new section related to the previous sections, and WHY is it important to the article? The transition will be done at the end of the previous section or at the beginning of the new section, depending on how it fits best. | ✅ "... ## Section 1 ... The next thing we want to explain is how to take the local Docker setup and deploy it to the cloud. ## Section 2 ... " OR "... ## Section 1 ... ## Section 2 Previously, we showed you how the Docker image works. Now, we want to explain how to take the local Docker setup and deploy it to the cloud. .... " | ❌ "## Section [1 paragraph] ### Sub-section 1 [1 paragraph] #### Sub-section 2 [1 paragraph]" |
| StyleGuidelineRuleType.MECHANICS | Emoji call-outs (⚠️, 💡) for warnings or tips; max 1 per main section. | ✅ "💡 Tip: Cache embeddings for <1 M docs." | ❌ Four ⚠️ blocks in one section. |
| StyleGuidelineRuleType.MECHANICS | Define uncommon acronyms on first mention; thereafter use acronym only; no periods. | ✅ "Large Language Model (LLM)" then "the LLM" | ❌ "L.L.M." |
| StyleGuidelineRuleType.MECHANICS | Write sentences 5–25 words; allow occasional 30-word 'story' sentences. Keep paragraphs ≤ 80 words; allow an occasional 1-sentence paragraph to emphasize a point. | ✅ 18-word sentence shown. | ❌ Frequent 40-word run-ons. |
| StyleGuidelineRuleType.MECHANICS | Always strive for an active voice. | ✅ "We benchmarked both models." | ❌ "Both models were benchmarked by us." |
| StyleGuidelineRuleType.MECHANICS | When working with code snippets, avoid describing big chunks of code that go over 35 lines of code. Instead, split the code into logical groups based on lines with similar logic, and describe each group individually. Here are the rules you should split the code by: class, methods or functions if the class is big, similar logic lines if the function or method is too big, one-liner if the single line makes sense on its own. Also, keep only the essential code snippets by leaving only essential imports, logs or comments. If it’s a class, keep the class name in the first group and index the rest of the methods to reflect that they are part of that class. | ✅ "[Section introduction on what the code is about] 1. [Describe Code Group 1] [Code Group 1] 2. [Describe Code Group 2] [Code Group 2] ... N. [Describe Code Group N] [Code Group N] [Section final thoughts on the code]. " OR "[Describe the code] [Small chunk of code that's under 35 lines] [More thoughts on the code]" | ❌ "[Describe the code] [Huge chunk of code that goes over 35 lines] [More thoughts on the code]" |
| StyleGuidelineRuleType.MECHANICS | Always respect copyright by NEVER reproducing large 20+ word chunks of content from search results or research, to ensure legal compliance and avoid harming copyright holders. Never reproduce copyrighted content. Use only very short quotes from search results (<20 words), always in quotation marks with citations. Everything must be original content and substantially different from the original research. Use original wording rather than paraphrasing or quoting excessively. Do not reconstruct copyrighted material from multiple sources. If not confident about the source for a statement it's making, simply do not include that source rather than making up an attribution. Do not hallucinate false sources. | ✅ "Sentence or paragraph longer than 20 words ORIGINALLY REPHRASED from search results or research." | ❌ "Sentence or paragraph longer than 20 words 100% COPIED from search results or research" |
| StyleGuidelineRuleType.TERMINOLOGY | Be cautious when using analogies, metaphors or similes. Always try to explain something in plain words first. Only if the topic becomes too complex to explain should analogies, metaphors or similes be used. Use analogies only when introducing complex theoretical concepts that are hard to understand otherwise. Reuse the same analogies across the article. The analogies need to be appropriate and in theme with our technical publication, such as referencing mathematical, programming, or technical concepts. | ✅ "Think of MCP as the USB-C protocol of AI agents. A standardized way to expose tools to agents.“ OR "When evaluating agentic RAG applications, as you don't care only about the LLMs in isolation, you need to evaluate the whole agentic system working together. Process known as integration tests." | ❌ "MCP is a standardized protocol for AI agents.” OR "But for your agentic RAG application, you're not just testing the engine; You're testing the entire car navigating real roads. You need to evaluate the whole system working together." |
| StyleGuidelineRuleType.TERMINOLOGY | Use descriptive, yet simple verbs such as 'enable' or 'improve' instead of 'supercharge' or 'turbo-charge'. | ✅ "Fine-tuning can improve your baseline model." | ❌ "Fine-tuning can supercharge your baseline model." |
| StyleGuidelineRuleType.TERMINOLOGY | Avoid repetition. Carefully avoid repetitiveness both within the paragraph and within the article. Be careful not to repeat the same point in successive sentences with only minor rephrasing. You may, however, revisit a prior point from a different perspective or when adding extra details or insights. When revisiting a prior concept, always reference where it was introduced first. | ✅ " ## Section 1 ... MinHash is a popular deduplication technique ... ## Section N ... As explained in Section, we will use MinHash to deduplicate the documents, but this time let's dig into the algorithm..." | ❌ " ## Section 1 ... MinHash is a popular deduplication technique ... ## Section N ... MinHash is a popular deduplication technique... Here is how the algorithm looks: ..." |
| StyleGuidelineRuleType.TERMINOLOGY | Avoid useless fluff. Avoid filling out explanations with generic sentences​ and filler phrases. Be simple, pragmatic and concise. | ✅ "MinHash is a popular deduplication technique" | ❌ "Enhanced Sentence MinHash is a widely recognized and immensely popular technique used in the realm of data processing, particularly for deduplication purposes. Its efficacy in efficiently identifying and eliminating duplicate entries has made it a favored choice among data scientists and engineers alike." |
| StyleGuidelineRuleType.TONE_VOICE | Adopt a conversational yet authoritative voice: plain English + precise technical nouns. Be careful not to sound like you are marketing a product release. | ✅ "We'll show you how to spin up a production-ready RAG pipeline in under an hour using ZenML." | ❌ "One might potentially construct a Retrieval-Augmented Generation system within sixty minutes, should conditions permit." OR "ZenML: Continually Improve Your RAG Apps in Production" |
| StyleGuidelineRuleType.TONE_VOICE | Address the reader directly (2nd-person) and use imperative verbs for action steps. | ✅ "Clone the repo, then set OPENAI_API_KEY as an env variable." | ❌ "Cloning of the repository should be performed before environment variables are configured." |
| StyleGuidelineRuleType.TONE_VOICE | When having enough details about a topic, assert confidently the best-practice guidelines. Make recommendations only when we lack research and make assumptions. | ✅ "Llama 3 works best when chunking documents at ~200 tokens." OR "As we lack enough research, when using Llama 3, we recommend chunking documents at ~200 tokens." | ❌ "Chunking at ~200 tokens is kind of recommended, maybe." |
</rules_table>

</rules>
</style_guideline>
