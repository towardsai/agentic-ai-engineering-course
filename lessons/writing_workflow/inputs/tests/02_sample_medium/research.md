# Research

## Code Sources

<details>
<summary>Repository analysis for https://github.com/google-gemini/gemini-cli/blob/main/README.md</summary>

# Repository analysis for https://github.com/google-gemini/gemini-cli/blob/main/README.md

## Summary
Repository: google-gemini/gemini-cli
File: README.md
Lines: 211

Estimated tokens: 1.6k

## File tree
```Directory structure:
└── README.md

```

## Extracted content
================================================
FILE: README.md
================================================
# Gemini CLI

[![Gemini CLI CI](https://github.com/google-gemini/gemini-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/google-gemini/gemini-cli/actions/workflows/ci.yml)

![Gemini CLI Screenshot](./docs/assets/gemini-screenshot.png)

This repository contains the Gemini CLI, a command-line AI workflow tool that connects to your
tools, understands your code and accelerates your workflows.

With the Gemini CLI you can:

- Query and edit large codebases in and beyond Gemini's 1M token context window.
- Generate new apps from PDFs or sketches, using Gemini's multimodal capabilities.
- Automate operational tasks, like querying pull requests or handling complex rebases.
- Use tools and MCP servers to connect new capabilities, including [media generation with Imagen,
  Veo or Lyria](https://github.com/GoogleCloudPlatform/vertex-ai-creative-studio/tree/main/experiments/mcp-genmedia)
- Ground your queries with the [Google Search](https://ai.google.dev/gemini-api/docs/grounding)
  tool, built into Gemini.

## Quickstart

You have two options to install Gemini CLI.

### With Node

1. **Prerequisites:** Ensure you have [Node.js version 20](https://nodejs.org/en/download) or higher installed.
2. **Run the CLI:** Execute the following command in your terminal:

   ```bash
   npx https://github.com/google-gemini/gemini-cli
   ```

   Or install it with:

   ```bash
   npm install -g @google/gemini-cli
   ```

   Then, run the CLI from anywhere:

   ```bash
   gemini
   ```

### With Homebrew

1. **Prerequisites:** Ensure you have [Homebrew](https://brew.sh/) installed.
2. **Install the CLI** Execute the following command in your terminal:

   ```bash
   brew install gemini-cli
   ```

   Then, run the CLI from anywhere:

   ```bash
   gemini
   ```

### Common Configuration steps

3. **Pick a color theme**
4. **Authenticate:** When prompted, sign in with your personal Google account. This will grant you up to 60 model requests per minute and 1,000 model requests per day using Gemini.

You are now ready to use the Gemini CLI!

### Use a Gemini API key:

The Gemini API provides a free tier with [100 requests per day](https://ai.google.dev/gemini-api/docs/rate-limits#free-tier) using Gemini 2.5 Pro, control over which model you use, and access to higher rate limits (with a paid plan):

1. Generate a key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set it as an environment variable in your terminal. Replace `YOUR_API_KEY` with your generated key.

   ```bash
   export GEMINI_API_KEY="YOUR_API_KEY"
   ```

3. (Optionally) Upgrade your Gemini API project to a paid plan on the API key page (will automatically unlock [Tier 1 rate limits](https://ai.google.dev/gemini-api/docs/rate-limits#tier-1))

### Use a Vertex AI API key:

The Vertex AI API provides a [free tier](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview) using express mode for Gemini 2.5 Pro, control over which model you use, and access to higher rate limits with a billing account:

1. Generate a key from [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys).
2. Set it as an environment variable in your terminal. Replace `YOUR_API_KEY` with your generated key and set GOOGLE_GENAI_USE_VERTEXAI to true

   ```bash
   export GOOGLE_API_KEY="YOUR_API_KEY"
   export GOOGLE_GENAI_USE_VERTEXAI=true
   ```

3. (Optionally) Add a billing account on your project to get access to [higher usage limits](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas)

For other authentication methods, including Google Workspace accounts, see the [authentication](./docs/cli/authentication.md) guide.

## Examples

Once the CLI is running, you can start interacting with Gemini from your shell.

You can start a project from a new directory:

```sh
cd new-project/
gemini
> Write me a Gemini Discord bot that answers questions using a FAQ.md file I will provide
```

Or work with an existing project:

```sh
git clone https://github.com/google-gemini/gemini-cli
cd gemini-cli
gemini
> Give me a summary of all of the changes that went in yesterday
```

### Next steps

- Learn how to [contribute to or build from the source](./CONTRIBUTING.md).
- Explore the available **[CLI Commands](./docs/cli/commands.md)**.
- If you encounter any issues, review the **[troubleshooting guide](./docs/troubleshooting.md)**.
- For more comprehensive documentation, see the [full documentation](./docs/index.md).
- Take a look at some [popular tasks](#popular-tasks) for more inspiration.
- Check out our **[Official Roadmap](./ROADMAP.md)**

### Troubleshooting

Head over to the [troubleshooting guide](docs/troubleshooting.md) if you're
having issues.

## Popular tasks

### Explore a new codebase

Start by `cd`ing into an existing or newly-cloned repository and running `gemini`.

```text
> Describe the main pieces of this system's architecture.
```

```text
> What security mechanisms are in place?
```

```text
> Provide a step-by-step dev onboarding doc for developers new to the codebase.
```

```text
> Summarize this codebase and highlight the most interesting patterns or techniques I could learn from.
```

```text
> Identify potential areas for improvement or refactoring in this codebase, highlighting parts that appear fragile, complex, or hard to maintain.
```

```text
> Which parts of this codebase might be challenging to scale or debug?
```

```text
> Generate a README section for the [module name] module explaining what it does and how to use it.
```

```text
> What kind of error handling and logging strategies does the project use?
```

```text
> Which tools, libraries, and dependencies are used in this project?
```

### Work with your existing code

```text
> Implement a first draft for GitHub issue #123.
```

```text
> Help me migrate this codebase to the latest version of Java. Start with a plan.
```

### Automate your workflows

Use MCP servers to integrate your local system tools with your enterprise collaboration suite.

```text
> Make me a slide deck showing the git history from the last 7 days, grouped by feature and team member.
```

```text
> Make a full-screen web app for a wall display to show our most interacted-with GitHub issues.
```

### Interact with your system

```text
> Convert all the images in this directory to png, and rename them to use dates from the exif data.
```

```text
> Organize my PDF invoices by month of expenditure.
```

### Uninstall

Head over to the [Uninstall](docs/Uninstall.md) guide for uninstallation instructions.

## Terms of Service and Privacy Notice

For details on the terms of service and privacy notice applicable to your use of Gemini CLI, see the [Terms of Service and Privacy Notice](./docs/tos-privacy.md).

</details>


## YouTube Video Transcripts

<details>
<summary>**Andrej Karpathy: Software in the Era of AI**</summary>

**Andrej Karpathy: Software in the Era of AI**

(On-screen text: Y Combinator Presents AI Startup School)

**Speaker 1:** Please welcome former Director of AI, Tesla, Andrej Karpathy.

(The screen updates to show a black and white photo of Andrej Karpathy with the title "Software in the era of AI". Andrej Karpathy walks onto the stage to applause.)

**Andrej Karpathy:** Hello, hello.

(Applause continues.)

**Andrej Karpathy:** Wow, a lot of people here. Hello.

Um, okay, yeah, so I'm excited to be here today to talk to you about software in the era of AI. And I'm told that many of you are students, like bachelors, masters, PhD, and so on, and you're about to enter the industry. And I think it's actually like an extremely unique and very interesting time to enter the industry right now. And I think fundamentally the reason for that is that, um, software is changing again.

[00:00:30]
(The slide on the screen behind the speaker now shows the text "Software is changing." with "(again)" written in a smaller font below it.)

And I say again because I actually gave this talk already. Um, but the problem is that software keeps changing, so I actually have a lot of material to create new talks. And I think it's changing quite fundamentally. I think roughly speaking, software has not changed much on such a fundamental level for 70 years. And then it's changed, I think, about twice quite rapidly in the last few years. And so there's just a huge amount of work to do, a huge amount of software to write and rewrite.

So let's take a look at maybe the realm of software. (A slide appears titled '"Map of GitHub"'. It displays a dark blue map-like visualization with clusters of interconnected points, each cluster representing a different "dominion" or "land" like "Pythonian Dominion" or "Javian Land".) So if we kind of think of this as like the map of software, this is a really cool tool called Map of GitHub.

[01:00:00]
This is kind of like all the software that's written. Uh these are instructions to the computer for carrying out tasks in the digital space. (The slide zooms into a cluster labeled "Promptoria", showing a dense network of interconnected nodes.) So if you zoom in here, these are all different kinds of repositories and this is all the code that has been written.

And a few years ago, I kind of observed that um, software was kind of changing and there was kind of like a new new type of software around, and I called this software 2.0 at the time. (A new slide appears titled "Software 2.0". It contrasts "Software 1.0 = code" with a picture of source code, against "Software 2.0 = weights" with a diagram of a neural network. It references a blog post by Andrej Karpathy from November 11, 2017.) And the idea here was that software 1.0 is the code you write for the computer. Software 2.0 are basically neural networks, and in particular the weights of a neural network.

[01:30:00]
And you're not writing this code directly. You are most you're more kind of like tuning the data sets and then you're running an optimizer to create the parameters of this neural net. And I think at the time, neural nets were kind of seen as like just a different kind of classifier, like a decision tree or something like that. And so I think it was kind of like um, I think this framing was a lot more appropriate. And now actually what we have is kind of like an equivalent of GitHub in the realm of software 2.0. And I think the Hugging Face uh is basically equivalent of GitHub in software 2.0. (The slide now shows the "Map of GitHub" on the left, labeled "(Software 1.0) computer code", and on the right, a colorful, complex network visualization titled "HuggingFace Model Atlas", labeled "(Software 2.0) neural network weights".) And there's also Model Atlas and you can visualize all the code written there. In case you're curious by the way, the giant circle, the point in the middle, uh these are the parameters of Flux, the image generator.

[02:00:00]
And so anytime someone tunes a LoRA on top of a Flux model, you basically create a Git commit uh in this space and uh you create a different kind of uh image generator. So basically what we have is software 1.0 is the computer code that programs a computer. (A new slide appears titled "Software 1.0", showing a diagram where "computer code" programs a "computer". It includes a black-and-white photo of an early computer operator from the 1940s.) Software 2.0 are the weights which program neural networks. (A second column appears titled "Software 2.0", showing a diagram where "weights" program a "neural net". It includes a diagram of the AlexNet architecture.) And here's an example of AlexNet, an image recognizer neural network. Now, so far, all of the neural networks that we've been familiar with until recently were kind of like fixed-function computers.

[02:30:00]
Image to uh categories or something like that. And I think what's changed and I think is a quite fundamental change is that neural networks became programmable with large language models. (A third column appears titled "Software 3.0", showing a diagram where "prompts" program an "LLM". It includes a diagram of a Transformer architecture.) And so I I see this as quite new, unique. It's a new kind of a computer and uh so in my mind it's uh worth giving it a new designation of software 3.0. And basically your prompts are now programs that program the LLM. And uh remarkably, uh these uh prompts are written in English.

[03:00:00]
So it's kind of a very interesting programming language.

*In this section, Andrej Karpathy introduces the evolution of software from Software 1.0 (explicit code), to Software 2.0 (neural network weights), and finally to Software 3.0 (natural language prompts for LLMs).*

Um, so maybe uh to uh summarize the difference, if you're doing sentiment classification, for example, you can imagine uh writing some uh amount of Python to to basically do sentiment classification, or you can train a neural net, or you can prompt a large language model. (A slide titled "Example: Sentiment Classification" contrasts the three software paradigms for the same task: 1.0 uses Python code, 2.0 trains a classifier on examples, and 3.0 uses a detailed few-shot prompt.) Uh so here I'm this is a few shot prompt and you can imagine changing it and programming the computer in a slightly different way.

[03:30:00]
So basically we have software 1.0, software 2.0, and I think what we're seeing, and maybe you've seen a lot of GitHub code is not just like code anymore. There's a bunch of like English interspersed with code, and so I think kind of there's a growing category of new kind of code. (The "Map of GitHub" slide reappears, now with a third element labeled "(Software 3.0) LLM prompts, in English" shown growing and encroaching on the other two.) So not only is it a new programming paradigm, it's also remarkable to me that it's in our native language of English. And so when this blew my mind a few uh I guess years ago now, uh I tweeted this and um I think it captured the attention of a lot of people and this is my currently pinned tweet uh is that remarkably we're now programming computers in English. (A slide shows a screenshot of Andrej Karpathy's pinned tweet from January 24, 2023: "The hottest new programming language is English".)

Now when I was at uh Tesla, um we were working on the uh autopilot and uh we were trying to get the car to drive. And I sort of showed this slide at the time where you can imagine that the inputs to the car are on the bottom and they're going through a software stack to produce the steering and acceleration.

[04:30:00]
(A slide appears titled "Software is eating the world / Software 2.0 eating Software 1.0." It visually represents the Tesla Autopilot stack as a mix of 1.0 code and a growing portion of 2.0 code (neural networks) for processing camera inputs.) And I made the observation at the time that there was a ton of C++ code around in the autopilot, which was the software 1.0 code, and then there were some neural nets in there doing image recognition. And uh I kind of observed that over time as we made the autopilot better, basically the neural network grew in capability and size and in addition to that, all the C++ code was being deleted and kind of like was um and a lot of the kind of capabilities and functionality that was originally written in 1.0 was migrated to 2.0. So as an example, a lot of the stitching up of information across images from the different cameras and across time was done by a neural network and we were able to delete a lot of code.

[05:00:00]
And so the software 2.0 stack literally ate through the software stack of the autopilot. So I thought this was really remarkable at the time. And I think we're seeing the same thing again where uh basically we have a new kind of software and it's eating through the stack. We have three completely different programming paradigms. And I think if you're entering the industry, it's a very good idea to be fluent in all of them because they all have slight pros and cons and you may want to program some functionality in 1.0 or 2.0 or 3.0. Are you going to train a neural net? Are you going to just prompt an LLM?

[05:30:00]
Uh should this be a piece of code that's explicit, etc. So we all have to make these decisions and actually potentially uh fluidly transition between these paradigms.

*Karpathy explains how Software 2.0 (neural nets) began replacing Software 1.0 (traditional code) in complex systems like Tesla's Autopilot, and now Software 3.0 (LLMs) is repeating this pattern.*

(A large summary slide appears with several diagrams and images from the previous slides, including the 1.0/2.0/3.0 diagram, the LLM OS diagram, and others.)

So, in summary so far, LLM labs fab LLMs, I think it's accurate language to use, but LLMs are complicated operating systems. They're circa 1960s in computing and we're redoing computing all over again. And they're currently available via time sharing and distributed like a utility. What is new and unprecedented is that they're not in the hands of a few governments and corporations, they're in the hands of all of us because we all have a computer and it's all just software.

[06:00:00]
(A new slide appears titled "Part 1 Summary", listing key takeaways: LLM labs fab LLMs; LLMs are like Operating Systems from the 1960s; they are available via time-sharing like a utility; and billions of people now have access to program them.)

And ChatGPT was beamed down to our computers like to billions of people like instantly and overnight. And this is insane. Uh and it's kind of like insane to me that this is the case. And now it is our time to enter the industry and program these computers. This is crazy. So I think this is quite remarkable. Before we program LLMs, we have to kind of like spend some time to think about what these things are and I especially like to kind of talk about their psychology.

So, the way I like to think about LLMs is that they're kind of like people spirits. Uh they are stochastic simulations of people.

[06:30:00]
Um and the simulator in this case happens to be an autoregressive Transformer. (A slide appears with the title "LLMs are 'people spirits': stochastic simulations of people." It features an artistic image of a glowing humanoid figure made of data streams, next to a diagram of a Transformer neural network.) So Transformer is a neural network, uh it's and it just kind of like, goes on the level of tokens, it goes chunk, chunk, chunk, chunk, and there's an almost equal amount of compute for every single chunk. Um, and uh this simulator of course is is just, there's some weights involved and we fit it to all of text that we have on the internet and so on. And you end up with this kind of a simulator. And because it is trained on humans, it's got this emergent psychology that is humanlike. So the first thing you'll notice is of course, um, LLMs have encyclopedic knowledge and memory. (A slide shows an image of a studious young man in a library next to the movie poster for "Rain Man", with the title "Encyclopedic knowledge/memory, ...") And they can remember lots of things, a lot more than any single individual human can because they've read so many things.

[07:30:00]
It's actually kind of reminds me of this movie Rain Man, which I actually really recommend people watch. It's an amazing movie. I love this movie. And Dustin Hoffman here is an autistic savant who has almost perfect memory. So he can read a a he can read like a phonebook and remember all of the names and uh phone numbers. And I kind of feel like LLMs are kind of like very similar. They can remember SHA hashes and lots of different things very, very easily. So they certainly have superpowers in some in some respects, but they also have a bunch of, I would say, cognitive uh deficits. So they hallucinate quite a bit, um, and they kind of make up stuff and don't have a very good, um, internal model of self-knowledge, not sufficient at least. And this has gotten better, but not perfect. They display jagged intelligence. So they're going to be superhuman in some problem-solving domains, and then they're going to make mistakes that basically no human will make.

[08:00:00]
Like, you know, they will insist that 9.11 is greater than 9.9, or that there are two 'r's in strawberry. These are some famous examples, but basically there are rough edges that you can trip on. So that's kind of I think also kind of unique. They also kind of suffer from anterograde amnesia. Um, so and I think I'm alluding to the fact that if you have a coworker who joins your organization, this coworker will over time, uh learn your organization and uh they will understand and gain like a huge amount of context on the organization and they go home and they sleep and they consolidate knowledge and develop expertise over time.

[08:30:00]
LLMs don't natively do this and this is not something that has really been solved in the R&D of LLMs, I think. Um, and so context windows are really kind of like working memory and you have to sort of program the working memory quite directly because they don't just kind of like get smarter by by default. And I think a lot of people get tripped up by the analogies um in this way. In popular culture, I recommend people watch these two movies, uh Memento and 50 First Dates. In both of these movies, the protagonists, their weights are fixed and their context windows gets wiped every single morning. And it's really problematic to go to work or have relationships when this happens. And this happens to LLMs all the time. I guess one more thing I would point to is security uh kind of related limitations of the use of LLMs.

[09:00:00]
So, for example, LLMs are quite gullible. Uh they are susceptible to prompt injection risks. They might leak your data, etc. And so um, and there's many other considerations security related. So, basically, long story short, you have to load your you have to simultaneously think through this superhuman thing that has a bunch of cognitive deficits and issues. How do we and yet they are extremely like useful. And so how do we program them and how do we work around their deficits and enjoy their superhuman powers?

*Karpathy uses analogies to describe LLM psychology, framing them as savants with encyclopedic knowledge but also cognitive deficits like hallucination, jagged intelligence, anterograde amnesia (lack of continual learning), and gullibility (prompt injection risk).*

So, what I want to switch to now is talk about the opportunities of how do we use these models and what are some of the biggest opportunities? This is not a comprehensive list, just some of the things that I thought were interesting for this talk.

[09:30:00]
The first thing I'm kind of excited about is what I would call partial autonomy apps. So, for example, let's work with the example of coding. You can certainly go to ChatGPT directly and you can start copy-pasting code around and copy-pasting bug reports and stuff around and getting code and copy-pasting everything around. Why would you why would you do that? Why would you go directly to the operating system? It makes a lot more sense to have an app dedicated for this. And so I think many of you, uh, use, uh, Cursor. I do as well. Uh, and, uh, Cursor is kind of like the thing you want instead. You don't want to just directly go to the ChatGPT.

[10:00:00]
(A slide appears titled "Example: anatomy of Cursor", showing a screenshot of the Cursor code editor UI, split into a "Traditional interface" section and an "LLM integration" section.)

And I think Cursor is a very good example of an early LLM app that has a bunch of properties that I think are, uh, useful across all the LLM apps. So, in particular, you will notice that we have a traditional interface that allows a human to go in and do all the work manually, just as before. But in addition to that, we now have this LLM integration that allows us to go in bigger chunks. And so some of the properties of LLM apps that I think are shared and useful to point out. Number one, the LLMs basically do a ton of the context management. Um, number two, they orchestrate multiple calls to LLMs, right? So in the case of Cursor, there's under the hood embedding models for all your files, the actual chat models, models that apply diffs to the code and this is all orchestrated for you.

[10:30:00]
A really big one that I think also maybe not fully appreciated always is application-specific GUI. And the importance of it. Because you don't want to just talk to the operating system directly in text. Text is very hard to read, interpret, understand. And also, like, you don't want to take some of these actions natively in text. So it's much better to just see a diff as like red and green change and you can see what's being added and subtracted. It's much easier to just do Command Y to accept or Command N to reject. I shouldn't have to type it in text, right? So a GUI allows a human to audit the work of these fallible systems and to go faster. I'm going to come back to this point a little bit later as well.

[11:00:00]
And the last kind of feature I want to point out is that there's what I call the autonomy slider. (An "autonomy slider" graphic is added to the Cursor UI screenshot.) So for example in Cursor, you can just do tab completion, you're mostly in charge. You can select a chunk of code and command K to change just that chunk of code. You can do command L to change the entire file, or you can do command I, which just, you know, let it rip, do whatever you want in the entire repo. And that's the sort of full autonomy agentic version. And so you are in charge of the autonomy slider, and depending on the complexity of the task at hand, you can uh tune the amount of autonomy that you're willing to give up for that task. Maybe to show one more example of a fairly successful LLM app, uh, Perplexity.

[11:30:00]
(A slide appears titled "Example: Anatomy of Perplexity", showing the search engine's UI and highlighting similar features like context packaging, model orchestration, a custom GUI, and an autonomy slider for "search", "research", and "deep research".)

Uh it it also has very similar features to what I've just pointed out in Cursor. Uh it packages up a lot of the information, it orchestrates multiple LLMs, it's got a GUI that allows you to audit some of its work. So for example, it will cite sources and you can imagine inspecting them. And it's got an autonomy slider. You can either just do a quick search or you can do research or you can do deep research and come back 10 minutes later. So this is all just varying levels of autonomy that you give up to the tool. So I guess my question is, what does all software look like in the partial autonomy world? (A slide shows screenshots of Adobe Photoshop and Unreal Engine, with text asking: "Can an LLM 'see' all the things the human can? Can an LLM 'act' in all the ways a human can? How can a human supervise and stay in the loop?") I feel like all of software will become partially autonomous.

[12:00:00]
And I'm trying to think through like what does that look like? And for many of you who maintain products and services, how are you going to make your products and services partially autonomous? Can an LLM see everything that a human can see? Can an LLM act in all the ways that a human could act? And can humans supervise and stay in the loop of this activity? Because again, these are fallible systems that aren't yet perfect. And what does a diff look like in Photoshop or something like that, you know? And also a lot of the traditional software right now, it has all these switches and all this kind of stuff that's all designed for a human. All this has to change and become accessible to LLMs.

*He argues that successful LLM applications are partial autonomy products, featuring an "autonomy slider" that allows users to control the level of AI involvement, and emphasizes the importance of custom GUIs to facilitate the fast, human-in-the-loop verification of AI-generated work.*

</details>

<details>
<summary>(The video begins with an intro animation for the "AI Engineer SUMMIT".)</summary>

(The video begins with an intro animation for the "AI Engineer SUMMIT".)

[00:00] Hey everyone. Uh, my name is Jerry, co-founder and CEO of LlamaIndex, and today we'll be talking about how to build production-ready RAG applications. Um, I think there's still time for a raffle for the bucket hats. So if you guys stop by our booth, uh please fill out the Google form. (The presenter, Jerry Liu, is on stage. The title slide is displayed behind him, titled "Building Production-Ready RAG Applications" with his name and company, LlamaIndex.) Okay.

*The speaker introduces himself and the topic of building production-ready Retrieval-Augmented Generation (RAG) applications.*

[00:30] Let's get started. So everybody knows that there's been a ton of amazing use cases in GenAI recently. You know, um, knowledge search and QA, conversational agents, uh workflow automation, document processing. (A slide titled "GenAI - Enterprise Use-cases" shows four diagrams: "Document Processing Tagging & Extraction", "Knowledge Search & QA", "Conversational Agent", and "Workflow Automation".) These are all things that you can build uh especially using the reasoning capabilities of LLMs uh over your data. So if we just do a quick refresher, in terms of like paradigms for how do you actually get language models to understand data that hasn't been trained over, there's really like two main paradigms.

*The speaker outlines common enterprise use cases for Generative AI and introduces two primary paradigms for incorporating knowledge into language models.*

[01:00] One is retrieval augmentation, where you like fix the model and you basically create a data pipeline to put context into the prompt from some data source into the input prompt of the language model. (A slide titled "Paradigms for inserting knowledge" shows a diagram for "Retrieval Augmentation". It illustrates data from a source like Notion being put into an input prompt, which is then fed to an LLM.) Um, so like a vector database, uh you know, like unstructured text, SQL database, etc. The next paradigm here is fine-tuning. How can we bake knowledge into the weights of the network by actually updating the weights of the model itself, some adapter on top of the model, but basically some sort of training process over some new data to actually incorporate knowledge.

*He explains the two main paradigms for inserting knowledge into LLMs: retrieval augmentation, which adds context to the prompt, and fine-tuning, which bakes knowledge into the model's weights.*

[01:30] (A new slide shows "Fine-tuning", with a diagram of a feedback loop between a data source and an LLM, labeled with terms like "RLHF, Adam, SGD, etc.") We'll probably talk a little bit more about retrieval augmentation, but this is just like to help you get uh started and really understanding the mission statement of of the company. Okay. (A slide with the title "RAG Stack" appears.) Let's talk about RAG, retrieval-augmented generation. Um, it's become kind of a buzzword recently. But we'll first walk through the current RAG stack for building a QA system. (A slide appears titled "Current RAG Stack for building a QA System". It displays a flowchart divided into two sections: "Data Ingestion" which shows a document being broken into chunks and stored in a vector database, and "Data Querying" where chunks are retrieved from the database and fed to an LLM.) This really consists of two main components, uh data ingestion as well as data querying, which contains retrieval and synthesis. Uh, if you're just getting started in LlamaIndex, you can basically do this in around like five-ish lines of code.

*The speaker introduces the concept of the RAG (Retrieval-Augmented Generation) stack, which consists of data ingestion and data querying (retrieval and synthesis).*

[02:00] Uh so you don't really need to think about it. But if you do want to learn some of the lower level components, and I do encourage like every engineer, uh AI engineer to basically just like learn how these components work under the hood, um I would encourage you to check out some of our docs to really understand how do you actually do data ingestion uh and data querying. Like how do you actually retrieve from a vector database and how do you synthesize that with an LLM?

*He encourages engineers to understand the underlying components of data ingestion and querying within the RAG pipeline.*

[02:30] (A new slide appears with the title "Challenges with 'Naive' RAG".) So that's basically the key stack that's kind of emerging these days like for every sort of like chatbot, like, you know, chat over your PDF, like over your unstructured data. Um a lot of these things are basically using the same principles of like how do you actually load data from some data source and actually, you know, um uh retrieve and query over it. But I think as developers are actually developing these applications, they're realizing that this isn't quite enough.

*The speaker notes that while the basic RAG stack is popular, developers often encounter challenges that require more than this "naive" implementation.*

[03:00] Uh like there's there's certain issues that you're running into that are blockers for actually being able to productionize these applications. And so what are these challenges with naive RAG? One aspect here is just like uh the response, and and this is the key thing that we're focused on, like the the response quality is not very good. You run into for instance like bad retrieval issues. Like uh during the retrieval stage from your vector database, if you're not actually returning the relevant chunks from your vector database, you're not going to be able to have the correct context actually put into the LLM. (A slide titled "Challenges with Naive RAG (Response Quality)" lists "Bad Retrieval" as a major issue, with sub-points for "Low Precision", "Low Recall", and "Outdated information".)

*He identifies poor response quality as a primary challenge with naive RAG, often stemming from bad retrieval, which includes issues of low precision, low recall, and outdated information.*

[03:30] So this includes certain issues like low precision, not all chunks in the retrieved set are relevant. Uh this leads to like hallucination, like lost in the middle problems, you have a lot of fluff in the return response. This could mean low recall, like your top K isn't high enough or basically like the the the set of like information that you need to actually answer the question is just not there. Um and of course, there's other issues too, like outdated information. And many of you who are building apps these days might be familiar with some like key concepts of like just why the LLM isn't always, you know, uh guaranteed to give you a correct answer.

*The speaker details the problems of bad retrieval, explaining that low precision can cause hallucinations and irrelevant responses, while low recall means the LLM lacks sufficient context.*

[04:00] (A second bullet point, "Bad Response Generation," is added to the slide, with sub-points for "Hallucination", "Irrelevance", and "Toxicity/Bias".) There's hallucination, irrelevance, like toxicity bias, there's a lot of issues on the LLM side as well. (A new slide titled "What do we do?" appears, showing the RAG pipeline diagram again. Above the diagram are four bullet points corresponding to the pipeline stages: "Data", "Embeddings", "Retrieval", and "Synthesis", each posing a question about optimization.) So, what can we do? Um what can we actually do to try to improve the performance of a retrieval augmented generation application? Um and and for many of you, like you might be running into certain issues and it really runs the gamut across like the entire pipeline. There's stuff you can do on the data, like can we store additional information beyond just like the raw text chunks, right, that that you're putting in the vector database? Can you optimize that data pipeline somehow?

*He transitions to solutions, outlining a holistic approach to improving RAG systems by optimizing every stage of the pipeline, from data handling and embeddings to retrieval and synthesis.*

[04:30] Play around with chunk sizes, that type of thing. Can you optimize the embedding representation itself? A lot of times when you're using a pre-trained embedding model, it's not really optimal for giving you the best performance. Um there's the retrieval algorithm. You know, the default thing you do is just look up the top K most similar elements from your vector database to return to the LLM. Um many times that's not enough and and what are kind of like both simple things you can do as well as hard things?

*The speaker suggests several optimization strategies, including tuning chunk sizes, optimizing embedding representations, and going beyond simple top-k retrieval algorithms.*

[05:00] Uh and there's also synthesis. Like, uh why is there Yeah, there's like a V in the. Anyways, so so can we use LLMs for more than generation? Um and so basically like you can um use the LLM to actually help you with like reasoning as opposed to just like pure um uh pure just like uh pure generation, right? You can actually use it to try to reason over given a question, can you break it down into simpler questions, route to different data sources and and kind of like have a a more sophisticated way of like querying your data.

*He discusses enhancing the synthesis stage by using LLMs for complex reasoning and query planning rather than just straightforward generation.*

[05:30] (Text appears at the bottom of the slide: "But before all this... We need a way to measure performance.") Of course, like if you've kind of been around some of my recent talks, like I always say before you actually try any of these techniques, you need to be pretty task specific and make sure that you need a way to that you actually have a way to measure performance. (A new slide with the single word "Evaluation" is shown.) So, I'll probably spend like two minutes talking about evaluation. Um, Simon, my co-founder just ran a workshop yesterday on really just like how to evaluate, uh build a data set, evaluate RAG systems, and help iterate on that.

*Before diving into specific optimization techniques, the speaker stresses the critical importance of having a robust evaluation framework to measure performance.*

[06:00] Uh if you missed the workshop, don't worry, I'll we'll have the slides and and materials uh available online so that you can take a look. (A new slide, also titled "Evaluation," shows the end of the RAG pipeline, from Vector Database to LLM, highlighting the "Retrieval" and "Synthesis" steps.) Um at a very high level in terms of evaluation, it's important because you basically need to define a benchmark for your system to understand how are you going to iterate on and improve it. Uh and there's like a few different ways you can try to do evaluation. I think Anton from from Chroma was was just saying some of this, but like you basically need a way to um evaluate both the end-to-end solution, like you have your input query as well as the output response. You also want to probably be able to evaluate like specific components.

*He introduces the two main approaches to evaluation: end-to-end testing of the entire system and isolated evaluation of individual components like retrieval and synthesis.*

[06:30] Like if you've diagnosed that the retrieval is the is like the portion that needs improving, you need like retrieval metrics to really understand how can you improve your retrieval system. Um so there's retrieval and there's synthesis. Let's talk a little bit just like 30 seconds on each one. (A slide titled "Evaluation in Isolation (Retrieval)" appears with a flowchart and bullet points.) Um evaluation on retrieval, what does this look like? You basically want to make sure that the stuff that's returned actually answers the query and that you're kind of, you know, not uh returning a bunch of fluff uh and that the stuff that you're returned is relevant to the question.

*Focusing on retrieval evaluation, he explains the goal is to ensure the retrieved chunks are relevant to the user's query without including unnecessary information.*

[07:00] Um so first you need an evaluation data set. A lot of people are uh have like human labeled data sets. If you're in uh building stuff in prod, you might have like user feedback as well. If not, you can synthetically generate a data set. This data set is input like query and output the IDs of like the returned documents or relevant to the query. So you need that somehow. Once you have that, you can measure stuff with ranking metrics, right? You can measure stuff like success rate, hit rate, MRR, NDCG, a variety of these things.

*He outlines the process for retrieval evaluation, which requires creating a dataset of queries and ground-truth relevant documents to measure ranking metrics like hit rate and MRR.*

[07:30] Uh and and so like once you are able to evaluate this, like this really isn't uh kind of like an LLM problem. This is like an IR problem and this has been around for at least like a decade or two. Um but a lot of this is becoming, you know, it's it's still very relevant in the face of actually building these LLM apps. The next piece here is um there's the retrieval portion, right? But then you generate a response from it. (A new slide titled "Evaluation E2E" is shown.) And then how do you actually evaluate the whole thing end to end? So evaluation of the final response uh given the input.

*He explains that after evaluating retrieval, the next step is end-to-end evaluation, which assesses the quality of the final generated response.*

[08:00] You still want to generate some sort of data set so you could do that through like human annotations, user feedback. You could have like ground truth reference answers given the query that really indicates like, hey, this is the proper answer to this question. Um and you can also just like, you know, synthetically generate it with like GPT4. Uh you run this through the full RAG pipeline that you built, the retrieval and synthesis, and you can run like LLM-based evals. Um so label-free evals, with label evals, there's a lot of uh projects these days uh going on about how do you actually properly evaluate the outputs, the predicted outputs of a language model.

*The speaker details the end-to-end evaluation process, which involves creating a dataset with queries and ground-truth answers, running it through the full pipeline, and then using LLM-based evaluators to score the final output.*

[08:30] (A new slide is shown with the title "Optimizing RAG Systems.") Once you've defined your eval benchmark, now you want to think about how do you actually optimize your RAG systems. (A new slide titled "From Simple to Advanced RAG" shows a horizontal spectrum from "Less Expressive" to "More Expressive" with four categories: "Table Stakes," "Advanced Retrieval," "Agentic Behavior," and "Fine-tuning.") So, I sent a teaser on the slide uh a few like yesterday. But the way I think about this is that when you want to actually improve your system, there's a million things that you can do to try to actually improve your RAG system. Uh and like you probably don't want to start with the hard stuff first just because like, you know, part of the value of like language models is how it's kind of democratized access to every developer.

*After establishing an evaluation benchmark, he introduces a spectrum of RAG optimization techniques, advising to start with simpler methods before moving to more complex ones.*

[09:00] It's really just made it easy for people to get up and running. And so if for instance you're running into some performance issues with RAG, I'd probably start with the basics. Like I call it like table stakes RAG techniques. Uh better parsing, um so that you don't just split by even chunks, like adjusting your chunk sizes, trying out stuff that's already integrated with the vector database like hybrid search, as well as like metadata filters. There's also like advanced retrieval methods uh that you could try. This is like a little bit more advanced.

*He categorizes optimization techniques, starting with "table stakes" methods like better parsing, adjusting chunk sizes, and using hybrid search and metadata filters.*

[09:30] Some of it pulls from like traditional IR, some of it's more like kind of uh really like uh new in in this age of like LLM-based apps. There's like uh reranking, um that's a traditional concept. There's also concepts in LlamaIndex like recursive retrieval, like dealing with embedded tables, like small to big retrieval, and a lot of other stuff that we have that help you potentially improve the performance of your application. And then the last bit like this kind of gets into more expressive stuff that might be harder to implement, might incur a higher latency and cost, but is potentially more powerful and forward looking is like agents.

*Moving along the spectrum, he introduces advanced retrieval techniques like reranking and recursive retrieval, as well as more complex and expressive approaches like agentic behavior and fine-tuning.*

[10:00] Like how do you incorporate agents towards better like RAG pipelines to better answer different types of questions and synthesize information. And how do you actually fine-tune stuff? (A new slide appears titled "Table Stakes: Chunk Sizes.") Let's talk a little bit about the table stakes first. So chunk sizes, tuning your chunk size can have outsized impacts on performance, right? Uh if you've kind of like played around with RAG systems, this may or may not be obvious to you. What's interesting though is that like more retrieved tokens does not always equal higher performance and that if you do like reranking of your retrieved tokens, it doesn't necessarily mean that your final generation response is going to be better.

*The speaker begins with "table stakes" optimizations, highlighting that tuning chunk size can significantly impact performance, and notes that simply retrieving more tokens or reranking them doesn't always lead to better results.*

[10:30] And this is again due to stuff like lost in the middle problems where stuff in the middle of the LLM context window tends to get lost whereas stuff at the end uh tends to be a little bit uh more well-remembered by the LLM. Um and so I think we did a workshop with like Arize a few uh a week ago where we basically showed, you know, uh there is kind of like an optimal chunk size given your data set. (The slide shows six bar charts comparing the percentage of incorrect QA evaluations based on different chunk sizes, retrieval methods, and the use of reranking.) And a lot of times when you try out stuff like reranking, it actually increases your error metrics.

*He explains that due to issues like the "lost in the middle" problem, where LLMs pay less attention to information in the middle of the context, an optimal chunk size exists and simple reranking can sometimes worsen performance.*

[11:00] (A new slide is shown titled "Table Stakes: Metadata Filtering.") Metadata filtering. Uh this is another like very table stakes thing that I think everybody should look into and I think vector databases, like, you know, Chroma, Pinecone, Weaviate, like these these uh vector databases are all implementing these uh capabilities under the hood. Metadata filtering is basically just like how can you add structured context uh to your your chunks, like your text chunks. And you can use this for both like embedding as well as synthesis, but it also integrates with the metadata filter capabilities of a vector database. Um so metadata is just like again structured JSON dictionary. It could be like page number, it could be the document title.

*He presents metadata filtering as another fundamental "table stakes" technique, where structured information like page numbers or document titles is associated with text chunks to improve retrieval.*

[11:30] It could be the summary of adjacent chunks, you can get creative with it too. You could hallucinate like questions uh that the chunk answers. Um and it can help retrieval, it can help augment your response quality, it also integrates with the vector database filters. (A slide is shown with the question: "Can you tell me the risk factors in 2021?" and a diagram illustrating how raw semantic search over a single collection of documents can have low precision.) So as an example, um let's say the question is over like the SEC uh like 10Q document and like, can you tell me the risk factors in 2021? If you just do raw semantic search, typically it's very low precision. You're going to return a bunch of stuff that may or may not match this.

*The speaker elaborates that metadata can help retrieval and response quality, explaining that raw semantic search is often imprecise, retrieving irrelevant information from different time periods or documents.*

[12:00] You might even return stuff from like other years if you have a bunch of documents from different years in the same vector collection. Um and so you're kind of like rolling the dice a little bit. (A new diagram shows how adding a "year": 2021 metadata tag allows filtering to only search within the relevant year's documents, increasing precision.) But one idea here is basically, you know, if you have access to the metadata of the documents, um and you ask this question like this, you basically combine structured query capabilities by inferring the metadata filters, like a where clause in a SQL statement, like a year equals 2021, and you combine that with semantic search to return the most relevant candidates given your query. And this improves the precision of your uh of your results.

*He demonstrates how metadata filtering acts like a SQL "WHERE" clause, allowing the system to pre-filter documents (e.g., by year) before performing semantic search, thereby increasing retrieval precision.*

[12:30] (A new slide appears titled "Advanced Retrieval: Small-to-Big".) Moving on to stuff that's maybe a bit more advanced, like advanced retrieval is one thing that we found generally helps is this idea of like small to big retrieval. Um so what does that mean? Basically, right now when you embed a big text chunk, you also synthesize over that text chunk. And so it's a little suboptimal because what if like the embedding representation is like biased? Because, you know, there's a bunch of fluff in that text chunk that contains a bunch of irrelevant information, you're not actually optimizing your retrieval quality. So, embedding a big text chunk sometimes feels a little suboptimal.

*The speaker introduces "small-to-big retrieval" as an advanced technique, arguing that embedding large, noisy text chunks is suboptimal for retrieval accuracy.*

[13:00] One thing that you could do is basically embed text at the sentence level or on a smaller level and then expand that window during synthesis time. (A diagram on the slide shows a question leading to an "Embedding Lookup" on a single sentence within a larger paragraph, but the entire paragraph ("Expanded Window") is what the "LLM Sees".) And so this is contained in a variety of like LlamaIndex abstractions. But the idea is that you return you retrieve on more granular pieces of information, so smaller chunks. This makes it so that these chunks are more likely to be retrieved when you actually ask a query over these specific pieces of context. But then you want to make sure that the LLM actually has access to more information to actually synthesize a proper result. (A new slide shows a comparison: "Sentence Window Retrieval (k=2)" successfully answers a question, while "Naive Retrieval (k=5)" fails, citing a "lost in the middle" problem.) This leads to like more precise retrieval, right? So we we tried this out and it helps avoid like some loss in the middle problems.

*He explains the "small-to-big" approach: retrieve based on smaller, more precise units of text (like sentences) but provide the LLM with the larger, surrounding context for better synthesis, which improves precision and avoids "lost in the middle" issues.*

[13:30] You can set a smaller top K value, like k equals two, uh whereas like uh over this data set, if you set k equals five for naive retrieval over big text chunks, you basically start returning a lot of context and that kind of leads into issues where uh, you know, maybe the relevant context is in the middle but you're not able to find out or or you're like the LLM is is not able to kind of synthesize over that information.

*This method allows for a smaller, more focused retrieval set (e.g., top-k=2) which is more precise than a larger set from naive retrieval that might bury the relevant information.*

[14:00] (A new slide with more diagrams clarifies the "small-to-big" concept further, showing how embedding a smaller "reference" to the parent chunk is used for retrieval, but the full parent chunk is used for synthesis.) A very related idea here is just like embedding a reference to the parent chunk, um as opposed to the actual text chunk itself. So for instance, if you want to embed like not just the raw text chunk or not the text chunk, but actually a smaller chunk, um or a summary or questions that answer the chunk, we have found that that actually helps to improve retrieval performance a decent amount. And it it kind of again goes along with this idea like a lot of times you want to embed something that's more amenable for embedding based retrieval, but then you want to return enough context so that the LLM can actually synthesize over that information.

*He refines the concept by suggesting embedding a *reference* to a larger chunk—such as a smaller piece of text, a summary, or a generated question—which optimizes for retrieval while still allowing the LLM to access the full context for synthesis.*

[14:30] (A slide titled "Agentic Behavior: Multi-Document Agents" is shown. It has a flowchart where a top-level agent queries several individual "Document Agents".) The next piece here is actually kind of even more advanced stuff, right? This goes on into agents and this goes on into that last pillar that I I mentioned, which is how can you use LLMs for for reasoning as opposed to just synthesis. The intuition here is that like for a lot of RAG, if you're just using the LLM at the end, you're one constrained by the quality of your retriever and you're really only able to do stuff like question answering. And there's certain types of questions and more advanced analysis that you might want to launch that like top K RAG can't really answer.

*The speaker transitions to using agents, explaining that standard RAG is limited to simple QA, whereas agentic systems can handle more complex questions that require reasoning and multi-step processes.*

[15:00] It it's not necessarily just a one-off question. You might need to have an entire sequence of reasoning steps to actually pull together a piece of information, or you might want to like summarize a document and compare it with other documents. So, one kind of architecture we're we're exploring right now is this idea of like multi-document agents. What if like instead of just like RAG, we moved a little bit more into agent territory. We modeled each document not just as a sequence of text chunks, but actually as a set of tools that contains the ability to both like summarize that document as well as to do QA over that document over specific facts.

*He proposes a "multi-document agent" architecture where each document is treated as an agent with its own set of tools (e.g., for summarization and QA), allowing for more complex, cross-document analysis.*

[15:30] Um, and of course, if you want to scale to like, you know, hundreds or thousands or millions of documents, um, typically an agent can only have access to a limited window of tools. So you probably want to do some sort of retrieval on these tools similar to how you want to retrieve like text chunks from a document. The main difference is that because these are tools, you actually want to act upon them, you want to use them as opposed to just taking the raw text and plugging it into the context window.

*To scale this agentic approach, he suggests that a master agent could retrieve the most relevant document agents (tools) for a given task, similar to how a RAG system retrieves text chunks.*

[16:00] So blending this combination of like um embedding-based retrieval or any sort of retrieval as well as like agent tool use is a very interesting paradigm that I think is really only possible with this age of LLMs and hasn't really existed before this. Another kind of advanced concept is this idea of fine-tuning. (A slide titled "Fine-Tuning: Embeddings" appears.) Um and so fine-tuning, you know, some other presenters have talked about this as well. But the idea of like fine-tuning in a RAG system is that it really optimizes specific pieces of this RAG pipeline for you to kind of better um improve the performance of either retriever or synthesis capabilities.

*The speaker concludes the section on agentic behavior and introduces fine-tuning as another advanced method to optimize specific components of the RAG pipeline for better retrieval and synthesis.*

[16:30] One thing you can do is fine-tune your embeddings. I think Anton was talking about this as well. Like if you just use a pre-trained model, the embedding representations are not going to be optimized over your specific data. So sometimes you're just going to retrieve the wrong wrong information. Um, if you can somehow tune these embeddings so that given any sort of like relevant question that the user might ask that you're actually returning the relevant response, then you're going to have like better performance.

*He explains that fine-tuning embeddings on a specific dataset can significantly improve retrieval performance by making the embedding representations more relevant to the domain.*

[17:00] So, the idea here, right, is to generate a synthetic query data set from raw text chunks using LLMs and use this to fine tune an embedding model. Um, you can do this like. (The speaker quickly skips forward to a slide titled "Fine-Tuning: LLMs", then back to the "Fine-Tuning: Embeddings" slide.) Uh if we go back real quick actually. Um you can do this by basically um fine-tuning the base model itself. You can also fine-tune an adapter on top of the model. Um and fine-tuning an adapter on top of the model has a few advantages in that you don't require the base model's weights to actually fine-tune stuff.

*To fine-tune embeddings, he suggests synthetically generating a query dataset from the source documents using an LLM, and notes that fine-tuning an adapter is often more practical than fine-tuning the entire base model.*

[17:30] And if you just fine-tune the query, you don't have to re-index your entire document corpus. (The speaker moves to the "Fine-Tuning: LLMs" slide, which shows a flowchart for using GPT-4 to generate a question-answer dataset from chunks of a document, which is then used to fine-tune a smaller model.) There's also fine-tuning LLMs, which of course like a lot of people are very interested in doing these days. Um an intuition here specifically for RAG is that if you have a weaker LLM, like 3.5 Turbo, like Llama 2 7B, like these weaker LLMs are bad at are are maybe a little bit worse at like response synthesis, reasoning, structured outputs, etc.

*The speaker discusses fine-tuning the language model itself, noting that it can help weaker LLMs improve at tasks like response synthesis, reasoning, and generating structured outputs.*

[18:00] um compared to like bigger models. So, a solution here is what if you can generate a synthetic data set using a bigger model like GPT-4, this is something we're exploring, and you actually distill that into 3.5 Turbo. So it gets better at chain of thought, longer response quality, um better structured outputs and a lot of other possibilities as well. Um, and of course, if you want to scale to like, you know, hundreds or thousands or millions of documents, um, typically an agent can only have access to a limited window of tools. So you probably want to do some sort of retrieval on these tools similar to how you want to retrieve like text chunks from a document. The main difference is that because these are tools, you actually want to act upon them, you want to use them as opposed to just taking the raw text and plugging it into the context window. So blending this combination of like um embedding based retrieval or any sort of retrieval as well as like agent tool use is a very interesting paradigm that I think is really only possible with this age of LLMs and hasn't really existed before this. Another kind of advanced concept is this idea of fine tuning. um and so fine tuning you know some other presenters have talked about this as well but the idea of like fine tuning in a RAG system is that it really optimizes specific pieces of this RAG pipeline for you to kind of better um improve the performance of either retriever or synthesis capabilities. One thing you can do is fine tune your embeddings I think anton was talking about this as well like if you just use a pre trained model the embedding representations are not going to be optimized over your specific data so sometimes you're just going to retrieve the wrong wrong information um if you can somehow tune these embeddings so that given any sort of like relevant question that the user might ask that you're actually returning the relevant response then you're going to have like better performance so the idea here right is to generate a synthetic query data set from raw text chunks using LLMs and use this to fine tune an embedding model um and you can do this like uh if we go back real quick actually um you can do this by basically um fine tuning the base model itself you can also fine tune an adapter on top of the model um and fine tuning an adapter on top of the model has a few advantages in that you don't require the base model's weights to actually fine tune stuff and if you just fine tune the query you don't have to reindex your entire document corpus there's also fine tuning LLMs which of course like a lot of people are very interested in doing these days um an intuition here specifically for RAG is that if you have a weaker LLM like 3.5 turbo like Llama 2 7B like these weaker LLMs are bad at are are maybe a little bit worse at like response synthesis reasoning structured outputs etc um compared to like bigger models so a solution here is what if you can generate a synthetic data set using a bigger model like GPT4 this is something we're exploring and you actually distill that into 3.5 turbo so it gets better at chain of thought longer response quality um better structured outputs and a lot of other possibilities as well. (A final slide titled "Resources" shows two QR codes and URLs for "Production RAG" and "Fine-tuning" documentation.) So all these things are in our docs. There's production RAG, uh there's fine-tuning and I have two seconds left. So thank you very much. (The audience applauds.)

*He proposes a solution where a larger, more capable model like GPT-4 generates a synthetic dataset, which is then used to fine-tune and distill knowledge into a smaller, more efficient LLM, improving its reasoning and synthesis capabilities.*

</details>

<details>
<summary>Real Agents vs. Workflows: The Truth Behind AI 'Agents'</summary>

Real Agents vs. Workflows: The Truth Behind AI 'Agents'

***

(The video opens with the speaker on the right side of the screen. On the left, a cartoon businessman shakes hands with a small white robot. A pixelated speech bubble above the businessman says "Hello, Mr.Agent". A red "IMPOSTER" stamp appears over the robot's face.)

**Speaker 1:** What most people call agents aren't agents. I've never really liked the term agent.

(The speaker gestures to the left, and the screen transitions to show a screenshot of an article from the company Anthropic titled "Building effective agents".)

**Speaker 1:** Until I saw this recent article by Anthropic, where I totally agree and now see how we can call something an agent.

*The speaker introduces the main argument: most things called "AI agents" are not true agents, a distinction clarified by a recent Anthropic article.*

(The scene changes to a black background with a small white robot and large white text that reads: "MOST AGENTS ARE JUST API CALL TO LLM".)

**Speaker 1:** The vast majority is simply API calls to a language model. This is this. (An animation of the robot appears next to a box containing a Python code snippet for an OpenAI API call.) A few lines of code and a prompt. This cannot act independently, make decisions or do anything. (Text boxes pop up next to the code: "Can't act independently", "Can't make decisions", "Just replies to users".) It simply replies to your users. Still, we call them agents.

[00:30]
(The scene returns to the speaker in his office. A text overlay appears at the bottom of the screen: "WE NEED REAL AGENTS".)

**Speaker 1:** But this isn't what we need. We need real agents. But what is a real agent? Before we dive into serious agentic stuff, if you are a student, writer, blogger, or content creator like me, or would like help becoming one, you will love the sponsor of today's video with a clever name, Originality.ai.

*The speaker defines what most people incorrectly call "agents" as simple, non-autonomous API calls to LLMs, and then introduces the video's sponsor.*

(The video transitions to the Originality.ai logo, a purple brain-shaped thought bubble, followed by a screen recording of their website homepage. The headline reads: "Our Accurate AI Checker, Plagiarism Checker and Fact Checker Lets You Publish with Integrity".)

**Speaker 1:** Originality.ai is an awesome tool designed to detect AI-generated content, check for plagiarism, grammar, readability, and even fact-check your work.

[01:00]
(The screen recording scrolls down to show the features, highlighting "Accurate AI Detection", "Plagiarism Checking", and "Fact Checking Aid".)

**Speaker 1:** Everything you need to publish with integrity. Simply upload a document and in seconds, it flags any AI-generated text, highlights plagiarism, and even checks grammar and readability with many useful tips and suggestions. (A demo shows text being pasted into the Originality.ai editor and scanned. The results show scores for AI detection, plagiarism, fact-checking, readability, and grammar.) I really love this feature. It also offers fact-checking, ensuring every claim in your content stands up to scrutiny. Pretty cool when you work on important or technical work. All based on the most state-of-the-art language models and systems. Try Originality.ai today with the first link in the description.

*The speaker explains the features of the sponsor, Originality.ai, a tool for checking AI content, plagiarism, grammar, and facts, before transitioning back to the main topic.*

(The scene changes to a black screen with a diagram. The speaker's face is in a circular frame in the bottom right corner.)

[01:30]
**Speaker 1:** So let's start over. We have an LLM accessed programmatically, which is through an API or accessed locally in your own server or machine. And then what? Well, we need it to take action or do something more than just generate text. How? By giving it access to tools and their documentation. (The diagram shows a "Query" icon pointing to a box for "SQL Query".) We give them access to a tool like the ability to execute SQL queries in a database to access private knowledge. Specifically, we code all that ourselves to have our LLM generate SQL queries. (Inside the "SQL Query" box, a flow appears: "Infer Schema" -> "Construct SQL Query".) And then our code will send and execute the query automatically in our database. (The flow continues: "Execute SQL Query" points to an icon representing "BigQuery Tables".) We then send back the outputs so that it uses them to answer the user.

[02:00]
(The diagram adds loops for "Self Correct" and "Optimize" around the query execution step. The result from the tables is sent back to the "LLM" box as "Content Retrieved", and the LLM then produces an "Answer".)

**Speaker 1:** This is what another great proportion of people call agents. (Large red text reading "NOT AN AGENT" is stamped repeatedly over the diagram.) They are still not agents. This is simply a process, hardcoded, or with small variations like routers that we discussed in the course. Of course, it's useful and it's super powerful. Yet, it's not an intelligent being or something independent. It's not an agent acting on our behalf. It's simply a program we made and control. (The word "WORKFLOW" is stamped over the diagram.) Or, as Anthropic calls it, a workflow.

*The speaker details a more complex system involving tools like SQL querying, which many people also call agents, but clarifies that these are still just predefined, hardcoded "workflows."*

[02:36]
**Speaker 1:** Don't get me wrong. A workflow is pretty damn useful, and it can be quite complex and advanced. We can implement intelligent routers to decide what tool to use and when to give it access to various databases. (A new diagram shows a workflow: "In" -> "LLM Call Router" which can route to one of three parallel "LLM Call" boxes, all leading to "Out".) Have it decide which one to query and when. Have it execute tasks through action tools, through code, and more. Plus, you can have as many workflows as you wish.

[03:00]
**Speaker 1:** Yet, I simply want to state how different it is than an actual agent. The type of agent we dream of and the type Ilya mentioned at a recent talk I attended at NeurIPS.

(A clip from a YouTube video of Ilya Sutskever's talk at NeurIPS 2024 is shown. The slide reads "What comes next? The long term" and "Superintelligence.")

**Ilya Sutskever:** So right now we have our incredible language models and the unbelievable chatbots and they can even do things but they're also kind of strangely unreliable and they get confused when while also having dramatically superhuman performance on evals so it's really unclear how to reconcile this. But eventually, sooner or later, the following will be achieved. Those systems are actually going to be agentic in a real ways whereas right now the systems are not agents in any meaningful sense. Just very that might be too strong. They're very, very slightly agentic. Just the beginning.

*The speaker explains that even complex, multi-tool systems are still just workflows, not true agents, and supports this by showing a clip of Ilya Sutskever discussing the future of truly "agentic" AI.*

[03:46]
**Speaker 1:** The next natural question might be, what exactly is a real agent? (A new screen appears with the text "WHAT EXACTLY IS A 'REAL AGENT'?".) In simple terms, a real agent is something that functions independently. (A graphic of a brain appears on screen with the text "SYSTEM 2".) More specifically, it's something capable of employing processes like our System 2 thinking, able to genuinely reason, reflect, and recognize when it lacks knowledge. This is almost the opposite of our System 1 thinking, which is fast, automatic, and based purely on patterns and learned responses.

[04:13]
**Speaker 1:** (Text appears next to the brain graphics, contrasting "System 2" and "System 1" thinking.) Like reflexes when you need to catch a dropping glass. By contrast, System 2 thinking might involve deciding whether to prevent the glass from falling in the first place, perhaps by using a nearby tool like a tray or moving the fragile object out of the way. (A green text box appears: "A real agent deliberately decides when and why to use tools with deliberate reasoning.") A real agent, then, will not only know how to use tools, but also decide when and why to use them based on deliberate reasoning. (Screenshots of OpenAI blog posts about their "o1" and "o3-mini" models appear.) OpenAI's new o1 and o3 series exemplify this shift as they begin exploring System 2-like approaches and try to make models reason by first discussing with themselves internally, mimicking a human-like approach to reasoning before speaking.

*A true agent is defined as an independent system capable of "System 2" thinking—deliberate reasoning and reflection—unlike current models that rely on "System 1" pattern recognition.*

[04:52]
(A diagram of a transformer model architecture is shown.)

**Speaker 1:** Unlike traditional language models that rely on next-word or next-token prediction, essentially a System 1 instant-thinking mechanism, purely based on what it knows and learned to guess the next instant thing to go with no plan. These new models aim to incorporate deeper reasoning capabilities, making a move toward the deliberate, reflective thinking associated with System 2. Something required for a true agent to be. But we are diverging a bit too much with this Kahneman parenthesis. Let me clarify what I mean by a real agent by going back to workflows and what they really are.

[05:30]
(A blue screen with text appears next to the speaker.)

**Speaker 1:** Workflows follow specific code lines and integrations and, other than the LLM's outputs, are pretty predictable. They are responsible for most of the advanced applications you see and use today, and for a reason. They are consistent, more predictable, and incredibly powerful when leveraged properly. As Anthropic wrote, "Workflows are systems where LLMs and tools are orchestrated through predefined code paths."

(The SQL query workflow diagram reappears.)

**Speaker 1:** Here's what a workflow looks like. We have our LLM, some tools or memory to retrieve for additional context, iterate a bit with multiple calls to the LLM, and then an output sent back to the user.

[06:07]
(The simpler router-based workflow diagram reappears.)

**Speaker 1:** As we discussed, when a system needs to sometimes do a task and sometimes another depending on conditions, workflows can use a router with various conditions to select the right tool or the right prompt to use. They can even work in parallel to be more efficient. (A more complex workflow diagram with an "Orchestrator" and "Synthesizer" appears.) Better, we can have some sort of main model, which we refer to as an orchestrator, that selects all the different fellow models to call for specific tasks and synthesize the results, such as our SQL example, where we'd have the main orchestrator getting the user query and could decide if it needs to query a dataset or not, and if it does, ask the SQL agent to generate the SQL query and query the dataset and get it back and synthesize the final answer thanks to all the information provided. This is a workflow. (A screenshot of the ChatGPT interface with Canvas is shown.) Just like ChatGPT is a workflow, sometimes using canvas and sometimes just straight up answering your question.

*The speaker defines workflows as systems with predefined code paths that, while powerful and predictable, are fundamentally different from agents.*

[07:01]
**Speaker 1:** Even if complex and advanced, it is still all hardcoded. If you know what you need your system to do, you need a workflow, however advanced it may be. (A screenshot of the CrewAI website's use cases is shown.) For instance, what CrewAI calls agents function like predefined workflows assigned to specific tasks. While Anthropic envisions an agent as a single system capable of reasoning through any task independently. (The speaker is shown next to a graphic comparing "Workflow" and "Agent".) Both approaches have merit. One is predictable and intuitive, while the other aims for flexibility and adaptability. However, the latter is far harder to achieve with current models and better fits an agent definition to me.

*Contrasting workflows with agents, the speaker explains that workflows are ideal for known, predefined tasks, while true agents, which are much harder to build, are designed for flexibility and independent reasoning.*

[07:39]
**Speaker 1:** So about those real agents. (The Anthropic article is shown again, highlighting the definition of "Agents".) Agents are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks themselves. This is what Anthropic wrote, and it is what I agree the most with. Real agents make a plan by exchanging with you and understanding your needs, iterating at a reasoning level to decide on the steps to take to solve the problem or query. Ideally, it will even ask you if it needs more information or clarification instead of hallucinating as with current LLMs.

[08:14]
(A new diagram appears showing a flow: "Human" has a dotted line to "LLM Call", which can go to "Stop" or take an "Action" in an "Environment", which returns "Feedback".)

**Speaker 1:** Still, they cannot be simply built. They require a very powerful LLM, better than those we have now, and an environment to evolve in, like a discussion with you and some extra powers like tools that they can use themselves whenever they see fit and iterate. (The workflow vs. agent comparison graphic reappears.) In short, you can see agents almost as replacing someone or a role and a workflow replacing a task one would do. There is no hardcoded path, the agentic system will make its decisions. They are much more advanced and complex things that we still haven't built very successfully yet.

*True agents are dynamic, reasoning systems that can plan, adapt, and control their own processes, essentially replacing a role rather than just a task, but current technology still struggles to build them reliably.*

[08:48]
**Speaker 1:** This independence and trust in your system obviously makes it more susceptible of failures, more expensive to run and use, added latency, and worst of all, the results aren't that exciting now. When they are, they are completely inconsistent.

[09:03]
**Speaker 1:** So what is an actual good example of an agent? Two examples that quickly come to my mind are Devin's and Anthropic's computer use. (Logos for Devin and Anthropic appear.) Yet, they are for now disappointing agents. If you're curious about Devin, there's a really good blog from Hamel Husain sharing his experience using it. (A screenshot of a blog about Devin is shown, followed by a screen recording of the Devin AI interface.) Devin offers an intriguing glimpse into the promise and challenges of agent-based systems. Designed as a fully autonomous software engineer with its own computing environment, it independently handles tasks like API integrations and real-time problem solving.

[09:37]
**Speaker 1:** However, as Hamel's extensive testing demonstrated, while Devin excelled at simpler, well-defined tasks, things that we can usually do quite easily, it struggled with complex or autonomous ones, often providing overcomplicated solutions and pursuing unfeasible paths, whereas advanced workflows like Cursor don't have as many issues.

[10:00]
**Speaker 1:** These limitations reflect the broader challenges of building reliable, context-aware agents with current LLMs, even if you raise millions and millions. Here, Devin aligns more with Anthropic's vision, showcasing the promise and challenges of a reasoning agent. (The Devin logo appears.) It can autonomously tackle complex problems but struggles with inconsistency. By contrast, workflows like those inspired by CrewAI are simpler and more robust for specific tasks but lack the flexibility of true reasoning systems. (The CrewAI logo appears.) Similarly, we have Anthropic's ambitious attempt at creating an autonomous agent having access to our computer, Anthropic computer use, which had lots of hype when it first came out and has since been quite forgotten.

[10:37]
(A screen recording of Anthropic's computer use agent is shown. It autonomously navigates a virtual desktop, opens a browser, and creates a website.)

**Speaker 1:** The system was undeniably complex and embodied the characteristics of a true agent: autonomous decision-making, dynamic tool usage, and the ability to interact with its environment. Its goal was also to replace anyone on a computer. Quite promising or scary. Still, its decline also serves as a reminder of the challenges in creating practical agentic systems that not only work as intended but do so systematically. In short, LLMs are simply not ready yet for becoming true agents, but it may be the case soon. For now, as with all things code related, we should always aim to find a solution to our problems that is as simple as possible.

[11:22]
**Speaker 1:** (A text overlay appears: "Find a solution that is as simple as possible".) One that we can iterate easily and debug easily. Simple LLM calls are often the way to go. And it is often what people and companies sell as being an agent. But you won't be fooled anymore. You may want to complement LLMs with some external knowledge through the use of retrieval systems or light fine-tuning, but your money and time aiming for true agents should be saved for really complex problems that cannot be solved otherwise.

[11:51]
**Speaker 1:** I hope this video helped you understand the difference between workflows and a real agent and when to use both. If you found it useful, please share it with a friend in the AI community and don't forget to subscribe for more in-depth AI content. Thank you for watching.

(The video ends with a blue screen with social media handles and a logo.)

*The speaker concludes by reiterating that while true agents are still unreliable and complex to build, simpler, more predictable workflows are highly effective for most current applications.*

</details>


## Additional Sources Scraped

<details>
<summary>a-developer-s-guide-to-building-scalable-ai-workflows-vs-age</summary>

# A Developer’s Guide to Building Scalable AI: Workflows vs Agents

Understanding the architectural trade-offs between autonomous agents and orchestrated workflows — because someone needs to make this decision, and it might as well be youhttps://towardsdatascience.com/wp-content/uploads/2025/06/agent-vs-workflow.jpegImage by author

There was a time not long ago — okay, like three months ago — when I fell deep into the agent rabbit hole.

I had just started experimenting with CrewAI and LangGraph, and it felt like I’d unlocked a whole new dimension of building. Suddenly, I didn’t just have tools and pipelines — I had _crews_. I could spin up agents that could reason, plan, talk to tools, and talk to each other. Multi-agent systems! Agents that summon other agents! I was practically architecting the AI version of a startup team.

Every use case became a candidate for a crew. Meeting prep? Crew. Slide generation? Crew. Lab report review? Crew.

It was exciting — until it wasn’t.

The more I built, the more I ran into questions I hadn’t thought through: _How do I monitor this? How do I debug a loop where the agent just keeps “thinking”? What happens when something breaks? Can anyone else even maintain this with me?_

That’s when I realized I had skipped a crucial question: _Did this really need to be agentic?_ Or was I just excited to use the shiny new thing?

Since then, I’ve become a lot more cautious — and a lot more practical. Because there’s a big difference (according to [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)) between:

- A **workflow**: a structured LLM pipeline with clear control flow, where you define the steps — use a tool, retrieve context, call the model, handle the output.
- And an **agent**: an autonomous system where the LLM decides what to do next, which tools to use, and when it’s “done.”

Workflows are more like you calling the shots and the LLM following your lead. Agents are more like hiring a brilliant, slightly chaotic intern who figures things out on their own — sometimes beautifully, sometimes in terrifyingly expensive ways.

This article is for anyone who’s ever felt that same temptation to build a multi-agent empire before thinking through what it takes to maintain it. It’s not a warning, it’s a reality check — and a field guide. Because there _are_ times when agents are exactly what you need. But most of the time? You just need a solid workflow.

* * *

## The State of AI Agents: Everyone’s Doing It, Nobody Knows Why

You’ve probably seen the stats. [95% of companies are now using generative AI, with 79% specifically implementing AI agents](https://www.bain.com/insights/survey-generative-ai-uptake-is-unprecedented-despite-roadblocks/), according to Bain’s 2024 survey. That sounds impressive — until you look a little closer and find out only _1%_ of them consider those implementations “mature.”

Translation: most teams are duct-taping something together and hoping it doesn’t explode in production.

I say this with love — I was one of them.

There’s this moment when you first build an agent system that works — even a small one — and it _feels like magic_. The LLM decides what to do, picks tools, loops through steps, and comes back with an answer like it just went on a mini journey. You think: “Why would I ever write rigid pipelines again when I can just let the model figure it out?”

And then the complexity creeps in.

You go from a clean pipeline to a network of tool-wielding LLMs reasoning in circles. You start writing logic to correct the logic of the agent. You build an agent to supervise the other agents. Before you know it, you’re maintaining a distributed system of interns with anxiety and no sense of cost.

Yes, there are real success stories. [Klarna’s agent handles the workload of 700 customer service reps](https://www.klarna.com/international/press/klarna-ai-assistant-handles-two-thirds-of-customer-service-chats-in-its-first-month/). [BCG built a multi-agent design system that cut shipbuilding engineering time by nearly half.](https://www.bcg.com/publications/2025/how-ai-can-be-the-new-all-star-on-your-team) These are not demos — these are production systems, saving companies real time and money.

But those companies didn’t get there by accident. Behind the scenes, they invested in infrastructure, observability, fallback systems, budget controls, and teams who could debug prompt chains at 3 AM without crying.

For most of us? We’re not Klarna. We’re trying to get something working that’s reliable, cost-effective, and doesn’t eat up 20x more tokens than a well-structured pipeline.

So yes, agents _can_ be amazing. But we have to stop pretending they’re a default. Just because the model _can_ decide what to do next doesn’t mean it _should_. Just because the flow is dynamic doesn’t mean the system is smart. And just because everyone’s doing it doesn’t mean you need to follow.

Sometimes, using an agent is like replacing a microwave with a sous chef — more flexible, but also more expensive, harder to manage, and occasionally makes decisions you didn’t ask for.

Let’s figure out when it actually makes sense to go that route — and when you should just stick with something that works.

## Technical Reality Check: What You’re Actually Choosing Between

Before we dive into the existential crisis of choosing between agents and workflows, let’s get our definitions straight. Because in typical tech fashion, everyone uses these terms to mean slightly different things.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/image-115.pngimage by author

### Workflows: The Reliable Friend Who Shows Up On Time

Workflows are orchestrated. You write the logic: maybe retrieve context with a vector store, call a toolchain, then use the LLM to summarize the results. Each step is explicit. It’s like a recipe. If it breaks, you know exactly where it happened — and probably how to fix it.

This is what most “RAG pipelines” or prompt chains are. Controlled. Testable. Cost-predictable.

The beauty? You can debug them the same way you debug any other software. Stack traces, logs, fallback logic. If the vector search fails, you catch it. If the model response is weird, you reroute it.

Workflows are your dependable friend who shows up on time, sticks to the plan, and doesn’t start rewriting your entire database schema because it felt “inefficient.”https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/image-112.pngImage by author, inspired by [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

In this example of a simple customer support task, this workflow always follows the same classify → route → respond → log pattern. It’s predictable, debuggable, and performs consistently.

```python
def customer_support_workflow(customer_message, customer_id):
    """Predefined workflow with explicit control flow"""

    # Step 1: Classify the message type
    classification_prompt = f"Classify this message: {customer_message}\nOptions: billing, technical, general"
    message_type = llm_call(classification_prompt)

    # Step 2: Route based on classification (explicit paths)
    if message_type == "billing":
        # Get customer billing info
        billing_data = get_customer_billing(customer_id)
        response_prompt = f"Answer this billing question: {customer_message}\nBilling data: {billing_data}"

    elif message_type == "technical":
        # Get product info
        product_data = get_product_info(customer_id)
        response_prompt = f"Answer this technical question: {customer_message}\nProduct info: {product_data}"

    else:  # general
        response_prompt = f"Provide a helpful general response to: {customer_message}"

    # Step 3: Generate response
    response = llm_call(response_prompt)

    # Step 4: Log interaction (explicit)
    log_interaction(customer_id, message_type, response)

    return response
```

The deterministic approach provides:

- **Predictable execution**: Input A always leads to Process B, then Result C
- **Explicit error handling**: “If this breaks, do that specific thing”
- **Transparent debugging**: You can literally trace through the code to find problems
- **Resource optimization**: You know exactly how much everything will cost

[Workflow implementations deliver consistent business value](https://ascendix.com/blog/salesforce-success-stories/): OneUnited Bank achieved 89% credit card conversion rates, while Sequoia Financial Group saved 700 hours annually per user. Not as sexy as “autonomous AI,” but your operations team will love you.

### Agents: The Smart Kid Who Sometimes Goes Rogue

Agents, on the other hand, are built around loops. The LLM gets a goal and starts reasoning about how to achieve it. It picks tools, takes actions, evaluates outcomes, and decides what to do next — all inside a recursive decision-making loop.

This is where things get… fun.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/image-111.pngImage by author, inspired by [Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

The architecture enables some genuinely impressive capabilities:

- **Dynamic tool selection**: “Should I query the database or call the API? Let me think…”
- **Adaptive reasoning**: Learning from mistakes within the same conversation
- **Self-correction**: “That didn’t work, let me try a different approach”
- **Complex state management**: Keeping track of what happened three steps ago

In the same example, the agent might decide to search the knowledge base first, then get billing info, then ask clarifying questions — all based on its interpretation of the customer’s needs. The execution path varies depending on what the agent discovers during its reasoning process:

```python
def customer_support_agent(customer_message, customer_id):
    """Agent with dynamic tool selection and reasoning"""

    # Available tools for the agent
    tools = {
        "get_billing_info": lambda: get_customer_billing(customer_id),
        "get_product_info": lambda: get_product_info(customer_id),
        "search_knowledge_base": lambda query: search_kb(query),
        "escalate_to_human": lambda: create_escalation(customer_id),
    }

    # Agent prompt with tool descriptions
    agent_prompt = f"""
    You are a customer support agent. Help with this message: "{customer_message}"

    Available tools: {list(tools.keys())}

    Think step by step:
    1. What type of question is this?
    2. What information do I need?
    3. Which tools should I use and in what order?
    4. How should I respond?

    Use tools dynamically based on what you discover.
    """

    # Agent decides what to do (dynamic reasoning)
    agent_response = llm_agent_call(agent_prompt, tools)

    return agent_response
```

Yes, that autonomy is what makes agents powerful. It’s also what makes them hard to control.

Your agent might:

- decide to try a new strategy mid-way
- forget what it already tried
- or call a tool 15 times in a row trying to “figure things out”

You can’t just set a breakpoint and inspect the stack. The “stack” is inside the model’s context window, and the “variables” are fuzzy thoughts shaped by your prompts.

When something goes wrong — and it will — you don’t get a nice red error message. You get a token bill that looks like someone mistyped a loop condition and summoned the OpenAI API 600 times. (I know, because I did this at least once where I forgot to cap the loop, and the agent just kept thinking… and thinking… until the entire system crashed with an “out of token” error).

* * *

To put it in simpler terms, you can think of it like this:

A **workflow** is a GPS.

You know the destination. You follow clear instructions. “Turn left. Merge here. You’ve arrived.” It’s structured, predictable, and you almost always get where you’re going — unless you ignore it on purpose.

An **agent** is different. It’s like handing someone a map, a smartphone, a credit card, and saying:

> “Figure out how to get to the airport. You can walk, call a cab, take a detour if needed — just make it work.”

They might arrive faster. Or they might end up arguing with a rideshare app, taking a scenic detour, and arriving an hour later with a $18 smoothie. (We all know someone like that).

**Both approaches can work**, but the real question is:

> **Do you actually need autonomy here, or just a reliable set of instructions?**

Because here’s the thing — agents _sound_ amazing. And they are, in theory. You’ve probably seen the headlines:

- “Deploy an agent to handle your entire support pipeline!”
- “Let AI manage your tasks while you sleep!”
- “Revolutionary multi-agent systems — your personal consulting firm in the cloud!”

These case studies are everywhere. And some of them are real. But most of them?

They’re like travel photos on Instagram. You see the glowing sunset, the perfect skyline. You don’t see the six hours of layovers, the missed train, the $25 airport sandwich, or the three-day stomach bug from the street tacos.

That’s what agent success stories often leave out: **the operational complexity, the debugging pain, the spiraling token bill**.

So yeah, agents _can_ take you places. But before you hand over the keys, make sure you’re okay with the route they might choose. And that you can afford the tolls.

## The Hidden Costs Nobody Talks About

On paper, agents seem magical. You give them a goal, and they figure out how to achieve it. No need to hardcode control flow. Just define a task and let the system handle the rest.

In theory, it’s elegant. In practice, it’s chaos in a trench coat.

Let’s talk about what it _really_ costs to go agentic — not just in dollars, but in complexity, failure modes, and emotional wear-and-tear on your engineering team.

### Token Costs Multiply — Fast

[According to Anthropic’s research](https://www.anthropic.com/engineering/built-multi-agent-research-system), agents consume 4x more tokens than simple chat interactions. Multi-agent systems? Try 15x more tokens. This isn’t a bug — it’s the whole point. They loop, reason, re-evaluate, and often talk to themselves several times before arriving at a decision.

Here’s how that math breaks down:

- **Basic workflows**: $500/month for 100k interactions
- **Single agent systems**: $2,000/month for the same volume
- **Multi-agent systems**: $7,500/month (assuming $0.005 per 1K tokens)

And that’s if everything is working as intended.

If the agent gets stuck in a tool call loop or misinterprets instructions? You’ll see spikes that make your billing dashboard look like a crypto pump-and-dump chart.

### Debugging Feels Like AI Archaeology

With workflows, debugging is like walking through a well-lit house. You can trace input → function → output. Easy.

With agents? It’s more like wandering through an unmapped forest where the trees occasionally rearrange themselves. You don’t get traditional logs. You get _reasoning traces_, full of model-generated thoughts like:

> “Hmm, that didn’t work. I’ll try another approach.”

That’s not a stack trace. That’s an AI diary entry. It’s poetic, but not helpful when things break in production.

The really “fun” part? **Error propagation in agent systems can cascade in completely unpredictable ways.** One incorrect decision early in the reasoning chain can lead the agent down a rabbit hole of increasingly wrong conclusions, like a game of telephone where each player is also trying to solve a math problem. Traditional debugging approaches — setting breakpoints, tracing execution paths, checking variable states — become much less helpful when the “bug” is that your AI decided to interpret your instructions creatively.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/image-113.pngImage by author, generated by GPT-4o

### New Failure Modes You’ve Never Had to Think About

[Microsoft’s research has identified](https://www.microsoft.com/en-us/security/blog/2025/04/24/new-whitepaper-outlines-the-taxonomy-of-failure-modes-in-ai-agents/) entirely **new failure modes that didn’t exist before agents**. Here are just a few that aren’t common in traditional pipelines:

- **Agent Injection**: Prompt-based exploits that hijack the agent’s reasoning
- **Multi-Agent Jailbreaks**: Agents colluding in unintended ways
- **Memory Poisoning**: One agent corrupts shared memory with hallucinated nonsense

These aren’t edge cases anymore — they’re becoming common enough that entire subfields of “LLMOps” now exist just to handle them.

If your monitoring stack doesn’t track token drift, tool spam, or emergent agent behavior, you’re flying blind.

### You’ll Need Infra You Probably Don’t Have

Agent-based systems don’t just need compute — they need new layers of tooling.

You’ll probably end up cobbling together some combo of:

- **LangFuse**, **Arize**, or **Phoenix** for observability
- **AgentOps** for cost and behavior monitoring
- Custom token guards and fallback strategies to stop runaway loops

This tooling stack _isn’t optional_. It’s required to keep your system stable.

And if you’re not already doing this? You’re not ready for agents in production — at least, not ones that impact real users or money.

* * *

So yeah. It’s not that agents are “bad.” They’re just a lot more expensive — financially, technically, and emotionally — than most people realize when they first start playing with them.

The tricky part is that none of this shows up in the demo. In the demo, it looks clean. Controlled. Impressive.

But in production, things leak. Systems loop. Context windows overflow. And you’re left explaining to your boss why your AI system spent $5,000 calculating the best time to send an email.

## When Agents Actually Make Sense

_\[Before we dive into agent success stories, a quick reality check: these are patterns observed from analyzing current implementations, not universal laws of software architecture. Your mileage may vary, and there are plenty of organizations successfully using workflows for scenarios where agents might theoretically excel. Consider these informed observations rather than divine commandments carved in silicon.\]_

Alright. I’ve thrown a lot of caution tape around agent systems so far — but I’m not here to scare you off forever.

Because sometimes, agents are _exactly_ what you need. They’re brilliant in ways that rigid workflows simply can’t be.

The trick is knowing the difference between “I want to try agents because they’re cool” and “this use case actually needs autonomy.”

Here are a few scenarios where agents genuinely earn their keep.

### Dynamic Conversations With High Stakes

Let’s say you’re building a customer support system. Some queries are straightforward — refund status, password reset, etc. A simple workflow handles those perfectly.

But other conversations? They require adaptation. Back-and-forth reasoning. Real-time prioritization of what to ask next based on what the user says.

That’s where agents shine.

In these contexts, you’re not just filling out a form — you’re navigating a situation. Personalized troubleshooting, product recommendations, contract negotiations — things where the next step depends entirely on what just happened.

Companies implementing agent-based customer support systems have reported wild ROI — we’re talking [112% to 457%](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai) increases in efficiency and conversions, depending on the industry. Because when done right, agentic systems _feel_ smarter. And that leads to trust.

### High-Value, Low-Volume Decision-Making

Agents are expensive. But sometimes, the decisions they’re helping with are _more_ expensive.

BCG helped a shipbuilding firm cut 45% of its engineering effort using a multi-agent design system. That’s worth it — because those decisions were tied to multi-million dollar outcomes.

If you’re optimizing how to lay fiber optic cable across a continent or analyzing legal risks in a contract that affects your entire company — burning a few extra dollars on compute isn’t the problem. The _wrong_ decision is.

Agents work here because the _cost of being wrong_ is way higher than the _cost of computing_.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/when-agents-win-683x1024.jpegImage by author

### Open-Ended Research and Exploration

There are problems where you literally can’t define a flowchart upfront — because you don’t know what the “right steps” are.

Agents are great at diving into ambiguous tasks, breaking them down, iterating on what they find, and adapting in real-time.

Think:

- Technical research assistants that read, summarize, and compare papers
- Product analysis bots that explore competitors and synthesize insights
- Research agents that investigate edge cases and suggest hypotheses

These aren’t problems with known procedures. They’re open loops by nature — and agents thrive in those.

### **Multi-Step, Unpredictable Workflows**

Some tasks have too many branches to hardcode — the kind where writing out all the “if this, then that” conditions becomes a full-time job.

This is where agent loops can actually _simplify_ things, because the LLM handles the flow dynamically based on context, not pre-written logic.

Think diagnostics, planning tools, or systems that need to factor in dozens of unpredictable variables.

If your logic tree is starting to look like a spaghetti diagram made by a caffeinated octopus — yeah, maybe it’s time to let the model take the wheel.

* * *

So no, I’m not anti-agent (I actually love them!) I’m pro-alignment — matching the tool to the task.

When the use case _needs_ flexibility, adaptation, and autonomy, then yes — bring in the agents. But only after you’re honest with yourself about whether you’re solving a real complexity… or just chasing a shiny abstraction.

## When Workflows Are Obviously Better (But Less Exciting)

_\[Again, these are observations drawn from industry analysis rather than ironclad rules. There are undoubtedly companies out there successfully using agents for regulated processes or cost-sensitive applications — possibly because they have specific requirements, exceptional expertise, or business models that change the economics. Think of these as strong starting recommendations, not limitations on what’s possible.\]_

Let’s step back for a second.

A lot of AI architecture conversations get stuck in hype loops — “Agents are the future!” “AutoGPT can build companies!” — but in actual production environments, most systems don’t need agents.

They need something that works.

That’s where workflows come in. And while they may not feel as futuristic, they are **incredibly effective** in the environments that most of us are building for.

### Repeatable Operational Tasks

If your use case involves clearly defined steps that rarely change — like sending follow-ups, tagging data, validating form inputs — a workflow will outshine an agent every time.

It’s not just about cost. It’s about stability.

You don’t want creative reasoning in your payroll system. You want the same result, every time, with no surprises. A well-structured pipeline gives you that.

There’s nothing sexy about “process reliability” — until your agent-based system forgets what year it is and flags every employee as a minor.

### Regulated, Auditable Environments

Workflows are deterministic. That means they’re traceable. Which means if something goes wrong, you can show exactly what happened — step-by-step — with logs, fallbacks, and structured output.

If you’re working in healthcare, finance, law, or government — places where **“we think the AI decided to try something new”** is not an acceptable answer — this matters.

You can’t build a safe AI system without transparency. Workflows give you that by default.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/when-workflows-win-683x1024.jpegImage by author

### High-Frequency, Low-Complexity Scenarios

There are entire categories of tasks where the **cost per request** matters more than the sophistication of reasoning. Think:

- Fetching info from a database
- Parsing emails
- Responding to FAQ-style queries

A workflow can handle thousands of these requests per minute, at predictable costs and latency, with zero risk of runaway behavior.

If you’re scaling fast and need to stay lean, a structured pipeline beats a clever agent.

### Startups, MVPs, and Just-Get-It-Done Projects

Agents require infrastructure. Monitoring. Observability. Cost tracking. Prompt architecture. Fallback planning. Memory design.

If you’re not ready to invest in all of that — and most early-stage teams aren’t — agents are probably too much, too soon.

Workflows let you move fast and learn how LLMs behave before you get into recursive reasoning and emergent behavior debugging.

Think of it this way: workflows are how you **get to production**. Agents are how you scale specific use cases once you understand your system deeply.

* * *

One of the best mental models I’ve seen (shoutout to [Anthropic’s engineering blog](https://www.anthropic.com/engineering/building-effective-agents)) is this:

> **Use workflows to build structure around the predictable. Use agents to explore the unpredictable.**

Most real-world AI systems are a mix — and many of them lean heavily on workflows because **production doesn’t reward cleverness**. It rewards **resilience**.

## A Decision Framework That Actually Works

Here’s something I’ve learned (the hard way, of course): most bad architecture decisions don’t come from a lack of knowledge — they come from moving too fast.

You’re in a sync. Someone says, “This feels a bit too dynamic for a workflow — maybe we just go with agents?”

Everyone nods. It sounds reasonable. Agents are flexible, right?

Fast forward three months: the system’s looping in weird places, the logs are unreadable, costs are spiking, and no one remembers who suggested using agents in the first place. You’re just trying to figure out why an LLM decided to summarize a refund request by booking a flight to Peru.

So, let’s slow down for a second.

This isn’t about picking the trendiest option — it’s about building something you can explain, scale, and actually maintain.

The framework below is designed to make you pause and think clearly before the token bills stack up and your nice prototype turns into a very expensive choose-your-own-adventure story.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/Mediamodifier-Design1.svgImage by author

### The Scoring Process: Because Single-Factor Decisions Are How Projects Die

This isn’t a decision tree that bails out at the first “sounds good.” It’s a structured evaluation. You go through **five dimensions**, score each one, and see what the system is really asking for — not just what sounds fun.

**Here’s how it works:**

> - Each dimension gives **+2 points** to either workflow or agents.
> - One question gives **+1 point** (reliability).
> - Add it all up at the end — and trust the result more than your agent hype cravings.

* * *

### Complexity of the Task (2 points)

Evaluate whether your use case has well-defined procedures. Can you write down steps that handle 80% of your scenarios without resorting to hand-waving?

- Yes → +2 for **workflows**
- No, there’s ambiguity or dynamic branching → +2 for **agents**

If your instructions involve phrases like “and then the system figures it out” — you’re probably in agent territory.

### Business Value vs. Volume (2 points)

Assess the cold, hard economics of your use case. Is this a high-volume, cost-sensitive operation — or a low-volume, high-value scenario?

- High-volume and predictable → +2 for **workflows**
- Low-volume but high-impact decisions → +2 for **agents**

Basically: if compute cost is more painful than getting something slightly wrong, workflows win. If being wrong is expensive and being slow loses money, agents might be worth it.

### Reliability Requirements (1 point)

Determine your tolerance for output variability — and be honest about what your business actually needs, not what sounds flexible and modern. How much output variability can your system tolerate?

- Needs to be consistent and traceable (audits, reports, clinical workflows) → +1 for **workflows**
- Can handle some variation (creative tasks, customer support, exploration) → +1 for **agents**

This one’s often overlooked — but it directly affects how much guardrail logic you’ll need to write (and maintain).

### Technical Readiness (2 points)

Evaluate your current capabilities without the rose-colored glasses of “we’ll figure it out later.” What’s your current engineering setup and comfort level?

- You’ve got logging, traditional monitoring, and a dev team that hasn’t yet built agentic infra → +2 for **workflows**
- You already have observability, fallback plans, token tracking, and a team that understands emergent AI behavior → +2 for **agents**

This is your system maturity check. Be honest with yourself. Hope is not a debugging strategy.

### Organizational Maturity (2 points)

Assess your team’s AI expertise with brutal honesty — this isn’t about intelligence, it’s about experience with the specific weirdness of AI systems. How experienced is your team with prompt engineering, tool orchestration, and LLM weirdness?

- Still learning prompt design and LLM behavior → +2 for **workflows**
- Comfortable with distributed systems, LLM loops, and dynamic reasoning → +2 for **agents**

You’re not evaluating intelligence here — just experience with a specific class of problems. Agents demand a deeper familiarity with AI-specific failure patterns.

* * *

### Add Up Your Score

After completing all five evaluations, calculate your total scores.

- **Workflow score ≥ 6** → Stick with workflows. You’ll thank yourself later.
- **Agent score ≥ 6** → Agents might be viable — _if_ there are no workflow-critical blockers.

**Important**: This framework doesn’t tell you what’s coolest. It tells you what’s sustainable.

A lot of use cases will lean workflow-heavy. That’s not because agents are bad — it’s because true agent readiness involves _many_ systems working in harmony: infrastructure, ops maturity, team knowledge, failure handling, and cost controls.

And if any one of those is missing, it’s usually not worth the risk — yet.

## The Plot Twist: You Don’t Have to Choose

Here’s a realization I wish I’d had earlier: you don’t have to pick sides. The magic often comes from **hybrid systems** — where workflows provide stability, and agents offer flexibility. It’s the best of both worlds.

Let’s explore how that actually works.

### Why Hybrid Makes Sense

Think of it as layering:

1. **Reactive layer** (your workflow): handles predictable, high-volume tasks
2. **Deliberative layer** (your agent): steps in for complex, ambiguous decisions

This is exactly how many real systems are built. The workflow handles the 80% of predictable work, while the agent jumps in for the 20% that needs creative reasoning or planning

### Building Hybrid Systems Step by Step

Here’s a refined approach I’ve used (and borrowed from hybrid best practices):

1. **Define the core workflow.**

Map out your predictable tasks — data retrieval, vector search, tool calls, response synthesis.
2. **Identify decision points.**

Where might you _need_ an agent to decide things dynamically?
3. **Wrap those steps with lightweight agents.**

Think of them as scoped decision engines — they plan, act, reflect, then return answers to the workflow .
4. **Use memory and plan loops wisely.**

Give the agent just enough context to make smart choices without letting it go rogue.
5. **Monitor and fail gracefully.**

If the agent goes wild or costs spike, fall back to a default workflow branch. Keep logs and token meters running.
6. **Human-in-the-loop checkpoint.**

Especially in regulated or high-stakes flows, pause for human validation before agent-critical actions

### When to Use Hybrid Approach

| Scenario | Why Hybrid Works |
| --- | --- |
| Customer support | Workflow does easy stuff, agents adapt when conversations get messy |
| Content generation | Workflow handles format and publishing; agent writes the body |
| Data analysis/reporting | Agents summarize & interpret; workflows aggregate & deliver |
| High-stakes decisions | Use agent for exploration, workflow for execution and compliance |

When to use hybrid approach

This aligns with how systems like WorkflowGen, n8n, and Anthropic’s own tooling advise building — stable pipelines with scoped autonomy.

### Real Examples: Hybrid in Action

#### A Minimal Hybrid Example

Here’s a scenario I used with LangChain and LangGraph:

- **Workflow stage**: fetch support tickets, embed & search
- **Agent cell**: decide whether it’s a refund question, a complaint, or a bug report
- **Workflow**: run the correct branch based on agent’s tag
- **Agent stage**: if it’s a complaint, summarize sentiment and suggest next steps
- **Workflow**: format and send response; log everything

The result? Most tickets flow through without agents, saving cost and complexity. But when ambiguity hits, the agent steps in and adds real value. No runaway token bills. Clear traceability. Automatic fallbacks.

This pattern splits the logic between a structured workflow and a scoped agent. ( **Note: this is a high-level demonstration**)

```python
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

# 1. Workflow: set up RAG pipeline
embeddings = OpenAIEmbeddings()
vectordb = FAISS.load_local(
    "docs_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectordb.as_retriever()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([\
    ("system", system_prompt),\
    ("human", "{input}"),\
])

llm = init_chat_model("openai:gpt-4.1", temperature=0)
qa_chain = create_retrieval_chain(
    retriever,
    create_stuff_documents_chain(llm, prompt)
)

# 2. Agent: Set up agent with Tavily search
search = TavilySearchResults(max_results=2)
agent_llm = init_chat_model("anthropic:claude-3-7-sonnet-latest", temperature=0)
agent = create_react_agent(
    model=agent_llm,
    tools=[search]
)

# Uncertainty heuristic
def is_answer_uncertain(answer: str) -> bool:
    keywords = [\
        "i don't know", "i'm not sure", "unclear",\
        "unable to answer", "insufficient information",\
        "no information", "cannot determine"\
    ]
    return any(k in answer.lower() for k in keywords)

def hybrid_pipeline(query: str) -> str:
    # RAG attempt
    rag_out = qa_chain.invoke({"input": query})
    rag_answer = rag_out.get("answer", "")

    if is_answer_uncertain(rag_answer):
        # Fallback to agent search
        agent_out = agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return agent_out["messages"][-1].content

    return rag_answer

if __name__ == "__main__":
    result = hybrid_pipeline("What are the latest developments in AI?")
    print(result)

```

**What’s happening here:**

- The workflow takes the first shot.
- If the result seems weak or uncertain, the agent takes over.
- You only pay the agent cost when you really need to.

Simple. Controlled. Scalable.

#### Advanced: Workflow-Controlled Multi-Agent Execution

If your problem _really_ calls for multiple agents — say, in a research or planning task — structure the system as a **graph**, not a soup of recursive loops. ( **Note: this is a high level demonstration**)

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AnyMessage

# 1. Define your graph's state
class TaskState(TypedDict):
    input: str
    label: str
    output: str

# 2. Build the graph
graph = StateGraph(TaskState)

# 3. Add your classifier node
def classify(state: TaskState) -> TaskState:
    # example stub:
    state["label"] = "research" if "latest" in state["input"] else "summary"
    return state

graph.add_node("classify", classify)
graph.add_edge(START, "classify")

# 4. Define conditional transitions out of the classifier node
graph.add_conditional_edges(
    "classify",
    lambda s: s["label"],
    path_map={"research": "research_agent", "summary": "summarizer_agent"}
)

# 5. Define the agent nodes
research_agent = ToolNode([create_react_agent(...tools...)])
summarizer_agent = ToolNode([create_react_agent(...tools...)])

# 6. Add the agent nodes to the graph
graph.add_node("research_agent", research_agent)
graph.add_node("summarizer_agent", summarizer_agent)

# 7. Add edges. Each agent node leads directly to END, terminating the workflow
graph.add_edge("research_agent", END)
graph.add_edge("summarizer_agent", END)

# 8. Compile and run the graph
app = graph.compile()
final = app.invoke({"input": "What are today's AI headlines?", "label": "", "output": ""})
print(final["output"])

```

This pattern gives you:

- **Workflow-level control** over routing and memory
- **Agent-level reasoning** where appropriate
- **Bounded loops** instead of infinite agent recursion

This is how tools like LangGraph are designed to work: **structured autonomy**, not free-for-all reasoning.

## Production Deployment — Where Theory Meets Reality

All the architecture diagrams, decision trees, and whiteboard debates in the world won’t save you if your AI system falls apart the moment real users start using it.

Because that’s where things get messy — the inputs are noisy, the edge cases are endless, and users have a magical ability to break things in ways you never imagined. Production traffic has a personality. It will test your system in ways your dev environment never could.

And that’s where most AI projects stumble.

The demo works. The prototype impresses the stakeholders. But then you go live — and suddenly the model starts hallucinating customer names, your token usage spikes without explanation, and you’re ankle-deep in logs trying to figure out why everything broke at 3:17 a.m. (True story!)

This is the gap between a cool proof-of-concept and a system that actually holds up in the wild. It’s also where the difference between workflows and agents stops being philosophical and starts becoming very, very operational.

Whether you’re using agents, workflows, or some hybrid in between — once you’re in production, it’s a different game.

You’re no longer trying to prove that the AI _can_ work.

You’re trying to make sure it works **reliably, affordably, and safely** — every time.

So what does that actually take?

Let’s break it down.

### Monitoring (Because “It Works on My Machine” Doesn’t Scale)

Monitoring an agent system isn’t just “nice to have” — it’s survival gear.

You can’t treat agents like regular apps. Traditional APM tools won’t tell you why an LLM decided to loop through a tool call 14 times or why it burned 10,000 tokens to summarize a paragraph.

You need observability tools that speak the agent’s language. That means tracking:

- token usage patterns,
- tool call frequency,
- response latency distributions,
- task completion outcomes,
- and cost per interaction — **in real time**.

This is where tools like **LangFuse**, **AgentOps**, and **Arize Phoenix** come in. They let you peek into the black box — see what decisions the agent is making, how often it’s retrying things, and what’s going off the rails before your budget does.

Because when something breaks, “the AI made a weird choice” is not a helpful bug report. You need traceable reasoning paths and usage logs — not just vibes and token explosions.

Workflows, by comparison, are way easier to monitor.

You’ve got:

- response times,
- error rates,
- CPU/memory usage,
- and request throughput.

All the usual stuff you already track with your standard APM stack — Datadog, Grafana, Prometheus, whatever. No surprises. No loops trying to plan their next move. Just clean, predictable execution paths.

So yes — both need monitoring. But agent systems demand a whole new layer of visibility. If you’re not prepared for that, production will make sure you learn it the hard way.https://contributor.insightmediagroup.io/wp-content/uploads/2025/06/image-116.pngImage by author

### Cost Management (Before Your CFO Stages an Intervention)

Token consumption in production can spiral out of control faster than you can say “autonomous reasoning.”

It starts small — a few extra tool calls here, a retry loop there — and before you know it, you’ve burned through half your monthly budget debugging a single conversation. Especially with agent systems, costs don’t just add up — they compound.

That’s why smart teams treat **cost management like infrastructure**, not an afterthought.

Some common (and necessary) strategies:

- **Dynamic model routing** — Use lightweight models for simple tasks, save the expensive ones for when it actually matters.
- **Caching** — If the same question comes up a hundred times, you shouldn’t pay to answer it a hundred times.
- **Spending alerts** — Automated flags when usage gets weird, so you don’t learn about the problem from your CFO.

With agents, this matters even more.

Because once you hand over control to a reasoning loop, you lose visibility into how many steps it’ll take, how many tools it’ll call, and how long it’ll “think” before returning an answer.

If you don’t have real-time cost tracking, per-agent budget limits, and graceful fallback paths — you’re just one prompt away from a very expensive mistake.

Agents are smart. But they’re not cheap. Plan accordingly.

Workflows need cost management too.

If you’re calling an LLM for every user request, especially with retrieval, summarization, and chaining steps — the numbers add up. And if you’re using GPT-4 everywhere out of convenience? You’ll feel it on the invoice.

But workflows are _predictable_. You know how many calls you’re making. You can precompute, batch, cache, or swap in smaller models without disrupting logic. Cost scales linearly — and predictably.

### Security (Because Autonomous AI and Security Are Best Friends)

AI security isn’t just about guarding endpoints anymore — it’s about preparing for systems that can make their own decisions.

That’s where the concept of **shifting left** comes in — bringing security earlier into your development lifecycle.

> Instead of bolting on security after your app “works,” shift-left means designing with security from day one: during prompt design, tool configuration, and pipeline setup.

With **agent-based systems**, you’re not just securing a predictable app. You’re securing something that can autonomously decide to call an API, access private data, or trigger an external action — often in ways you didn’t explicitly program. That’s a very different threat surface.

This means your security strategy needs to evolve. You’ll need:

- **Role-based access control** for every tool an agent can access
- **Least privilege enforcement** for external API calls
- **Audit trails** to capture every step in the agent’s reasoning and behavior
- **Threat modeling** for novel attacks like prompt injection, agent impersonation, and collaborative jailbreaking (yes, that’s a thing now)

Most traditional app security frameworks assume the code defines the behavior. But with agents, the behavior is dynamic, shaped by prompts, tools, and user input. If you’re building with autonomy, you need **security controls designed for unpredictability**.

* * *

But what about **workflows**?

They’re easier — but not risk-free.

Workflows are deterministic. You define the path, you control the tools, and there’s no decision-making loop that can go rogue. That makes security simpler and more testable — especially in environments where compliance and auditability matter.

Still, workflows touch sensitive data, integrate with third-party services, and output user-facing results. Which means:

- Prompt injection is still a concern
- Output sanitation is still essential
- API keys, database access, and PII handling still need protection

For workflows, “shifting left” means:

- Validating input/output formats early
- Running prompt tests for injection risk
- Limiting what each component can access, even if it “seems safe”
- Automating red-teaming and fuzz testing around user inputs

It’s not about paranoia — it’s about protecting your system before things go live and real users start throwing unexpected inputs at it.

* * *

Whether you’re building agents, workflows, or hybrids, the rule is the same:

> **If your system can generate actions or outputs, it can be exploited.**

So build like someone _will_ try to break it — because eventually, someone probably will.

### Testing Methodologies (Because “Trust but Verify” Applies to AI Too)

Testing production AI systems is like quality-checking a very smart but slightly unpredictable intern.

They mean well. They usually get it right. But every now and then, they surprise you — and not always in a good way.

That’s why you need **layers of testing**, especially when dealing with agents.

For **agent systems**, a single bug in reasoning can trigger a whole chain of weird decisions. One wrong judgment early on can snowball into broken tool calls, hallucinated outputs, or even data exposure. And because the logic lives inside a prompt, not a static flowchart, you can’t always catch these issues with traditional test cases.

A solid testing strategy usually includes:

- **Sandbox environments** with carefully designed mock data to stress-test edge cases
- **Staged deployments** with limited real data to monitor behavior before full rollout
- **Automated regression tests** to check for unexpected changes in output between model versions
- **Human-in-the-loop reviews** — because some things, like tone or domain nuance, still need human judgment

For agents, this isn’t optional. It’s the only way to stay ahead of unpredictable behavior.

* * *

But what about **workflows**?

They’re easier to test — and honestly, that’s one of their biggest strengths.

Because workflows follow a deterministic path, you can:

- Write unit tests for each function or tool call
- Mock external services cleanly
- Snapshot expected inputs/outputs and test for consistency
- Validate edge cases without worrying about recursive reasoning or planning loops

You still want to test prompts, guard against prompt injection, and monitor outputs — but the surface area is smaller, and the behavior is traceable. You know what happens when Step 3 fails, because you wrote Step 4.

**Workflows don’t remove the need for testing — they make it testable.**

That’s a big deal when you’re trying to ship something that won’t fall apart the moment it hits real-world data.

## The Honest Recommendation: Start Simple, Scale Intentionally

If you’ve made it this far, you’re probably not looking for hype — you’re looking for a system that actually works.

So here’s the honest, slightly unsexy advice:

> **Start with workflows. Add agents only when you can clearly justify the need.**

Workflows may not feel revolutionary, but they are reliable, testable, explainable, and cost-predictable. They teach you how your system behaves in production. They give you logs, fallback paths, and structure. And most importantly: **they scale.**

That’s not a limitation. That’s maturity.

It’s like learning to cook. You don’t start with molecular gastronomy — you start by learning how to not burn rice. Workflows are your rice. Agents are the foam.

And when you do run into a problem that actually _needs_ dynamic planning, flexible reasoning, or autonomous decision-making — you’ll know. It won’t be because a tweet told you agents are the future. It’ll be because you hit a wall workflows can’t cross. And at that point, you’ll be ready for agents — and your infrastructure will be, too.

Look at the Mayo Clinic. [They run **14 algorithms on every ECG**](https://newsnetwork.mayoclinic.org/discussion/mayo-clinic-launches-new-technology-platform-ventures-to-revolutionize-diagnostic-medicine/#:~:text=Mayo%20Clinic%20and%20AI%2Ddriven%20health%20technology%20company,to%20Mayo%27s%20deep%20repository%20of%20medical%20data.)— not because it’s trendy, but because it improves diagnostic accuracy at scale. Or take [Kaiser Permanente](https://healthinnovation.ucsd.edu/news/11-health-systems-leading-in-ai), which says its AI-powered clinical support systems have helped save _hundreds of lives each year_.

These aren’t tech demos built to impress investors. These are real systems, in production, handling millions of cases — quietly, reliably, and with huge impact.

The secret? It’s not about choosing agents or workflows.

It’s about understanding the problem deeply, picking the right tools deliberately, and building for resilience — not for flash.

Because in the real world, value comes from what works.

Not what wows.

* * *

**Now go forth and make informed architectural decisions.** The world has enough AI demos that work in controlled environments. What we need are AI systems that work in the messy reality of production — regardless of whether they’re “cool” enough to get upvotes on Reddit.

</details>

<details>
<summary>build-production-agentic-rag-with-llmops-at-its-core</summary>

LLM-powered agents combine a **language model, tools, and memory** to process information and take action.

They don’t just generate text—they **reason, retrieve data, and interact with external systems** to complete tasks.

At its core, an agent takes in an input, analyzes what needs to be done, and decides the best way to respond. Instead of working in isolation, it can tap into external tools like APIs, databases, or plugins to enhance its capabilities.

With the reasoning power of LLMs, the agent doesn’t just react—it strategizes. It breaks down the task, plans the necessary steps, and takes action to get the job done efficiently.

[https://substackcdn.com/image/fetch/$s_!gLNT!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F67ffe267-55f2-4af7-9910-7410c7605550_1220x754.png](https://substackcdn.com/image/fetch/$s_!gLNT!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F67ffe267-55f2-4af7-9910-7410c7605550_1220x754.png) Figure 1: The components of an LLM-powered agent

The most popular way to design agents is by using the ReAct framework, which models the agent as follows:

- **act:** the LLM calls specific tools
- **observe:** pass the tool output back to the LLM
- **reason:** the LLM reason about the tool output to decide what to do next (e.g., call another tool or respond directly)

Now, let’s understand how agents and RAG fit together.

Unlike a traditional RAG setup's linear, step-by-step nature, Agentic RAG puts an agent at the center of decision-making.

Instead of passively retrieving and generating responses,the agent actively directs the process—deciding what to search for, how to refine queries, and when to use external tools, such as SQL, vector, or graph databases.

[https://substackcdn.com/image/fetch/$s_!ZnV_!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c59d9df-d60f-47bc-81de-cfd4fdebf5f8_1210x704.png](https://substackcdn.com/image/fetch/$s_!ZnV_!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4c59d9df-d60f-47bc-81de-cfd4fdebf5f8_1210x704.png) Figure 2: The single-agent RAG system architecture

For example, instead of querying the vector database just once (what we usually do in a standard RAG workflow), the agent might decide that after its first query, it doesn’t have enough information to provide an answer, making another request to the vector database with a different query.

Now that we’ve explored LLM-powered agents and Agentic RAGs, let’s take a step back and look at a broader question: “ **How do agents differ from workflows?”** While both help automate tasks, they operate in fundamentally different ways.

A workflow follows a fixed, predefined sequence—every step is planned in advance, making it reliable but rigid (more similar to classic programming).

In contrast, an agent **dynamically decides** what to do next **based on reasoning,** memory, and available tools. Instead of just executing steps, it adapts, learns, and makes decisions on the fly.

Think of a workflow as an assembly line, executing tasks in order, while an agent is like an intelligent assistant, capable of adjusting its approach in real time. This flexibility makes agents powerful for handling unstructured, complex problems that require dynamic decision-making.

[https://substackcdn.com/image/fetch/$s_!yBni!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5e64d5e0-7ef1-4e7f-b441-3bf1fef4ff9a_1276x818.png](https://substackcdn.com/image/fetch/$s_!yBni!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5e64d5e0-7ef1-4e7f-b441-3bf1fef4ff9a_1276x818.png) Figure 3: Differences between workflows and agents

Therefore, the trade-off between reliability and adaptability is key—workflows offer stability but are rigid, while agents provide flexibility by making dynamic decisions at the cost of consistency.

Now that we understand the basics of working with agents, let’s dive into the architecture of our Second Brain agent.

When architecting the Agentic RAG module, the goal is to build an intelligent system that efficiently combines retrieval, reasoning, and summarization to generate high-quality responses tailored to user queries.

#### What’s the interface of the pipeline?

The pipeline takes a user query as input (submitted through the Gradio UI).

The output is a refined answer generated by the agent after reasoning, retrieving relevant context from **[MongoDB](https://www.mongodb.com/products/platform/atlas-vector-search?utm_campaign=ai-pilot&utm_medium=creator&utm_term=iusztin&utm_source=blog)** through semantic search, and processing it through the summarization tool.

#### Offline vs. online pipelines

The Agentic RAG module fundamentally differs from the offline ML pipelines we’ve built in previous lessons.

This module is entirely decoupled from the pipelines in Lessons 1-5. It lives in a separate **[second-brain-online](https://github.com/decodingml/second-brain-ai-assistant-course/tree/main/apps/second-brain-online)** folder within our repository as its own standalone Python application.

This separation is intentional—by keeping the offline pipelines (feature and training) fully independent from the online inference system, we ensure a clean architectural divide.

As a quick reminder from Lesson 1, **offline pipelines** are batch pipelines that run on a schedule or trigger. They process input data and store the output artifacts in storage, allowing other pipelines or clients to consume them as needed.

These include the data collection pipeline, ETL pipeline, RAG feature pipeline, dataset generation pipeline, and training pipeline. They operate independently and are decoupled through various storage solutions such as document databases, vector databases, data registries, or model registries.

The Agentic RAG module, on the other hand, belongs to the category of **online pipelines**. It directly interacts with the user and must remain available at all times. The online pipelines available in this project are the agentic inference pipeline, the summarization inference pipeline, and the observability pipeline.

Unlike offline pipelines, these do not require orchestration and function similarly to RESTful APIs, ensuring minimal latency and efficient responses.

#### What does the pipeline’s architecture look like?

The Agentic RAG module operates in real time, instantly responding to user queries without redundant processing.

This module's core is an agent-driven system that reasons independently and dynamically invokes tools to handle user queries. They serve as extensions of the LLM model powering the agent, allowing it to perform tasks it wouldn’t efficiently handle on its own without specialized fine-tuning.

Our agent relies on three main components:

1. **The what can I do tool**, which helps users understand the usages of the system
2. **The retriever tool** that queries MongoDB’s vector index pre-populated during our offline processing
3. **The summarization tool** uses a REST API to call a different model specialized in summarizing web documents.

We specifically picked these ones as they are a perfect use case for showing how to use a tool that runs only with Python, one that calls a database, and one that calls an API (three of the most common scenarios).

The agent layer is powered by the **[SmolAgents](https://github.com/huggingface/smolagents)** framework (by Hugging Face) and orchestrates the reasoning process. A maximum number of steps can be set to ensure the reasoning remains focused and does not take unnecessary iterations to reach a response (avoiding skyrocketing bills).

To provide a seamless user experience, we integrated the agentic inference pipeline with a **[Gradio UI](https://www.gradio.app/)**, making interactions intuitive and accessible. This setup ensures that users can engage with the assistant as naturally as possible, simulating a conversational AI experience.

The interface allows us to track how the agent selects and uses tools during interactions.

For instance, we can see when it calls the **[MongoDB vector search tool](https://www.mongodb.com/products/platform/atlas-vector-search?utm_campaign=ai-pilot&utm_medium=creator&utm_term=iusztin&utm_source=blog)** to retrieve relevant data and how it cycles between retrieving information and reasoning before generating a response.

[https://substackcdn.com/image/fetch/$s_!bqEU!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb30a1f89-0c60-4a28-b87a-5390262f1500_1170x1065.png](https://substackcdn.com/image/fetch/$s_!bqEU!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb30a1f89-0c60-4a28-b87a-5390262f1500_1170x1065.png) Figure 4: The architecture of the Agentic RAG module

The agentic inference pipeline is designed to handle user queries in real time, orchestrating a seamless data flow from input to response. To understand how information moves through the system, we break down the interaction between the user, the retrieval process, and the summarization mechanism.

When a user submits a query through the **Gradio UI**, the **Agentic Layer**, an LLM-powered agent, dynamically determines the most suitable tool to process the request.

If additional context is required, the **Retriever Tool** fetches relevant information from the MongoDB vector database, extracting the most relevant chunks. This vector database was previously populated through the RAG feature pipeline in Lesson 5, ensuring the system has preprocessed, structured knowledge readily available for retrieval.

The retrieved data is then refined using the **Summarization Tool**, which enhances clarity before generating the final response. For summarization, we can choose between a custom Summarization Inference Pipeline, which is powered by the Hugging Face model we trained in Lesson 4, or an OpenAI model.

The agent continues reasoning iteratively until it reaches the predefined step limit or it decides it has the final answer, ensuring efficiency while maintaining high response quality.

As a side note, given the simplicity of our use case, the Second Brain AI assistant could have been implemented using a traditional workflow, directly retrieving and responding to queries without an agentic approach.

However, by embracing this modular strategy, we achieve greater scalability and flexibility, allowing the system to integrate new data sources or tools easily in the future.

Now that we understand how the agent works, let’s dig into how we can evaluate it and then into the implementation.

When evaluating an Agentic RAG application, it’s important to distinguish between two primary evaluation approaches: **LLM evaluation** and **Application/RAG evaluation**. Each serves a different purpose, and while LLM evaluation assesses the model in isolation, Application/RAG evaluation tests the entire application as a system.

In this case, our primary focus is evaluating the RAG pipeline as a black-box system, assessing how retrieval and reasoning work together to generate the final output.

However, we also provide a brief refresher on key insights from LLM evaluation in Lesson 4 to highlight its role in the broader evaluation process.

#### LLM evaluation

As a brief reminder, LLM evaluation measures response quality without retrieval. In Lesson 4, we tested this by analyzing the model’s ability to generate answers from its internal knowledge.

Popular methods for LLM evaluation include **benchmark-based evaluation** (using standardized datasets), **heuristic evaluation**(ROUGE, BLEU, regex matching, or custom heuristics), semantic-based evaluation (BERT Score), and **LLM-as-a-judge**, where another LLM evaluates the generated outputs.

Each method has strengths and trade-offs. Benchmark-based evaluation provides standardized comparisons but may not fully capture real-world performance, while heuristic methods may offer quick, interpretable insights but often fail to assess deeper contextual understanding. Additionally, LLM-as-a-judge is flexible and scalable, though it introduces potential biases from the evaluating model itself.

#### RAG evaluation

Unlike LLM evaluation, which assesses the model’s ability to generate responses from internal knowledge, RAG evaluation focuses on how well the retrieval and generation processes work together.

Evaluating a RAG application requires analyzing how different components interact. We focus on four key dimensions:

- **User input** – The query submitted by the user.
- **Retrieved context** – The passages or documents fetched from the vector database.
- **Generated output**– The final response produced by the LLM based on retrieved information.
- **Expected output** – The ideal or ground-truth answer, if available, for comparison.

By evaluating these dimensions, we can determine whether the retrieved context is relevant, the response is grounded in the retrieved data, and the system generates complete and accurate answers.

As mentioned, we break the process into two key steps to evaluate a RAG application correctly: retrieval and generation.

Since RAG applications rely on retrieving relevant documents before generating responses, retrieval quality plays a critical role in overall performance. If the retrieval step fails, the LLM will either generate incorrect answers or hallucinate information.

To assess **retrieval step** effectiveness, we can use various ranking-based metrics, including:

- **NDCG (Normalized Discounted Cumulative Gain)** – Measures how well the retrieved documents are ranked, prioritizing the most relevant ones at the top.
- **MRR (Mean Reciprocal Rank)** – Evaluates how early the first relevant document appears in the retrieved results, ensuring high-ranking relevance.

Another option is to visualize the embedding from your vector index (using algorithms such as t-SNE or UMAP) to see if there are any meaningful clusters within your vector space.

On the other hand, during **the generation step**, you can leverage similar strategies we looked at in the LLM evaluation subsection while considering the context dimension.

#### LLM application evaluation

For LLM application evaluation, we take a black-box approach, meaning we assess the entire Agentic RAG module rather than isolating individual components.

We evaluate the entire system by analyzing the input, output, and retrieved context instead of separating retrieval and generation into independent evaluations.

This approach allows us to identify system-wide failures and measure how well the retrieved knowledge contributes to generating accurate and relevant responses.

By evaluating the entire module, we can detect common RAG issues, such as hallucinations caused by missing context or low retrieval recall leading to incomplete answers, ensuring the system performs reliably in real-world scenarios.

#### **How many samples do we need to evaluate our LLM app?**

Naturally, using too few samples for evaluation can lead to misleading conclusions. For example, 5-10 examples are insufficient for capturing meaningful patterns, while 30-50 examples provide a reasonable starting point for evaluation.

Ideally, a dataset of over 400 samples ensures a more comprehensive assessment, helping to uncover biases and edge cases.

#### What else should be monitored along the LLM outputs?

Beyond output quality, **system performance metrics** like latency, throughput, reliability, and costs should be tracked to ensure scalability.

Additionally, **business metrics**—such as conversion rates, user engagement, or behavior influenced by the assistant—help measure the real-world impact of the LLM application.

#### Popular evaluation tools

Several tools specialize in RAG and LLM evaluation, offering similar capabilities for assessing retrieval quality and model performance.

For RAG evaluation, **RAGAS** is widely used to assess retrieval-augmented models, while **ARES** focuses on measuring how well the retrieved context supports the generated response.

**[Opik](https://github.com/comet-ml/opik)** stands out as an open-source solution that provides structured evaluations, benchmarking, and observability for LLM applications, ensuring assessment transparency and consistency.

Other proprietary alternatives include **Langfuse**, **Langsmith**, which is deeply integrated into the LangChain ecosystem for debugging and evaluation, and **Phoenix**.

​In our observability pipeline, implemented with **[Opik](https://github.com/comet-ml/opik)**, we combine monitoring and evaluation to ensure our application runs smoothly. Monitoring tracks all activities, while evaluation assesses performance and correctness.

#### What’s the interface of the pipeline?

LLMOps observability pipelines consist of two parts: one for monitoring prompts and another for evaluating the RAG module. These pipelines help us track system performance and ensure the application remains reliable.

The **prompt monitoring pipeline** captures entire prompt traces and metadata, such as prompt templates or models used within the chain. It also logs latency and system behavior while providing structured insights through dashboards that help detect and resolve inefficiencies.

The **RAG evaluation pipeline** tests the agentic RAG module using heuristics and LLM judges to assess performance. It receives a set of evaluation prompts and processes them to evaluate accuracy and reasoning quality. The pipeline outputs accuracy assessments, quality scores, and alerts for performance issues, helping maintain system reliability.

We utilize **[Opik](https://github.com/comet-ml/opik)** (by **[Comet ML](https://www.comet.com/site/products/ml-experiment-tracking?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)**), an open-source platform, to handle both the monitoring and evaluation of our application. Opik offers comprehensive tracing, automated evaluations, and production-ready dashboards, making it an ideal choice for our needs.

For evaluation, Opik automates performance assessments using both built-in and custom metrics. Users can define a threshold for any metric and configure alerts for immediate intervention if performance falls below the set value.

Now that we have an overview of the interfaces and components let’s dive into more details about each of the 2 pipelines.

#### **The prompt monitoring pipeline**

This component logs and monitors prompt traces. Prompt monitoring is essential to understand how our application interacts with users and identify areas for improvement. By tracking prompts and responses, we can debug issues in LLM reasoning or other issues like latency and costs.

Opik enables us to monitor latency across every phase of the generation process—pre-generation, generation, and post-generation—ensuring our application responds promptly to user inputs. ​

Latency is crucial to the user experience, as it includes multiple factors such as Time to First Token (TTFT), Time Between Tokens (TBT), Tokens Per Second (TPS), and Total Latency. Tracking these metrics helps us optimize response generation and manage hosting costs effectively.

Figure 5 provides an overview of how **[Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)** logs and tracks prompt traces, helping us analyze inputs, outputs, and execution times for better performance monitoring.

[https://substackcdn.com/image/fetch/$s_!8RLK!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff794b64c-1318-41f7-85be-6fc66181b969_2896x1142.png](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons) Figure 5: [Opik dashboard](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons) displaying the logged prompt traces

You can **[visualize](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)** details of the execution flow of a prompt, including its input, output, and latency at each step, as displayed in Figure 6. It helps us track the steps taken during processing, analyze latency at each stage, and identify potential inefficiencies.

[https://substackcdn.com/image/fetch/$s_!qZDa!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F49ea5358-cfbd-4c22-80fb-e57555c9f5e4_2538x1356.png](https://substackcdn.com/image/fetch/$s_!qZDa!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F49ea5358-cfbd-4c22-80fb-e57555c9f5e4_2538x1356.png) Figure 6: Details of a prompt trace stored by [Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)

Finally, in Figure 7, we can also **[visualize](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)** key metadata like retrieval parameters, system prompts, and model settings, providing deeper insights into prompt execution context:

[https://substackcdn.com/image/fetch/$s_!3qUW!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F06a65f34-04cd-4daf-9044-6ff24083ce35_2522x1726.png](https://substackcdn.com/image/fetch/$s_!3qUW!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F06a65f34-04cd-4daf-9044-6ff24083ce35_2522x1726.png) Figure 7: The metadata of a prompt trace in [Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)

The last step is to understand the RAG evaluation pipeline.

#### **The RAG evaluation pipeline**

As previously mentioned, the RAG evaluation pipeline assesses the performance of our agentic RAG module, which performs application/RAG evaluation.

The pipeline uses built-in heuristics such as Hallucination, Answer Relevance, and Moderation to evaluate response quality. Additionally, we define and integrate a custom metric and LLM judge, which assesses if the LLM's output has appropriate length and density.

This flow can also run as an offline batch pipeline during development to assess performance on test sets. Additionally, it integrates into the CI/CD pipeline to test the RAG application before deployment, ensuring any issues are identified early (similar to integration tests).

Post-deployment, it can run on a schedule to evaluate random samples from production, maintaining consistent application performance. If metrics fall below a certain threshold, we can hook an alarm system that notifies us to address potential issues promptly.

Figure 8 illustrates the results of an evaluation experiment conducted on our RAG module. It displays both the built-in and the custom performance metrics configured by us.

[https://substackcdn.com/image/fetch/$s_!S27v!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fddae553e-205c-4c68-98a4-e52a9281c7bc_2908x1096.png](https://substackcdn.com/image/fetch/$s_!S27v!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fddae553e-205c-4c68-98a4-e52a9281c7bc_2908x1096.png) Figure 8: Results of a RAG evaluation experiment stored in [Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)

**[Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)** allows us to compare multiple experiments side by side. This comparison helps track performance trends over time, making refining and improving our models easier.

[https://substackcdn.com/image/fetch/$s_!Dj6n!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fca348d2f-f918-4c2e-9868-ad82b8f6bde1_2912x1064.png](https://substackcdn.com/image/fetch/$s_!Dj6n!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fca348d2f-f918-4c2e-9868-ad82b8f6bde1_2912x1064.png) Figure 9: Comparing 2 experiments in [Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons)

By implementing these components with Opik, we maintain a robust observability pipeline that ensures our application operates efficiently.

A final note is how similar a prompt management tool, such as Opik, is to more standard experiment tracking tools, such as Comet, W&B and MLFlow. But instead of being focused on simple metrics, it’s built around the prompts as their first-class citizen.

Finally, let’s dig into the implementation.

Now that we’ve understood what it takes to build the agentic RAG and observability pipelines, let’s start implementing them.

The agentic RAG module is implemented using the SmolAgents Hugging Face frame, to build an agent that utilizes three key tools: the MongoDB retriever, the summarizer, and the "What Can I Do" tool.

Since prompt monitoring is closely tied to agent execution, here we will also cover how the system logs input/output data, latency, and other key details for each tool, ensuring full observability with Opik.

#### Building the agent

The core of our agentic RAG module starts with `get_agent()`, a method responsible for initializing the agent:

```
def get_agent(retriever_config_path: Path) -> "AgentWrapper":
    agent = AgentWrapper.build_from_smolagents(
        retriever_config_path=retriever_config_path
    )
    return agent
```

This function builds an `AgentWrapper`, which is a custom class we implemented that extends the agent's functionality by incorporating Opik for tracking all the agent’s interactions.

Building the agent requires a retriever configuration to create the MongoDB retriever tool. As a reminder from Lesson 5, we support multiple retrieval strategies based on retriever type (e.g., parent or contextual), embedding models, and other parameters.

> _**Note**: The retrieval setup is essentially copied from the offline Second Brain app in Lesson 5, ensuring consistency in document search and retrieval methods. This means the retriever is loaded exactly as it was implemented in the previous version, preserving the same retrieval logic and configurations._

#### Wrapping the agent for monitoring

The `AgentWrapper` class extends the base agent to incorporate metadata tracking with Opik. This ensures that every action taken by the agent is logged and traceable:

```
class AgentWrapper:
    def __init__(self, agent: MultiStepAgent) -> None:
        self.__agent = agent

    @property
    def input_messages(self) -> list[dict]:
        return self.__agent.input_messages

    @property
    def agent_name(self) -> str:
        return self.__agent.agent_name

    @property
    def max_steps(self) -> str:
        return self.__agent.max_steps
```

We use composition to wrap the `MultiStepAgent` from SmolAgents and expose its properties. The `MultiStepAgent` enables our agent to execute multi-step reasoning and decision-making processes.

Next, we define a method to build the agent, specifying the retriever configuration and integrating the 3 tools necessary for execution:

```
@classmethod
    def build_from_smolagents(cls, retriever_config_path: Path) -> "AgentWrapper":
        retriever_tool = MongoDBRetrieverTool(config_path=retriever_config_path)
        if settings.USE_HUGGINGFACE_DEDICATED_ENDPOINT:
            logger.warning(
                f"Using Hugging Face dedicated endpoint as the summarizer with URL: {settings.HUGGINGFACE_DEDICATED_ENDPOINT}"
            )
            summarizer_tool = HuggingFaceEndpointSummarizerTool()
        else:
            logger.warning(
                f"Using OpenAI as the summarizer with model: {settings.OPENAI_MODEL_ID}"
            )
            summarizer_tool = OpenAISummarizerTool(stream=False)

        model = LiteLLMModel(
            model_id=settings.OPENAI_MODEL_ID,
            api_base="https://api.openai.com/v1",
            api_key=settings.OPENAI_API_KEY,
        )

        agent = ToolCallingAgent(
            tools=[what_can_i_do, retriever_tool, summarizer_tool],
            model=model,
            max_steps=3,
            verbosity_level=2,
        )

        return cls(agent)
```

This method builds the agent by selecting the retriever configuration, which defines how the MongoDB retriever tool is created and configured.

> **It’s critical** that the retriever config matches the one used during the RAG feature pipeline used to populate the MongoDB vector index.

Next, we build the summarizer tool, which can either be the custom model trained in Lesson 4 and deployed on Hugging Face or an OpenAI model, depending on the settings.

After that, we initialize the LiteLLM model, which powers our AI agent.

Finally, all tools, along with the LLM model, are wrapped inside a `ToolCallingAgent` class with a maximum of three reasoning steps, ensuring structured decision-making and controlled execution flow.

Now that our agent is built, we can define its run function:

```
@opik.track(name="Agent.run")
    def run(self, task: str, **kwargs) -> Any:
        result = self.__agent.run(task, **kwargs)

        model = self.__agent.model
        metadata = {
            "system_prompt": self.__agent.system_prompt,
            "system_prompt_template": self.__agent.system_prompt_template,
            "tool_description_template": self.__agent.tool_description_template,
            "tools": self.__agent.tools,
            "model_id": self.__agent.model.model_id,
            "api_base": self.__agent.model.api_base,
            "input_token_count": model.last_input_token_count,
            "output_token_count": model.last_output_token_count,
        }
        if hasattr(self.__agent, "step_number"):
            metadata["step_number"] = self.__agent.step_number
        opik_context.update_current_trace(
            tags=["agent"],
            metadata=metadata,
        )

        return result
```

The `run` method tracks every execution of the agent using Opik’s `@track()` decorator. It logs key metadata, including the system prompt, tool descriptions, model details, and token counts within the current trace.

Having the skeleton of our agent in place, we can dig into each of the 3 tools that our model calls.

#### Building the MongoDB retriever tool

The first tool integrated is the `MongoDBRetrieverTool`, which allows the agent to find relevant documents using semantic search.

It matches a user query to the most relevant stored documents, helping the agent retrieve context when needed.

To integrate the tool with our agent, we must inherit from the `Tool ` class from SmolAgents. We also have to specify the name, description, inputs, and output type that the LLM uses to infer what the tool does and what its interface is. These are critical elements in integrating your tool with an LLM, as they are the only properties used to integrate the tool with the LLM:

```
class MongoDBRetrieverTool(Tool):
    name = "mongodb_vector_search_retriever"
    description = """Use this tool to search and retrieve relevant documents from a knowledge base using semantic search.
    This tool performs similarity-based search to find the most relevant documents matching the query.
    Best used when you need to:
    - Find specific information from stored documents
    - Get context about a topic
    - Research historical data or documentation
    The tool will return multiple relevant document snippets."""

    inputs = {
        "query": {
            "type": "string",
            "description": """The search query to find relevant documents for using semantic search.
            Should be a clear, specific question or statement about the information you're looking for.""",
        }
    }
    output_type = "string"

    def __init__(self, config_path: Path, **kwargs):
        super().__init__(**kwargs)

        self.config_path = config_path
        self.retriever = self.__load_retriever(config_path)

    def __load_retriever(self, config_path: Path):
        config = yaml.safe_load(config_path.read_text())
        config = config["parameters"]

        return get_retriever(
            embedding_model_id=config["embedding_model_id"],
            embedding_model_type=config["embedding_model_type"],
            retriever_type=config["retriever_type"],
            k=5,
            device=config["device"],
        )
```

The retriever tool is initialized with parameters from one of the retriever config files defined in Lesson 5. The settings include essential parameters such as the embedding model and retrieval type.

Now, we get to the core part of the tool, which is the `forward` method. This method is called when the AI agent uses the tool to search for information.

The `forward` method takes a query from the agent, searches for relevant documents, and returns the results in a format the agent can use.

The method is decorated with `@track`, which means its performance is being monitored with Opik. Before performing the actual search, the method first extracts important search parameters:

```
@track(name="MongoDBRetrieverTool.forward")
    def forward(self, query: str) -> str:
        if hasattr(self.retriever, "search_kwargs"):
            search_kwargs = self.retriever.search_kwargs
        else:
            try:
                search_kwargs = {
                    "fulltext_penalty": self.retriever.fulltext_penalty,
                    "vector_score_penalty": self.retriever.vector_penalty,
                    "top_k": self.retriever.top_k,
                }
            except AttributeError:
                logger.warning("Could not extract search kwargs from retriever.")

                search_kwargs = {}

        opik_context.update_current_trace(
            tags=["agent"],
            metadata={
                "search": search_kwargs,
                "embedding_model_id": self.retriever.vectorstore.embeddings.model,
            },
        )
```

First, we check what type of retriever is used and extract the relevant search parameters. Different retrievers might have different ways of configuring searches, so this code handles various cases.

The key parameters being extracted include:

- `fulltext_penalty`: Adjusts how much weight is given to exact text matches
- `vector_score_penalty`: Influences how semantic similarity affects the ranking
- `top_k`: Determines how many search results to return

These parameters significantly impact the search results. For example, a higher vector score penalty might prioritize results that match the semantic meaning of the query over those with exact keyword matches.

After setting up tracking, the method parses the query, performs the actual search, and formats the results:

```
 try:
            query = self.__parse_query(query)
            relevant_docs = self.retriever.invoke(query)

            formatted_docs = []
            for i, doc in enumerate(relevant_docs, 1):
                formatted_docs.append(
                    f"""
<document id="{i}">
<title>{doc.metadata.get("title")}</title>
<url>{doc.metadata.get("url")}</url>
<content>{doc.page_content.strip()}</content>
</document>
"""
                )

            result = "\n".join(formatted_docs)
            result = f"""
<search_results>
{result}
</search_results>
When using context from any document, also include the document URL as reference, which is found in the <url> tag.
"""
            return result
        except Exception:
            logger.opt(exception=True).debug("Error retrieving documents.")

            return "Error retrieving documents."
```

In this code snippet, we search for documents that match the query and format them in an XML-like structure. Each document includes a title, URL, and content. Additionally, the results are wrapped in tags to make them easy for the AI agent to read.

#### Creating the summarizer tool

In our agentic RAG module, we provide two summarization options: one using Hugging Face’s API and another using OpenAI’s models. Both tools inherit from `Tool` in SmolAgents and are tracked by Opik, ensuring that every summarization step is logged and monitored.

The first option for summarization is the Hugging Face endpoint-based summarizer.

This tool sends the text to an external Hugging Face model that generates a concise summary. The model deployed on Hugging Face is the one we trained in Lesson 4, which was explicitly fine-tuned for document summarization.

```
class HuggingFaceEndpointSummarizerTool(Tool):
    name = "huggingface_summarizer"
    description = """Use this tool to summarize a piece of text. Especially useful when you need to summarize a document."""

    inputs = {
        "text": {
            "type": "string",
            "description": """The text to summarize.""",
        }
    }
    output_type = "string"

    SYSTEM_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful assistant specialized in summarizing documents. Generate a concise TL;DR summary in markdown format having a maximum of 512 characters of the key findings from the provided documents, highlighting the most significant insights

### Input:
{content}

### Response:
"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert settings.HUGGINGFACE_ACCESS_TOKEN is not None, (
            "HUGGINGFACE_ACCESS_TOKEN is required to use the dedicated endpoint. Add it to the .env file."
        )
        assert settings.HUGGINGFACE_DEDICATED_ENDPOINT is not None, (
            "HUGGINGFACE_DEDICATED_ENDPOINT is required to use the dedicated endpoint. Add it to the .env file."
        )

        self.__client = OpenAI(
            base_url=settings.HUGGINGFACE_DEDICATED_ENDPOINT,
            api_key=settings.HUGGINGFACE_ACCESS_TOKEN,
        )
```

The code snippet above initializes the Hugging Face summarizer tool. It verifies that the necessary API credentials are available before setting up the client connection to Hugging Face’s inference endpoint.

To generate a summary, we implement the `forward` method, which is tracked by Opik for monitoring:

```
@track
    def forward(self, text: str) -> str:
        result = self.__client.chat.completions.create(
            model="tgi",
            messages=[\
                {\
                    "role": "user",\
                    "content": self.SYSTEM_PROMPT.format(content=text),\
                },\
            ],
        )

        return result.choices[0].message.content
```

This function sends the input text to the Hugging Face API, applying the predefined system prompt. The generated response is then returned, providing a structured summary.

The second summarization option uses OpenAI’s models to generate summaries. It follows a similar structure to the Hugging Face summarizer but connects to OpenAI’s API instead.

```
class OpenAISummarizerTool(Tool):
    name = "openai_summarizer"
    description = """Use this tool to summarize a piece of text. Especially useful when you need to summarize a document or a list of documents."""

    inputs = {
        "text": {
            "type": "string",
            "description": """The text to summarize.""",
        }
    }
    output_type = "string"

    SYSTEM_PROMPT = """You are a helpful assistant specialized in summarizing documents.
Your task is to create a clear, concise TL;DR summary in plain text.
Things to keep in mind while summarizing:
- titles of sections and sub-sections
- tags such as Generative AI, LLMs, etc.
- entities such as persons, organizations, processes, people, etc.
- the style such as the type, sentiment and writing style of the document
- the main findings and insights while preserving key information and main ideas
- ignore any irrelevant information such as cookie policies, privacy policies, HTTP errors,etc.

Document content:
{content}

Generate a concise summary of the key findings from the provided documents, highlighting the most significant insights and implications.
Return the document in plain text format regardless of the original format.
"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.__client = OpenAI(
            base_url="https://api.openai.com/v1",
            api_key=settings.OPENAI_API_KEY,
        )
```

This summarizer connects to OpenAI’s API and uses a structured prompt to generate high-quality summaries.

Note that because the Hugging Face model was fine-tuned on summarizing documents, it doesn't require careful prompt engineering for the desired results (it has the logic embedded into it), resulting in fewer tokens/requests, which translates to lower costs and better latencies.

#### The "What Can I Do" tool

The third and last integrated tool is the "What Can I Do" tool, which provides a list of available capabilities within the Second Brain assistant and helps users explore relevant topics.

```
@opik.track(name="what_can_i_do")
@tool
def what_can_i_do(question: str) -> str:
    """Returns a comprehensive list of available capabilities and topics in the Second Brain system.

    This tool should be used when:
    - The user explicitly asks what the system can do
    - The user asks about available features or capabilities
    - The user seems unsure about what questions they can ask
    - The user wants to explore the system's knowledge areas

    This tool should NOT be used when:
    - The user asks a specific technical question
    - The user already knows what they want to learn about
    - The question is about a specific topic covered in the knowledge base

    Args:
        question: The user's query about system capabilities. While this parameter is required,
                 the function returns a standard capability list regardless of the specific question.

    Returns:
        str: A formatted string containing categorized lists of example questions and topics
             that users can explore within the Second Brain system.

    Examples:
        >>> what_can_i_do("What can this system do?")
        >>> what_can_i_do("What kind of questions can I ask?")
        >>> what_can_i_do("Help me understand what I can learn here")
    """

    return """
You can ask questions about the content in your Second Brain, such as:

Architecture and Systems:
- What is the feature/training/inference (FTI) architecture?
- How do agentic systems work?
- Detail how does agent memory work in agentic applications?

LLM Technology:
- What are LLMs?
- What is BERT (Bidirectional Encoder Representations from Transformers)?
- Detail how does RLHF (Reinforcement Learning from Human Feedback) work?
- What are the top LLM frameworks for building applications?
- Write me a paragraph on how can I optimize LLMs during inference?

RAG and Document Processing:
- What tools are available for processing PDFs for LLMs and RAG?
- What's the difference between vector databases and vector indices?
- How does document chunk overlap affect RAG performance?
- What is chunk reranking and why is it important?
- What are advanced RAG techniques for optimization?
- How can RAG pipelines be evaluated?

Learning Resources:
- Can you recommend courses on LLMs and RAG?
"""
```

This tool is useful when users are unsure about what they can ask or want to explore different capabilities within the system. Like other tools, it is tracked by Opik for monitoring and observability.

To see our agentic RAG module in action, check out the video below, where we query our agent using the Gradio UI, visualizing how the agent reasons and calls the tools to construct the answer to our question:

Having the agentic module tested, we can check out the results of the tracking done by Opik in Figure 10:

[https://substackcdn.com/image/fetch/$s_!77pD!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F224265c3-6fd2-4b64-ac1e-39630ab9df4a_2522x1726.png](https://substackcdn.com/image/fetch/$s_!77pD!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F224265c3-6fd2-4b64-ac1e-39630ab9df4a_2522x1726.png) Figure 10: Example of a prompt trace in [Opik](https://github.com/comet-ml/opik)

Here, we can see that the agent calls the MongoDB retriever tool, which in turn invokes the `forward` function. Each step is logged with latency values, providing insight into execution times at different stages.

Furthermore, all metadata related to the trace—including the system prompt, tool configurations, and token usage—is captured to ensure complete observability.

Now that we have implemented the agentic RAG module, we need a structured way to evaluate its performance. This is where the **LLM evaluation pipeline** comes in, ensuring that our agentic RAG module consistently meets quality and reliability standards.

The evaluation pipeline is built using Opik, which helps us log, analyze, and score the agent’s responses. We will focus strictly on Opik's evaluation logic and how it tracks our agent’s outputs.

Before evaluating our agent, we first need to gather a suitable evaluation dataset. This dataset will help us consistently test performance and track improvements

#### Creating the evaluation dataset

To evaluate the agent properly, we use a dataset of ~30 predefined prompts that cover various scenarios the agent might encounter. This dataset allows us to consistently test our agent’s performance across different iterations, ensuring that changes do not degrade its capabilities.

```
EVALUATION_PROMPTS: List[str] = [\
    """\
Write me a paragraph on the feature/training/inference (FTI) pipelines architecture following the next structure:\
\
- introduction\
- what are its main components\
- why it's powerful\
\
Retrieve the sources when compiling the answer. Also, return the sources you used as context.\
""",\
    "What is the feature/training/inference (FTI) pipelines architecture?",\
    "What is the Tensorflow Recommenders Python package?",\
    """How does RLHF: Reinforcement Learning from Human Feedback work?\
\
Explain to me:\
- what is RLHF\
- how it works\
- why it's important\
- what are the main components\
- what are the main challenges\
""",\
    "List 3 LLM frameworks for building LLM applications and why they are important.",\
    "Explain how does Bidirectional Encoder Representations from Transformers (BERT) work. Focus on what architecture it uses, how it's different from other models and how they are trained.",\
    "List 5 ways or tools to process PDFs for LLMs and RAG",\
    """How can I optimize my LLMs during inference?\
\
Provide a list of top 3 best practices, while providing a short explanation for each, which contains why it's important.\
""",\
    "Explain to me in more detail how does an Agent memory work and why do we need it when building Agentic apps.",\
    "What is the difference between a vector database and a vector index?",\
    "Recommend me a course on LLMs and RAG",\
    "How Document Chunk overlap affects a RAG pipeline and it's performance?",\
    """What is the importance of reranking chunks for RAG?\
Explain to me:\
- what is reranking\
- how it works\
- why it's important\
- what are the main components\
- what are the main trade-offs\
""",\
    "List the most popular advanced RAG techniques to optimize RAG performance and why they are important.",\
    "List what are the main ways of evaluating a RAG pipeline and why they are important.",\
]
```

We could have added more samples, but for the first iteration, having 30 samples is a sweet spot. The core idea is to expand this split with edge case samples you find while developing the application.

We use Opik to store and manage the dataset, as shown in the following code:

```
def get_or_create_dataset(name: str, prompts: list[str]) -> opik.Dataset | None:
    client = opik.Opik()
    try:
        dataset = client.get_dataset(name=name)
    except Exception:
        dataset = None

    if dataset:
        logger.warning(f"Dataset '{name}' already exists. Skipping dataset creation.")

        return dataset

    assert prompts, "Prompts are required to create a dataset."

    dataset_items = []
    for prompt in prompts:
        dataset_items.append(
            {
                "input": prompt,
            }
        )

    dataset = create_dataset(
        name=name,
        description="Dataset for evaluating the agentic app.",
        items=dataset_items,
    )

    return dataset
```

This function ensures the dataset is created if it doesn’t exist, avoiding unnecessary duplication. It logs whether the dataset is new or previously stored and ensures that each prompt is properly formatted before evaluation.

#### Evaluating the agent

The core of the evaluation pipeline is the `evaluate_agent()` function. This function runs the set of predefined prompts through our agent and scores its responses using a combination of built-in and custom metrics.

```
def evaluate_agent(prompts: list[str], retriever_config_path: Path) -> None:
    assert settings.COMET_API_KEY, (
        "COMET_API_KEY is not set. We need it to track the experiment with Opik."
    )

    logger.info("Starting evaluation...")
    logger.info(f"Evaluating agent with {len(prompts)} prompts.")

    def evaluation_task(x: dict) -> dict:
        """Call agentic app logic to evaluate."""
        agent = agents.get_agent(retriever_config_path=retriever_config_path)
        response = agent.run(x["input"])
        context = extract_tool_responses(agent)

        return {
            "input": x["input"],
            "context": context,
            "output": response,
        }
```

In this code section, we first ensure that Opik can log the experiment by asserting that the necessary API keys are set.

Then, we define the `evaluation_task()`, a method that retrieves an instance of our agent, runs an input prompt through it, and captures both the output and retrieval context.

Before running the actual evaluation, we either fetch an existing dataset or create a new one to store our evaluation prompts:

```
# Get or create dataset
    dataset_name = "second_brain_rag_agentic_app_evaluation_dataset"
    dataset = opik_utils.get_or_create_dataset(name=dataset_name, prompts=prompts)
```

Here, `opik_utils.get_or_create_dataset()` is used to manage the datasets dynamically, as detailed earlier in this section.

Once the dataset is set up, we retrieve our agent instance and configure the experiment. The `experiment_config` dictionary defines key parameters for tracking and logging the evaluation:

```
# Evaluate
    agent = agents.get_agent(retriever_config_path=retriever_config_path)
    experiment_config = {
        "model_id": settings.OPENAI_MODEL_ID,
        "retriever_config_path": retriever_config_path,
        "agent_config": {
            "max_steps": agent.max_steps,
            "agent_name": agent.agent_name,
        },
    }
```

Next, we define the scoring metrics used to evaluate the agent's performance. Opik provides built-in evaluation metrics, but we also include custom ones for deeper analysis.

```
scoring_metrics = [\
        Hallucination(),\
        AnswerRelevance(),\
        Moderation(),\
        SummaryDensityHeuristic(),\
        SummaryDensityJudge(),\
    ]
```

The scoring process evaluates the agent’s performance across multiple dimensions:

- **Hallucination**: Measures whether the agent generates false or misleading information.
- **Answer Relevance**: Scores the relevance of the agent's response to the given prompt.
- **Moderation**: Detects potentially inappropriate or unsafe content in responses.

In addition to these built-in Opik metrics, we include two custom components. Both compute the response density (whether the answer is too long or too short) but with different techniques: heuristics or LLM-as-Judges. This is a good example of understanding the difference between the two.

- **SummaryDensityHeuristic**: Evaluates whether a response is too short, too long, or appropriately balanced.
- **SummaryDensityJudge**: Uses an external LLM to judge response density and conciseness.

Finally, we execute the evaluation process using the metrics defined and our evaluation dataset:

```
if dataset:
        evaluate(
            dataset=dataset,
            task=evaluation_task,
            scoring_metrics=scoring_metrics,
            experiment_config=experiment_config,
            task_threads=2,
        )
    else:
        logger.error("Can't run the evaluation as the dataset items are empty.")
```

This code ensures that evaluation runs only when a dataset is available. The `evaluate()` function runs the agent using the `evaluation_task()` method on the evaluation dataset and measures the defined scoring metrics. The [results are then logged in Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons) for further analysis and comparison.

#### The summary density heuristic

In our evaluation pipeline, we include a custom metric called **summary density heuristic**.

This metric assesses whether an LLM-generated response is appropriately concise and informative. It extends `BaseMetric` from Opik, allowing us to integrate it seamlessly into our evaluation framework.

The purpose of this heuristic is to ensure that responses are neither too short nor excessively long. A well-balanced response provides sufficient detail without unnecessary verbosity.

```
class SummaryDensityHeuristic(base_metric.BaseMetric):
    """
    A metric that evaluates whether an LLM's output has appropriate length and density.

    This metric uses an heuristic to determine if the output length is appropriate for the given instruction.
    It returns a normalized score between 0 and 1, where:
    - 0.0 (Poor): Output is either too short and incomplete, or too long with unnecessary information
    - 0.5 (Good): Output has decent length balance but still slightly too short or too long
    - 1.0 (Excellent): Output length is appropriate, answering the question concisely without being verbose
    """

    def __init__(
        self,
        name: str = "summary_density_heuristic",
        min_length: int = 128,
        max_length: int = 1024,
    ) -> None:
        self.name = name
        self.min_length = min_length
        self.max_length = max_length
```

This snippet initializes the metric with a name, minimum length, and maximum length. The `min_length` and `max_length` parameters define the acceptable range for a response's length.

To evaluate the response length, we define the `score()` function, which compares the output against the predefined length limits:

```
 def score(
        self, input: str, output: str, **ignored_kwargs: Any
    ) -> score_result.ScoreResult:
        """
        Score the output of an LLM.

        Args:
            input: The input prompt given to the LLM.
            output: The output of an LLM to score.
            **ignored_kwargs: Any additional keyword arguments.

        Returns:
            ScoreResult: The computed score with explanation.
        """

        length_score = self._compute_length_score(output)

        reason = f"Output length: {len(output)} chars. "
        if length_score == 1.0:
            reason += "Length is within ideal range."
        elif length_score >= 0.5:
            reason += "Length is slightly outside ideal range."
        else:
            reason += "Length is significantly outside ideal range."

        return score_result.ScoreResult(
            name=self.name,
            value=length_score,
            reason=reason,
        )
```

The `score()` function determines how well the LLM's response fits within the acceptable length range. It assigns a normalized score between 0 and 1 based on whether the output is too short, too long, or appropriately balanced.

The core logic of this metric lies in `_compute_length_score()`, which calculates a numerical score based on response length:

```
    def _compute_length_score(self, text: str) -> float:
        """
        Compute a score based on text length relative to min and max boundaries.

        Args:
            text: The text to evaluate.

        Returns:
            float: A score between 0 and 1, where:
                - 0.0: Text length is significantly outside the boundaries
                - 0.5: Text length is slightly outside the boundaries
                - 1.0: Text length is within the ideal range
        """
        length = len(text)

        # If length is within bounds, return perfect score
        if self.min_length <= length <= self.max_length:
            return 1.0

        if length < self.min_length:
            deviation = (self.min_length - length) / self.min_length
        else:
            deviation = (length - self.max_length) / self.max_length

        # Convert deviation to a score between 0 and 1
        # deviation <= 0.5 -> score between 0.5 and 1.0
        # deviation > 0.5 -> score between 0.0 and 0.5
        score = max(0.0, 1.0 - deviation)

        return score
```

This function ensures that responses falling within the predefined length range receive a perfect score of 1.0. If a response deviates too far from the range, its score is gradually reduced to reflect the severity of the deviation.

#### The summary density judge

The summary density judge is a custom evaluation component that builds upon the summary density metric by using an external LLM to assess response length.

Instead of relying on a manually calculated heuristic, this judge uses an AI model to determine if the length of an output is appropriate for a given input.

This approach allows us to incorporate more nuanced and context-aware judgments into our evaluation pipeline. Like the heuristic, it integrates seamlessly with Opik’s evaluation framework:

```
class LLMJudgeStyleOutputResult(BaseModel):
    score: int
    reason: str

class SummaryDensityJudge(base_metric.BaseMetric):
    """
    A metric that evaluates whether an LLM's output has appropriate length and density.

    This metric uses another LLM to judge if the output length is appropriate for the given instruction.
    It returns a normalized score between 0 and 1, where:
    - 0.0 (Poor): Output is either too short and incomplete, or too long with unnecessary information
    - 0.5 (Good): Output has decent length balance but still slightly too short or too long
    - 1.0 (Excellent): Output length is appropriate, answering the question concisely without being verbose
    """

    def __init__(
        self,
        name: str = "summary_density_judge",
        model_name: str = settings.OPENAI_MODEL_ID,
    ) -> None:
        self.name = name
        self.llm_client = LiteLLMChatModel(model_name=model_name)
        self.prompt_template = """
        You are an impartial expert judge. Evaluate the quality of a given answer to an instruction based on how long the answer it is.

How to decide wether the lengths of the answer is appropriate:
1 (Poor): Too short, does not answer the question OR too long, it contains too much noise and unrequired information, where the answer could be more concise.
2 (Good): Good lengthbalance of the answer, but the answer is still too short OR too long.
3 (Excellent): The length of the answer is appropriate, it answers the question and is not too long or too short.

Example of bad answer that is too short:
<answer>
LangChain, LlamaIndex, Haystack
</answer>

Example of bad answer that is too long:
<answer>
LangChain is a powerful and versatile framework designed specifically for building sophisticated LLM applications. It provides comprehensive abstractions for essential components like prompting, memory management, agent behaviors, and chain orchestration. The framework boasts an impressive ecosystem with extensive integrations across various tools and services, making it highly flexible for diverse use cases. However, this extensive functionality comes with a steeper learning curve that might require dedicated time to master.

LlamaIndex (which was formerly known as GPTIndex) has carved out a specialized niche in the LLM tooling landscape, focusing primarily on data ingestion and advanced indexing capabilities for Large Language Models. It offers a rich set of sophisticated mechanisms to structure and query your data, including vector stores for semantic similarity search, keyword indices for traditional text matching, and tree indices for hierarchical data organization. While it particularly shines in Retrieval-Augmented Generation (RAG) applications, its comprehensive feature set might be excessive for more straightforward implementation needs.

Haystack stands out as a robust end-to-end framework that places particular emphasis on question-answering systems and semantic search capabilities. It provides a comprehensive suite of document processing tools and comes equipped with production-ready pipelines that can be deployed with minimal configuration. The framework includes advanced features like multi-stage retrieval, document ranking, and reader-ranker architectures. While these capabilities make it powerful for complex information retrieval tasks, new users might find the initial configuration and architecture decisions somewhat challenging to navigate.

Each of these frameworks brings unique strengths to the table while sharing some overlapping functionality. The choice between them often depends on specific use cases, technical requirements, and team expertise. LangChain offers the broadest general-purpose toolkit, LlamaIndex excels in data handling and RAG, while Haystack provides the most streamlined experience for question-answering systems.
</answer>

Example of excellent answer that is appropriate:
<answer>
1. LangChain is a powerful framework for building LLM applications that provides abstractions for prompting, memory, agents, and chains. It has extensive integrations with various tools and services, making it highly flexible but potentially complex to learn.
2. LlamaIndex specializes in data ingestion and indexing for LLMs, offering sophisticated ways to structure and query your data through vector stores, keyword indices, and tree indices. It excels at RAG applications but may be overkill for simpler use cases.
3. Haystack is an end-to-end framework focused on question-answering and semantic search, with strong document processing capabilities and ready-to-use pipelines. While powerful, its learning curve can be steep for beginners.
</answer>

Instruction: {input}

Answer: {output}

Provide your evaluation in JSON format with the following structure:
{{
    "accuracy": {{
        "reason": "...",
        "score": 0
    }},
    "style": {{
        "reason": "...",
        "score": 0
    }}
}}
"""
```

In this snippet, we initialize the summary density judge, specifying the model it will use to evaluate responses. The `prompt_template` provides clear instructions for the external LLM, defining the criteria for scoring an answer.

The judge’s scoring function uses the external LLM to analyze a response and assign a score based on how well its length aligns with the expected range:

```
    def score(self, input: str, output: str, **ignored_kwargs: Any):
        prompt = self.prompt_template.format(input=input, output=output)

        model_output = self.llm_client.generate_string(
            input=prompt, response_format=LLMJudgeStyleOutputResult
        )

        return self._parse_model_output(model_output)
```

The `score()` function formats the prompt and sends it to the external LLM. The model then evaluates the response and provides a structured output with a score and explanation.

Once the external model returns a score, we process it to ensure consistency and normalize the values.

```
    def _parse_model_output(self, content: str) -> score_result.ScoreResult:
        try:
            dict_content = json.loads(content)
        except Exception:
            raise exceptions.MetricComputationError("Failed to parse the model output.")

        score = dict_content["score"]
        try:
            assert 1 <= score <= 3, f"Invalid score value: {score}"
        except AssertionError as e:
            raise exceptions.MetricComputationError(str(e))

        score = (score - 1) / 2.0  # Normalize the score to be between 0 and 1

        return score_result.ScoreResult(
            name=self.name,
            value=score,
            reason=dict_content["reason"],
        )
```

The `_parse_model_output()` function ensures that the returned score is valid and within the expected range. The score is then normalized between 0 and 1 for consistency with other evaluation metrics.

#### Evaluation results

We [track evaluation results in Opik](https://www.comet.com/opik?utm_source=paul_2nd_brain_course&utm_campaign=opik&utm_medium=lessons), allowing us to compare different agent versions and detect performance regressions.

Figure 11 shows a sample evaluation run, displaying the scores across all metrics:

[https://substackcdn.com/image/fetch/$s_!KzPS!,w_1456,c_limit,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F918a538b-8ac6-4fb7-864b-557a70ab1fe0_2908x1096.png](https://substackcdn.com/image/fetch/$s_!KzPS!,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F918a538b-8ac6-4fb7-864b-557a70ab1fe0_2908x1096.png) Figure 11: Outcome of an evaluation run in [Opik](https://github.com/comet-ml/opik)

By implementing this evaluation pipeline, we ensure that our agentic RAG module continues to improve while maintaining accuracy, relevance, and overall quality.

</details>

<details>
<summary>building-effective-ai-agents-anthropic</summary>

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

For the remainder of this post, we'll assume each LLM call has access to these augmented capabilities.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

### Workflow: Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.

**When to use this workflow:** This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

**Examples where evaluator-optimizer is useful:**

- Literary translation where there are nuances that the translator LLM might not capture initially, but where an evaluator LLM can provide useful critiques.
- Complex search tasks that require multiple rounds of searching and analysis to gather comprehensive information, where the evaluator decides whether further searches are warranted.

### Agents

Agents are emerging in production as LLMs mature in key capabilities—understanding complex inputs, engaging in reasoning and planning, using tools reliably, and recovering from errors. Agents begin their work with either a command from, or interactive discussion with, the human user. Once the task is clear, agents plan and operate independently, potentially returning to the human for further information or judgement. During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress. Agents can then pause for human feedback at checkpoints or when encountering blockers. The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully.

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

</details>

<details>
<summary>google-announces-gemini-cli-your-open-source-ai-agent</summary>

For developers, the command line interface (CLI) isn't just a tool; it's home. The terminal’s efficiency, ubiquity and portability make it the go-to utility for getting work done. And as developers' reliance on the terminal endures, so does the demand for integrated AI assistance.

That’s why we’re introducing [Gemini CLI](http://github.com/google-gemini/gemini-cli), an open-source AI agent that brings the power of Gemini directly into your terminal. It provides lightweight access to Gemini, giving you the most direct path from your prompt to our model. While it excels at coding, we built Gemini CLI to do so much more. It’s a versatile, local utility you can use for a wide range of tasks, from content generation and problem solving to deep research and task management.

We’ve also integrated Gemini CLI with Google’s AI coding assistant, [Gemini Code Assist](https://codeassist.google/), so that all developers — on free, Standard, and Enterprise Code Assist plans — get prompt-driven, AI-first coding in both VS Code and Gemini CLI.

## Unmatched usage limits for individual developers

To use Gemini CLI free-of-charge, simply login with a personal Google account to get a free Gemini Code Assist license. That free license gets you access to Gemini 2.5 Pro and its massive 1 million token context window. To ensure you rarely, if ever, hit a limit during this preview, we offer the industry’s largest allowance: 60 model requests per minute and 1,000 requests per day at no charge.

If you’re a professional developer who needs to run multiple agents simultaneously, or if you prefer to use specific models, you can use a [Google AI Studio](https://aistudio.google.com/apikey) or [Vertex AI](https://console.cloud.google.com/vertex-ai/studio/multimodal) key for usage-based billing or get a Gemini Code Assist Standard or Enterprise license.

Gemini CLI offers the industry’s largest usage allowance at 60 model requests per minute and 1,000 model requests per day at no charge

## Powerful models in your command line

Now in preview, Gemini CLI provides powerful AI capabilities, from code understanding and file manipulation to command execution and dynamic troubleshooting. It offers a fundamental upgrade to your command line experience, enabling you to write code, debug issues and streamline your workflow with natural language.

Its power comes from built-in tools allowing you to:

- **Ground prompts with Google Search** so you can fetch web pages and provide real-time, external context to the model
- **Extend Gemini CLI’s capabilities** through built-in support for the Model Context Protocol (MCP) or bundled extensions
- **Customize prompts and instructions** to tailor Gemini for your specific needs and workflows
- **Automate tasks and integrate with existing workflows** by invoking Gemini CLI non-interactively within your scripts

Gemini CLI can be used for a wide variety of tasks, including making a short video showing the story of a ginger cat’s adventures around Australia with Veo and Imagen

## Open and extensible

Because Gemini CLI is fully [open source (Apache 2.0)](https://github.com/google-gemini/gemini-cli/blob/main/LICENSE), developers can inspect the code to understand how it works and verify its security implications. We fully expect (and welcome!) a global community of developers to [contribute to this project](https://github.com/google-gemini/gemini-cli/blob/main/CONTRIBUTING.md) by reporting bugs, suggesting features, continuously improving security practices and submitting code improvements. [Post your issues](http://github.com/google-gemini/gemini-cli/issues) or [submit your ideas](http://github.com/google-gemini/gemini-cli/discussions) in our GitHub repo.

We also built Gemini CLI to be extensible, building on emerging standards like MCP, system prompts (via GEMINI.md) and settings for both personal and team configuration. We know the terminal is a personal space, and everyone deserves the autonomy to make theirs unique.

## Shared technology with Gemini Code Assist

Sometimes, an IDE is the right tool for the job. When that time comes, you want all the capabilities of a powerful AI agent by your side to iterate, learn and overcome issues quickly.

[Gemini Code Assist](https://codeassist.google/), Google’s AI coding assistant for students, hobbyists and professional developers, now shares the same technology with Gemini CLI. In VS Code, you can place any prompt into the chat window using agent mode, and Code Assist will relentlessly work on your behalf to write tests, fix errors, build out features or even migrate your code. Based on your prompt, Code Assist’s agent will build a multi-step plan, auto-recover from failed implementation paths and recommend solutions you may not have even imagined.

Gemini Code Assist’s chat agent is a multi-step, collaborative, reasoning agent that expands the capabilities of simple-command response interactions

Gemini Code Assist agent mode is available at no additional cost for all plans (free, Standard and Enterprise) through the [Insiders channel](https://developers.google.com/gemini-code-assist/docs/use-agentic-chat-pair-programmer#before-you-begin). If you aren’t already using Gemini Code Assist, give it a try. Its free tier has the highest usage limit in the market today, and only takes less than a minute to [get started](https://codeassist.google/).

</details>