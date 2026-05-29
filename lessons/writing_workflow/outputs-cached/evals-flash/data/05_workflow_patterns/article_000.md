# Lesson 5: Basic Workflow Ingredients
### Building Modular, Parallel, and Dynamic LLM Workflows

I remember a time, not so long ago, when building complex AI applications felt like trying to assemble intricate LEGO sets with a single, massive block. You'd craft one enormous prompt, hoping an LLM would magically understand every instruction, generate every detail, and handle every edge case. It was a chaotic, often frustrating process that rarely delivered reliable results in production.

This "single-prompt-does-all" approach was a trap. It led to inconsistent outputs, made debugging a nightmare, and often ballooned costs as we crammed more and more context into each call. We realized that for AI engineering to move beyond prototypes, we needed a more structured way to build. We needed to break down problems, just like we do in traditional software engineering.

This is where the fundamental workflow patterns come into play. By chaining LLM calls, running tasks in parallel, and implementing dynamic routing and orchestrator-worker patterns, we can build robust, efficient, and truly intelligent systems. These techniques are the bedrock for constructing sophisticated LLM applications and more advanced AI agents.

In this lesson, you will learn:

*   The inherent challenges of relying on complex, single LLM calls for multi-step tasks.
*   The power of modularity through prompt chaining and how it enhances accuracy and debuggability.
*   How to build a sequential workflow for generating FAQs, breaking it into manageable steps.
*   Strategies for optimizing sequential workflows using parallel processing to reduce latency.
*   The principles of dynamic behavior through routing and conditional logic in LLM applications.
*   How to implement a basic routing workflow for customer service intent classification.
*   The orchestrator-worker pattern, enabling dynamic task decomposition and specialized processing.

## The Challenge with Complex Single LLM Calls

Trying to solve a complex, multi-step problem with a single, monolithic LLM call often leads to more headaches than solutions. Imagine asking an LLM to generate an entire FAQ list, including questions, answers, and source citations, all in one go. The model struggles to maintain focus across all these sub-tasks, leading to several common failure modes [[1]](https://arxiv.org/pdf/2309.08181), [[2]](https://www.getambassador.io/blog/prompt-engineering-for-llms), [[3]](https://proceedings.neurips.cc/paper_files/paper/2023/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf), [[4]](https://www.york.ac.uk/assuring-autonomy/news/blog/part-one-using-large-language-models/), [[5]](https://arxiv.org/html/2410.23884v1).

First, pinpointing errors or specific failures becomes nearly impossible. If the output is wrong, where in that giant prompt did the model go astray? There is no modularity, making it hard to update or improve specific parts of the task without affecting everything else. Furthermore, complex prompts with long contexts increase the likelihood of "lost in the middle" issues, where the LLM overlooks important instructions or information buried within the prompt [[2]](https://www.getambassador.io/blog/prompt-engineering-for-llms), [[5]](https://arxiv.org/html/2410.23884v1). This also leads to higher token consumption, as the prompt tries to do too much, ultimately resulting in less reliable outputs for complex multi-step tasks.

Let's look at an example. We set up our environment by initializing the Gemini client and defining the model ID. These are standard setup instructions we use across our lessons.

1.  We begin by importing the necessary `genai` library and loading environment variables.
    ```python
    import os
    import json
    from google import genai
    from google.genai import types # Used for structured outputs
    
    # Assuming env.load is defined elsewhere to load GOOGLE_API_KEY
    # For demonstration, we'll simulate it here.
    # os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
    
    client = genai.Client()
    MODEL_ID = "gemini-2.5-flash"
    ```

2.  Next, we define three mock webpages containing content about renewable energy.
    ```python
    webpage_1 = {
        "title": "The Benefits of Solar Energy",
        "content": """
        Solar energy is a renewable powerhouse, offering numerous environmental and economic benefits.
        By converting sunlight into electricity through photovoltaic (PV) panels, it reduces reliance on fossil fuels,
        thereby cutting down greenhouse gas emissions. Homeowners who install solar panels can significantly
        lower their monthly electricity bills, and in some cases, sell excess power back to the grid.
        While the initial installation cost can be high, government incentives and long-term savings make
        it a financially viable option for many. Solar power is also a key component in achieving energy
        independence for nations worldwide.
        """,
    }
    
    webpage_2 = {
        "title": "Understanding Wind Turbines",
        "content": """
        Wind turbines are towering structures that capture kinetic energy from the wind and convert it into
        electrical power. They are a critical part of the global shift towards sustainable energy.
        Turbines can be installed both onshore and offshore, with offshore wind farms generally producing more
        consistent power due to stronger, more reliable winds. The main challenge for wind energy is its
        intermittency—it only generates power when the wind blows. This necessitates the use of energy
        storage solutions, like large-scale batteries, to ensure a steady supply of electricity.
        """,
    }
    
    webpage_3 = {
        "title": "Energy Storage Solutions",
        "content": """
        Effective energy storage is the key to unlocking the full potential of renewable sources like solar
        and wind. Because these sources are intermittent, storing excess energy when it's plentiful and
        releasing it when it's needed is crucial for a stable power grid. The most common form of
        large-scale storage is pumped-hydro storage, but battery technologies, particularly lithium-ion,
        are rapidly becoming more affordable and widespread. These batteries can be used in homes, businesses,
        and at the utility scale to balance energy supply and demand, making our energy system more
        resilient and reliable.
        """,
    }
    
    all_sources = [webpage_1, webpage_2, webpage_3]
    combined_content = "\n\n".join(
        [f"Source Title: {source['title']}\nContent: {source['content']}" for source in all_sources]
    )
    ```

3.  We then construct a complex prompt that attempts to generate FAQs, answers, and source citations all at once.
    ```python
    class FAQ(types.BaseModel):
        question: str = types.Field(description="The question to be answered")
        answer: str = types.Field(description="The answer to the question")
        sources: list[str] = types.Field(description="The sources used to answer the question")
    
    class FAQList(types.BaseModel):
        faqs: list[FAQ] = types.Field(description="A list of FAQs")
    
    n_questions = 4
    prompt_complex = f"""
    Based on the provided content from three webpages, generate a list of exactly {n_questions} frequently asked questions (FAQs).
    For each question, provide a concise answer derived ONLY from the text.
    After each answer, you MUST include a list of the 'Source Title's that were used to formulate that answer.
    
    <provided_content>
    {combined_content}
    </provided_content>
    """.strip()
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=FAQList
    )
    response_complex = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt_complex,
        config=config
    )
    result_complex = response_complex.parsed
    ```

4.  Here is a partial output of the LLM's response.
    ```json
    {
      "question": "What are the primary environmental and economic benefits of solar energy?",
      "answer": "Solar energy reduces reliance on fossil fuels, thereby cutting down greenhouse gas emissions. Economically, homeowners can lower electricity bills and sell excess power to the grid.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    }
    ```

While this output might seem acceptable, the more complex the instructions become, the more inaccuracies we introduce. For example, sometimes answers are sourced from multiple documents, but the model only lists one. Humans naturally break down complex tasks into smaller, more manageable sub-tasks. We should apply the same thinking to LLM workflows.

## The Power of Modularity: Why Chain LLM Calls?

To overcome the limitations of complex single LLM calls, we introduce prompt chaining. This concept involves connecting multiple LLM calls, or other processing steps, sequentially. The output of one step becomes the input for the next, creating a structured flow for complex tasks [[25]](https://www.vellum.ai/blog/what-is-prompt-chaining), [[26]](https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84), [[27]](https://www.voiceflow.com/blog/prompt-chaining), [[28]](https://blog.promptlayer.com/what-is-prompt-chaining/). It is a divide-and-conquer strategy that makes complex problems more manageable.

Prompt chaining offers several benefits:
*   **Improved modularity**: Each LLM call focuses on a specific, well-defined sub-task. This makes the system easier to control and test at the step level [[25]](https://www.vellum.ai/blog/what-is-prompt-chaining), [[26]](https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84).
*   **Enhanced accuracy**: Simpler, targeted prompts for each step generally lead to better, more reliable outputs. The LLM can dedicate its full attention to one specific goal [[26]](https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84).
*   **Easier debugging**: If something goes wrong, you can isolate issues to specific links in the chain, reducing debugging time and improving maintenance [[25]](https://www.vellum.ai/blog/what-is-prompt-chaining), [[26]](https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84).
*   **Increased flexibility**: Individual components can be swapped, updated, or optimized independently. This supports structured workflows and clearer boundaries between tasks [[26]](https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84).
*   **Potential for optimization**: You can use different models for different steps. For example, use a cheaper or faster model for a simple classification step and reserve a more powerful model for complex generation or reasoning tasks, reducing overall cost without sacrificing quality [[25]](https://www.vellum.ai/blog/what-is-prompt-chaining).

However, prompt chaining also has downsides [[6]](https://aisdr.com/blog/what-is-prompt-chaining), [[7]](https://www.humanfirst.ai/blog/prompt-chaining), [[8]](https://blog.promptlayer.com/what-is-prompt-chaining), [[9]](https://ai.plainenglish.io/prompt-chaining-is-dead-long-live-prompt-stuffing-58a1c08820c5):
*   **Loss of meaning**: Some instructions make sense only when grouped. Splitting them into multiple prompts or steps can cause the LLM to lose the broader context or intent, leading to fragmented or less coherent results [[7]](https://www.humanfirst.ai/blog/prompt-chaining).
*   **Higher costs**: More LLM calls generally mean more tokens processed, which translates to increased API costs [[8]](https://blog.promptlayer.com/what-is-prompt-chaining).
*   **Increased latency**: Each LLM call adds to the total time to completion. A long chain can significantly slow down the overall workflow, impacting real-time applications [[8]](https://blog.promptlayer.com/what-is-prompt-chaining).
*   **Information loss**: Information can be lost or misinterpreted as it passes through multiple steps in a prompt chain. For example, a summary from the first prompt might lose critical details when translated by a second prompt [[7]](https://www.humanfirst.ai/blog/prompt-chaining).

Despite these drawbacks, the benefits of modularity, accuracy, and debuggability often outweigh the costs, making prompt chaining a fundamental pattern in LLM workflow design.

## Building a Sequential Workflow: FAQ Generation Pipeline

Let's refactor our previous complex FAQ generation example into a sequential workflow. We will break it down into three distinct steps: Generate Questions, Answer Questions, and Find Sources. This sequential approach produces more consistent and traceable results.

Image 1: A flowchart illustrating a sequential FAQ generation pipeline.

1.  We define a `QuestionList` Pydantic model and a prompt to generate questions.
    ```python
    class QuestionList(types.BaseModel):
        questions: list[str] = types.Field(description="A list of questions")
    
    prompt_generate_questions = """
    Based on the content below, generate a list of {n_questions} relevant and distinct questions that a user might have.
    
    <provided_content>
    {combined_content}
    </provided_content>
    """.strip()
    
    def generate_questions(content: str, n_questions: int = 10) -> list[str]:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QuestionList
        )
        response_questions = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_generate_questions.format(n_questions=n_questions, combined_content=content),
            config=config
        )
        return response_questions.parsed.questions
    ```
    This function focuses solely on creating relevant questions based on the provided material.

2.  Next, we define a prompt and function to generate answers for individual questions, using only the provided content.
    ```python
    prompt_answer_question = """
    Using ONLY the provided content below, answer the following question.
    The answer should be concise and directly address the question.
    
    <question>
    {question}
    </question>
    
    <provided_content>
    {combined_content}
    </provided_content>
    """.strip()
    
    def answer_question(question: str, content: str) -> str:
        answer_response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_answer_question.format(question=question, combined_content=content),
        )
        return answer_response.text
    ```

3.  Finally, we define a `SourceList` Pydantic model, a prompt, and a function to identify which sources were used to generate an answer.
    ```python
    class SourceList(types.BaseModel):
        sources: list[str] = types.Field(description="A list of source titles that were used to answer the question")
    
    prompt_find_sources = """
    You will be given a question and an answer that was generated from a set of documents.
    Your task is to identify which of the original documents were used to create the answer.
    
    <question>
    {question}
    </question>
    
    <answer>
    {answer}
    </answer>
    
    <provided_content>
    {combined_content}
    </provided_content>
    """.strip()
    
    def find_sources(question: str, answer: str, content: str) -> list[str]:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SourceList
        )
        sources_response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_find_sources.format(question=question, answer=answer, combined_content=content),
            config=config
        )
        return sources_response.parsed.sources
    ```

4.  We combine these three functions into a sequential workflow. Each step executes one after another for each question.
    ```python
    def sequential_workflow(content, n_questions=10) -> list[FAQ]:
        questions = generate_questions(content, n_questions)
        final_faqs = []
        for question in questions:
            answer = answer_question(question, content)
            sources = find_sources(question, answer, content)
            faq = FAQ(
                question=question,
                answer=answer,
                sources=sources
            )
            final_faqs.append(faq)
        return final_faqs
    
    import time
    start_time = time.monotonic()
    sequential_faqs = sequential_workflow(combined_content, n_questions=4)
    end_time = time.monotonic()
    print(f"Sequential processing completed in {end_time - start_time:.2f} seconds")
    ```
    It outputs:
    ```text
    Sequential processing completed in 22.20 seconds
    ```

5.  Here is a partial output of the final generated FAQs.
    ```json
    {
      "question": "What are the primary financial benefits of installing solar panels for homeowners, and are there any initial costs to consider?",
      "answer": "The primary financial benefits of installing solar panels for homeowners are significantly lowered monthly electricity bills and, in some cases, the ability to sell excess power back to the grid. The initial installation cost can be high.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    }
    ```
This sequential workflow is clearer and more debuggable. Each step has a single, well-defined responsibility. However, executing steps one by one can be slow, especially for many questions.

## Optimizing Sequential Workflows With Parallel Processing

While the sequential workflow is effective, we can significantly reduce the overall processing time by running some steps in parallel [[21]](https://www.anthropic.com/research/building-effective-agents), [[23]](https://fme.safe.com/guides/ai-agent-architecture/ai-agentic-workflows/), [[24]](https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns). For instance, after generating all questions, we can answer and find sources for each question concurrently.

1.  We implement asynchronous versions of our `answer_question` and `find_sources` functions using Python's `asyncio` library.
    ```python
    import asyncio
    
    async def answer_question_async(question: str, content: str) -> str:
        prompt = prompt_answer_question.format(question=question, combined_content=content)
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    
    async def find_sources_async(question: str, answer: str, content: str) -> list[str]:
        prompt = prompt_find_sources.format(question=question, answer=answer, combined_content=content)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SourceList
        )
        response = await client.aio.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.parsed.sources
    
    async def process_question_parallel(question: str, content: str) -> FAQ:
        answer = await answer_question_async(question, content)
        sources = await find_sources_async(question, answer, content)
        return FAQ(
            question=question,
            answer=answer,
            sources=sources
        )
    ```

2.  We then execute the complete parallel workflow.
    ```python
    async def parallel_workflow(content: str, n_questions: int = 10) -> list[FAQ]:
        questions = generate_questions(content, n_questions)
        tasks = [process_question_parallel(question, content) for question in questions]
        parallel_faqs = await asyncio.gather(*tasks)
        return parallel_faqs
    
    start_time = time.monotonic()
    # In a real async environment, you would run this with asyncio.run(parallel_workflow(...))
    # For a Jupyter environment, await is sufficient.
    parallel_faqs = await parallel_workflow(combined_content, n_questions=4)
    end_time = time.monotonic()
    print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")
    ```
    It outputs:
    ```text
    Parallel processing completed in 8.98 seconds
    ```
    This demonstrates a significant reduction in processing time compared to the sequential approach.

3.  Here is a partial output of the final generated FAQs.
    ```json
    {
      "question": "What are the primary environmental and economic benefits of using solar energy?",
      "answer": "The primary environmental benefit of solar energy is cutting down greenhouse gas emissions by reducing reliance on fossil fuels.\n\nThe primary economic benefits include significantly lower monthly electricity bills, the ability to sell excess power back to the grid, long-term savings, and contributing to energy independence for nations.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    }
    ```
Parallel processing can drastically improve the speed of your LLM applications, especially for tasks with independent sub-components. However, it introduces more complex error handling and demands careful management of API rate limits. For example, models with free tiers often have limits like 20 calls per minute. You need to implement retries with exponential backoff and potentially queue or batch requests to stay within these quotas [[11]](https://platform.openai.com/docs/guides/rate-limits), [[12]](https://cloud.google.com/vertex-ai/generative-ai/docs/quotas), [[13]](https://learn.microsoft.com/azure/ai-services/openai/quotas-limits), [[14]](https://platform.openai.com/docs/guides/error-codes), [[15]](https://docs.anthropic.com/en/docs/build-with-claude/rate-limits).

Beyond reducing latency, parallel LLM calls can also act as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs [[44]](https://arxiv.org/html/2503.15838v2), [[45]](https://cameronrwolfe.substack.com/p/prompt-ensembles-make-llms-more-reliable), [[46]](https://arxiv.org/html/2502.18036v1), [[47]](https://openreview.net/forum?id=OIEczoib6t). By generating multiple candidate responses in parallel from diverse LLMs or prompts and then aggregating their agreement, we can select a consensus output that is more reliable and consistent.

## Introducing Dynamic Behavior: Routing and Conditional Logic

Not all inputs or intermediate states in an LLM workflow should be processed in the same way. Imagine a customer service system where a user asks about a billing issue, a technical problem, or general information. Each of these queries requires a different specialized response. This is where routing and conditional logic become essential [[34]](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html), [[35]](https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns), [[36]](https://arize.com/docs/phoenix/learn).

Routing is a method that categorizes an input and then sends it to a specific task designed to handle that type of input [[37]](https://www.revanthquicklearn.com/post/understanding-workflow-design-patterns-in-ai-systems). An LLM call itself can be used to make the routing decision, for example, by classifying the input's intent or an intermediate result [[39]](https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/). This creates "branching" in a workflow, allowing the system to follow different paths based on dynamic conditions.

Routing is preferable to trying to optimize a single prompt for multiple types of inputs because it adheres to the principle of "divide and conquer." We keep prompts as specialized as possible, ideally with a single responsibility. This separation of concerns improves accuracy, simplifies maintenance, and allows for more focused optimization of each processing path [[38]](https://mikulskibartosz.name/ai-workflow-design-patterns), [[40]](https://arize.com/blog/best-practices-for-building-an-ai-agent-router/), [[41]](https://www.requesty.ai/blog/intelligent-llm-routing-in-enterprise-ai-uptime-cost-efficiency-and-model), [[42]](https://arxiv.org/html/2506.16655v1), [[43]](https://www.youtube.com/watch?v=HMXVMpJTW6o).

## Building a Basic Routing Workflow

Let's build a basic routing workflow for a customer service system. The goal is to classify a user's query intent and then route it to a specialized prompt or handler.

Image 2: A flowchart illustrating a routing workflow for customer service intent classification.

1.  First, we define an `IntentEnum` for possible categories and a `UserIntent` Pydantic model for the classification output. We also create a `classify_intent` function.
    ```python
    from enum import Enum
    
    class IntentEnum(str, Enum):
        TECHNICAL_SUPPORT = "Technical Support"
        BILLING_INQUIRY = "Billing Inquiry"
        GENERAL_QUESTION = "General Question"
    
    class UserIntent(types.BaseModel):
        intent: IntentEnum = types.Field(description="The intent of the user's query")
    
    prompt_classification = """
    Classify the user's query into one of the following categories.
    
    <categories>
    {categories}
    </categories>
    
    <user_query>
    {user_query}
    </user_query>
    """.strip()
    
    def classify_intent(user_query: str) -> IntentEnum:
        prompt = prompt_classification.format(
            user_query=user_query,
            categories=[intent.value for intent in IntentEnum]
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=UserIntent
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.parsed.intent
    ```
    This function uses an LLM to classify a user query into predefined categories.

2.  Next, we define specialized prompts for each intent.
    ```python
    prompt_technical_support = """
    You are a helpful technical support agent.
    
    Here's the user's query:
    <user_query>
    {user_query}
    </user_query>
    
    Provide a helpful first response, asking for more details like what troubleshooting steps they have already tried.
    """.strip()
    
    prompt_billing_inquiry = """
    You are a helpful billing support agent.
    
    Here's the user's query:
    <user_query>
    {user_query}
    </user_query>
    
    Acknowledge their concern and inform them that you will need to look up their account, asking for their account number.
    """.strip()
    
    prompt_general_question = """
    You are a general assistant.
    
    Here's the user's query:
    <user_query>
    {user_query}
    </user_query>
    
    Apologize that you are not sure how to help.
    """.strip()
    ```

3.  Finally, we create a `handle_query` function that routes the user's query to the appropriate handler based on the classified intent.
    ```python
    def handle_query(user_query: str, intent: str) -> str:
        if intent == IntentEnum.TECHNICAL_SUPPORT:
            prompt = prompt_technical_support.format(user_query=user_query)
        elif intent == IntentEnum.BILLING_INQUIRY:
            prompt = prompt_billing_inquiry.format(user_query=user_query)
        elif intent == IntentEnum.GENERAL_QUESTION:
            prompt = prompt_general_question.format(user_query=user_query)
        else:
            prompt = prompt_general_question.format(user_query=user_query) # Fallback
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        return response.text
    
    query_1 = "My internet connection is not working."
    query_2 = "I think there is a mistake on my last invoice."
    query_3 = "What are your opening hours?"
    
    intent_1 = classify_intent(query_1)
    intent_2 = classify_intent(query_2)
    intent_3 = classify_intent(query_3)
    
    response_1 = handle_query(query_1, intent_1)
    response_2 = handle_query(query_2, intent_2)
    response_3 = handle_query(query_3, intent_3)
    
    print(f"Question 1: {query_1}\nIntent 1: {intent_1}\nResponse 1: {response_1}\n")
    print(f"Question 2: {query_2}\nIntent 2: {intent_2}\nResponse 2: {response_2}\n")
    print(f"Question 3: {query_3}\nIntent 3: {intent_3}\nResponse 3: {response_3}\n")
    ```
    It outputs:
    ```text
    Question 1: My internet connection is not working.
    Intent 1: IntentEnum.TECHNICAL_SUPPORT
    Response 1: Hello there! I'm sorry to hear you're having trouble with your internet connection. That can definitely be frustrating.
    
    To help me understand what's going on and assist you best, could you please provide a few more details?
    
    1.  **What exactly are you experiencing?** For example, are you not seeing your Wi-Fi network, is your Wi-Fi connected but no websites are loading, or are there any specific error messages?
    2.  **What device are you trying to connect with?** (e.g., a laptop, phone, desktop PC)
    3.  **Have you already tried any troubleshooting steps yourself?** For instance, have you tried:
        *   Restarting your computer or device?
        *   Restarting your Wi-Fi router and modem (unplugging them for 30 seconds and plugging them back in)?
        *   Checking if other devices can connect to the internet?
    
    Once I have a bit more information, I'll be happy to guide you through some potential solutions.
    
    Question 2: I think there is a mistake on my last invoice.
    Intent 2: IntentEnum.BILLING_INQUIRY
    Response 2: I'm sorry to hear you think there might be a mistake on your last invoice. I can definitely help you look into that!
    
    To access your account and investigate the charges, could you please provide your account number?
    
    Question 3: What are your opening hours?
    Intent 3: IntentEnum.GENERAL_QUESTION
    Response 3: I apologize, but I'm not sure how to help with that. As an AI, I don't have a physical location or opening hours.
    ```
This basic routing workflow effectively directs diverse user queries to specialized handlers, ensuring appropriate and focused responses.

## Orchestrator-Worker Pattern: Dynamic Task Decomposition

The orchestrator-worker pattern is a step beyond fixed routing or parallelization. In this workflow, a central LLM, the **orchestrator**, dynamically breaks down complex tasks into smaller, often unpredictable, sub-tasks. It then delegates these sub-tasks to specialized **worker** LLMs and synthesizes their results into a cohesive final response [[21]](https://www.anthropic.com/research/building-effective-agents), [[22]](https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows), [[23]](https://fme.safe.com/guides/ai-agent-architecture/ai-agentic-workflows/).

Image 3: A flowchart illustrating the Orchestrator-Worker pattern.

This pattern is well-suited for complex tasks where you cannot predict the sub-tasks needed in advance. For example, in a coding project, the number of files that need to be changed and the nature of those changes depend entirely on the initial request [[21]](https://www.anthropic.com/research/building-effective-agents). The key difference from parallelization is its flexibility: sub-tasks are not pre-defined but determined by the orchestrator based on the specific input, allowing for dynamic planning and delegation [[21]](https://www.anthropic.com/research/building-effective-agents), [[22]](https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows).

However, the orchestrator-worker pattern also introduces practical implementation challenges [[29]](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-orchestration.html), [[30]](https://www.anthropic.com/research/building-effective-agents), [[31]](https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns), [[32]](https://javaaidev.com/docs/agentic-patterns/patterns/orchestrator-workers-workflow/), [[33]](https://bootcamptoprod.com/spring-ai-orchestrator-workers-workflow-guide/):
*   **Orchestrator bottleneck**: The central orchestrator can become a single point of failure and a bottleneck if planning, decomposition, and synthesis are not efficiently handled, especially in complex, hierarchical workflows [[29]](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-orchestration.html), [[30]](https://www.anthropic.com/research/building-effective-agents).
*   **Synthesis errors**: Errors can occur when aggregating heterogeneous worker outputs, such as inconsistent formats or contradictory findings, leading to incorrect final responses [[30]](https://www.anthropic.com/research/building-effective-agents), [[31]](https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns), [[33]](https://bootcamptoprod.com/spring-ai-orchestrator-workers-workflow-guide/).
*   **Task assignment mismatches**: If worker roles are ambiguous or tools overlap, the orchestrator might misroute sub-tasks, leading to degraded intermediate results [[29]](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-orchestration.html), [[32]](https://javaaidev.com/docs/agentic-patterns/patterns/orchestrator-workers-workflow/).

Despite these challenges, the orchestrator-worker pattern is powerful for highly dynamic and complex problems.

Let's look at an example using a customer service query that involves multiple tasks: a billing inquiry, a product return, and an order status update.

1.  We define `QueryTypeEnum` and Pydantic models for `Task` and `TaskList`. We also create the `orchestrator` function.
    ```python
    import random
    
    class QueryTypeEnum(str, Enum):
        BILLING_INQUIRY = "BillingInquiry"
        PRODUCT_RETURN = "ProductReturn"
        STATUS_UPDATE = "StatusUpdate"
    
    class Task(types.BaseModel):
        query_type: QueryTypeEnum = types.Field(description="The type of query to be handled.")
        invoice_number: str | None = types.Field(description="The invoice number for the billing inquiry.", default=None)
        product_name: str | None = types.Field(description="The name of the product to be returned.", default=None)
        reason_for_return: str | None = types.Field(description="The reason for returning the product.", default=None)
        order_id: str | None = types.Field(description="The order ID for the status update.", default=None)
    
    class TaskList(types.BaseModel):
        tasks: list[Task] = types.Field(description="A list of tasks to be performed.")
    
    prompt_orchestrator = f"""
    You are a master orchestrator. Your job is to break down a complex user query into a list of sub-tasks.
    Each sub-task must have a "query_type" and its necessary parameters.
    
    The possible "query_type" values and their required parameters are:
    1. "{QueryTypeEnum.BILLING_INQUIRY.value}": Requires "invoice_number".
    2. "{QueryTypeEnum.PRODUCT_RETURN.value}": Requires "product_name" and "reason_for_return".
    3. "{QueryTypeEnum.STATUS_UPDATE.value}": Requires "order_id".
    
    Here's the user's query.
    
    <user_query>
    {{query}}
    </user_query>
    """.strip()
    
    def orchestrator(query: str) -> list[Task]:
        prompt = prompt_orchestrator.format(query=query)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TaskList
        )
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=config
        )
        return response.parsed.tasks
    ```
    The orchestrator analyzes the input and identifies what types of actions need to be taken.

2.  We then define the `BillingTask` Pydantic model and the `handle_billing_worker` function.
    ```python
    class BillingTask(types.BaseModel):
        query_type: QueryTypeEnum = types.Field(description="The type of task to be performed.", default=QueryTypeEnum.BILLING_INQUIRY)
        invoice_number: str = types.Field(description="The invoice number for the billing inquiry.")
        user_concern: str = types.Field(description="The concern or question the user has voiced about the invoice.")
        action_taken: str = types.Field(description="The action taken to address the user's concern.")
        resolution_eta: str = types.Field(description="The estimated time to resolve the concern.")
    
    prompt_billing_worker_extractor = """
    You are a specialized assistant. A user has a query regarding invoice '{invoice_number}'.
    From the full user query provided below, extract the specific concern or question the user has voiced about this particular invoice.
    Respond with ONLY the extracted concern/question. If no specific concern is mentioned beyond a general inquiry about the invoice, state 'General inquiry regarding the invoice'.
    
    Here's the user's query:
    <user_query>
    {original_user_query}
    </user_query>
    
    Extracted concern about invoice {invoice_number}:
    """.strip()
    
    def handle_billing_worker(invoice_number: str, original_user_query: str) -> BillingTask:
        extraction_prompt = prompt_billing_worker_extractor.format(
            invoice_number=invoice_number, original_user_query=original_user_query
        )
        response = client.models.generate_content(model=MODEL_ID, contents=extraction_prompt)
        extracted_concern = response.text
        investigation_id = f"INV_CASE_{random.randint(1000, 9999)}"
        eta_days = 2
        task = BillingTask(
            invoice_number=invoice_number,
            user_concern=extracted_concern,
            action_taken=f"An investigation (Case ID: {investigation_id}) has been opened regarding your concern.",
            resolution_eta=f"{eta_days} business days",
        )
        return task
    ```
    This worker handles invoice-related inquiries, extracts concerns, and simulates opening an investigation.

3.  Next is the `ReturnTask` Pydantic model and the `handle_return_worker` function.
    ```python
    class ReturnTask(types.BaseModel):
        query_type: QueryTypeEnum = types.Field(description="The type of task to be performed.", default=QueryTypeEnum.PRODUCT_RETURN)
        product_name: str = types.Field(description="The name of the product to be returned.")
        reason_for_return: str = types.Field(description="The reason for returning the product.")
        rma_number: str = types.Field(description="The RMA number for the return.")
        shipping_instructions: str = types.Field(description="The shipping instructions for the return.")
    
    def handle_return_worker(product_name: str, reason_for_return: str) -> ReturnTask:
        rma_number = f"RMA-{random.randint(10000, 99999)}"
        shipping_instructions = (
            "Please pack the '{product_name}' securely in its original packaging if possible. "
            "Include all accessories and manuals. Write the RMA number ({rma_number}) clearly on the outside of the package. "
            "Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."
        ).format(product_name=product_name, rma_number=rma_number)
        task = ReturnTask(
            product_name=product_name,
            reason_for_return=reason_for_return,
            rma_number=rma_number,
            shipping_instructions=shipping_instructions,
        )
        return task
    ```
    This worker handles product return requests by generating RMA numbers and providing shipping instructions.

4.  We define the `StatusTask` Pydantic model and the `handle_status_worker` function.
    ```python
    class StatusTask(types.BaseModel):
        query_type: QueryTypeEnum = types.Field(description="The type of task to be performed.", default=QueryTypeEnum.STATUS_UPDATE)
        order_id: str = types.Field(description="The order ID for the status update.")
        current_status: str = types.Field(description="The current status of the order.")
        carrier: str = types.Field(description="The carrier of the order.")
        tracking_number: str = types.Field(description="The tracking number of the order.")
        expected_delivery: str = types.Field(description="The expected delivery date of the order.")
    
    def handle_status_worker(order_id: str) -> StatusTask:
        possible_statuses = [
            {"status": "Processing", "carrier": "N/A", "tracking": "N/A", "delivery_estimate": "3-5 business days"},
            {"status": "Shipped", "carrier": "SuperFast Shipping", "tracking": f"SF{random.randint(100000, 999999)}", "delivery_estimate": "Tomorrow"},
            {"status": "Delivered", "carrier": "Local Courier", "tracking": f"LC{random.randint(10000, 99999)}", "delivery_estimate": "Delivered yesterday"},
            {"status": "Delayed", "carrier": "Standard Post", "tracking": f"SP{random.randint(10000, 99999)}", "delivery_estimate": "Expected in 2-3 additional days"},
        ]
        status_details = random.choice(possible_statuses)
        task = StatusTask(
            order_id=order_id,
            current_status=status_details["status"],
            carrier=status_details["carrier"],
            tracking_number=status_details["tracking"],
            expected_delivery=status_details["delivery_estimate"],
        )
        return task
    ```
    This worker retrieves and formats order status information.

5.  Next, we define the `prompt_synthesizer` and the `synthesizer` function.
    ```python
    prompt_synthesizer = """
    You are a master communicator. Combine several distinct pieces of information from our support team into a single, well-formatted, and friendly email to a customer.
    
    Here are the points to include, based on the actions taken for their query:
    <points>
    {formatted_results}
    </points>
    
    Combine these points into one cohesive response.
    Start with a friendly greeting (e.g., "Dear Customer," or "Hi there,") and end with a polite closing (e.g., "Sincerely," or "Best regards,").
    Ensure the tone is helpful and professional.
    """.strip()
    
    def synthesizer(results: list[Task]) -> str:
        bullet_points = []
        for res in results:
            point = f"Regarding your {res.query_type}:\n"
            if res.query_type == QueryTypeEnum.BILLING_INQUIRY:
                res: BillingTask = res
                point += f"  - Invoice Number: {res.invoice_number}\n"
                point += f'  - Your Stated Concern: "{res.user_concern}"\n'
                point += f"  - Our Action: {res.action_taken}\n"
                point += f"  - Expected Resolution: We will get back to you within {res.resolution_eta}."
            elif res.query_type == QueryTypeEnum.PRODUCT_RETURN:
                res: ReturnTask = res
                point += f"  - Product: {res.product_name}\n"
                point += f'  - Reason for Return: "{res.reason_for_return}"\n'
                point += f"  - Return Authorization (RMA): {res.rma_number}\n"
                point += f"  - Instructions: {res.shipping_instructions}"
            elif res.query_type == QueryTypeEnum.STATUS_UPDATE:
                res: StatusTask = res
                point += f"  - Order ID: {res.order_id}\n"
                point += f"  - Current Status: {res.current_status}\n"
                if res.carrier != "N/A":
                    point += f"  - Carrier: {res.carrier}\n"
                if res.tracking_number != "N/A":
                    point += f"  - Tracking Number: {res.tracking_number}\n"
                point += f"  - Delivery Estimate: {res.expected_delivery}"
            bullet_points.append(point)
        formatted_results = "\n\n".join(bullet_points)
        prompt = prompt_synthesizer.format(formatted_results=formatted_results)
        response = client.models.generate_content(model=MODEL_ID, contents=prompt)
        return response.text
    ```
    The synthesizer combines structured results from all workers into a single, cohesive, and customer-friendly response.

6.  Finally, the `process_user_query` function coordinates the entire orchestrator-worker workflow.
    ```python
    def process_user_query(user_query):
        print(f"User query: {user_query}")
        tasks_list = orchestrator(user_query)
        if not tasks_list:
            print("Orchestrator did not return any tasks. Exiting.")
            return
    
        for i, task in enumerate(tasks_list, start=1):
            print(f"Deconstructed task {i}: {task.model_dump_json(indent=2)}")
    
        worker_results = []
        if tasks_list:
            for task in tasks_list:
                if task.query_type == QueryTypeEnum.BILLING_INQUIRY:
                    worker_results.append(handle_billing_worker(task.invoice_number, user_query))
                elif task.query_type == QueryTypeEnum.PRODUCT_RETURN:
                    worker_results.append(handle_return_worker(task.product_name, task.reason_for_return))
                elif task.query_type == QueryTypeEnum.STATUS_UPDATE:
                    worker_results.append(handle_status_worker(task.order_id))
                else:
                    print(f"Warning: Unknown query_type '{task.query_type}' found in orchestrator tasks.")
    
            if worker_results:
                for i, res in enumerate(worker_results, start=1):
                    print(f"Worker result {i}: {res.model_dump_json(indent=2)}")
            else:
                print("No valid worker tasks to run.")
        else:
            print("No tasks to run for workers.")
    
        if worker_results:
            final_user_message = synthesizer(worker_results)
            print(f"Final synthesized response: {final_user_message}")
        else:
            print("Skipping synthesis because there were no worker results.")
    
    complex_customer_query = """
    Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.
    Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.
    Finally, can you give me an update on my order #A-12345?
    """.strip()
    
    process_user_query(complex_customer_query)
    ```
    It outputs:
    ```text
    User query: Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.
    Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.
    Finally, can you give me an update on my order #A-12345?
    Deconstructed task 1: {
      "query_type": "BillingInquiry",
      "invoice_number": "INV-7890",
      "product_name": null,
      "reason_for_return": null,
      "order_id": null
    }
    Deconstructed task 2: {
      "query_type": "ProductReturn",
      "invoice_number": null,
      "product_name": "SuperWidget 5000",
      "reason_for_return": "not compatible with my system",
      "order_id": null
    }
    Deconstructed task 3: {
      "query_type": "StatusUpdate",
      "invoice_number": null,
      "product_name": null,
      "reason_for_return": null,
      "order_id": "A-12345"
    }
    Worker result 1: {
      "query_type": "BillingInquiry",
      "invoice_number": "INV-7890",
      "user_concern": "It seems higher than I expected.",
      "action_taken": "An investigation (Case ID: INV_CASE_5678) has been opened regarding your concern.",
      "resolution_eta": "2 business days"
    }
    Worker result 2: {
      "query_type": "ProductReturn",
      "product_name": "SuperWidget 5000",
      "reason_for_return": "not compatible with my system",
      "rma_number": "RMA-12345",
      "shipping_instructions": "Please pack the 'SuperWidget 5000' securely in its original packaging if possible. Include all accessories and manuals. Write the RMA number (RMA-12345) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."
    }
    Worker result 3: {
      "query_type": "StatusUpdate",
      "order_id": "A-12345",
      "current_status": "Shipped",
      "carrier": "SuperFast Shipping",
      "tracking_number": "SF987654",
      "expected_delivery": "Tomorrow"
    }
    Final synthesized response: Dear Customer,
    
    Thank you for reaching out! Here's an update on your recent inquiries:
    
    Regarding your BillingInquiry:
      - Invoice Number: INV-7890
      - Your Stated Concern: "It seems higher than I expected."
      - Our Action: An investigation (Case ID: INV_CASE_5678) has been opened regarding your concern.
      - Expected Resolution: We will get back to you within 2 business days.
    
    Regarding your ProductReturn:
      - Product: SuperWidget 5000
      - Reason for Return: "not compatible with my system"
      - Return Authorization (RMA): RMA-12345
      - Instructions: Please pack the 'SuperWidget 5000' securely in its original packaging if possible. Include all accessories and manuals. Write the RMA number (RMA-12345) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765.
    
    Regarding your StatusUpdate:
      - Order ID: A-12345
      - Current Status: Shipped
      - Carrier: SuperFast Shipping
      - Tracking Number: SF987654
      - Delivery Estimate: Tomorrow
    
    Best regards,
    Customer Support Team
    ```
This complete execution flow demonstrates how the orchestrator-worker pattern can handle complex, multi-faceted user queries by dynamically decomposing them into specialized sub-tasks and synthesizing the results into a single, coherent response.

## Conclusion

In this lesson, we explored the fundamental building blocks for sophisticated LLM applications: chaining, parallelization, routing, and the orchestrator-worker pattern. You learned why breaking down complex tasks into modular steps is often more effective than relying on a single, monolithic LLM call, leading to improved accuracy, debuggability, and flexibility. We demonstrated how to implement sequential and parallel workflows for FAQ generation, showcasing the performance benefits of concurrent processing. Furthermore, we delved into dynamic behavior with routing for customer service intent classification and introduced the powerful orchestrator-worker pattern for dynamic task decomposition.

Mastering these workflow patterns is a crucial step for any AI engineer. They provide the necessary structure and adaptability to build reliable, production-grade AI systems, moving you beyond simple prototypes to shipping AI products that truly work. In the next lessons, we will build on these foundations, exploring how to give your LLMs the ability to take actions, reason about the world, and manage memory effectively.

## References

1.  (n.d.). *Prompt-only approaches (single-prompt LLM calls) can yield superficially plausible but inconsistent, hallucinated, and ontology-mismatched outputs, limiting reliability for structured tasks without additional controls or fine-tuning.* https://arxiv.org/pdf/2309.08181
2.  (n.d.). *Identifies common prompt-level failure modes in production: ambiguous instructions lead to inconsistent or incorrect outputs; important instructions placed deep in long prompts may be missed, a manifestation related to “lost in the middle.”* https://www.getambassador.io/blog/prompt-engineering-for-llms
3.  (n.d.). *Emphasizes stochasticity: querying with the same prompt multiple times yields varied lists, underscoring instability and variability of single-prompt outputs and the need for multiple runs to surface diverse failure modes.* https://proceedings.neurips.cc/paper_files/paper/2023/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf
4.  (n.d.). *Highlights that careful instruction design and simplified schemas reduce error surface area, implying that complex prompts with extensive requirements increase the likelihood of hallucination and format deviation in one-shot calls.* https://www.york.ac.uk/assuring-autonomy/news/blog/part-one-using-large-language-models/
5.  (n.d.). *Finds failures in long-term causal reasoning: performance drops on long narratives with many events, consistent with “lost in the middle” effects where mid-sequence information is under-attended and multi-hop dependencies degrade.* https://arxiv.org/html/2410.23884v1
6.  (n.d.). *Lists explicit drawbacks of prompt chaining: Management difficulty: Coordinating “a series of interrelated prompts” becomes challenging as chains grow long and intricate, increasing room for error, especially when multiple models are involved.* https://aisdr.com/blog/what-is-prompt-chaining/
7.  (n.d.). *Highlights that chaining requires robust data transformation between steps because LLM outputs are often unstructured; without proper structuring, downstream steps may misinterpret or lose information, leading to degraded results.* https://www.humanfirst.ai/blog/prompt-chaining
8.  (n.d.). *Identifies disadvantages (summary points): Increased complexity: Managing multiple interconnected prompts adds design and maintenance burden compared to single-shot prompting.* https://blog.promptlayer.com/what-is-prompt-chaining/
9.  (n.d.). *Argues that prompt chaining had early value but came with maintenance difficulties and required glue code to connect steps, increasing engineering complexity.* https://ai.plainenglish.io/prompt-chaining-is-dead-long-live-prompt-stuffing-58a1c08820c5
10. Starks, A. (2023, May 1). *How to easily add an AI-Powered Chat Assistant Tool to your Web Application*.
11. (n.d.). *OpenAI enforces per-minute and per-day rate limits and may return 429 errors when limits are exceeded or during transient capacity issues. Clients should implement retries with exponential backoff and jitter, and respect the Retry-After header when present.* https://platform.openai.com/docs/guides/rate-limits
12. (n.d.). *Vertex AI applies quotas for requests-per-minute, requests-per-day, and tokens-per-minute; exceeding quotas results in 429 Too Many Requests. Clients should implement exponential backoff with jitter and respect Retry-After for smooth recovery.* https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
13. (n.d.). *Azure OpenAI enforces per-deployment rate limits measured in RPM and TPM; exceeding them yields 429. Applications should implement exponential backoff and gradually reduce concurrency when 429 frequency increases.* https://learn.microsoft.com/azure/ai-services/openai/quotas-limits
14. (n.d.). *429 Too Many Requests occurs when you exceed rate limits or the system is temporarily overloaded; resolution is to back off and retry respecting Retry-After and using exponential backoff with jitter.* https://platform.openai.com/docs/guides/error-codes
15. (n.d.). *Anthropic enforces per-minute request and token limits; exceeding limits results in 429s. Implement exponential backoff with randomized jitter and honor Retry-After when provided.* https://docs.anthropic.com/en/docs/build-with-claude/rate-limits
16. (n.d.). *Use an LLM-based intent classifier when you need few-shot learning, fast iteration, or multilingual coverage; LLM classifiers can work with only a handful of examples per intent and are quick to train, making it easier to bootstrap and update routing workflows as new intents emerge. This supports rapid deployment and maintenance of customer service routers across languages and channels.* https://rasa.com/docs/rasa/next/llms/llm-intent/
17. (n.d.). *Use a two-stage architecture for classification quality and efficiency: first retrieve top-N candidate intents via an encoder model using intent names and descriptions, then prompt an LLM to select the best match from those candidates. This narrows the decision space and improves routing precision in customer service flows.* https://www.voiceflow.com/pathways/5-tips-to-optimize-your-llm-intent-classification-prompts
18. (n.d.). *Use hierarchical intent classification to manage large intent sets typical in customer service; classify in stages by the biggest differentiators (e.g., product vs. account vs. troubleshooting) to reduce overlap and ambiguity, then refine to more granular intents. This improves routing by focusing the LLM’s decision on one variable at a time.* https://developer.vonage.com/en/blog/how-to-build-an-intent-classification-hierarchy
19. (n.d.). *Define a clear, precise, and comprehensive intent taxonomy to guide LLM classification; a well-structured, hierarchical taxonomy covering the breadth of user intents improves accurate identification and downstream response quality, which is crucial for routing workflows.* https://arxiv.org/html/2402.02136v2
20. (n.d.). *Build a high-quality, representative dataset for intent classification: collect customer queries from relevant channels (support chats, voice transcripts), and annotate with clear labeling guidelines to ensure consistency—foundational for reliable LLM routing.* https://spotintelligence.com/2023/11/03/intent-classification-nlp/
21. (n.d.). *Definition and scope: Anthropic distinguishes between predefined code-path “workflows” and more flexible “agents.” In a predefined workflow, LLMs and tools are orchestrated through fixed logic; in an orchestrator-workers agent workflow, a central LLM dynamically breaks down tasks, delegates to worker LLMs, and synthesizes results.* https://www.anthropic.com/research/building-effective-agents
22. (n.d.). *Orchestrator-workers architecture: A central LLM “Orchestrator” performs real-time task analysis to determine necessary subtasks, selects appropriate specialized workers, provides task-specific objectives and output formats, and then synthesizes worker outputs into a coherent response.* https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows
23. (n.d.). *Pre-defined parallelization workflow: Suitable when several subtasks are independent and known up front; agents work simultaneously on multiple predefined subtasks and their outputs are collated at the end. Example: processing multiple research papers in parallel to extract specific information, then aggregating findings.* https://fme.safe.com/guides/ai-agent-architecture/ai-agentic-workflows/
24. (n.d.). *Parallelization pattern characteristics: Emphasizes efficient concurrent processing of multiple LLM operations with automated aggregation. Best when you can define parallel tasks up front (e.g., analyzing multiple stakeholder groups simultaneously) and aggregate results programmatically.* https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns
25. (n.d.). *Prompt chaining breaks a complex task into smaller, linked steps, where each prompt’s output feeds the next, improving overall LLM performance.* https://www.vellum.ai/blog/what-is-prompt-chaining
26. (n.d.). *Prompt chaining is a systematic approach that breaks down complex tasks into smaller, manageable sequences, helping maintain context and guide the model to more accurate and relevant responses.* https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84
27. (n.d.). *Prompt chaining links multiple prompts where each output becomes the next input, enabling LLMs to handle complex tasks more effectively.* https://www.voiceflow.com/blog/prompt-chaining
28. (n.d.). *Prompt chaining divides complex tasks into interconnected prompts, guiding the LLM through a nuanced reasoning process for more accurate and comprehensive results.* https://blog.promptlayer.com/what-is-prompt-chaining/
29. (n.d.). *The central orchestrator must plan, decompose, delegate, monitor, and synthesize across multiple specialized workers; this concentration of responsibilities can create a single‑point bottleneck if planning and progress monitoring are serialized or poorly parallelized, especially in complex, hierarchical workflows where subtasks vary in scope and reasoning type.* https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-orchestration.html
30. (n.d.). *The orchestrator dynamically determines subtasks at runtime rather than using predefined parallel tasks; this flexibility can introduce variability and planning errors, making it harder to guarantee coverage of all necessary subtasks and increasing the chance of omissions that surface during synthesis.* https://www.anthropic.com/research/building-effective-agents
31. (n.d.). *Implementation centers on a class that: (1) lets the orchestrator analyze the task and determine subtasks, (2) runs workers in parallel, and (3) combines results; practical issues include ensuring safe concurrency for parallel workers and correctly merging partial results into a coherent final response structure.* https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns
32. (n.d.). *The orchestrator commonly uses a reasoning model (e.g., o1, o3‑mini) for planning, while workers use standard models (e.g., gpt‑4o, gpt‑4o‑mini); mismatched capabilities and costs can create a planning throughput bottleneck and higher latency at the orchestrator, as well as inconsistencies when workers’ outputs lack the reasoning depth expected by the synthesizer.* https://javaaidev.com/docs/agentic-patterns/patterns/orchestrator-workers-workflow/
33. (n.d.). *The pattern introduces three components—Orchestrator, Workers, Synthesizer—with the orchestrator deciding tasks at runtime; this dynamism raises coordination overhead and the risk of planning inaccuracies that surface when the synthesizer tries to assemble incomplete or overlapping worker outputs.* https://bootcamptoprod.com/spring-ai-orchestrator-workers-workflow-guide/
34. (n.d.). *AWS describes the routing workflow as a foundational pattern where a first-pass LLM acts as a classifier or dispatcher to interpret input intent or category and then route the request to a specialized downstream task, agent, tool, or workflow.* https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html
35. (n.d.). *Spring AI documents routing as intelligent task distribution: an LLM analyzes input content and routes to the most appropriate specialized prompt or handler, fitting complex tasks where different inputs are better handled by specialized processes.* https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns
36. (n.d.). *Arize Phoenix frames workflows as the backbone of LLM apps, offering structure and predictability compared to fully autonomous agents—key for building reliable agentic systems.* https://arize.com/docs/phoenix/learn
37. (n.d.). *This overview attributes five workflow patterns to Anthropic and details prompt chaining and routing as core designs.* https://www.revanthquicklearn.com/post/understanding-workflow-design-patterns-in-ai-systems
38. (n.d.). *This guide summarizes Anthropic-style routing and chaining with a concrete example: a routing agent classifies a question and selects a downstream workflow, while the downstream tasks can be implemented as separate prompt-chaining workflows.* https://mikulskibartosz.name/ai-workflow-design-patterns
39. (n.d.). *Define routing approach upfront: use static routing when tasks map cleanly to distinct UI components or flows (e.g., separate modules for text generation vs. insight extraction); this improves modularity and simplifies swapping models per task but is less adaptable to evolving needs.* https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/
40. (n.d.). *Choose a routing technique to match system constraints: function calling with an LLM, intent-based routing, or pure code routing; select based on complexity, scalability, performance, and maintenance needs.* https://arize.com/blog/best-practices-for-building-an-ai-agent-router/
41. (n.d.). *Plan routing strategy and criteria early: decide between rule-based routing (keywords, categories), learned classifiers, or a router LLM; begin with simple categories (e.g., code, general chat, analytics, fallback) and map each to models optimizing for speed, accuracy, or cost.* https://www.requesty.ai/blog/intelligent-llm-routing-in-enterprise-ai-uptime-cost-efficiency-and-model
42. (n.d.). *Align routing with human preferences via a structured Domain–Action taxonomy: first resolve high-level domain (e.g., legal, finance), then action (e.g., summarization, code generation); this hierarchy reduces semantic ambiguity and provides natural fallbacks when actions are unclear.* https://arxiv.org/html/2506.16655v1
43. (n.d.). *When to use an agent router: introduce routing as complexity grows (multiple skills/services) to streamline workflows and performance; match the approach to application needs.* https://www.youtube.com/watch?v=HMXVMpJTW6o
44. (n.d.). *Proposes an ensemble framework for LLM-based code generation that generates multiple candidate programs from different LLMs and selects a final answer via a structured voting mechanism, improving reliability beyond latency reduction.* https://arxiv.org/html/2503.15838v2
45. (n.d.). *Discusses prompt ensembles and why naive majority voting over multiple prompts can fail: LLM errors are not i.i.d.; outputs often cluster around the same wrong answer, making majority vote unreliable.* https://cameronrwolfe.substack.com/p/prompt-ensembles-make-llms-more-reliable
46. (n.d.). *Provides a taxonomy of LLM ensembles that clarifies how parallel calls can be leveraged for quality and robustness: Ensemble-before-inference: route to the best model among candidates (hard voting analogue) to improve accuracy and reliability.* https://arxiv.org/html/2502.18036v1
47. (n.d.). *Introduces an algorithm (EnsemW2S) that combines multiple weaker LLMs by adjusting token probabilities through a voting mechanism, showing that ensemble-based probability fusion can approach or exceed a strong single model in some cases.* https://openreview.net/forum?id=OIEczoib6t