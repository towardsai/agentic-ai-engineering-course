**Global Context**

- **What I’m planning to share:** This article explores the fundamental components for building LLM workflows: chaining multiple LLM calls and implementing routing or conditional logic. We will explain why breaking down complex tasks into chained calls is often more effective than relying on a single, large LLM call. Practical demonstrations will show how to build a simple sequential workflow (e.g., Summarize -> Translate) and a basic routing workflow (e.g., Classify Intent -> Route to specific prompt) using just the base OpenAI Python library (Chat Completions API).
- **Why I think it’s valuable:** For an AI Engineer, mastering chaining and routing is the first step towards constructing sophisticated and reliable LLM applications. These techniques provide modularity, improve accuracy, and allow for more controlled and adaptable processing, forming the building blocks for both deterministic workflows and more complex agentic systems.

**Outline**

- **Section 1: The Power of Modularity: Why Chain LLM Calls?**
    - Explain the challenge: Why a single large LLM call for a complex, multi-step task can be problematic.
        - Reduced accuracy and coherence, at least using non-thinking models. Thinking models are better at following complex prompts.
        - Difficulty in pinpointing errors or specific failures.
        - Lack of modularity; hard to update or improve specific parts.
        - Increased likelihood of "lost in the middle" issues with long contexts.
        - Potentially higher token consumption for prompts trying to do too much.
    - Show an example prompt of this, with an example output where we pinpoint problems with it (i.e. parts of the complex instructions are not satisfied with the result). Show a minimal (but runnable) code example using the OpenAI Python library (https://github.com/openai/openai-python/blob/main/README.md) with the “completion” endpoint. Here you should explain how to set an environment variable in google Colab (as the students will run the code from Google Colab) and how to set the “OPENAI_API_KEY” environment variable (it can be retrieved with `from google.colab import userdata`, and it will be used by the openai Python client), and then proceed by defining the prompt and using it with the openai library to generate a completion. When talking about the "OPENAI_API_KEY" api key, you should explain also how to get it and that you have to buy some credits to use the openai models via API, with a minimum purchase of 5$. Explain also how to retrieve the API key. You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.
    - Introduce Chaining: The concept of connecting multiple LLM calls (or other processing steps) sequentially, where the output of one step becomes the input for the next. (https://www.promptingguide.ai/techniques/prompt_chaining)
    - List the benefits of chaining:
        - **Improved Modularity:** Each LLM call focuses on a specific, well-defined sub-task.
        - **Enhanced Accuracy:** Simpler, targeted prompts for each step generally lead to better, more reliable outputs.
        - **Easier Debugging:** Isolate issues to specific links in the chain.
        - **Increased Flexibility:** Individual components can be swapped, updated, or optimized independently.
        - **Potential for Optimization:** Use different models for different steps (e.g., a cheaper/faster model for a simple classification step, a more powerful model for complex generation).
- **Section 2: Building a Sequential Workflow: Summarize -> Translate (with OpenAI Chat Completions)**
    - Show how the previous prompt example can be split into multiple prompts/steps with prompt chaining. Show how the final output is better, more manageable, debugging, etc. (https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/claude-4-best-practices, https://www.anthropic.com/engineering/building-effective-agents, https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts). You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.
    - Discuss considerations for each step: e.g., prompt clarity, handling potential API errors.
    - List the downsides:
        - For example, some instructions may have sense only “together” and they lose meaning when split into multiple prompts/steps.
        - More costs (as more tokens are used).
        - Higher time to completion, as we have to wait for two LLM calls to complete.
        - Some information may be loss after doing multiple steps in a prompt chain (e.g. the first prompt may ask to summarize, while the second prompt may ask to translate, and it may lose some information from the summary while translating). This can also happen when not using prompt chaining though, but here it is more likely the higher the number of prompts in the chain.
- **Section 3: Introducing Dynamic Behavior: Routing and Conditional Logic**
    - Explain the need for routing: Not all inputs or intermediate states should be processed the same way.
    - Introduce conditional logic (e.g., Python if/elif/else statements) as the mechanism to direct the workflow's path.
    - Discuss how an LLM call itself can be used to make the routing decision (e.g., by classifying input or an intermediate result).
    - Explain the concept of "branching" in a workflow.
- **Section 4: Building a Basic Routing Workflow: Classify Intent -> Route (with OpenAI Chat Completions)**
    - Define a clear use case: e.g., a preliminary step in a customer service system that classifies the user's query intent and then routes it to a specialized prompt or handler.
    - You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells. Step-by-step implementation guide:
        - **Step 1: Intent Classification Call**
            - Design a prompt that instructs the LLM to classify the input text into predefined categories (e.g., "Technical Support," "Billing Inquiry," "General Question").
            - Specify how the LLM should output the classification (e.g., a single keyword).
            - Make the API call.
            - Extract and parse the classification result.
            - Include example Python code snippet.
        - **Step 2: Conditional Logic in Python**
            - Use if/elif/else statements based on the classified intent.
        - **Step 3: Route to Specific Prompts/Handlers**
            - Based on the intent, construct and execute a different, specialized LLM call (or trigger a non-LLM action).
            - Example:
                - If intent is "Technical Support," use prompt_tech_support.
                - If intent is "Billing Inquiry," use prompt_billing.
            - Include example Python code snippets for the conditional routing and subsequent calls.
    - Discuss challenges: Ensuring robust classification, handling ambiguous or out-of-scope intents, designing effective prompts for each branch.
- **Section 5: Orchestrator-Worker**
    - Define the orchestrator worker pattern:
        - With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers. (https://www.anthropic.com/engineering/building-effective-agents, https://langchain-ai.github.io/langgraphjs/tutorials/workflows/#routing)
            
            In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
            
            When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.
            
    - Show a code example of this pattern. Make it again a customer care example: the customer asks a query involving multiple questions and multiple actions by the customer care agents, and the LLM will extract them from the question (so, the output here is a string which will be parsed as a list). Mention that we’ll see later in the course a way to make the LLM output specific data types with a technique named “structured outputs”, but for now we simply parse the string results to explain all the steps. Each element of the list must have a “query_type” field and other parameters relevant to the specific item. Then, for each element of the list, the code will check the element to see if that element has a particular “query_type”, and the code will route that element to the specific Python function to manage it (which must involve using an LLM again). This can be done concurrently using async/await. Last, when each element of the list has received an answer, there must be a last LLM step where all the answers are collected and a final message/recap is synthesized and sent to the user. You'll find the code of this section below in these guidelines. Comment also the outputs of the code cells if relevant, they are in the code below as well between cells.
    - Explain the pros and cons of this pattern.

When copying the code from below, try to keep the code cells the same, without splitting their code while explaining them. Remember to comment also the outputs of the cells (it's fine to also comment just a part of the output).

Notebook code for the lesson:
<notebook-code>
```python
!pip install -q openai
```


```python
import os
import json
import asyncio
import random
from openai import OpenAI, AsyncOpenAI
from google.colab import userdata


# Initialize the OpenAI clients (sync and async)
# The clients automatically read the OPENAI_API_KEY from the environment
try:
    client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))
    aclient = AsyncOpenAI(api_key=userdata.get('OPENAI_API_KEY'))
    print("OpenAI clients initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    print("Please ensure your OPENAI_API_KEY is set correctly in Colab.")
```

**Output:**
```
OpenAI clients initialized successfully.
```


# The problem with a single, large LLM call

```python
# Here we define our mock webpage content. Each source has a title and text.

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
    """
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
    """
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
    """
}

all_sources = [webpage_1, webpage_2, webpage_3]

# We'll combine the content for the LLM to process
combined_content = "\n\n".join([f"Source Title: {source['title']}\nContent: {source['content']}" for source in all_sources])
```


```python
# This prompt tries to do everything at once: generate questions, find answers,
# and cite sources. This complexity can often confuse the model.
prompt_complex = f"""
Based on the provided content from three webpages, generate a list of exactly 10 frequently asked questions (FAQs).
For each question, provide a concise answer derived ONLY from the text.
After each answer, you MUST include a list of the 'Source Title's that were used to formulate that answer.

Your final output should be a JSON array where each object has three keys: "question", "answer", and "sources" (which is an array of strings).

Provided Content:
---
{combined_content}
---
""".strip()

response_complex = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "You are an expert at creating FAQs from provided documents."},
        {"role": "user", "content": prompt_complex}
    ],
    response_format={"type": "json_object"}
)
# Note: Even with JSON mode, the model might fail to follow all instructions perfectly.
result_complex = response_complex.choices[0].message.content
print("Complex prompt result (might be inconsistent):")
print(json.dumps(json.loads(result_complex), indent=2))
```

**Output:**
```
Complex prompt result (might be inconsistent):
{
  "faqs": [
    {
      "question": "What are the main benefits of solar energy?",
      "answer": "Solar energy reduces reliance on fossil fuels, cuts greenhouse gas emissions, lowers electricity bills, and can allow homeowners to sell excess power back to the grid. It is also a key component in achieving energy independence for nations.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    },
    {
      "question": "How do wind turbines generate electricity?",
      "answer": "Wind turbines capture kinetic energy from the wind and convert it into electrical power, functioning as a critical part of sustainable energy systems.",
      "sources": [
        "Understanding Wind Turbines"
      ]
    },
    {
      "question": "Where can wind turbines be installed?",
      "answer": "Wind turbines can be installed both onshore and offshore, with offshore turbines generally producing more consistent power due to stronger, more reliable winds.",
      "sources": [
        "Understanding Wind Turbines"
      ]
    },
    {
      "question": "What is a major challenge of wind energy?",
      "answer": "The main challenge for wind energy is its intermittency\u2014it only generates power when the wind blows, requiring energy storage solutions to ensure a steady supply.",
      "sources": [
        "Understanding Wind Turbines"
      ]
    },
    {
      "question": "Why is energy storage important for renewable energy sources?",
      "answer": "Effective energy storage allows excess energy generated when supplies are plentiful to be stored and released when needed, making the power grid more stable and reliable.",
      "sources": [
        "Energy Storage Solutions"
      ]
    },
    {
      "question": "What are common energy storage methods?",
      "answer": "The most common large-scale storage method is pumped-hydro storage, and lithium-ion batteries are also increasingly affordable and widespread for use in various settings.",
      "sources": [
        "Energy Storage Solutions"
      ]
    },
    {
      "question": "How do batteries enhance energy system reliability?",
      "answer": "Batteries, especially lithium-ion ones, help balance energy supply and demand, making the energy system more resilient and reliable.",
      "sources": [
        "Energy Storage Solutions"
      ]
    },
    {
      "question": "What are the economic considerations of solar energy?",
      "answer": "While the initial installation cost can be high, government incentives and long-term savings make solar energy a financially viable option.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    },
    {
      "question": "How does solar energy contribute to environmental benefits?",
      "answer": "Solar energy reduces greenhouse gas emissions by converting sunlight into electricity without fossil fuels.",
      "sources": [
        "The Benefits of Solar Energy"
      ]
    },
    {
      "question": "What role does wind power play in sustainable energy development?",
      "answer": "Wind power is a critical part of the global shift towards sustainable energy, capturing wind's kinetic energy for electricity generation and helping reduce reliance on fossil fuels.",
      "sources": [
        "Understanding Wind Turbines"
      ]
    }
  ]
}
```


# Building a sequential workflow

```python
# Prompts
prompt_generate_questions = """
Based on the content below, generate a list of 10 relevant and distinct questions that a user might have.
Return these questions as a JSON array of strings.

Provided Content:
---
{combined_content}
---
""".strip()

prompt_answer_question = """
Using ONLY the provided content below, answer the following question.
The answer should be concise and directly address the question.

Question:
"{question}"

Provided Content:
---
{combined_content}
---
""".strip()

prompt_find_sources = """
You will be given a question and an answer that was generated from a set of documents.
Your task is to identify which of the original documents were used to create the answer.
Return a JSON object with a single key "sources" which is a list of the titles of the relevant documents.

Question: "{question}"
Answer: "{answer}"

Documents:
---
{combined_content}
---
""".strip()
```


```python
# Step 1: Generate questions
response_questions = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that generates questions based on text."},
        {"role": "user", "content": prompt_generate_questions.format(combined_content=combined_content)}
    ],
    response_format={"type": "json_object"}
)

generated_questions = json.loads(response_questions.choices[0].message.content)['questions']
print(f"Successfully generated {len(generated_questions)} questions.")

# Steps 2 & 3: Answer and find sources for each question
final_faqs = []
for question in generated_questions:
    print(f"  - Processing: '{question[:50]}...'")

    # Step 2: Generate an answer for the current question
    prompt_answer_question = f"""
    Using ONLY the provided content below, answer the following question.
    The answer should be concise and directly address the question.

    Question:
    "{question}"

    Provided Content:
    ---
    {combined_content}
    ---
    """
    answer_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You answer questions based *only* on the provided context."},
            {"role": "user", "content": prompt_answer_question.format(combined_content=combined_content)}
        ]
    )
    answer = answer_response.choices[0].message.content

    # Step 3: Identify the sources for the generated answer
    sources_response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are an expert at identifying document sources for a given text."},
            {"role": "user", "content": prompt_find_sources.format(question=question, answer=answer, combined_content=combined_content)}
        ],
        response_format={"type": "json_object"}
    )
    sources = json.loads(sources_response.choices[0].message.content)['sources']

    final_faqs.append({
        "question": question,
        "answer": answer,
        "sources": sources
    })

# Final result
print("\nGenerated FAQ List:")
print(json.dumps(final_faqs, indent=2))
```

**Output:**
```
Successfully generated 10 questions.
  - Processing: 'What are the main environmental benefits of using ...'
  - Processing: 'How do photovoltaic (PV) panels convert sunlight i...'
  - Processing: 'What financial incentives are available for homeow...'
  - Processing: 'How does wind energy contribute to sustainable ene...'
  - Processing: 'What are the differences between onshore and offsh...'
  - Processing: 'Why is wind energy considered intermittent, and ho...'
  - Processing: 'What role do energy storage solutions play in rene...'
  - Processing: 'How do batteries like lithium-ion enhance the stab...'
  - Processing: 'What are the advantages and disadvantages of pumpe...'
  - Processing: 'What challenges are associated with integrating re...'

Generated FAQ List:
[
  {
    "question": "What are the main environmental benefits of using solar energy?",
    "answer": "The main environmental benefits of using solar energy are reducing reliance on fossil fuels and cutting down greenhouse gas emissions.",
    "sources": [
      "The Benefits of Solar Energy"
    ]
  },
  {
    "question": "How do photovoltaic (PV) panels convert sunlight into electricity?",
    "answer": "Photovoltaic (PV) panels convert sunlight into electricity by using solar cells that generate electrical current when exposed to sunlight.",
    "sources": [
      "The Benefits of Solar Energy"
    ]
  },
  {
    "question": "What financial incentives are available for homeowners to install solar panels?",
    "answer": "Homeowners can benefit from government incentives for installing solar panels.",
    "sources": [
      "The Benefits of Solar Energy"
    ]
  },
  {
    "question": "How does wind energy contribute to sustainable energy goals worldwide?",
    "answer": "Wind energy contributes to sustainable energy goals worldwide by providing a renewable source of electricity, reducing reliance on fossil fuels, and lowering greenhouse gas emissions. It helps achieve energy independence and supports the global shift toward environmentally friendly power generation.",
    "sources": [
      "Understanding Wind Turbines"
    ]
  },
  {
    "question": "What are the differences between onshore and offshore wind turbines?",
    "answer": "Onshore wind turbines are installed on land, while offshore wind turbines are installed at sea, typically producing more consistent power due to stronger winds.",
    "sources": [
      "Understanding Wind Turbines"
    ]
  },
  {
    "question": "Why is wind energy considered intermittent, and how does this affect power supply?",
    "answer": "Wind energy is considered intermittent because it only generates power when the wind blows. This affects power supply by necessitating energy storage solutions to ensure a steady and reliable supply of electricity.",
    "sources": [
      "Understanding Wind Turbines",
      "Energy Storage Solutions"
    ]
  },
  {
    "question": "What role do energy storage solutions play in renewable energy systems?",
    "answer": "Energy storage solutions play a crucial role in renewable energy systems by storing excess energy generated from sources like solar and wind when it is plentiful, and releasing it when needed. This helps to address the intermittency of renewable sources and ensures a stable, reliable power supply.",
    "sources": [
      "Energy Storage Solutions",
      "Understanding Wind Turbines"
    ]
  },
  {
    "question": "How do batteries like lithium-ion enhance the stability of renewable energy sources?",
    "answer": "Batteries like lithium-ion enhance the stability of renewable energy sources by storing excess energy when production is high and releasing it when demand is high or generation is low, ensuring a steady and reliable power supply.",
    "sources": [
      "Energy Storage Solutions",
      "Understanding Wind Turbines"
    ]
  },
  {
    "question": "What are the advantages and disadvantages of pumped-hydro storage versus battery storage?",
    "answer": "Advantages of pumped-hydro storage include large capacity and long-term efficiency. Disadvantages are high initial infrastructure costs and limited site availability.  \nAdvantages of battery storage include lower initial costs and flexibility in installation. Disadvantages are generally smaller capacity compared to pumped-hydro and higher cost per unit of stored energy over large scales.",
    "sources": [
      "Energy Storage Solutions"
    ]
  },
  {
    "question": "What challenges are associated with integrating renewable energy into existing power grids?",
    "answer": "The challenges associated with integrating renewable energy into existing power grids include managing intermittency, which requires energy storage solutions like batteries to ensure a steady supply of electricity.",
    "sources": [
      "Understanding Wind Turbines",
      "Energy Storage Solutions"
    ]
  }
]
```


# Building a routing workflow

```python
prompt_classification = """
Classify the user's query into one of the following categories:
"Technical Support", "Billing Inquiry", or "General Question".
Return only the category name and nothing else.
User Query: "{user_query}"
""".strip()

def classify_intent(user_query):
    """Uses an LLM to classify a user query."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert at classifying user intents."},
            {"role": "user", "content": prompt_classification.format(user_query=user_query)}
        ]
    )
    intent = response.choices[0].message.content.strip()
    return intent

query_1 = "My internet connection is not working."
query_2 = "I think there is a mistake on my last invoice."
query_3 = "What are your opening hours?"

intent_1 = classify_intent(query_1)
print(f"Query: {query_1}\nIntent: {intent_1}\n")
intent_2 = classify_intent(query_2)
print(f"Query: {query_2}\nIntent: {intent_2}\n")
intent_3 = classify_intent(query_3)
print(f"Query: {query_3}\nIntent: {intent_3}\n")
```

**Output:**
```
Query: My internet connection is not working.
Intent: Technical Support

Query: I think there is a mistake on my last invoice.
Intent: Billing Inquiry

Query: What are your opening hours?
Intent: General Question
```


```python
prompt_technical_support = """
You are a helpful technical support agent.
The user says: '{user_query}'.
Provide a helpful first response, asking for more details like what troubleshooting steps they have already tried.
""".strip()

prompt_billing_inquiry = """
You are a helpful billing support agent.
The user says: '{user_query}'.
Acknowledge their concern and inform them that you will need to look up their account, asking for their account number.
""".strip()

prompt_general_question = """
You are a general assistant.
The user says: '{user_query}'.
Apologize that you are not sure how to help and ask them to rephrase their question.
""".strip()

prompt_general_question = """
You are a general assistant.
The user says: '{user_query}'.
Apologize that you are not sure how to help and ask them to rephrase their question.
""".strip()

def handle_query(user_query, intent):
    """Routes a query to the correct handler based on its classified intent."""
    if intent == "Technical Support":
        prompt = prompt_technical_support.format(user_query=user_query)
    elif intent == "Billing Inquiry":
        prompt = prompt_billing_inquiry.format(user_query=user_query)
    elif intent == "General Question":
        prompt = prompt_general_question.format(user_query=user_query)
    else:
        prompt = prompt_general_question.format(user_query=user_query)
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "system", "content": prompt}]
    )
    return response.choices[0].message.content

response_1 = handle_query(query_1, intent_1)
print(f"Query: {query_1}\nIntent: {intent_1}\nResponse: {response_1}\n")

response_2 = handle_query(query_2, intent_2)
print(f"Query: {query_2}\nIntent: {intent_2}\nResponse: {response_2}\n")

response_3 = handle_query(query_3, intent_3)
print(f"Query: {query_3}\nIntent: {intent_3}\nResponse: {response_3}\n")
```

**Output:**
```
Query: My internet connection is not working.
Intent: Technical Support
Response: I'm sorry to hear that your internet connection isn't working. Could you please provide a bit more detail? For example, are you using a wired or wireless connection? Have you tried any troubleshooting steps so far, such as restarting your modem/router or checking if other devices can connect? This info will help me assist you better.

Query: I think there is a mistake on my last invoice.
Intent: Billing Inquiry
Response: I'm sorry to hear that you think there may be a mistake on your last invoice. To assist you further, could you please provide me with your account number? That way, I can look up your account and review the details.

Query: What are your opening hours?
Intent: General Question
Response: I'm sorry, I'm not sure how to help with that. Could you please rephrase your question or provide more details?
```


# Orchestrator-worker pattern

```python
# Orchestrator
prompt_orchestrator = """
You are a master orchestrator. Your job is to break down a complex user query into a JSON array of objects.
Each object represents one sub-task and must have a "query_type" and relevant parameters.

The possible "query_type" values are:
1. "BillingInquiry": Requires "invoice_number".
2. "ProductReturn": Requires "product_name" and "reason_for_return".
3. "StatusUpdate": Requires "order_id".

User Query:
---
{complex_query}
---

Return ONLY the JSON array and nothing else. Start with "[", end with "]" and separate objects with a comma.
""".strip()

async def orchestrator(complex_query):
    """Breaks down a complex query into a list of tasks."""
    response = await aclient.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt_orchestrator.format(complex_query=complex_query)}]
    )
    tasks_str = response.choices[0].message.content
    try:
        return json.loads(tasks_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from orchestrator: {e}")
        print(f"Orchestrator raw response: {tasks_str}")
        return []
```


```python
# Worker for Billing Inquiry
prompt_billing_worker_extractor = """
You are a specialized assistant. A user has a query regarding invoice '{invoice_number}'.
From the full user query provided below, extract the specific concern or question the user has voiced about this particular invoice.
Respond with ONLY the extracted concern/question. If no specific concern is mentioned beyond a general inquiry about the invoice, state 'General inquiry regarding the invoice'.

Full User Query:
---
{original_user_query}
---

Extracted concern about invoice {invoice_number}:
""".strip()

async def handle_billing_worker(invoice_number, original_user_query):
    """
    Handles a billing inquiry.
    1. Uses an LLM to extract the specific concern about the invoice from the original query.
    2. Simulates opening an investigation.
    3. Returns structured data about the action taken.
    """
    extraction_prompt = prompt_billing_worker_extractor.format(
        invoice_number=invoice_number,
        original_user_query=original_user_query
    )
    response = await aclient.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": extraction_prompt}]
    )
    extracted_concern = response.choices[0].message.content.strip()

    # Simulate backend action: opening an investigation
    print(f"  [Billing Worker] Action: Investigating invoice {invoice_number} for concern: '{extracted_concern}'")
    investigation_id = f"INV_CASE_{random.randint(1000,9999)}"
    eta_days = 2

    return {
        "task": "Billing Inquiry",
        "invoice_number": invoice_number,
        "user_concern": extracted_concern,
        "action_taken": f"An investigation (Case ID: {investigation_id}) has been opened regarding your concern.",
        "resolution_eta": f"{eta_days} business days"
    }
```


```python
# Worker for Product Return
async def handle_return_worker(product_name, reason_for_return):
    """
    Handles a product return request.
    1. Simulates generating an RMA number and providing return instructions.
    2. Returns structured data.
    """
    # Simulate backend action: generating RMA and getting instructions
    rma_number = f"RMA-{random.randint(10000, 99999)}"
    shipping_instructions = (
        "Please pack the '{product_name}' securely in its original packaging if possible. "
        "Include all accessories and manuals. Write the RMA number ({rma_number}) clearly on the outside of the package. "
        "Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."
    ).format(product_name=product_name, rma_number=rma_number)
    print(f"  [Return Worker] Action: Generated RMA {rma_number} for {product_name} (Reason: {reason_for_return})")

    return {
        "task": "Product Return",
        "product_name": product_name,
        "reason_for_return": reason_for_return,
        "rma_number": rma_number,
        "shipping_instructions": shipping_instructions
    }
```


```python
# Worker for Status Update
async def handle_status_worker(order_id):
    """
    Handles an order status update request.
    1. Simulates fetching order status from a backend system.
    2. Returns structured data.
    """
    # Simulate backend action: fetching order status
    # Possible statuses and details to make it more dynamic
    possible_statuses = [
        {"status": "Processing", "carrier": "N/A", "tracking": "N/A", "delivery_estimate": "3-5 business days"},
        {"status": "Shipped", "carrier": "SuperFast Shipping", "tracking": f"SF{random.randint(100000,999999)}", "delivery_estimate": "Tomorrow"},
        {"status": "Delivered", "carrier": "Local Courier", "tracking": f"LC{random.randint(10000,99999)}", "delivery_estimate": "Delivered yesterday"},
        {"status": "Delayed", "carrier": "Standard Post", "tracking": f"SP{random.randint(10000,99999)}", "delivery_estimate": "Expected in 2-3 additional days"}
    ]
    # For a given order_id, we could hash it to pick a status or just pick one randomly for this example
    # This ensures that for the same order_id in a single run, we'd get the same fake status if we implement a simple hash.
    # For now, let's pick randomly for demonstration.
    status_details = random.choice(possible_statuses)
    print(f"  [Status Worker] Action: Fetched status for order {order_id}: {status_details['status']}")

    return {
        "task": "Status Update",
        "order_id": order_id,
        "current_status": status_details["status"],
        "carrier": status_details["carrier"],
        "tracking_number": status_details["tracking"],
        "expected_delivery": status_details["delivery_estimate"]
    }
```


```python
# Synthesizer
prompt_synthesizer = """
You are a master communicator. Combine several distinct pieces of information from our support team into a single, well-formatted, and friendly email to a customer.

Here are the points to include, based on the actions taken for their query:
---
{formatted_results}
---

Combine these points into one cohesive response. Start with a friendly greeting (e.g., "Dear Customer," or "Hi there,") and end with a polite closing (e.g., "Sincerely," or "Best regards,").
Ensure the tone is helpful and professional.
""".strip()

async def synthesizer(results):
    """Combines structured results from workers into a single user-facing message."""
    bullet_points = []
    for res in results:
        point = f"Regarding your {res['task']}:\n"
        if res['task'] == 'Billing Inquiry':
            point += f"  - Invoice Number: {res['invoice_number']}\n"
            point += f"  - Your Stated Concern: \"{res['user_concern']}\"\n"
            point += f"  - Our Action: {res['action_taken']}\n"
            point += f"  - Expected Resolution: We will get back to you within {res['resolution_eta']}."
        elif res['task'] == 'ProductReturn':
            point += f"  - Product: {res['product_name']}\n"
            point += f"  - Reason for Return: \"{res['reason_for_return']}\"\n"
            point += f"  - Return Authorization (RMA): {res['rma_number']}\n"
            point += f"  - Instructions: {res['shipping_instructions']}"
        elif res['task'] == 'Status Update':
            point += f"  - Order ID: {res['order_id']}\n"
            point += f"  - Current Status: {res['current_status']}\n"
            if res['carrier'] != "N/A":
                point += f"  - Carrier: {res['carrier']}\n"
            if res['tracking_number'] != "N/A":
                point += f"  - Tracking Number: {res['tracking_number']}\n"
            point += f"  - Delivery Estimate: {res['expected_delivery']}"
        bullet_points.append(point)

    formatted_results = "\n\n".join(bullet_points)
    prompt = prompt_synthesizer.format(formatted_results=formatted_results)
    response = await aclient.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```


```python
# Test with customer query
complex_customer_query = """
Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.
Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.
Finally, can you give me an update on my order #A-12345?
""".strip()

async def process_user_query(user_query):
    """Processes a query using the Orchestrator-Worker-Synthesizer pattern."""

    print(f"User query:\n---\n{user_query}\n---")

    # 1. Run orchestrator
    tasks_list = await orchestrator(user_query)
    if not tasks_list:
        print("\nOrchestrator did not return any tasks. Exiting.")
        return

    print("\nDeconstructed tasks from Orchestrator:")
    print(json.dumps(tasks_list, indent=2))

    # 2. Run workers
    worker_results = []
    if tasks_list:
        worker_coroutines = []
        print(f"\nDispatching {len(tasks_list)} workers...")
        for task in tasks_list:
            if task['query_type'] == 'BillingInquiry':
                worker_coroutines.append(handle_billing_worker(task['invoice_number'], user_query))
            elif task['query_type'] == 'ProductReturn':
                # Ensure reason_for_return is present, provide a default if not (though orchestrator should capture it)
                reason = task.get('reason_for_return', 'Not specified by user')
                worker_coroutines.append(handle_return_worker(task['product_name'], reason))
            elif task['query_type'] == 'StatusUpdate':
                worker_coroutines.append(handle_status_worker(task['order_id']))
            else:
                print(f"Warning: Unknown query_type '{task.get('query_type')}' found in orchestrator tasks.")

        if worker_coroutines:
            print(f"Running {len(worker_coroutines)} workers concurrently...")
            worker_results = await asyncio.gather(*worker_coroutines)
            print("\nWorkers finished their jobs. Results:")
            for i, res in enumerate(worker_results):
                print(f"--- Worker Result {i+1} ---")
                print(json.dumps(res, indent=2))
                print("----------------------")
        else:
            print("\nNo valid worker tasks to run.")
    else:
        print("\nNo tasks to run for workers.")

    # 3. Run synthesizer
    if worker_results:
        print("\nSynthesizing final response...")
        final_user_message = await synthesizer(worker_results)
        print("\n--- Final Synthesized Response ---")
        print(final_user_message)
        print("---------------------------------")
    else:
        print("\nSkipping synthesis because there were no worker results.")

await process_user_query(complex_customer_query)
```

**Output:**
```
User query:
---
Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.
Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.
Finally, can you give me an update on my order #A-12345?
---

Deconstructed tasks from Orchestrator:
[
  {
    "query_type": "BillingInquiry",
    "invoice_number": "INV-7890"
  },
  {
    "query_type": "ProductReturn",
    "product_name": "SuperWidget 5000",
    "reason_for_return": "not compatible with my system"
  },
  {
    "query_type": "StatusUpdate",
    "order_id": "A-12345"
  }
]

Dispatching 3 workers...
Running 3 workers concurrently...
  [Return Worker] Action: Generated RMA RMA-63742 for SuperWidget 5000 (Reason: not compatible with my system)
  [Status Worker] Action: Fetched status for order A-12345: Delayed
  [Billing Worker] Action: Investigating invoice INV-7890 for concern: 'The invoice amount seems higher than expected.'

Workers finished their jobs. Results:
--- Worker Result 1 ---
{
  "task": "Billing Inquiry",
  "invoice_number": "INV-7890",
  "user_concern": "The invoice amount seems higher than expected.",
  "action_taken": "An investigation (Case ID: INV_CASE_1468) has been opened regarding your concern.",
  "resolution_eta": "2 business days"
}
----------------------
--- Worker Result 2 ---
{
  "task": "Product Return",
  "product_name": "SuperWidget 5000",
  "reason_for_return": "not compatible with my system",
  "rma_number": "RMA-63742",
  "shipping_instructions": "Please pack the 'SuperWidget 5000' securely in its original packaging if possible. Include all accessories and manuals. Write the RMA number (RMA-63742) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."
}
----------------------
--- Worker Result 3 ---
{
  "task": "Status Update",
  "order_id": "A-12345",
  "current_status": "Delayed",
  "carrier": "Standard Post",
  "tracking_number": "SP40802",
  "expected_delivery": "Expected in 2-3 additional days"
}
----------------------

Synthesizing final response...

--- Final Synthesized Response ---
Hi there,

Thank you for reaching out to us. I wanted to provide you with an update regarding your recent queries.

Concerning your billing inquiry about Invoice Number INV-7890, we understand that the amount appeared higher than expected. To address this, we have opened an investigation under Case ID INV_CASE_1468. Our team is actively looking into this and will get back to you within the next 2 business days with a resolution.

Regarding your order status for Order ID A-12345, please note that the shipment with Standard Post is currently delayed. The tracking number SP40802 indicates that delivery is now expected within the next 2 to 3 days. We appreciate your patience as we work to get your order to you as promptly as possible.

If you have any further questions or need assistance, please don’t hesitate to reach out.

Best regards,  
[Your Name]  
Customer Support Team
---------------------------------
```
</notebook-code>