# Stop Babysitting Your LLM
### Mastering LLM chaining and routing

## Stop Babysitting Your LLM: Master Chaining and Routing for Production-Ready AI

If you're building Large Language Model (LLM) applications, you've probably been tempted to stuff all your logic into a single, massive prompt. It feels simple, but it’s an engineering trap. Relying on one giant prompt to handle a complex task is inefficient, brittle, and a nightmare to debug when it inevitably fails. It’s the equivalent of writing an entire application in one function.

The hype cycle wants you to believe that a bigger prompt is a better prompt. The reality is, we build production-grade AI systems on principles of modularity and control. This article cuts through the noise and gives you a practical, engineering-focused guide to the fundamental components for building robust LLM workflows: chaining and routing. We’ll do this with nothing but the standard OpenAI Python library, showing you how to build reliable systems from the ground up.

## The Power of Modularity: Why a Single Large Language Model Call is a Bad Idea

Trying to make a single LLM call perform a complex, multi-step task is a common mistake. This approach often leads to reduced accuracy, as the model struggles to follow a long list of intricate instructions. You will also run into "lost in the middle" issues, where instructions buried deep within a long context are ignored.

This monolithic design lacks modularity, making it incredibly difficult to update or debug specific parts of your logic without rewriting the entire prompt. Furthermore, attempting too much in one prompt can lead to higher token consumption, increasing costs. Let's demonstrate this with a concrete example. We will set up a simple Google Colab environment and use the OpenAI Python library to process content from three mock webpages. Our goal is to generate a list of exactly 10 Frequently Asked Questions (FAQs), answer them using only the provided text, and cite the sources for each answer, all within a single API call.

First, you need to set up your environment. Install the OpenAI library and configure your API key. You can get a key from the OpenAI platform after adding credits; a minimum purchase of $5 is required. In Google Colab, you can store your key securely using the secrets manager and access it with `userdata`.
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
Now, let's define our mock data and the complex prompt.
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
Here is the single, complex prompt that attempts to do everything in one go.
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
    model="gpt-4o-mini",
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
    ...
  ]
}
```
While the output might look reasonable at first glance, it often fails in subtle ways. For instance, the model might not generate *exactly* 10 questions, or it might hallucinate an answer not strictly based on the provided text. Debugging this is a pain because the entire process is a black box.

This is where prompt chaining comes in. It's the technique of breaking down a complex task into a sequence of smaller, focused steps, where the output of one step becomes the input for the next. This approach gives us more control and reliability [1](https://www.promptingguide.ai/techniques/prompt_chaining). The benefits of prompt chaining are clear:
*   **Improved Modularity:** Each LLM call focuses on a specific, well-defined sub-task.
*   **Enhanced Accuracy:** Simpler, targeted prompts for each step generally lead to better, more reliable outputs.
*   **Easier Debugging:** You can isolate issues to specific links in the chain, making failures much easier to pinpoint and fix.
*   **Increased Flexibility:** Individual components can be swapped, updated, or optimized independently without affecting the entire workflow.
*   **Potential for Optimization:** You can use different models for different steps, such as a cheaper and faster model for a simple classification step, and a more powerful model for complex generation.

## Building a Sequential Workflow: A Practical FAQ Generation Example

We will now solve the problem we encountered earlier by building a sequential workflow, also known as a prompt chain. Instead of a single, complex prompt, we will use three distinct LLM calls. The first generates FAQs. The second answers each question. The third identifies the sources. This modular design provides more control and simplifies debugging.

Here are the prompts for each step in our chain.
```python
# Prompts
prompt_generate_questions = """
Based on the content below, generate a list of 10 relevant and distinct questions that a user might have.
Return these questions as a JSON array of strings inside a "questions" key.

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
Now, we execute this prompt chain. We start by generating the questions, then iterate through each question to get an answer and identify its sources.

1.  **Generate Questions**
    First, we make an LLM call to generate a list of 10 relevant questions based on the combined content. We specify `response_format={"type": "json_object"}` to ensure the output is a structured JSON array.
    ```python
    # Step 1: Generate questions
    response_questions = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates questions based on text."},
            {"role": "user", "content": prompt_generate_questions.format(combined_content=combined_content)}
        ],
        response_format={"type": "json_object"}
    )

    generated_questions = json.loads(response_questions.choices[0].message.content)['questions']
    print(f"Successfully generated {len(generated_questions)} questions.")
    ```
2.  **Answer Questions and Find Sources**
    Next, we loop through each generated question. For every question, we perform two additional LLM calls. One generates a concise answer using only the provided content. The other identifies the source documents for that answer. This ensures each answer is grounded in the original text and properly attributed.
    ```python
    # Steps 2 & 3: Answer and find sources for each question
    final_faqs = []
    for question in generated_questions:
        print(f"  - Processing: '{question[:50]}...'\n")

        # Step 2: Generate an answer for the current question
        answer_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You answer questions based *only* on the provided context."},
                {"role": "user", "content": prompt_answer_question.format(question=question, combined_content=combined_content)}
            ]
        )
        answer = answer_response.choices[0].message.content

        # Step 3: Identify the sources for the generated answer
        sources_response = client.chat.completions.create(
            model="gpt-4o-mini",
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
    ```
3.  **Display Final Result**
    Finally, we print the complete list of generated FAQs, each with its answer and corresponding sources.
    ```python
    # Final result
    print("\nGenerated FAQ List:")
    print(json.dumps(final_faqs, indent=2))
    ```
**Output:**
```
Successfully generated 10 questions.
  - Processing: 'What are the environmental benefits of solar energy...'
  - Processing: 'What are the economic benefits for homeowners who ...'
  - Processing: 'How do wind turbines convert wind into electricit...'
  - Processing: 'What is the main challenge associated with wind en...'
  - Processing: 'Why is energy storage crucial for renewable energy...'
  - Processing: 'What are the most common forms of large-scale ene...'
  - Processing: 'How do lithium-ion batteries contribute to a stab...'
  - Processing: 'What is the financial viability of installing sol...'
  - Processing: 'Where are wind turbines typically installed and wh...'
  - Processing: 'How does energy storage improve the resilience and...'

Generated FAQ List:
[
  {
    "question": "What are the environmental benefits of solar energy?",
    "answer": "The main environmental benefits of solar energy are the reduction of reliance on fossil fuels and the cutting down of greenhouse gas emissions.",
    "sources": [
      "The Benefits of Solar Energy"
    ]
  },
  ...
]
```
This sequential approach yields a more reliable and well-structured output. Each step is focused and easier to manage. If source identification fails, you know exactly which prompt and API call to inspect. This modularity also allows for optimization; you can use a cheaper, faster model for simple tasks and reserve a more powerful model for complex generation.

💡 **Warning:** Prompt chaining is not a silver bullet. It introduces trade-offs, primarily latency and cost with each additional API call. You can also lose or distort information between steps. For example, an early summarization step might omit a detail that a subsequent step needs, leading to a less accurate final output. You must weigh the benefits of modularity against these potential downsides.

## Beyond Sequential Logic: Introducing Dynamic Routing

Sequential chains are powerful, but they operate on a fixed path where every input follows the same sequence of steps. What if your workflow needs to adapt dynamically based on the input it receives? This is where routing becomes essential. Routing introduces conditional logic, enabling your application to take different execution paths based on the content it processes.

Think of routing as adding `if/else` statements to your LLM workflow. Not all inputs should be treated equally; a customer asking for technical support requires a different response than someone with a billing question. Routing enables this by first classifying an input and then directing it to a specialized sub-workflow.

An LLM can even make the routing decision itself. You can design a dedicated LLM call to classify an input into a predefined category. This classification then becomes the condition in your Python code that determines the next step. This creates a more intelligent and adaptable application, where the system can dynamically select the most appropriate path or model based on the LLM's analysis [2](https://www.anthropic.com/engineering/building-effective-agents), [3](https://platform.openai.com/docs/guides/chat-completions), [4](https://github.com/openai/openai-python).

## Building a Basic Routing Workflow: Classify Intent and Route

Let's build a simple routing workflow for a customer service use case. This system first classifies a user's intent and then routes the query to a specialized prompt for a more context-aware response.

An LLM can make the routing decision by classifying an input into a predefined category. This classification becomes the condition in your Python code that determines the next step, allowing the LLM to steer its own path through the system [3](https://platform.openai.com/docs/guides/chat-completions). First, we'll create the classifier. It will take a user query and categorize it as "Technical Support," "Billing Inquiry," or "General Question."
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
        model="gpt-4o-mini",
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
With the intent classified, you can now use simple Python `if/elif/else` statements to route the query to the appropriate handler. Each handler uses a prompt tailored to that specific intent.
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

def handle_query(user_query, intent):
    """Routes a query to the correct handler based on its classified intent."""
    if intent == "Technical Support":
        prompt = prompt_technical_support.format(user_query=user_query)
    elif intent == "Billing Inquiry":
        prompt = prompt_billing_inquiry.format(user_query=user_query)
    else:
        # Default to a general response for "General Question" or any unhandled intent
        prompt = prompt_general_question.format(user_query=user_query)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
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
Response: I'm sorry to hear your internet connection isn't working. To help you better, could you tell me if you've already tried any troubleshooting steps, like restarting your modem and router?

Query: I think there is a mistake on my last invoice.
Intent: Billing Inquiry
Response: I understand your concern regarding your last invoice. To look into this for you, could you please provide your account number?

Query: What are your opening hours?
Intent: General Question
Response: I'm sorry, I'm not sure how to help with that. Could you please rephrase your question?
```
This approach is far more robust than a single prompt. Each branch can be developed and tested independently. However, the success of the entire system hinges on the accuracy of the initial classification step. You must design a robust classifier and have a fallback plan for ambiguous or out-of-scope intents.

## Advanced Workflows: The Orchestrator-Worker Pattern

For highly complex and unpredictable tasks, we can use a more advanced pattern: the orchestrator-worker. In this workflow, a central orchestrator LLM dynamically breaks down a user's query into multiple sub-tasks. It then delegates these sub-tasks to specialized worker functions, which can run concurrently.

Finally, a synthesizer LLM gathers the results from the workers and composes a single, cohesive response. This pattern excels in situations where you cannot predefine the workflow, such as handling a multi-part customer query that requires several different actions. We will implement this for our customer service example [2](https://www.anthropic.com/engineering/building-effective-agents), [5](https://langchain-ai.github.io/langgraphjs/tutorials/workflows/#routing).

First, the orchestrator's job is to deconstruct the user's query into a structured list of tasks. We use `asyncio` to handle concurrent operations.
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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_orchestrator.format(complex_query=complex_query)}]
    )
    tasks_str = response.choices[0].message.content
    try:
        # A more robust solution would use structured output (e.g. JSON mode),
        # but for now we parse the string.
        return json.loads(tasks_str)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from orchestrator: {e}")
        print(f"Orchestrator raw response: {tasks_str}")
        return []
```
Next, we define our worker functions. Each worker is an `async` function responsible for a specific task, like handling a billing inquiry, a product return, or an order status update. For this example, they simulate backend actions and return structured data.

Here is the worker for handling billing inquiries. It extracts the specific concern from the user's query and simulates an investigation.
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
    """Handles a billing inquiry by extracting the concern and simulating an investigation."""
    extraction_prompt = prompt_billing_worker_extractor.format(
        invoice_number=invoice_number,
        original_user_query=original_user_query
    )
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extraction_prompt}]
    )
    extracted_concern = response.choices[0].message.content.strip()

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
This worker handles product return requests by generating a Return Merchandise Authorization (RMA) number and providing shipping instructions.
```python
# Worker for Product Return
async def handle_return_worker(product_name, reason_for_return):
    """Handles a product return request by simulating RMA generation."""
    rma_number = f"RMA-{random.randint(10000, 99999)}"
    shipping_instructions = (
        f"Please pack the '{product_name}' securely. "
        f"Write the RMA number ({rma_number}) clearly on the outside of the package. "
        "Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."
    )
    print(f"  [Return Worker] Action: Generated RMA {rma_number} for {product_name} (Reason: {reason_for_return})")

    return {
        "task": "Product Return",
        "product_name": product_name,
        "reason_for_return": reason_for_return,
        "rma_number": rma_number,
        "shipping_instructions": shipping_instructions
    }
```
Finally, the status update worker simulates fetching the current status of an order, including carrier and tracking details.
```python
# Worker for Status Update
async def handle_status_worker(order_id):
    """Handles an order status update request by simulating a status fetch."""
    possible_statuses = [
        {"status": "Shipped", "carrier": "SuperFast Shipping", "tracking": f"SF{random.randint(100000,999999)}", "delivery_estimate": "Tomorrow"},
        {"status": "Delivered", "carrier": "Local Courier", "tracking": f"LC{random.randint(10000,99999)}", "delivery_estimate": "Delivered yesterday"},
    ]
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
The synthesizer takes the outputs from all the workers and crafts a final, user-friendly response.
```python
# Synthesizer
prompt_synthesizer = """
You are a master communicator. Combine several distinct pieces of information from our support team into a single, well-formatted, and friendly email to a customer.

Here are the points to include, based on the actions taken for their query:
---
{formatted_results}
---

Combine these points into one cohesive response. Start with a friendly greeting and end with a polite closing.
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
        elif res['task'] == 'Product Return':
            point += f"  - Product: {res['product_name']}\n"
            point += f"  - Reason for Return: \"{res['reason_for_return']}\"\n"
            point += f"  - Return Authorization (RMA): {res['rma_number']}\n"
            point += f"  - Instructions: {res['shipping_instructions']}"
        elif res['task'] == 'Status Update':
            point += f"  - Order ID: {res['order_id']}\n"
            point += f"  - Current Status: {res['current_status']}\n"
            point += f"  - Carrier: {res['carrier']}\n"
            point += f"  - Tracking Number: {res['tracking_number']}\n"
            point += f"  - Delivery Estimate: {res['expected_delivery']}"
        bullet_points.append(point)

    formatted_results = "\n\n".join(bullet_points)
    prompt = prompt_synthesizer.format(formatted_results=formatted_results)
    response = await aclient.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```
Finally, we tie it all together. A single function orchestrates the entire process: deconstruct, delegate, and synthesize.
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

    # 2. Run workers concurrently
    worker_coroutines = []
    print(f"\nDispatching {len(tasks_list)} workers...")
    for task in tasks_list:
        if task['query_type'] == 'BillingInquiry':
            worker_coroutines.append(handle_billing_worker(task['invoice_number'], user_query))
        elif task['query_type'] == 'ProductReturn':
            reason = task.get('reason_for_return', 'Not specified')
            worker_coroutines.append(handle_return_worker(task['product_name'], reason))
        elif task['query_type'] == 'StatusUpdate':
            worker_coroutines.append(handle_status_worker(task['order_id']))
    
    worker_results = await asyncio.gather(*worker_coroutines)
    print("\nWorkers finished their jobs.")

    # 3. Run synthesizer
    if worker_results:
        print("\nSynthesizing final response...")
        final_user_message = await synthesizer(worker_results)
        print("\n--- Final Synthesized Response ---")
        print(final_user_message)
        print("---------------------------------")

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
  [Billing Worker] Action: Investigating invoice INV-7890 for concern: 'The invoice seems higher than expected.'
  [Return Worker] Action: Generated RMA RMA-90033 for SuperWidget 5000 (Reason: not compatible with my system)
  [Status Worker] Action: Fetched status for order A-12345: Delivered

Workers finished their jobs.

Synthesizing final response...

--- Final Synthesized Response ---
Hi there,

Thank you for reaching out. Here is an update on your requests:

Regarding your Billing Inquiry:
  - Invoice Number: INV-7890
  - Your Stated Concern: "The invoice seems higher than expected."
  - Our Action: An investigation (Case ID: INV_CASE_5076) has been opened regarding your concern.
  - Expected Resolution: We will get back to you within 2 business days.

Regarding your Product Return:
  - Product: SuperWidget 5000
  - Reason for Return: "not compatible with my system"
  - Return Authorization (RMA): RMA-90033
  - Instructions: Please pack the 'SuperWidget 5000' securely. Write the RMA number (RMA-90033) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765.

Regarding your Status Update:
  - Order ID: A-12345
  - Current Status: Delivered
  - Carrier: Local Courier
  - Tracking Number: LC69724
  - Delivery Estimate: Delivered yesterday

If you have any more questions, feel free to ask.

Best regards,
The Support Team
---------------------------------
```
This pattern provides immense flexibility and power. The orchestrator can dynamically adapt to any combination of user requests, and the concurrent workers ensure efficiency. The main drawback is complexity; this is a more involved architecture and requires careful design of the orchestrator, workers, and the data structures they share.

## Conclusion: From Simple Scripts to Sophisticated AI Systems

We have moved beyond the flawed idea of a single, monolithic prompt to embrace robust, modular patterns. Mastering sequential chaining, dynamic routing, and the orchestrator-worker pattern provides the fundamental building blocks for any serious AI engineer. These aren't just clever tricks; they are core software engineering principles directly applied to LLM development.

By breaking down complex problems, you gain more than just a working application. You gain accuracy because each component does one thing well. You gain debuggability because you can isolate failures to a specific step. And you gain flexibility because you can update, swap, or optimize individual parts of your system without tearing the whole thing down. This is how you move beyond simple scripts and start engineering sophisticated, reliable AI systems that are truly ready for production.