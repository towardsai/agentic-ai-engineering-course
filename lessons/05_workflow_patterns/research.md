# Research

## Research Results

<details>
<summary>What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?</summary>

### Source [1]: https://arxiv.org/pdf/2309.08181

Query: What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?

Answer: - The paper investigates using LLMs for failure mode classification and documents several failure patterns when issuing single-shot prompts without fine-tuning. It finds that unconstrained prompting leads to output label inconsistency (e.g., “Fail to function” vs. “Failure to function”), undermining downstream usability due to lack of a stable ontology.[1]
- Even when constraining the label space by appending a list of valid labels to the prompt, the model still hallucinates non-existent labels (e.g., “Fail to open”) and continues to produce near-duplicate or variant labels, indicating brittle adherence to instructions and schema drift.[1]
- The study concludes that prompt-only approaches (single-prompt LLM calls) can yield superficially plausible but inconsistent, hallucinated, and ontology-mismatched outputs, limiting reliability for structured tasks without additional controls or fine-tuning.[1]

-----

-----

### Source [2]: https://www.getambassador.io/blog/prompt-engineering-for-llms

Query: What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?

Answer: - Identifies common prompt-level failure modes in production: **ambiguous instructions** lead to inconsistent or incorrect outputs; important instructions placed deep in long prompts may be missed, a manifestation related to “lost in the middle.”[2]
- Describes **overstuffed context windows**: when token limits are exceeded or approached, inputs may be truncated, causing lost context and incoherent or incomplete outputs—an explicit context window limitation.[2]
- Notes **high sensitivity to minor input changes**: small phrasing or ordering differences can cause dramatically different outputs due to probabilistic decoding, harming reproducibility and downstream integrations.[2]
- Recommends mitigation patterns (few-shot examples, formatting hints, schema-driven instructions) to reduce these failure rates, implying that single-prompt calls lacking such scaffolding are especially vulnerable.[2]

-----

-----

### Source [3]: https://proceedings.neurips.cc/paper_files/paper/2023/file/5d570ed1708bbe19cb60f7a7aff60575-Paper-Conference.pdf

Query: What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?

Answer: - Presents MULTIMON for mass-producing failures in multimodal systems using language models; shows that LLMs can reliably enumerate systematic failure categories from examples within a single prompt, highlighting that models consistently output lists of failure patterns under a fixed instruction.[3]
- Example failure types captured include systematic semantic confusions such as **negation handling** and **temporal differences**, indicating that single-prompt analyses can expose model tendencies to conflate semantically distinct inputs.[3]
- Emphasizes stochasticity: querying with the same prompt multiple times yields varied lists, underscoring instability and variability of single-prompt outputs and the need for multiple runs to surface diverse failure modes.[3]
- Notes the constraint that prompts must fit within the **model’s context window**, implicitly acknowledging that exceeding context limits prevents the method from operating, a practical context-window failure mode consideration.[3]

-----

-----

### Source [4]: https://www.york.ac.uk/assuring-autonomy/news/blog/part-one-using-large-language-models/

Query: What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?

Answer: - Discusses building FLAGPT for failure logic analysis and reports practical issues tied to LLM output control in single-prompt workflows: necessity to specify level of detail and chain-of-thought to avoid unhelpful or incorrect outputs.[4]
- Describes switching output targets (from Visio code to LaTeX TiKZ) to reduce complexity and mitigate **hallucinations due to output token length**, connecting longer outputs to higher error rates in generation—an output-length-related failure pattern in single-pass generations.[4]
- Highlights that careful instruction design and simplified schemas reduce error surface area, implying that complex prompts with extensive requirements increase the likelihood of hallucination and format deviation in one-shot calls.[4]

-----

-----

### Source [5]: https://arxiv.org/html/2410.23884v1

Query: What are the common failure modes of complex, single-prompt LLM calls, such as "lost in the middle" or context window limitations?

Answer: - Studies LLMs’ causal reasoning on narratives and identifies precise failure modes relevant to single-prompt comprehension: reliance on **topological order heuristics** (assuming earlier events cause later ones), leading to errors when narration order differs from causal order.[5]
- Finds failures in **long-term causal reasoning**: performance drops on long narratives with many events, consistent with “lost in the middle” effects where mid-sequence information is under-attended and multi-hop dependencies degrade.[5]
- Shows models **over-rely on parametric knowledge** over provided context; when the narrative contradicts prior knowledge, models err—an instruction-following/context-grounding failure common in single-prompt settings.[5]
- Reports that explicit intermediate structure (e.g., generating a **causal graph**) improves performance, while naive chain-of-thought does not reliably fix these issues, suggesting that unstructured, single-prompt reasoning is fragile for complex inputs.[5]

-----

</details>

<details>
<summary>What are the trade-offs and disadvantages of prompt chaining, such as increased latency, cost, and potential for information loss between steps?</summary>

### Source [6]: https://aisdr.com/blog/what-is-prompt-chaining/

Query: What are the trade-offs and disadvantages of prompt chaining, such as increased latency, cost, and potential for information loss between steps?

Answer: - Lists explicit drawbacks of prompt chaining:  
  - **Management difficulty**: Coordinating “a series of interrelated prompts” becomes challenging as chains grow long and intricate, increasing room for error, especially when multiple models are involved.[1]  
  - **Hyper-dependency on the first prompt**: Chain quality relies on earlier steps; a flawed initial prompt can trigger “a cascade of failures” that compound downstream.[1]  
  - **Time-consuming**: Prompt chaining amplifies the iterative nature of prompting; you must repeatedly test each link, and issues become harder to locate and debug across steps.[1]  
- Implicit trade-offs tied to latency and cost: more steps and repeated testing increase overall execution time and operational overhead, even though the article primarily frames these as effort/management burdens.[1]  
- Notes a potential mitigation for cost via caching repeated chains (can “reduce your LLM costs up to 90%”), implying that without such caching, chains can be more expensive to run repeatedly.[1]

-----

-----

### Source [7]: https://www.humanfirst.ai/blog/prompt-chaining

Query: What are the trade-offs and disadvantages of prompt chaining, such as increased latency, cost, and potential for information loss between steps?

Answer: - Highlights that chaining requires robust **data transformation between steps** because LLM outputs are often unstructured; without proper structuring, downstream steps may misinterpret or lose information, leading to degraded results.[2]  
- Notes the **slight unpredictable nature of LLMs**, where a model can produce multiple valid but different responses to the same prompt; in chains this can cause **cascading undesired responses**, propagating unexpected data through later steps—an information integrity risk.[2]  
- Indicates real-world applications involve **complex and parallel multi-step tasks**, which challenge single-chain designs, increasing system complexity and maintenance overhead.[2]  
- The need for run-time prompt customization without model training is presented as a benefit, but it also implies operational complexity in managing variable prompts across steps, which can introduce inconsistency and debugging difficulty in chains.[2]

-----

-----

### Source [8]: https://blog.promptlayer.com/what-is-prompt-chaining/

Query: What are the trade-offs and disadvantages of prompt chaining, such as increased latency, cost, and potential for information loss between steps?

Answer: - Identifies disadvantages (summary points):  
  - **Increased complexity**: Managing multiple interconnected prompts adds design and maintenance burden compared to single-shot prompting.[3]  
  - **Latency and cost**: Although not expanded in detail, more steps imply additional model calls, which generally increases total response time and usage cost in chained workflows.[3]  
- Motivations for chaining (context): overcoming context length and reducing hallucinations via stepwise control; however, these benefits come with the above complexity and operational trade-offs.[3]  
- Error isolation is easier in theory, but operationalizing this still requires instrumentation and evaluation across steps, which contributes to development and maintenance overhead.[3]

-----

-----

### Source [9]: https://ai.plainenglish.io/prompt-chaining-is-dead-long-live-prompt-stuffing-58a1c08820c5

Query: What are the trade-offs and disadvantages of prompt chaining, such as increased latency, cost, and potential for information loss between steps?

Answer: - Argues that prompt chaining had early value but came with **maintenance difficulties** and required **glue code** to connect steps, increasing engineering complexity.[4]  
- With modern large context windows, the article claims it can be preferable to consolidate into single prompts (“prompt stuffing”), implicitly to reduce multi-step **latency**, **coordination overhead**, and potential **information loss** between steps.[4]  
- Notes that changing practices can trigger **refactors with cost implications**, implying chained systems risk higher ongoing maintenance and potential cost “explosions” if not updated to new capabilities.[4]  
- While opinionated, it explicitly contrasts chaining vs. single-shot in the context of building complex JSON, suggesting that chaining’s step boundaries can introduce fragility and overhead compared to larger-context single prompts.[4]

-----

</details>

<details>
<summary>How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?</summary>

### Source [11]: https://platform.openai.com/docs/guides/rate-limits

Query: How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?

Answer: - OpenAI enforces per-minute and per-day rate limits and may return 429 errors when limits are exceeded or during transient capacity issues. Clients should implement retries with exponential backoff and jitter, and respect the Retry-After header when present.[1]
- Recommended retry policy: use a limited number of retries (e.g., 5), exponential backoff starting around a few hundred milliseconds, add randomized jitter, and fail fast for non-retryable errors (e.g., 4xx other than 429).[1]
- Handle concurrency by queueing or batching requests to keep within tokens-per-minute (TPM) and requests-per-minute (RPM) quotas; tune parallelism dynamically based on current error rates and headers.[1]
- Inspect response headers for rate limit state (e.g., limit and remaining) to adjust throughput; throttle proactively as you approach limits to avoid bursts that trigger 429s.[1]
- Different models and accounts have distinct limits; production systems should read limits at startup and adapt, including per-model routing if some models have higher quotas.[1]
- For streaming responses, count tokens as they are generated to avoid exceeding TPM; cancel or pause upstream producers if downstream backpressure appears.[1]
- Use idempotency keys for safe retries of create-like operations to prevent duplicates when network or 5xx errors lead to replay.[1]
- Monitor and alert on 429s, timeouts, and 5xx; log request IDs to correlate with support when investigating rate-limit events.[1]

-----

-----

### Source [12]: https://cloud.google.com/vertex-ai/generative-ai/docs/quotas

Query: How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?

Answer: - Vertex AI applies quotas for requests-per-minute, requests-per-day, and tokens-per-minute; exceeding quotas results in 429 Too Many Requests. Clients should implement exponential backoff with jitter and respect Retry-After for smooth recovery.[2]
- Guidance: reduce request concurrency when receiving 429s; scale back parallel calls dynamically rather than retrying immediately at full rate.[2]
- Use client-side flow control: cap the number of in-flight requests per model/region, and implement token budgeting to remain within TPM limits when making parallel LLM calls.[2]
- Batch requests or use streaming when appropriate to lower request counts while meeting latency goals.[2]
- Monitor quota usage via Cloud Monitoring; set alerts near 80–90% utilization to preempt errors, and request quota increases if sustained demand requires it.[2]
- Treat 5xx as transient and retry with backoff; do not retry 4xx other than 429. Implement per-try timeouts and overall deadlines to avoid retry storms.[2]
- For long-running operations, use idempotency and deduplication on the client to avoid double work when retries occur.[2]

-----

-----

### Source [13]: https://learn.microsoft.com/azure/ai-services/openai/quotas-limits

Query: How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?

Answer: - Azure OpenAI enforces per-deployment rate limits measured in RPM and TPM; exceeding them yields 429. Applications should implement exponential backoff and gradually reduce concurrency when 429 frequency increases.[3]
- Best practices: distribute load across multiple deployments of the same model, perform client-side queueing, and pace requests to honor TPM; prefer larger batch sizes rather than many small parallel calls when feasible.[3]
- Error handling: retry on 429 and 5xx with bounded retries and jitter; do not retry client errors such as 400/401/403. Observe Retry-After and X-RateLimit headers to calibrate throughput.[3]
- Use asynchronous patterns and circuit breakers. Open the circuit when consecutive failures rise, then probe with limited traffic before resuming normal rates.[3]
- Monitor X-RateLimit-Limit and X-RateLimit-Remaining to adapt in real time; when remaining drops, slow producers to prevent bursts causing throttling.[3]
- For streaming completions, enforce per-connection limits and propagate backpressure to upstream tasks to avoid cascading 429s.[3]

-----

-----

### Source [14]: https://platform.openai.com/docs/guides/error-codes

Query: How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?

Answer: - 429 Too Many Requests occurs when you exceed rate limits or the system is temporarily overloaded; resolution is to back off and retry respecting Retry-After and using exponential backoff with jitter.[4]
- 5xx errors (e.g., 500, 502, 503) indicate transient server issues; safe to retry with backoff. Implement a maximum retry count and overall timeout to avoid infinite retries.[4]
- 4xx errors (e.g., 400, 401, 403, 404) are client errors; correct the request rather than retrying. Only 409/429 may be retried after changes or delay.[4]
- Timeouts and network errors should be treated as transient; use idempotency keys for create operations to avoid duplication on retry.[4]
- Include request IDs from response headers in logs to aid debugging and support escalation.[4]

-----

-----

### Source [15]: https://docs.anthropic.com/en/docs/build-with-claude/rate-limits

Query: How to handle API rate limiting and errors when implementing parallel LLM calls in a production environment?

Answer: - Anthropic enforces per-minute request and token limits; exceeding limits results in 429s. Implement exponential backoff with randomized jitter and honor Retry-After when provided.[5]
- Recommended client strategies: limit concurrent requests, use token budgeting to stay within TPM, and adjust concurrency dynamically based on recent 429 rates.[5]
- Streaming responses still count toward token quotas; manage producers to avoid exceeding TPM mid-stream and consider chunking large tasks if needed.[5]
- Use idempotency keys for safe retries on message creation; this prevents duplicate outputs when a retry succeeds after a partial failure.[5]
- Monitor rate limit headers (e.g., anthropic-ratelimit-*) to observe remaining capacity and throttle proactively.[5]
- Avoid unbounded parallelism; prefer work queues with worker pools sized by measured sustainable throughput for your account.[5]

### : https://docs.mistral.ai/platform/limits/
- Mistral applies RPM and TPM limits per account and model; exceeding them returns 429. Clients should implement exponential backoff, respect Retry-After, and reduce concurrency when limits are approached.
- Guidance includes using rate-limit headers to adapt, batching where possible, and monitoring 429s as signals to downshift throughput.
- Treat 5xx as transient with retries; do not retry other 4xx. Use idempotency keys to make retries safe for non-idempotent endpoints.
- For high-throughput workloads, request quota increases and distribute calls across regions or models where supported.

-----

</details>

<details>
<summary>What are best practices for using a large language model to perform intent classification for routing workflows in customer service?</summary>

### Source [16]: https://rasa.com/docs/rasa/next/llms/llm-intent/

Query: What are best practices for using a large language model to perform intent classification for routing workflows in customer service?

Answer: - Use an LLM-based intent classifier when you need few-shot learning, fast iteration, or multilingual coverage; LLM classifiers can work with only a handful of examples per intent and are quick to train, making it easier to bootstrap and update routing workflows as new intents emerge. This supports rapid deployment and maintenance of customer service routers across languages and channels.[5]
- Treat LLM-based intent classification as an experimental component to be evaluated and monitored in production; performance varies by model and language, so continuous improvement and testing are essential before scaling to high-stakes workflows.[5]
- Combine LLMs with traditional NLU components where appropriate; leverage LLMs for few-shot adaptability while maintaining deterministic rules or statistical classifiers for high-volume, well-defined intents to balance robustness and cost.[5]
- Prepare high-quality examples for each intent; even with few-shot capability, representative examples remain critical for accuracy, especially for edge cases and similar intents common in customer service routing (e.g., billing vs. refund).[5]
- Plan for multilingual routing; train on multilingual data to allow a single classifier to route across locales, but validate per language due to model-dependent variance.[5]
- Optimize for speed and cost at orchestration time; because training is fast and lightweight, prioritize runtime efficiency (e.g., batching, caching) and choose LLMs that meet latency constraints for real-time routing.[5]

-----

-----

### Source [17]: https://www.voiceflow.com/pathways/5-tips-to-optimize-your-llm-intent-classification-prompts

Query: What are best practices for using a large language model to perform intent classification for routing workflows in customer service?

Answer: - Use a two-stage architecture for classification quality and efficiency: first retrieve top-N candidate intents via an encoder model using intent names and descriptions, then prompt an LLM to select the best match from those candidates. This narrows the decision space and improves routing precision in customer service flows.[4]
- Improve prompt inputs modestly by tuning descriptions: adding prefixes/suffixes to intent descriptions, including a dedicated None_intent description, and allowing AI-generated descriptions can yield small but measurable gains. Use temperature near 0.1 to reduce variance in outcomes.[4]
- Allocate effort where it matters most: larger accuracy gains come from better training examples, few-shot examples, and thorough handling of edge cases rather than minor wording tweaks. Invest in structured formatting of prompts, which has shown larger impact on accuracy.[4]
- Operationalize evaluation: benchmark across multiple LLMs and datasets, run multiple trials to quantify variance, and keep classification settings consistent to measure improvements reliably—practices that translate directly to production routing scenarios.[4]

-----

-----

### Source [18]: https://developer.vonage.com/en/blog/how-to-build-an-intent-classification-hierarchy

Query: What are best practices for using a large language model to perform intent classification for routing workflows in customer service?

Answer: - Use hierarchical intent classification to manage large intent sets typical in customer service; classify in stages by the biggest differentiators (e.g., product vs. account vs. troubleshooting) to reduce overlap and ambiguity, then refine to more granular intents. This improves routing by focusing the LLM’s decision on one variable at a time.[2]
- Sanitize intents to avoid overlap and keep training sets consistent; ensure similar phrases that can belong to multiple intents (e.g., “want to speak with”) are proportionally represented to reduce confusion in downstream stages.[2]
- Grouping strategies matter: group intents by major nouns or verbs depending on which maximally separates your customer queries in the first stage, then subdivide further. This staged design reduces misroutes in complex workflows.[2]

-----

-----

### Source [19]: https://arxiv.org/html/2402.02136v2

Query: What are best practices for using a large language model to perform intent classification for routing workflows in customer service?

Answer: - Define a clear, precise, and comprehensive intent taxonomy to guide LLM classification; a well-structured, hierarchical taxonomy covering the breadth of user intents improves accurate identification and downstream response quality, which is crucial for routing workflows.[3]
- Ensure consistency and coherence of categories; precise, consistent categorization helps LLMs map ambiguous customer language to the correct routing intent, reducing dissatisfaction caused by misinterpretation.[3]
- Design for versatility across applications; an intent taxonomy applicable across domains and contexts supports scalable customer service routers that handle technical, transactional, and conversational intents consistently.[3]

-----

-----

### Source [20]: https://spotintelligence.com/2023/11/03/intent-classification-nlp/

Query: What are best practices for using a large language model to perform intent classification for routing workflows in customer service?

Answer: - Build a high-quality, representative dataset for intent classification: collect customer queries from relevant channels (support chats, voice transcripts), and annotate with clear labeling guidelines to ensure consistency—foundational for reliable LLM routing.[1]
- Ensure diversity and coverage to reduce bias and improve real-world performance: include a range of demographics, phrasing styles, and outlier cases so the classifier handles unexpected inputs and edge scenarios common in customer service.[1]
- Use data augmentation judiciously to expand coverage of rare intents or edge phrasing, improving generalization for robust routing under noisy or varied inputs.[1]

-----

</details>

<details>
<summary>What are the key differences between a pre-defined parallel workflow and a dynamic orchestrator-worker pattern for LLM task decomposition?</summary>

### Source [21]: https://www.anthropic.com/research/building-effective-agents

Query: What are the key differences between a pre-defined parallel workflow and a dynamic orchestrator-worker pattern for LLM task decomposition?

Answer: - Definition and scope: Anthropic distinguishes between predefined code-path “workflows” and more flexible “agents.” In a predefined workflow, LLMs and tools are orchestrated through fixed logic; in an orchestrator-workers agent workflow, a central LLM dynamically breaks down tasks, delegates to worker LLMs, and synthesizes results[2].  
- Key difference vs parallelization: Parallelization executes a known set of independent subtasks concurrently; orchestrator-workers is flexible because subtasks are not pre-defined but are determined by the orchestrator based on the specific input[2].  
- When to use parallelization vs orchestrator-workers: Orchestrator-workers is recommended when you cannot predict the subtasks needed (e.g., coding tasks where the number and nature of file changes depend on the input) or search tasks that require gathering and analyzing information from multiple sources with variable scope[2].  
- Practical implication: The orchestrator-workers pattern adapts the number, type, and sequencing of subtasks at run time, whereas pre-defined parallel workflows presume a known decomposition and fixed fan-out, making them less suitable for inputs with highly variable structure and requirements[2].

-----

-----

### Source [22]: https://huggingface.co/blog/dcarpintero/design-patterns-for-building-agentic-workflows

Query: What are the key differences between a pre-defined parallel workflow and a dynamic orchestrator-worker pattern for LLM task decomposition?

Answer: - Orchestrator-workers architecture: A central LLM “Orchestrator” performs real-time task analysis to determine necessary subtasks, selects appropriate specialized workers, provides task-specific objectives and output formats, and then synthesizes worker outputs into a coherent response[1].  
- Dynamic decomposition vs pre-defined flows: Unlike pre-defined workflows, the orchestrator decides the decomposition based on task complexity, available resources, and optimal delegation; it does not follow a static set of parallel branches[1].  
- Use cases favoring dynamic orchestration: Business intelligence (variable query complexity, multi-source synthesis) and software development (deciding which files to modify; coordinating code generation, testing, debugging, documentation) benefit from dynamic planning and delegation[1].  
- Operational considerations: Orchestrator can become a bottleneck and single point of failure; production systems may require context compression, external memory, control loops, robust error handling, and observability to manage many turns and agents—complexities less pronounced in simple pre-defined parallel flows[1].  
- Evaluation: Multi-agent, dynamically decomposed systems need specialized evaluation approaches (e.g., small-scale tests, LLM-as-a-judge), reflecting higher variability than deterministic parallel workflows[1].

-----

-----

### Source [23]: https://fme.safe.com/guides/ai-agent-architecture/ai-agentic-workflows/

Query: What are the key differences between a pre-defined parallel workflow and a dynamic orchestrator-worker pattern for LLM task decomposition?

Answer: - Pre-defined parallelization workflow: Suitable when several subtasks are independent and known up front; agents work simultaneously on multiple predefined subtasks and their outputs are collated at the end. Example: processing multiple research papers in parallel to extract specific information, then aggregating findings[3].  
- Orchestrator-worker workflow: Used when dynamic decision-making prevents defining flow logic at initialization. An orchestrator agent creates a plan, determines execution sequence, and delegates tasks to helper agents. The number of workers is dynamic and depends on orchestrator decisions; in contrast, the parallelization pattern’s path is pre-defined[3].  
- Comparative essence: Parallelization assumes a fixed decomposition and concurrency plan; orchestrator-worker adapts both decomposition and degree of parallelism during execution based on input and intermediate results[3].  
- Example illustrating dynamism: A travel booking agent chooses among flights, trains, rental cars based on the specific route, then delegates to sub-agents to check availability and fares before collating results—demonstrating dynamic branching and worker allocation absent in a pre-defined parallel fan-out[3].

-----

-----

### Source [24]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: What are the key differences between a pre-defined parallel workflow and a dynamic orchestrator-worker pattern for LLM task decomposition?

Answer: - Parallelization pattern characteristics: Emphasizes efficient concurrent processing of multiple LLM operations with automated aggregation. Best when you can define parallel tasks up front (e.g., analyzing multiple stakeholder groups simultaneously) and aggregate results programmatically[4].  
- Routing vs orchestration context: Spring’s “Routing” pattern shows deterministic input analysis to select specialized handlers—illustrating fixed control logic compared to the dynamic planning and delegation of an orchestrator-workers agent pattern. This underscores that pre-defined workflows rely on pre-specified routes/branches rather than run-time task decomposition[4].  
- Practical takeaway: In pre-defined parallel workflows, the developer specifies the fan-out and aggregation strategies in code; execution is predictable and easier to test. Orchestrator-worker patterns require an LLM to decide decomposition and sequencing at run time, trading predictability for adaptability[4].

-----

</details>

<details>
<summary>What are the primary benefits of prompt chaining for improving LLM performance, specifically regarding modularity, debuggability, and the ability to use different models for different steps?</summary>

### Source [25]: https://www.vellum.ai/blog/what-is-prompt-chaining

Query: What are the primary benefits of prompt chaining for improving LLM performance, specifically regarding modularity, debuggability, and the ability to use different models for different steps?

Answer: - Prompt chaining breaks a complex task into smaller, linked steps, where each prompt’s output feeds the next, improving overall LLM performance.[2]
- Benefits tied to modularity: By decomposing tasks into discrete prompts, each subtask is isolated, making the system easier to control and test at the step level (i.e., controllability).[2]
- Benefits tied to debuggability: It’s easier to debug and test each step of the “chain,” and if something fails, you can locate the error and remedy it at the specific step that produced the issue.[2]
- Benefits tied to using different models per step: You can trade cost/latency for quality by assigning cheaper/faster models (e.g., GPT‑3.5‑turbo, Claude 3 Haiku) to simpler steps while reserving stronger models for harder steps, reducing overall cost without sacrificing quality.[2]
- Additional reliability benefit: Chaining raises reliability since you can validate intermediate outputs and recover from failures at specific steps rather than rerunning the entire pipeline.[2]

-----

-----

### Source [26]: https://dev.to/kapusto/enhancing-large-language-model-performance-with-prompt-chaining-2p84

Query: What are the primary benefits of prompt chaining for improving LLM performance, specifically regarding modularity, debuggability, and the ability to use different models for different steps?

Answer: - Prompt chaining is a systematic approach that breaks down complex tasks into smaller, manageable sequences, helping maintain context and guide the model to more accurate and relevant responses.[1]
- Modularity: By segmenting the interaction into distinct steps, each prompt becomes a module whose inputs/outputs are explicit, which supports structured workflows and clearer boundaries between tasks.[1]
- Debuggability: The modular segmentation makes it straightforward to isolate and fix problematic prompts, reducing debugging time and improving maintenance.[1]
- Model selection across steps: Although not explicitly prescribing model swapping, the description of chaining as connected, focused prompts implies the flexibility to assign different LLMs (or configurations) per step, because each step is independently specified and consumed; this modularity supports substituting models where appropriate (inference based on the described modular structure).[1]
- Risk reduction: Chaining helps prevent “context hallucinations” by keeping strict control of context across steps, which also supports easier verification of intermediate outputs during debugging.[1]

-----

-----

### Source [27]: https://www.voiceflow.com/blog/prompt-chaining

Query: What are the primary benefits of prompt chaining for improving LLM performance, specifically regarding modularity, debuggability, and the ability to use different models for different steps?

Answer: - Prompt chaining links multiple prompts where each output becomes the next input, enabling LLMs to handle complex tasks more effectively.[3]
- Modularity: Segmenting tasks ensures each part gets focused attention, aligning to a modular design where each prompt targets a specific subproblem.[3]
- Debuggability and explainability: Step-by-step outputs increase explainability and make it easier to trace how conclusions are reached, which supports diagnosing where an error occurred in the chain.[3]
- Context management: Increased context retention across steps maintains coherence, allowing validation at each stage and facilitating targeted adjustments without overhauling the entire flow.[3]
- Model flexibility: While the article does not explicitly state model-swapping, the structured, stepwise nature supports assigning different tools or models to different steps where they best fit (inference grounded in the stepwise architecture described).[3]

-----

-----

### Source [28]: https://blog.promptlayer.com/what-is-prompt-chaining/

Query: What are the primary benefits of prompt chaining for improving LLM performance, specifically regarding modularity, debuggability, and the ability to use different models for different steps?

Answer: - Prompt chaining divides complex tasks into interconnected prompts, guiding the LLM through a nuanced reasoning process for more accurate and comprehensive results.[5]
- Modularity: Segmenting into smaller, digestible pieces creates clear, independent steps that can be managed and improved separately.[5]
- Debuggability: It simplifies fault analysis by isolating problems to individual steps, making it easier to pinpoint and rectify errors and thereby increasing reliability.[5]
- Context handling: Helps overcome context length constraints and mitigates context hallucination by maintaining context through the chain, which also enables intermediate validation.[5]
- Model-per-step flexibility: While not explicitly prescribing different models per step, the modular, independent-step framing supports substituting different LLMs or configurations for specific steps based on cost/latency/quality needs (inference grounded in the modular description).[5]

-----

</details>

<details>
<summary>What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?</summary>

### Source [29]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-orchestration.html

Query: What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?

Answer: - The **central orchestrator** must plan, decompose, delegate, monitor, and synthesize across multiple specialized workers; this concentration of responsibilities can create a single‑point bottleneck if planning and progress monitoring are serialized or poorly parallelized, especially in complex, hierarchical workflows where subtasks vary in scope and reasoning type[4].  
- The pattern targets complex, multidisciplinary tasks with structured decomposition; misapplication to tasks that are not cleanly divisible can cause coordination overhead, excessive back‑and‑forth, and degraded throughput as the orchestrator struggles to modularize responsibilities[4].  
- Reliance on **role-based behavior** and **distinct capabilities/toolsets** across worker agents requires careful interface design; ambiguous roles or overlapping tool scopes increase failure risk in delegation and in later synthesis due to inconsistent assumptions and outputs[4].  
- In multiturn planning settings (for example, software development copilots), the orchestrator must maintain and update plans over time; inadequate progress tracking or state management leads to drift, duplicate work, or missed dependencies when synthesizing worker outputs[4].  
- Hierarchical or multilevel delegation adds coordination depth; without clear escalation and aggregation rules, synthesis errors can propagate upward, compounding inaccuracies in the final result[4].  
- While orchestration enables scalability and reuse, it requires explicit monitoring of worker progress and result quality; insufficient monitoring increases chances that low‑quality or incompatible worker outputs degrade the final synthesized answer[4].

-----

-----

### Source [30]: https://www.anthropic.com/research/building-effective-agents

Query: What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?

Answer: - The orchestrator dynamically determines subtasks at runtime rather than using predefined parallel tasks; this flexibility can introduce variability and planning errors, making it harder to guarantee coverage of all necessary subtasks and increasing the chance of omissions that surface during synthesis[5].  
- The workflow suits coding changes across multiple files and multi‑source search/analysis; in these domains, failure modes include incomplete decomposition (missing impacted files or sources), over‑decomposition (too many small subtasks causing overhead), or conflicting worker outputs that are difficult to reconcile during synthesis[5].  
- Because the orchestrator also synthesizes results, errors in aggregating heterogeneous worker outputs—such as inconsistent formats or contradictory findings—can lead to incorrect final responses if not paired with robust validation or a reviewer step[5].  
- Compared with simple parallelization, dynamic tasking increases coordination complexity; without safeguards, the orchestrator can become a throughput bottleneck as it iteratively plans and delegates based on evolving input and intermediate findings[5].

-----

-----

### Source [31]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?

Answer: - Implementation centers on a class that: (1) lets the orchestrator analyze the task and determine subtasks, (2) runs workers in parallel, and (3) combines results; practical issues include ensuring safe concurrency for parallel workers and correctly merging partial results into a coherent final response structure[3].  
- Clear boundaries between the central LLM and specialized workers are emphasized to maintain reliability; weak boundaries (unclear inputs/outputs per worker) increase coupling and raise the risk of synthesis errors and misrouted subtasks[3].  
- Using a common ChatClient abstraction for all LLM interactions highlights operational risks: inconsistent prompt schemas across orchestrator/workers can yield incompatible outputs that fail during the combination step or require expensive normalization passes[3].  
- The pattern is presented to “maintain control” over more complex agent behavior; lacking explicit evaluation or guardrails (e.g., schema validation of worker outputs before merge) can cause silent errors to slip into the final aggregation stage[3].

-----

-----

### Source [32]: https://javaaidev.com/docs/agentic-patterns/patterns/orchestrator-workers-workflow/

Query: What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?

Answer: - The orchestrator commonly uses a **reasoning model** (e.g., o1, o3‑mini) for planning, while workers use standard models (e.g., gpt‑4o, gpt‑4o‑mini); mismatched capabilities and costs can create a planning throughput bottleneck and higher latency at the orchestrator, as well as inconsistencies when workers’ outputs lack the reasoning depth expected by the synthesizer[1].  
- Effective implementation requires carefully crafted **prompt templates** and “agents as tools”; brittle templates or poorly defined tool registration can cause delegation mistakes and synthesis errors when outputs don’t align with the orchestrator’s expected structure[1].  
- In incident/outage handling, the orchestrator aggregates logs, metrics, traces, and code history; failure modes include noisy or incomplete data retrieval by workers, followed by incorrect causal synthesis when the orchestrator infers relationships from heterogeneous signals without sufficient validation[1].  
- While reasoning models simplify instruction complexity, overreliance on their planning can lead to under‑specified subtasks or insufficient guidance to workers, increasing rework and synthesis ambiguity in later stages[1].

-----

-----

### Source [33]: https://bootcamptoprod.com/spring-ai-orchestrator-workers-workflow-guide/

Query: What are the practical implementation challenges and failure modes of the orchestrator-worker pattern in LLM workflows, such as the orchestrator becoming a bottleneck or errors in synthesizing worker outputs?

Answer: - The pattern introduces three components—**Orchestrator**, **Workers**, **Synthesizer**—with the orchestrator deciding tasks at runtime; this dynamism raises coordination overhead and the risk of planning inaccuracies that surface when the synthesizer tries to assemble incomplete or overlapping worker outputs[2].  
- The synthesizer’s role to combine outputs into a single cohesive response is a distinct step; practical challenges include resolving conflicts between specialized outputs, normalizing formats, and ensuring that the final product reflects all critical subtasks, otherwise users receive fragmented or inconsistent results[2].  
- Contrasted with the Parallelization pattern, where tasks are predefined, orchestrator‑led decomposition may repeatedly adjust tasks; without guardrails, this can produce oscillation in subtask definitions and delays that accumulate at the synthesis step[2].  
- The manager–specialist analogy underscores the need for correct routing to “experts”; misrouting or poorly matched worker expertise leads to weak intermediate results that degrade the synthesizer’s ability to produce a high‑quality final answer[2].

-----

</details>

<details>
<summary>How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?</summary>

### Source [34]: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html

Query: How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?

Answer: AWS describes the routing workflow as a foundational pattern where a first-pass LLM acts as a classifier or dispatcher to interpret input intent or category and then route the request to a specialized downstream task, agent, tool, or workflow.[1] This enables quick triage across tasks (e.g., search, summarization, booking, calculations) and supports preprocessing or normalization before handing off to specialized flows—crucial building blocks for larger agent systems where correct task delegation drives quality and efficiency.[1] Routing supports modular expansion of capabilities and allows routes to invoke distinct workflows or even other agent patterns, making it a compositional primitive for multi-agent systems and composite architectures.[1] AWS highlights its use in domain-specific copilots, customer-support bots, enterprise service routers, and multimodal agents acting as conversational switchboards—roles directly analogous to the “controller” or “manager” agents used in multi-agent orchestration and ReAct-style tool selection.[1] The pattern also aligns with dynamic tool selection (e.g., choosing search vs. code generation) and decision trees enhanced by LLM reasoning, which underpin ReAct-like prompt-tool loops and hierarchical multi-agent task allocation.[1]

-----

-----

### Source [35]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?

Answer: Spring AI documents routing as intelligent task distribution: an LLM analyzes input content and routes to the most appropriate specialized prompt or handler, fitting complex tasks where different inputs are better handled by specialized processes.[4] By defining route-specific prompts (e.g., billing, technical, general) and invoking a RoutingWorkflow, developers compose modular expert behaviors—an essential precursor to multi-agent architectures where specialized agents handle subproblems and a router coordinates them.[4] The post also presents the chain workflow pattern—breaking complex tasks into simpler, manageable steps—capturing “prompt chaining” where outputs feed subsequent steps.[4] Chaining serves as a deterministic scaffold for multi-step reasoning and tool-use sequences seen in ReAct-style approaches, where the model iteratively observes tool results and proceeds stepwise. Together, routing (dispatch to the right specialist) and chaining (structured multi-step decomposition) provide the control flow primitives that can be composed into more advanced agent systems: routers select pathways/agents; chains implement the within-agent or within-route reasoning sequence.[4] The inclusion of parallelization further shows how these patterns aggregate multiple LLM operations—a capability often embedded in multi-agent systems to run specialists concurrently and combine results.[4]

-----

-----

### Source [36]: https://arize.com/docs/phoenix/learn

Query: How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?

Answer: Arize Phoenix frames workflows as the backbone of LLM apps, offering structure and predictability compared to fully autonomous agents—key for building reliable agentic systems.[5] It defines agent routing as directing a task or query to the most appropriate agent based on context, skills, domain expertise, or tools—core to multi-agent systems where specialization and capability-aware dispatch improve efficiency and accuracy.[5] It defines prompt chaining as breaking a complex task into multiple steps so the output of one prompt becomes input to the next, enabling multi-step reasoning and maintaining context across steps—capabilities integral to architectures like ReAct that simulate iterative thinking and tool-use loops.[5] By positioning routing and chaining as practical workflows used across agent frameworks, Phoenix emphasizes their role as composable building blocks: routing determines which agent or toolchain should act; chaining structures the agent’s internal reasoning/execution plan. These primitives underpin more advanced designs by enabling deterministic orchestration, dynamic agent selection, and context-carrying multi-step execution that can be extended into ReAct or broader multi-agent pipelines.[5]

-----

-----

### Source [37]: https://www.revanthquicklearn.com/post/understanding-workflow-design-patterns-in-ai-systems

Query: How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?

Answer: This overview attributes five workflow patterns to Anthropic and details prompt chaining and routing as core designs.[2] Prompt chaining decomposes a complex task into a fixed series of LLM calls where each call’s output feeds the next, allowing precise task framing—mirroring the stepwise reasoning and observation-action loops common to ReAct-like strategies when tools are inserted between steps.[2] Routing uses an LLM as a decision-maker to determine which specialized model should handle a task, enabling separation of concerns and decision autonomy while maintaining structured workflows—akin to controller agents dispatching to specialists in multi-agent systems.[2] Together, chaining (sequential decomposition) and routing (specialist selection) are presented as orthogonal primitives that, when combined, create scalable architectures: a router chooses the appropriate specialized chain (or agent), and the chain executes the multi-step plan—directly mapping to advanced agentic patterns that coordinate multiple experts and iterative reasoning.[2]

-----

-----

### Source [38]: https://mikulskibartosz.name/ai-workflow-design-patterns

Query: How do foundational workflow patterns like chaining and routing serve as the building blocks for more advanced AI agent architectures like ReAct or multi-agent systems?

Answer: This guide summarizes Anthropic-style routing and chaining with a concrete example: a routing agent classifies a question and selects a downstream workflow, while the downstream tasks can be implemented as separate prompt-chaining workflows.[3] It characterizes routing as classifying an input and directing it to a specialized follow-up task, enabling separation of concerns and specialized prompts—matching the dispatcher role in multi-agent orchestration.[3] The example shows a router deciding among HR, Tech Support, and Finance, returning a structured decision type that determines which branch (specialized workflow) to execute—illustrating how routing composes with chaining, where each branch is a chain tailored to the domain.[3] This compositional pattern mirrors how advanced agent architectures are built: a top-level router (or manager agent) selects a specialist agent/toolchain, and the specialist executes a chained sequence of reasoning/tool steps, aligning with ReAct-like iterative workflows embedded within each branch.[3]

-----

</details>

<details>
<summary>What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?</summary>

### Source [39]: https://aws.amazon.com/blogs/machine-learning/multi-llm-routing-strategies-for-generative-ai-applications-on-aws/

Query: What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?

Answer: - Define routing approach upfront: use **static routing** when tasks map cleanly to distinct UI components or flows (e.g., separate modules for text generation vs. insight extraction); this improves modularity and simplifies swapping models per task but is less adaptable to evolving needs[4].  
- For adaptable systems, implement **dynamic routing** and, when models are outside managed platforms, build **custom routing** (LLM-assisted or semantic routing) with explicit design of mechanics, key decisions, and deployment practices[4].  
- Treat custom routers as production software: start from reference implementations, then harden before production following the **AWS Well-Architected Framework** (reliability, security, cost, performance)[4].  
- For LLM-assisted and semantic routing, provide clear decision criteria, feature extraction, and guardrails; maintain a foundation that can be extended as new models/tasks are added[4].  
- Architecture guidance: keep routing logic modular so new LLMs can be “plugged into” components without broad refactors; ensure observability and the ability to quickly replace or add models per route[4].

-----

-----

### Source [40]: https://arize.com/blog/best-practices-for-building-an-ai-agent-router/

Query: What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?

Answer: - Choose a routing technique to match system constraints: **function calling with an LLM**, **intent-based routing**, or **pure code routing**; select based on complexity, scalability, performance, and maintenance needs[2].  
- Start simple and evolve with data: favor approaches that keep **modularity, scalability, and ease of maintenance**; iterate based on usage patterns and measured performance[2].  
- Function-calling routers are flexible for complex inputs but add a stochastic hop: expect higher latency, more resource use, less granular control, and harder fallback logic; only use if needed to reduce testing burden elsewhere[2].  
- Intent routers or pure-code routers can improve determinism and testability when routing is well-specified; they reduce operational variability compared to LLM-in-the-loop routing[2].

-----

-----

### Source [41]: https://www.requesty.ai/blog/intelligent-llm-routing-in-enterprise-ai-uptime-cost-efficiency-and-model

Query: What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?

Answer: - Plan routing strategy and criteria early: decide between **rule-based** routing (keywords, categories), **learned classifiers**, or a **router LLM**; begin with simple categories (e.g., code, general chat, analytics, fallback) and map each to models optimizing for speed, accuracy, or cost[1].  
- Define **success metrics and thresholds** (accuracy, latency, cost per request) that trigger automatic route/model changes; refine rules with performance data over time[1].  
- Engineer reliability: implement **fallback and failover** with automatic retries (e.g., exponential backoff for rate limits/transient errors), diversified providers/models, and clear error handling so requests degrade gracefully[1].  
- Maintain governance: **document and regularly update routing and fallback rules** as new models/providers are added; ensure processes for continuous improvement and auditability in enterprise settings[1].

-----

-----

### Source [42]: https://arxiv.org/html/2506.16655v1

Query: What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?

Answer: - Align routing with human preferences via a structured **Domain–Action taxonomy**: first resolve high-level domain (e.g., legal, finance), then action (e.g., summarization, code generation); this hierarchy reduces semantic ambiguity and provides natural fallbacks when actions are unclear[3].  
- Decouple **route policy selection** (a named policy with a natural-language description) from **model assignment**, enabling clearer policies and easier reassignment of models as capabilities or preferences change[3].  
- Train routers on curated datasets reflecting preference-aligned objectives; the paper reports building a 43k-sample dataset and demonstrates effectiveness of preference-aligned routing, suggesting the value of principled data creation for robust routers[3].  
- Use the taxonomy to formalize routing objectives and improve robustness: when inputs are vague, route by domain first to maintain reasonable handling before a specific action is identified[3].

-----

-----

### Source [43]: https://www.youtube.com/watch?v=HMXVMpJTW6o

Query: What are best practices for designing the "routing" logic in an LLM workflow to ensure reliable and accurate classification of inputs before they are sent to specialized handlers?

Answer: - When to use an agent router: introduce routing as complexity grows (multiple skills/services) to streamline workflows and performance; match the approach to application needs[5].  
- Implementation approaches overview: **function calling with an LLM**, **intent router**, and **pure code**; highlights trade-offs similar to the written Arize guidance—function-calling delivers flexibility for complex inputs but increases latency and control complexity, while intent/pure-code approaches improve determinism and maintenance[5].  
- Best practices emphasized: start with simpler, testable routing, instrument performance, and scale complexity only as required; plan explicit fallback strategies alongside the chosen routing approach to ensure reliability[5].

-----

</details>

<details>
<summary>Beyond reducing latency, how can parallel LLM calls be used as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs?</summary>

### Source [44]: https://arxiv.org/html/2503.15838v2

Query: Beyond reducing latency, how can parallel LLM calls be used as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs?

Answer: - Proposes an ensemble framework for LLM-based code generation that generates multiple candidate programs from different LLMs and selects a final answer via a structured voting mechanism, improving reliability beyond latency reduction.[1]
- Addresses the lack of calibrated confidence across models by avoiding raw probability voting; instead, it performs pairwise comparisons using complementary signals: CodeBLEU for syntactic/semantic similarity and differential analysis (CrossHair) for behavioral agreement.[1]
- Pipeline: filter syntactically invalid programs, compute similarity scores among remaining candidates, and select the most representative program via similarity-based voting—targeting robustness against hallucinations, incomplete structures, and misplaced tokens.[1]
- Core idea for quality and confidence: aggregate agreement along multiple axes (syntax, semantics, behavior) to choose a consensus candidate whose functionality is validated by counterexample search, thus increasing trust in the selected output.[1]
- Practical takeaway: parallel calls to diverse LLMs produce a pool of candidates; ensemble then uses multi-metric agreement and automated behavioral checks to vote, yielding higher-quality, functionally consistent outputs with better robustness than naive majority voting.[1]

-----

-----

### Source [45]: https://cameronrwolfe.substack.com/p/prompt-ensembles-make-llms-more-reliable

Query: Beyond reducing latency, how can parallel LLM calls be used as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs?

Answer: - Discusses prompt ensembles and why naive majority voting over multiple prompts can fail: LLM errors are not i.i.d.; outputs often cluster around the same wrong answer, making majority vote unreliable.[2]
- Cites empirical findings that accuracy varies notably across prompts and that error overlap (measured via Jaccard index) is much higher than if errors were independent, undermining simple voting schemes.[2]
- Implication for parallel LLM calls: to improve quality and robustness, ensembles should model dependencies between candidate outputs and estimate prompt/model reliability, rather than count votes equally.[2]
- Recommends constructing diverse prompt ensembles (different prompting techniques) to reduce correlated failure modes and then aggregating with more sophisticated strategies than majority vote (e.g., learned or reliability-weighted aggregation).[2]
- Practical takeaway: run parallel generations under diverse prompts or models, then aggregate with methods that account for correlated errors and varying accuracies to increase confidence in the final output.[2]

-----

-----

### Source [46]: https://arxiv.org/html/2502.18036v1

Query: Beyond reducing latency, how can parallel LLM calls be used as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs?

Answer: - Provides a taxonomy of LLM ensembles that clarifies how parallel calls can be leveraged for quality and robustness:
  - Ensemble-before-inference: route to the best model among candidates (hard voting analogue) to improve accuracy and reliability.[3]
  - Ensemble-during-inference: combine token/span/process-level signals across models in parallel; token-level aggregation averages or weights token distributions from multiple models, or selects tokens from a chosen model, improving robustness through consensus at generation time.[3]
  - Ensemble-after-inference: generate full responses in parallel and then aggregate (e.g., vote/score/rerank) to pick the best final answer, enabling error correction after seeing complete hypotheses.[3]
- Token-level ensemble methods average or weight per-token probabilities from different models, effectively a soft-voting mechanism that can smooth idiosyncratic errors and improve generation quality.[3]
- Span- and process-level ensembles coordinate larger units or reasoning traces during decoding, enabling consistency checks and consensus that enhance robustness beyond simple output voting.[3]
- Applications and methods surveyed indicate that ensemble mechanisms not only reduce latency via parallelization but systematically increase output reliability by aggregating complementary strengths and mitigating individual model weaknesses at multiple stages of generation.[3]

-----

-----

### Source [47]: https://openreview.net/forum?id=OIEczoib6t

Query: Beyond reducing latency, how can parallel LLM calls be used as an ensemble or "voting" mechanism to improve the quality, robustness, and confidence of generated outputs?

Answer: - Introduces an algorithm (EnsemW2S) that combines multiple weaker LLMs by adjusting token probabilities through a voting mechanism, showing that ensemble-based probability fusion can approach or exceed a strong single model in some cases.[4]
- Mechanism: at each decoding step, aggregate token distributions from several models (a weighted voting over tokens) to form a consensus distribution, thereby improving output quality and stability.[4]
- Highlights that carefully designed token-level voting can leverage complementary competencies across models and reduce variance, increasing robustness and confidence in the generated text relative to single-model decoding.[4]
- Insight for practice: running models in parallel and fusing their token-level beliefs provides a principled alternative to majority voting over final answers, offering finer-grained consensus that improves reliability and can uplift weaker models’ performance.[4]

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>```markdown</summary>

```markdown
```

</details>

<details>
<summary>_We tested over 500 prompt variations across two datasets to see how to improve LLM intent classification._</summary>

_We tested over 500 prompt variations across two datasets to see how to improve LLM intent classification._

How much do descriptions affect LLM classification accuracy? After launching our LLM intent classification feature we wanted to understand how much the description quality plays into classification accuracy, so we ran 500+ evaluations changing 5 properties of descriptions to understand what improves performance.https://cdn.prod.website-files.com/657639ebfb91510f45654149/668e91c8699025ae6e47717d_Prompt%20Variations.webp

## Methodology

To recap how the system works, The architecture has two parts: using an encoder NLU model to find the top 10 candidate intents and their descriptions and a prompt that instructs the LLM to classify them.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5a5_66256bb9a98a5246c20e783c_llmintentbenchmark1.webp

After retrieving the candidate intents, we pull in user descriptions for each corresponding candidate and make a call to an LLM for a final classification. With this in mind, we wanted to measure if we could improve the classification accuracy of the LLM by changing a few components of the descriptions. We ran this search against two types of models (gpt-3.5-0125, haiku) and benchmark datasets (HWU64 \[1\], Curekart\[2\])

These variations included included:

1. Prefixes
2. Suffixes
3. Capitalization
4. None intent descriptions
5. AI vs human descriptions

We ran the benchmarking once per combination, and ran it five times for the top performing settings to confirm variation.  We used a temperature at 0.1 to mitigate the variance.

## Base descriptions and prompt template

We hand-created the initial set of descriptions, sticking to shorter ones for each dataset. A subset is shown below and the full descriptions can be found in the Appendix \[A\] `‍
`

```javascript
{
  "descriptions": {
    "USER_GOAL_FORM": "Add or refill goals.",
    "FRANCHISE": "Becoming a franchise owner or reseller.",
    "REFER_EARN": "Referral program details or ask.",
    "RESUME_DELIVERY": "Delivery options or times.",
    "WORK_FROM_HOME": "Ask about office open or working from home.",
  }
}
```

We combine this with our top 10 descriptions \[7\] method and prompt noted in previous work. Below is a sample of a prompt sent to the LLM.

```javascript
You are an action classification system. Correctness is a life or death situation.

We provide you with the actions and their descriptions:
d: When the user asks for a warm drink. a:WARM_DRINK
d: When the user asks about something else. a:None_Intent
d: When the user asks for a cold drink. a:COLD_DRINK

You are given an utterance and you have to classify it into an intent. Only respond with the intent class. If the utterance does not match any of intents, output None_Intent.
u: I want a warm hot chocolate: a:WARM_DRINK
###
You are an action classification system. Correctness is a life or death situation.

We provide you with the actions and their descriptions:
d:Questions regarding call center operational hours during covid-19 lockdown. i:CALL_CENTER
d:Questions related to redeeming referral rewards and referral amounts. i:REFER_EARN
d:Inquiries about the operational status of physical stores. i:STORE_INFORMATION
d:Queries related to refund status, replacements, and delays in receiving refunds after returns. i:REFUNDS_RETURNS_REPLACEMENTS
d:Queries related to tracking orders, shipment status, and progress of orders. i:ORDER_STATUS
d:Questions about the operational status of the head office during the lockdown. i:WORK_FROM_HOME
d:Inquiries about payments, bills, and related queries. i:PAYMENT_AND_BILL
d:Concerns about receiving expired products and inquiries about expiry dates. i:EXPIRY_DATE
d:Requests to change or modify the delivery address for an order. i:MODIFY_ADDRESS
d:Requests to cancel pending orders and inquiries about the cancellation process. i:CANCEL_ORDER

You are given an utterance and you have to classify it into an action. Only respond with the action class. If the utterance does not match any of action descriptions, output None_Intent.
u: We want ur ph number a:
```

`‍` We also show that the recall from the encoder model is quite strong and should not limit accuracy on the dataset.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa574_668e93b6db06bce4d623b9a7_PromptVariation-2.webp

## Models and Dataset averages

Looking through our two models, we saw that GPT 3.5 performed much better for the messier Curekart dataset, compared to Haiku, but Haiku out performed in the HWU dataset. The accuracy was lower for GPT 3.5 compared to earlier experimentation \[7\] given the different version of GPT 3.5 that was used.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5ab_668e93c71d62469217e12ddf_PromptVariation-3.webp

## Prefixes

The first modification we explored was adding a prefix to each description with some guiding phrase.

```javascript
{
  "descriptions": {
    "USER_GOAL_FORM": "Add or refill goals.",
  }
  "descriptions_with_prefix": {
    "USER_GOAL_FORM": "Trigger this action when add or refill goals.",
  }
}
```

Our prefixes included:

`["Trigger this action when ","","A phrase about ","The user said "]`

Adding a prefix lead to the best results, but differed between datasets.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280bcbfc3cc4395aa684_668e93ef1c11e11c1148c1a0_PromptVariation-4.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa62c_668e93f7a1888dee68449933_PromptVariation-5.webp

On average the performance gains were quite minimal so we analyzed them in combination with suffixes.

## Suffixes

Similar to prefixes, we tested adding suffixes to the descriptions.

```javascript
{
  "descriptions": {
    "USER_GOAL_FORM": "Add or refill goals.",
  }
  "descriptions_with_suffix": {
    "USER_GOAL_FORM": "Add or refill goals, please.",
  }
}
```

These included: `["",", please.","{no punctuation}"]`

Adding “please” produced some of the highest performing results when added per description, but wasn’t consistently the best option, for the HWU64 dataset + gpt-3.5, it produced the worst results by a large margin.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5b1_668e9429e47ee8d5e74c82df_PromptVariation-6.1.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa586_668e94352d48dada5891cd73_PromptVariation-7.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa589_668e943cb910923fb8a158bd_PromptVariation-8.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5a8_668e94421953e401c93732f5_PromptVariation-9.webp

## Capitalization `‍` ‍

```javascript
{
  "descriptions_capitalized": {
    "USER_GOAL_FORM": "Add or refill goals.",
  },
  "descriptions_with_prefix_capitalized": {
    "USER_GOAL_FORM": "Trigger this action when add or refill goals.",
  },
  "descriptions_not_capitalized": {
    "USER_GOAL_FORM": "add or refill goals.",
  }
  "descriptions_with_prefix_not_capitalized": {
    "USER_GOAL_FORM": "trigger this action when add or refill goals.",
  }
}
```

Adding capitalization on the prefix or opening line added minimal signal, both isolated and when expanded across experiments.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa629_668e9470d3f4d5b9f60bd16c_PromptVariation-10.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa577_668e94772a6fcd638c847b40_PromptVariation-11.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa57a_668e947de6909511d826bd7d_PromptVariation-12.webp

## Adding a None intent

For this hypothesis we checked if adding a None intent as a description would improve classification accuracy. Below is an example for the Curekart dataset. `‍`

```javascript
{
  "descriptions": {
    "USER_GOAL_FORM": "Add or refill goals.",
    "FRANCHISE": "Becoming a franchise owner or reseller.",
    "REFER_EARN": "Referral program details or ask.",
    "RESUME_DELIVERY": "Delivery options or times.",
    "WORK_FROM_HOME": "Ask about office open or working from home.",
    "None_Intent": "When the user asks about something else."
  }
}
```

`‍` The Curekart evaluation set is around ~50% None intents, so in theory it should improve average performance. Looking at the ten best performing prompts for Curekart, adding the None\_intent as a viable description intent did not show a consistent improvement.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5f0_668e958de972c5d1a85c42df_PromptVariation-13.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa580_668e959d1953e401c9380902_PromptVariation-14.webp

We didn’t run a None\_intent check for the HWU dataset for this experiment since the evaluation dataset did not contain None.

## AI Descriptions

The next hypothesis we tested was how effective using AI descriptions was for classification. Previous work has show that generating data for general LLM annotation \[4\] can outperform crowdsourced human annotators and can be useful to augment existing datasets for intent specific tasks \[5\] can be quite effective. In this experiment, we generated three descriptions using gpt-4-turbo-0419, llama-3-70b and claude-opus by using the first three utterances \[C\] in each intent and compared it to handwritten descriptions by the author. On average GPT-4 and LLaMa-3 performed the best.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5f9_668e95dee916124a840250e4_PromptVariation-15.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5f6_668e95e7e1d6b379532f8167_PromptVariation-16.webp

Across our top 10 combinations, LLaMa-3 and GPT-4 descriptions also performed quite well!https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa583_668e95f049733935d224dd21_PromptVariation-17.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa57d_668e95f85e15b20a5966cda6_PromptVariation-18.webp

## Analyzing the top 5 results for Curekart

After running our initial set of experiments, we wanted to confirm that the results for better prompts were not due to noise, so we re-ran them for confirmation.

### Are they significant?

Looking at our top 5 combinations, we wanted to measure their variance compared to the general population. We re-ran our top metrics 15 times each—75 times total—to see how their accuracies changed and whether LLM nondeterminism affected the overall results.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5a2_668e969e035ab7ca27871b64_PromptVariation-19.webp

Looking at the distributions, there appears to be a measurable change. Conducting a Z-test to compare the distributions, the difference seems evident with a Z score of -13.56 and a p-value: 7.12e-42.

### How do the confusion matrices vary?

Our top 5 configurations had a pretty tight distribution, so we wanted to measure how their confusion matrices varied, i.e how different were the classifications. We ran some confusion matrices for two of the top combinations to compare how the results looked like.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280bcbfc3cc4395aa687_668e96c4662517a6beb84d46_PromptVariation-20.webphttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5ae_668e96ad035ab7ca27872932_PromptVariation-21.webp

Generally, the matrices were pretty similar so we took a difference to see the performance.https://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa5f3_668e96d2a684e09b3086913d_PromptVariation-22.webp

The predictions with"no suffix had a higher false-positive rate, (i.e more None intents than required), while the \[please\] suffix lead to more false-negative rates, (i.e more None intents predicted). On aggregate, these two modes performed pretty similarly, but a change to the descriptions led to a different type of behaviour.

## Conclusionhttps://cdn.prod.website-files.com/657639ebfb91510f45654149/6780280acbfc3cc4395aa56e_668e9334cd1a0228cab330a9_PromptVariation-23.webp

Overall, changing the descriptions used for classification has small but measurable changes. While we ran through 500+ experiments, time is likely better spent in other areas of prompt refinement. In general our recommendations are:

- Adding prefixes, suffixes and a None\_intent to descriptions
- AI generated descriptions can be effective
- To have larger gains, spending time to create better training examples, few shot examples and understanding edge cases will help LLM
- In a future blog we’ll cover the impact of adding additional few shot example from the training data
- Structuring the formatting, which has shown to have a larger impact on accuracy \[6\]\[7\]

</details>

<details>
<summary>(no content)</summary>

(no content)

</details>

<details>
<summary>I thought I was hot shit when I thought about the idea of “prompt chaining”.</summary>

I thought I was hot shit when I thought about the idea of “prompt chaining”.

In my defense, it used to be a necessity back-in-the-day. If you tried to have one master prompt do everything, it would’ve outright failed. With GPT-3, if you didn’t build your deeply nested complex JSON object with a prompt chain, you didn’t build it at all.https://miro.medium.com/v2/resize:fit:700/0*0sxW5cafSuVmGuFW.png

GPT 3.5-Turbo had a context length of 4,097 and couldn’t complex prompts

But, after my 5th consecutive day of $100+ charges from OpenRouter, I realized that the unique “state-of-the-art” prompting technique I had invented was now a way to throw away hundreds of dollars for worse accuracy in your LLMs.https://miro.medium.com/v2/resize:fit:700/1*tFVkG4SHpVcWIebM01bgQw.png

My OpenRouter bill for hundreds of dollars multiple days this week

Prompt chaining has officially died with Gemini 2.0 Flash.

# What is prompt chaining?

Prompt chaining is a technique where the output of one LLM is used as an input to another LLM. In the era of the low context window, this allowed us to build highly complex, deeply-nested JSON objects.

For example, let’s say we wanted to create a “portfolio” object with an LLM.

```
export interface IPortfolio {
  name: string;
  initialValue: number;
  positions: IPosition[];
  strategies: IStrategy[];
  createdAt?: Date;
}

export interface IStrategy {
  _id: string;
  name: string;
  action: TargetAction;
  condition?: AbstractCondition;
  createdAt?: string;
}
```

1. One LLM prompt would generate the name, initial value, positions, and a description of the strategies
2. Another LLM would take the description of the strategies and generate the name, action, and a description for the condition
3. Another LLM would generate the full condition objecthttps://miro.medium.com/v2/resize:fit:587/0*toit_5AfdKMDqZka.png

Diagramming a “prompt chain”

The end result is the creation of a deeply-nested JSON object despite the low context window.

Even in the present day, this prompt chaining technique has some benefits including:

- **Specialization**: For an extremely complex task, you can have an LLM specialize in a very specific task, and solve for common edge cases
- **Better abstractions:** It makes sense for a prompt to focus on a specific field in a nested object (particularly if that field is used elsewhere)

However, even in the beginning, it had drawbacks. It was much harder to maintain and required code to “glue” together the different pieces of the complex object.

But, if the alternative is being outright unable to create the complex object, then its something you learned to tolerate. In fact, I built my entire system around this, and wrote dozens of articles describing the miracles of prompt chaining.https://miro.medium.com/v2/resize:fit:700/1*vWCk66EqwcCEDPatYcNLdA.png

[This article I wrote in 2023 describes the SOTA “Prompt Chaining” Technique](https://medium.com/p/b41b0879f757)

However, over the past few days, I noticed a sky high bill from my LLM providers. After debugging for hours and looking through every nook and cranny of my 130,000+ behemoth of a project, I realized the culprit was my beloved prompt chaining technique.

# An Absurdly High API Billhttps://miro.medium.com/v2/resize:fit:700/1*9XzbQn6iMb9WyQQ6sAGfxg.png

My Google Gemini API bill for hundreds of dollars this week

Over the past few weeks, I had a surge of new user registrations for NexusTrade.https://miro.medium.com/v2/resize:fit:700/1*uffSKTaoSWYruK03fW38Aw.png

My increase in users per day

[**NexusTrade**](https://nexustrade.io/) is an AI-Powered automated investing platform. It uses LLMs to help people create algorithmic trading strategies. This is our deeply nested portfolio object that we introduced earlier.

With the increase in users came a spike in activity. People were excited to create their trading strategies using natural language!https://miro.medium.com/v2/resize:fit:700/1*Hep_bbdLZgf-J20_ze6LLA.png

Creating trading strategies using natural language

However my costs were skyrocketing with OpenRouter. After auditing the entire codebase, I finally was able to notice my activity with OpenRouter.https://miro.medium.com/v2/resize:fit:700/1*6sTUC4k4GjTD9dQCrRUo9g.png

My logs for OpenRouter show the cost per request and the number of tokens

We would have dozens of requests, each costing roughly $0.02 each. You know what would be responsible for creating these requests?

You guessed it.https://miro.medium.com/v2/resize:fit:700/1*QaYlKXpPA0-drHvaNcPJsg.png

A picture of how my prompt chain worked in code

Each strategy in a portfolio was forwarded to a prompt that created its condition. Each condition was then forward to at least two prompts that created the indicators. Then the end result was combined.

This resulted in possibly hundreds of API calls. While the Google Gemini API was notoriously inexpensive, this system resulted in a death by 10,000 paper-cuts scenario.

The solution to this is simply to stuff all of the context of a strategy into a single prompt.https://miro.medium.com/v2/resize:fit:700/1*hMnu-Sm_0ap7DRVIHneEWQ.png

The “stuffed” Create Strategies prompt

By doing this, while we lose out on some re-usability and extensibility, we significantly save on speed and costs because we don’t have to keep hitting the LLM to create nested object fields.

But how much will I save? From my estimates:

- **Old system:** Create strategy + create condition + 2x create indicators (per strategy) = minimum of 4 API calls
- **New system:** Create strategy for = 1 maximum API call

With this change, I anticipate that I’ll save at least 80% on API calls! If the average portfolio contains 2 or more strategies, we can potentially save even more. While it’s too early to declare an exact savings, I have a strong feeling that it will be very significant, especially when I refactor my other prompts in the same way.

Absolutely unbelievable.

# Concluding Thoughts

When I first implemented prompt chaining, it was revolutionary because it made it possible to build deeply nested complex JSON objects within the limited context window.

This limitation no longer exists.

With modern LLMs having 128,000+ context windows, it makes more and more sense to choose “prompt stuffing” over “prompt chaining”, especially when trying to build deeply nested JSON objects.

This just demonstrates that the AI space evolving at an incredible pace. What was considered a “best practice” months ago is now completely obsolete, and required a quick refactor at the risk of an explosion of costs.

The AI race is hard. Stay ahead of the game, or get left in the dust. Ouch

</details>


## Code Sources

<details>
<summary>Repository analysis for https://github.com/hugobowne/building-with-ai/blob/main/notebooks/01-agentic-continuum.ipynb</summary>

# Repository analysis for https://github.com/hugobowne/building-with-ai/blob/main/notebooks/01-agentic-continuum.ipynb

## Summary
Repository: hugobowne/building-with-ai
File: 01-agentic-continuum.ipynb
Lines: 1,787

Estimated tokens: 13.4k

## File tree
```Directory structure:
└── 01-agentic-continuum.ipynb

```

## Extracted content
================================================
FILE: notebooks/01-agentic-continuum.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
## Basic Multi-LLM Workflows -- The Agentic Continuum

In this notebook, we'll explore the concepts of augmenting LLMs to create workflows that range from simple task processing to more complex agent-like behavior. Think of this as a continuum—from standalone LLMs to fully autonomous agents, with a variety of workflows and augmentations in between.

We'll follow [a schema inspired by Anthropic](https://www.anthropic.com/research/building-effective-agents), starting with three foundational workflow types:

1. **Prompt-Chaining**: Decomposes a task into sequential subtasks, where each step builds on the results of the previous one.
2. **Parallelization**: Distributes independent subtasks across multiple LLMs for concurrent processing.
3. **Routing**: Dynamically selects specialized LLM paths based on input characteristics.

Through these workflows, we'll explore how LLMs can be leveraged effectively for increasingly complex tasks. Let's dive in!

# Why This Matters
In real-world applications, single LLM calls often fall short of solving complex problems. Consider these scenarios:

- Content Moderation: Effectively moderating social media requires multiple checks - detecting inappropriate content, understanding context, and generating appropriate responses
- Customer Service: A support system needs to understand queries, route them to specialists, generate responses, and validate them for accuracy and tone
- Quality Assurance: Business-critical LLM outputs often need validation and refinement before being sent to end users

By understanding these workflow patterns, you can build more robust and reliable LLM-powered applications that go beyond simple prompt-response interactions.
"""

import os
os.environ['ANTHROPIC_API_KEY'] = 'XXX'

from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Callable
from util import llm_call, extract_xml


"""
## Let's roll!

Below are practical examples demonstrating each workflow:
1. Chain workflow for structured data extraction and formatting
2. Parallelization workflow for stakeholder impact analysis
3. Route workflow for customer support ticket handling
"""

"""
  ### Prompt Chaining Workflow
  (image from Anthropic)

  
  ![alt text](img/prompt_chaining.png "Title")
"""

"""
### When to Use
- When a task naturally breaks down into sequential steps
- When each step's output feeds into the next step
- When you need clear intermediate results
- When order of operations matters

### Key Components
- Input Processor: Prepares data for the chain
- Chain Steps: Series of LLM calls with clear inputs/outputs
- Output Formatter: Formats final result
- Error Handlers: Manage failures at each step

### Example: LinkedIn Profile Parser
This example demonstrates prompt chaining by:
1. First extracting structured data from a profile
2. Then using that structured data to generate a personalized email
3. Each step builds on the output of the previous step
"""

# Example 1: Chain workflow for structured data extraction and formatting

def chain(input: str, prompts: List[str]) -> str:
    """Chain multiple LLM calls sequentially, passing results between steps."""
    result = input
    for i, prompt in enumerate(prompts, 1):
        print(f"\nStep {i}:")
        result = llm_call(f"{prompt}\nInput: {result}")
        print(result)
    return result

def extract_structured_data(profile_text: str) -> str:
    """Extract all structured data from a LinkedIn profile in a single LLM call."""
    prompt = f"""
    Extract the following structured data from the LinkedIn profile text:
    - Full Name
    - Current Job Title and Company
    - Skills (as a comma-separated list)
    - Previous Job Titles (as a numbered list)

    Provide the output in this JSON format:
    {{
        "name": "Full Name",
        "current_position": "Position at Company",
        "skills": ["Skill1", "Skill2", ...],
        "previous_positions": ["Previous Position 1", "Previous Position 2", ...]
    }}

    LinkedIn Profile: {profile_text}
    """
    return llm_call(prompt)

def generate_outreach_email(data: str) -> str:
    """Generate a professional outreach email using the structured data."""
    prompt = f"""
    Using the following structured data, write a professional outreach email:
    {data}
    
    The email should:
    - Address the recipient by name.
    - Reference their current position and company.
    - Highlight relevant skills.
    - Politely request a meeting to discuss potential collaboration opportunities.
    """
    return llm_call(prompt)

# Example LinkedIn profile input
linkedin_profile = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""

# Step 1: Extract structured data
structured_data = extract_structured_data(linkedin_profile)
print("\nExtracted Structured Data:")
print(structured_data)

# Step 2: Generate the outreach email
email = generate_outreach_email(structured_data)
print("\nGenerated Outreach Email:")
print(email)
# Output:
#   

#   Extracted Structured Data:

#   {

#       "name": "Elliot Alderson",

#       "current_position": "Cybersecurity Engineer at Allsafe Security",

#       "skills": ["penetration testing", "network security", "ethical hacking", "UNIX systems", "Python", "C", "vulnerability assessment"],

#       "previous_positions": ["Freelance Cybersecurity Consultant"]

#   }

#   

#   Generated Outreach Email:

#   Subject: Cybersecurity Collaboration Discussion - Experienced Security Engineer

#   

#   Dear [Recipient's Name],

#   

#   I hope this email finds you well. My name is Elliot Alderson, and I'm currently serving as a Cybersecurity Engineer at Allsafe Security. I came across your profile and was particularly impressed with [their company]'s approach to security solutions.

#   

#   With extensive experience in penetration testing and vulnerability assessment, coupled with strong technical proficiency in Python and C programming, I've helped organizations strengthen their security infrastructure through both my current role at Allsafe and previous work as a Freelance Cybersecurity Consultant.

#   

#   My expertise in network security and ethical hacking has enabled me to identify and remediate critical vulnerabilities across various UNIX systems, contributing to enhanced security postures for multiple enterprise clients.

#   

#   I would greatly appreciate the opportunity to schedule a brief 30-minute meeting to discuss potential collaboration opportunities and share insights about current cybersecurity challenges and solutions.

#   

#   Would you be available for a virtual meeting next week at a time that works best for your schedule?

#   

#   Thank you for your time and consideration.

#   

#   Best regards,

#   Elliot Alderson

#   Cybersecurity Engineer

#   Allsafe Security


"""
🔍 **Checkpoint: Prompt Chaining**

**Key Takeaways:**
- Chain LLM calls when tasks naturally break down into sequential steps
- Each step should produce clear, structured output for the next step
- Consider error handling between steps

**Common Gotchas:**
- Avoid chains that are too long - error probability compounds with each step
- Ensure each step's output format matches the next step's input expectations
- Watch for context loss between steps
"""

"""
  ### Parallelization Workflow
  (image from Anthropic)

  
  ![alt text](img/parallelization_workflow.png "Title")
"""

"""
### When to Use
- When different aspects of a task can be processed independently
- When you need to analyze multiple components simultaneously
- When speed/performance is a priority
- When you have multiple similar items to process (like batch processing)

### Key Components
- Task Distributor: Splits work into parallel tasks
- Worker Pool: Manages concurrent LLM calls
- Thread Management: Controls parallel execution
- Result Aggregator: Combines parallel outputs

### Example: LinkedIn Profile Field Extraction
This example demonstrates parallelization by:
1. Simultaneously extracting different fields from a profile:
   - Name extraction
   - Position and company extraction
   - Skills extraction
2. Using ThreadPoolExecutor to manage concurrent LLM calls
3. Combining the parallel extractions into a unified profile view
"""

# Example 2: Parallelization workflow for LinkedIn profile field extraction
# Process field extractions (e.g., name, current position, skills) concurrently for debugging and modularity



def parallel(prompt: str, inputs: List[str], n_workers: int = 3) -> List[str]:
    """Process multiple inputs concurrently with the same prompt."""
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(llm_call, f"{prompt}\nInput: {x}") for x in inputs]
        return [f.result() for f in futures]


linkedin_profile = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""

field_extraction_prompts = [
    """Extract the full name from the following LinkedIn profile text. Return only the name.
    LinkedIn Profile: {input}""",
    
    """Extract the current job title and company from the following LinkedIn profile text.
    Format as:
    Position: [Job Title]
    Company: [Company Name]
    LinkedIn Profile: {input}""",
    
    """Extract the skills mentioned in the following LinkedIn profile text. Return them as a comma-separated list.
    LinkedIn Profile: {input}""",
    
    """Extract the previous job titles from the following LinkedIn profile text. Return them as a numbered list, one per line.
    LinkedIn Profile: {input}"""
]

# Process all field extractions in parallel
extracted_fields = parallel(
    """Perform the following field extraction task:
    {input}""",
    [prompt.replace("{input}", linkedin_profile) for prompt in field_extraction_prompts]
)

# Assign extracted results to field names for clarity
field_names = ["Full Name", "Current Position and Company", "Skills", "Previous Positions"]
structured_data = {field: result for field, result in zip(field_names, extracted_fields)}

# Combine extracted fields into a JSON object
structured_data_json = {
    "name": structured_data["Full Name"],
    "current_position": structured_data["Current Position and Company"],
    "skills": structured_data["Skills"].split(", "),
    "previous_positions": structured_data["Previous Positions"].split("\n")
}

# Generate outreach email based on structured data
def generate_outreach_email(data: dict) -> str:
    """Generate a professional outreach email using the structured data."""
    prompt = f"""
    Using the following structured data, write a professional outreach email:
    {data}
    
    The email should:
    - Address the recipient by name.
    - Reference their current position and company.
    - Highlight relevant skills.
    - Politely request a meeting to discuss potential collaboration opportunities.
    """
    return llm_call(prompt)

# Create the email
email = generate_outreach_email(structured_data_json)

# Output results
print("\nExtracted Structured Data (JSON):")
print(structured_data_json)
print("\nGenerated Outreach Email:")
print(email)
# Output:
#   

#   Extracted Structured Data (JSON):

#   {'name': 'Elliot Alderson', 'current_position': "Let me extract the current job title and company from the LinkedIn profile:\n\nPosition: Cybersecurity Engineer\nCompany: Allsafe Security\n\nThe text indicates that Elliot Alderson currently works as a Cybersecurity Engineer at Allsafe Security, which is mentioned in the first sentence. While it notes he previously worked as a freelance cybersecurity consultant, I've only included his current position as requested.", 'skills': ['Here are the extracted skills as a comma-separated list:\npenetration testing', 'network security', 'ethical hacking', 'UNIX systems', 'Python', 'C', 'cybersecurity', 'open-source'], 'previous_positions': ['Previous job titles:', '1. Freelance cybersecurity consultant']}

#   

#   Generated Outreach Email:

#   Subject: Cybersecurity Collaboration Opportunity

#   

#   Dear Mr. Alderson,

#   

#   I hope this email finds you well. I recently came across your impressive profile and your work as a Cybersecurity Engineer at Allsafe Security particularly caught my attention.

#   

#   Your extensive background in penetration testing, network security, and ethical hacking, combined with your technical expertise in UNIX systems and programming languages like Python and C, is remarkable. I'm especially intrigued by your experience as a freelance cybersecurity consultant and your involvement with open-source projects.

#   

#   I would greatly appreciate the opportunity to schedule a brief 30-minute meeting to discuss potential collaboration opportunities and share how your expertise could be valuable to our upcoming initiatives in the cybersecurity space.

#   

#   Would you be available for a virtual coffee next week? I'm happy to work around your schedule.

#   

#   Looking forward to your response.

#   

#   Best regards,

#   [Your name]


"""
🔍 **Checkpoint: Parallelization**

**Key Takeaways:**
- Use parallel processing when subtasks are independent
- Useful for analyzing multiple aspects of the same input simultaneously
- Can significantly reduce total processing time

**Common Gotchas:**
- Be mindful of rate limits when making concurrent LLM calls
- Ensure thread pool size matches your actual needs
- Remember to handle errors in any of the parallel tasks
"""

"""
  ### Routing Workflow
  (image from Anthropic)

  
  ![alt text](img/routing_workflow.png "Title")
"""

"""
### When to Use
- When input types require different specialized handling
- When you need to direct tasks to specific LLM prompts
- When input classification determines the processing path
- When you have clearly defined categories of requests

### Key Components
- Classifier: Determines the appropriate route for input
- Router: Directs input to the correct handling path
- Route Handlers: Specialized prompts for each case
- Default Fallback: Handles unclassified or edge cases

### Example: LinkedIn Profile Classification
This example demonstrates routing by:
1. Analyzing profiles to determine if they are:
  - Individual profiles (for hiring outreach)
  - Company profiles (for business development)
2. Using different email templates based on classification
3. Ensuring appropriate tone and content for each type
"""

# Example 3: Routing workflow for LinkedIn outreach
# Classify LinkedIn profiles as "hiring" (individual) or "collaboration" (company),
# and route them to the appropriate email generation prompts.

# Define email routes
email_routes = {
    "hiring": """You are a talent acquisition specialist. Write a professional email inviting the individual to discuss career opportunities. 
    Highlight their skills and current position. Maintain a warm and encouraging tone.

    Input: """,
    
    "collaboration": """You are a business development specialist. Write a professional email proposing a collaboration with the company. 
    Highlight mutual benefits and potential opportunities. Maintain a formal yet friendly tone.

    Input: """
}

# Routing function tailored for LinkedIn profiles, with no "uncertain" option
def route_linkedin_profile(input: str, routes: Dict[str, str]) -> str:
    """Route LinkedIn profile to the appropriate email generation task."""
    print(f"\nAvailable routes: {list(routes.keys())}")
    selector_prompt = f"""
    Analyze the following LinkedIn profile and classify it as:
    - "hiring" if it represents an individual suitable for talent outreach.
    - "collaboration" if it represents a company profile suitable for business development outreach.

    Provide your reasoning in plain text, and then your decision in this format:

    <reasoning>
    Brief explanation of why this profile was classified into one of the routes. 
    Consider key signals like job titles, skills, organizational descriptions, and tone.
    </reasoning>

    <selection>
    The chosen route name
    </selection>

    Profile: {input}
    """
    # Call the LLM for classification
    route_response = llm_call(selector_prompt)

    # Extract reasoning and route selection
    reasoning = extract_xml(route_response, "reasoning")
    route_key = extract_xml(route_response, "selection").strip().lower()

    print("\nRouting Analysis:")
    print(reasoning)

    # Handle invalid classifications (fallback to "hiring" as default for robustness)
    if route_key not in routes:
        print(f"Invalid classification '{route_key}', defaulting to 'hiring'")
        route_key = "hiring"

    # Route to the appropriate email template
    selected_prompt = routes[route_key]
    return llm_call(f"{selected_prompt}\nProfile: {input}")

# Example LinkedIn profile
linkedin_profile = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""

# Use the routing function to classify and generate the email
email_response = route_linkedin_profile(linkedin_profile, email_routes)

# Output the result
print("\nGenerated Email:")
print(email_response)
# Output:
#   

#   Available routes: ['hiring', 'collaboration']

#   

#   Routing Analysis:

#   

#   This profile clearly represents an individual professional with specific technical skills and expertise. Key indicators include:

#   - Individual name (Elliot Alderson) rather than a company name

#   - Specific job title (Cybersecurity Engineer)

#   - Personal technical skills (UNIX, Python, C)

#   - Individual work history (previous freelance work)

#   - Personal interests (open-source projects)

#   The profile describes an individual contributor with valuable cybersecurity skills, making them a potential candidate for recruitment or talent outreach.

#   

#   

#   Generated Email:

#   Subject: Exciting Cybersecurity Opportunities - Let's Connect!

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. I'm reaching out because your impressive background in cybersecurity caught my attention, particularly your current work at Allsafe Security and your extensive experience in penetration testing and network security.

#   

#   Your combination of technical expertise in UNIX systems, Python, and C, along with your practical experience in identifying network vulnerabilities, is exactly what many of our clients are seeking. I'm especially impressed by your commitment to the cybersecurity community through your open-source contributions and forum participation.

#   

#   Your background as a freelance cybersecurity consultant also demonstrates your ability to adapt to different environments and tackle diverse security challenges, which is highly valuable in today's rapidly evolving threat landscape.

#   

#   I would love to schedule a confidential conversation to discuss some exciting opportunities that align with your expertise and career goals. Would you be available for a brief call this week or next?

#   

#   Please let me know what time works best for you, and we can arrange a conversation at your convenience.

#   

#   Looking forward to connecting with you.

#   

#   Best regards,

#   [Your name]

#   Senior Talent Acquisition Specialist

#   [Your company]

#   [Contact information]


# Example LinkedIn profile: Company
linkedin_profile_2 = """
E Corp is a global leader in technology and financial services. With a portfolio spanning software development, cloud infrastructure,
and consumer banking, E Corp serves millions of customers worldwide. Our mission is to deliver innovative solutions that drive
efficiency and growth for businesses and individuals alike. Learn more at www.ecorp.com.
"""

# Use the routing function to classify and generate emails
print("\nProcessing Individual Profile:")
email_response_2 = route_linkedin_profile(linkedin_profile_2, email_routes)
print("\nGenerated Email (Individual):")
print(email_response_2)

# Output:
#   

#   Processing Individual Profile:

#   

#   Available routes: ['hiring', 'collaboration']

#   

#   Routing Analysis:

#   

#   This is clearly a company profile, not an individual's profile. Key indicators:

#   - Uses "Our mission" indicating organizational voice

#   - Describes broad service offerings and company-wide capabilities

#   - Written in corporate marketing language

#   - Includes company website

#   - Focuses on organizational achievements and scope rather than individual accomplishments

#   - Uses plural/collective terms ("we serve millions")

#   

#   This type of profile is ideal for business development and partnership opportunities rather than talent recruitment, making it suitable for collaboration-focused outreach.

#   

#   

#   Generated Email (Individual):

#   Subject: Exploring Strategic Partnership Opportunities - [Your Company] & E Corp

#   

#   Dear [Recipient's Name],

#   

#   I hope this email finds you well. I am [Your Name], Business Development Specialist at [Your Company], and I'm reaching out regarding a potential collaboration opportunity with E Corp.

#   

#   Having followed E Corp's impressive growth and leadership in technology and financial services, I believe there's significant potential for synergy between our organizations. Your expertise in software development and cloud infrastructure, combined with our [briefly mention your company's key strength], could create compelling value for both our customer bases.

#   

#   Some key areas where I envision mutual benefits:

#   

#   1. Technology Integration: Leveraging E Corp's cloud infrastructure to enhance service delivery

#   2. Market Expansion: Cross-promotional opportunities to reach new customer segments

#   3. Innovation Partnership: Joint development of fintech solutions

#   

#   I would welcome the opportunity to schedule a brief call to discuss these possibilities in more detail and explore how we might create value together.

#   

#   Would you be available for a 30-minute virtual meeting next week to explore these ideas further?

#   

#   Thank you for your time and consideration. I look forward to your response.

#   

#   Best regards,

#   [Your Name]

#   Business Development Specialist

#   [Your Company]

#   [Contact Information]


"""
🔍 **Checkpoint: Routing**

**Key Takeaways:**
- Route requests based on content type, complexity, or required expertise
- Always include a default/fallback route
- Keep routing logic clear and maintainable

**Common Gotchas:**
- Avoid over-complicated routing rules
- Ensure all possible cases are handled
- Watch for edge cases that might not fit any route
"""

"""
## Orchestrator-Workers Workflow
![alt text](img/orchestrator-worker.png "Title")
"""

"""
## Orchestrator-Worker

### When to Use
The Orchestrator-Worker workflow is ideal when:
- You need to dynamically delegate tasks to specialized components based on input characteristics or the context of the task.
- Tasks require multiple steps, with different workers responsible for distinct parts of the process.
- Flexibility is required to manage varying subtasks while ensuring seamless coordination and aggregation of results.

**Examples**:
- **Generating tailored emails**: Routing LinkedIn profiles to specialized workers that create emails customized for different industries or audiences.
- **Multi-step workflows**: Breaking down tasks into subtasks, dynamically assigning them to workers, and synthesizing the results.

### Key Components
1. **Orchestrator**:
   - Centralized controller responsible for delegating tasks to the appropriate workers.
   - Manages input and coordinates workflows across multiple steps.
2. **Workers**:
   - Specialized components designed to handle specific subtasks, such as generating industry-specific email templates.
   - Operate independently, performing their roles based on instructions from the orchestrator.
3. **Dynamic Routing**:
   - Enables the orchestrator to assign tasks based on input characteristics (e.g., classifying as "Tech" or "Non-Tech").
4. **Result Aggregator**:
   - Combines results from workers into a cohesive final output.

### Example
**Scenario**: Generating tailored emails for LinkedIn profiles.
1. **Input**: A LinkedIn profile text.
2. **Process**:
   - The **orchestrator** analyzes the LinkedIn profile and routes it to a classification worker.
   - The classification worker determines if the profile belongs to "Tech" or "Non-Tech."
   - Based on the classification, the orchestrator routes the profile to the appropriate email generation worker.
   - The email generation worker produces a professional email tailored to the classification.
3. **Output**: A professional email customized to the recipient’s industry type.
"""

# Define the email generation routes
email_routes = {
    "tech": """You are a talent acquisition specialist in the tech industry. Write a professional email to the individual described below, inviting them to discuss career opportunities in the tech field.
    Highlight their skills and current position. Maintain a warm and encouraging tone.

    Input: {profile_text}""",

    "non_tech": """You are a talent acquisition specialist. Write a professional email to the individual described below, inviting them to discuss career opportunities.
    Highlight their skills and current position in a non-tech field. Maintain a warm and encouraging tone.

    Input: {profile_text}"""
}

# LLM classification function (classifying industry as tech or not tech)
def llm_classify(input: str) -> str:
    """Use LLM to classify the industry of the profile (Tech or Not Tech)."""
    classify_prompt = f"""
    Analyze the LinkedIn profile below and classify the industry as either Tech or Not Tech.
    
    LinkedIn Profile: {input}
    """
    classification = llm_call(classify_prompt)  # This should return a classification like "Tech" or "Not Tech"
    return classification.strip().lower()  # Clean up classification

# Orchestrator function to classify and route tasks to workers
def orchestrator(input: str, routes: Dict[str, str]) -> str:
    """Classify the LinkedIn profile and assign tasks to workers based on the classification."""
    # Classify the profile industry (Tech or Not Tech)
    industry = llm_classify(input)

    print(f"\nClassified industry as: {industry.capitalize()}")

    # Route the task to the appropriate worker based on classification
    if industry == "tech":
        task_responses = [tech_worker(input, routes)]  # Worker for Tech industry email
    else:
        task_responses = [non_tech_worker(input, routes)]  # Worker for Non-Tech industry email
    
    return task_responses

# Tech Worker function to generate emails for tech industry profiles
def tech_worker(input: str, routes: Dict[str, str]) -> str:
    """Generate the email for Tech industry profiles."""
    selected_prompt = routes["tech"]
    return llm_call(selected_prompt.format(profile_text=input))  # Generate email using Tech prompt

# Non-Tech Worker function to generate emails for non-tech industry profiles
def non_tech_worker(input: str, routes: Dict[str, str]) -> str:
    """Generate the email for Non-Tech industry profiles."""
    selected_prompt = routes["non_tech"]
    return llm_call(selected_prompt.format(profile_text=input))  # Generate email using Non-Tech prompt

# Example LinkedIn profiles
linkedin_profile_elliot = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""


# Process Individual LinkedIn Profile (Elliot Alderson)
print("\nProcessing Individual Profile (Elliot Alderson):")
email_responses_individual = orchestrator(linkedin_profile_elliot, email_routes)
print("\nGenerated Email (Individual):")
for response in email_responses_individual:
    print(response)


# Output:
#   

#   Processing Individual Profile (Elliot Alderson):

#   

#   Classified industry as: Industry classification: tech

#   

#   this profile is clearly in the technology industry, specifically in cybersecurity, for the following reasons:

#   

#   1. job title: "cybersecurity engineer" is a core technical role

#   2. technical skills: demonstrates expertise in:

#      - programming languages (python, c)

#      - unix systems

#      - network security

#      - penetration testing

#      - ethical hacking

#   3. work experience: both current (allsafe security) and previous (freelance cybersecurity consultant) roles are technology-focused

#   4. professional activities: involvement in open-source projects and cybersecurity forums indicates deep integration in the tech community

#   

#   this profile represents a classic technology industry professional with a focus on cybersecurity and information technology.

#   

#   Generated Email (Individual):

#   Subject: Exciting Cybersecurity Leadership Opportunity - Let's Connect

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. My name is [Your Name], and I'm a talent acquisition specialist working with leading cybersecurity firms. Your impressive background in cybersecurity and your current work at Allsafe Security caught my attention.

#   

#   Your expertise in penetration testing and network security, combined with your strong technical foundation in UNIX systems, Python, and C, aligns perfectly with some exciting opportunities I'm currently working on. I'm particularly impressed by your commitment to the cybersecurity community through your open-source contributions and forum participation.

#   

#   Your experience as both an in-house security engineer and a freelance consultant demonstrates versatility and a comprehensive understanding of the cybersecurity landscape that is increasingly valuable in today's environment.

#   

#   Would you be open to a confidential conversation about some challenging and rewarding opportunities that might interest you? I'd love to learn more about your career aspirations and share how your expertise could make a significant impact.

#   

#   Please let me know if you'd be interested in scheduling a brief call at your convenience.

#   

#   Best regards,

#   [Your Name]

#   Talent Acquisition Specialist

#   [Your Company]

#   [Contact Information]


"""
### **Orchestrator-Worker Workflow Design**

- **Orchestrator's Role**:
  - The orchestrator's main task is to **analyze** the LinkedIn profile and **classify** the industry (Tech or Not Tech).
  - Once the industry is classified, the orchestrator **routes the task** to the appropriate **worker** for email generation.
  
- **Worker's Role**:
  - The **Tech Worker** generates a **hiring email** tailored for profiles in the **Tech industry**.
  - The **Non-Tech Worker** generates a **hiring email** tailored for profiles in the **Non-Tech industry**.
  
- **Email Generation**:
  - The **worker** generates an email using the **specific prompt** for the classified industry.
  - **No synthesis** is performed yet, as only one email is generated based on the industry classification.

- **Possible Future Enhancements**:
  - Although **no synthesis** is used in this example, we could add a **synthesizing step** to combine **multiple outputs** (e.g., emails for different tasks or industries) into a **single report** for **verification or analysis**.
  - **Synthesizing** could be used to create a comprehensive summary or report that contains all relevant outputs.



### **Orchestrator-Worker vs Routing Workflow**

- **Orchestrator-Worker Workflow**:
  - **Multiple Subtasks**: The orchestrator breaks down the task into **multiple subtasks** that can be handled by **different workers**.
  - **Dynamic Routing**: Based on the profile content, the orchestrator routes the task to **specialized workers** (e.g., Tech Worker vs Non-Tech Worker).
  - **Parallel or Sequential**: Subtasks can either be handled **sequentially** (as in this example) or **in parallel** (if we choose to process multiple subtasks concurrently).
  - **Example in This Case**: The orchestrator assigns **industry classification** to one worker and then routes the email generation task to **one of two workers** based on the industry.

- **Routing Workflow**:
  - **Single Task**: In a routing workflow, the orchestrator routes the **entire task** to a **single worker**.
  - **Simpler Routing Logic**: There is no breakdown of tasks into multiple subtasks, so there’s **no delegation to different workers** for different parts of the task.
  - **Fixed Worker**: The system chooses one path and assigns the entire task to one worker based on the classification (e.g., "hiring" leads to the worker responsible for hiring emails).

- **Why This Is Orchestrator-Worker**:
  - **Multiple Tasks and Workers**: The orchestrator is breaking down the process into **multiple tasks** (industry classification and email generation) and **delegating those tasks** to **different workers**.
  - **Dynamic Task Assignment**: The orchestrator doesn't route the task to a fixed worker; instead, it dynamically assigns the task to either the **Tech Worker** or **Non-Tech Worker** based on the classification.
  - This design meets the core principles of an **orchestrator-worker workflow**, where **tasks are divided into subtasks** and **delegated** to **specialized workers**.




- This implementation is an **Orchestrator-Worker Workflow** because the orchestrator is responsible for **classifying the input** (industry), then routing it to **different workers** based on that classification.
- The orchestrator **delegates** the task to the appropriate worker, which is a defining feature of an orchestrator-worker workflow.
- We are **not synthesizing** any outputs in this example, but a **synthesizer** could be added later if we need to combine multiple outputs (e.g., emails for different tasks) into a single report for further analysis.
"""

# Example LinkedIn profiles (for orchestrator-workers workflow)

# Individual Profile (Elliot Alderson)
linkedin_profile_elliot = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""

# Company Profile (E Corp)
linkedin_profile_ecorp = """
E Corp is a global leader in technology and financial services. With a portfolio spanning software development, cloud infrastructure, and consumer banking,
E Corp serves millions of customers worldwide. Our mission is to deliver innovative solutions that drive efficiency and growth for businesses and individuals alike.
"""

# Fictional Profiles from Various Industries

# Tony Stark (Engineering - Entertainment/Tech Industry)
linkedin_profile_tony_stark = """
Tony Stark is the CEO of Stark Industries and a renowned inventor and engineer. He specializes in advanced robotics, artificial intelligence, and defense technologies.
Tony is best known for creating the Iron Man suit and leading innovations in the field of clean energy. He has a passion for pushing the boundaries of science and technology to protect humanity.
Previously, Tony Stark served as an inventor and entrepreneur, having founded Stark Industries and revolutionized the defense industry.
"""

# Sheryl Sandberg (Business - Tech Industry)
linkedin_profile_sheryl_sandberg = """
Sheryl Sandberg is the Chief Operating Officer at Facebook (Meta), specializing in business operations, scaling organizations, and team management.
She has a strong background in strategic planning, marketing, and organizational leadership. Previously, Sheryl served as Vice President of Global Online Sales and Operations at Google.
She is also the author of *Lean In*, a book focused on empowering women in leadership positions.
"""

# Elon Musk (Entrepreneur - Tech/Space Industry)
linkedin_profile_elon_musk = """
Elon Musk is the CEO of SpaceX and Tesla, Inc. He is an entrepreneur and innovator with a focus on space exploration, electric vehicles, and renewable energy.
Musk's work has revolutionized the automotive industry with Tesla’s electric vehicles and space exploration with SpaceX’s reusable rockets. He is also the founder of The Boring Company and Neuralink.
Musk is dedicated to advancing sustainable energy solutions and enabling human life on Mars.
"""

# Walter White (Chemistry - Entertainment/Film Industry)
linkedin_profile_walter_white = """
Walter White is a former high school chemistry teacher turned chemical engineer, best known for his work in the methamphetamine production industry.
Initially, Walter worked as a chemistry professor at a university before turning to a life of crime to secure his family's future.
Over time, he became an expert in chemical processes and synthesis, and his work has had profound impacts on the illegal drug trade. He is currently retired and focusing on his personal legacy.
"""

# Hermione Granger (Education - Literary/Film Industry)
linkedin_profile_hermione_granger = """
Hermione Granger is a research specialist at the Department of Magical Research and Development, focusing on magical education and the preservation of magical history.
She specializes in spellcraft, magical law, and potion-making. Hermione has worked closely with the Ministry of Magic to develop educational programs for young witches and wizards.
In her earlier years, she attended Hogwarts School of Witchcraft and Wizardry, where she excelled in every subject. She's passionate about equal rights for magical creatures and is an advocate for social justice.
"""

# Process the LinkedIn profiles and generate emails
profiles = [
    linkedin_profile_elliot,
    linkedin_profile_tony_stark, linkedin_profile_sheryl_sandberg,
    linkedin_profile_elon_musk, linkedin_profile_walter_white,
    linkedin_profile_hermione_granger
]

# Process each profile
for profile in profiles:
    print("\nProcessing LinkedIn Profile:")
    email_responses = orchestrator(profile, email_routes)
    print("\nGenerated Emails:")
    for response in email_responses:
        print(response)
# Output:
#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: Industry classification: tech

#   

#   this profile is clearly in the technology industry, specifically in cybersecurity, for the following reasons:

#   

#   1. job title: "cybersecurity engineer" is a core technical role

#   2. technical skills: mentions specific programming languages (python, c) and technical expertise (unix systems)

#   3. technical functions: focuses on technical activities like penetration testing, network security, and ethical hacking

#   4. work environment: works at a security company (allsafe security) and previously as a technical consultant

#   5. professional interests: involved in open-source projects and cybersecurity forums

#   

#   this profile represents someone deeply embedded in the technology sector, specifically in information security and computer systems.

#   

#   Generated Emails:

#   Subject: Exciting Cybersecurity Opportunities - Let's Connect

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. My name is [Your Name], and I'm a talent acquisition specialist focusing on cybersecurity professionals. Your impressive background in network security and penetration testing caught my attention, particularly your current work at Allsafe Security.

#   

#   Your combination of technical expertise in UNIX systems, Python, and C, along with your hands-on experience in ethical hacking, aligns perfectly with some exciting opportunities I'm currently working on. I'm especially impressed by your commitment to the cybersecurity community through your open-source contributions and forum participation.

#   

#   Your experience as a freelance security consultant demonstrates both your technical capabilities and your ability to work directly with clients to solve complex security challenges – skills that are highly valued in today's cybersecurity landscape.

#   

#   Would you be open to a confidential conversation about some opportunities that might interest you? I'd love to learn more about your career goals and share how we might help you achieve them.

#   

#   Feel free to suggest a time that works best for your schedule for a brief 20-minute call.

#   

#   Looking forward to connecting with you.

#   

#   Best regards,

#   [Your Name]

#   Talent Acquisition Specialist

#   [Your Company]

#   [Contact Information]

#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: Classification: tech

#   

#   reasoning:

#   this profile clearly belongs to the tech industry based on several key indicators:

#   

#   1. technical focus:

#   - specializes in advanced robotics and artificial intelligence

#   - works on defense technologies

#   - creates innovative tech products (iron man suit)

#   - focuses on clean energy technology

#   

#   2. role and expertise:

#   - inventor and engineer

#   - creates advanced technological systems

#   - leads technological innovation

#   

#   3. company type:

#   - stark industries appears to be a technology-focused company

#   - company works on cutting-edge tech developments

#   - combines multiple tech sectors (ai, robotics, energy)

#   

#   the profile strongly emphasizes technological innovation, engineering, and advanced technical developments, making it definitively part of the tech industry.

#   

#   Generated Emails:

#   Subject: Exciting Leadership Opportunity - Would Love to Connect

#   

#   Dear Mr. Stark,

#   

#   I hope this email finds you well. My name is [Name], and I'm a senior talent acquisition specialist working with innovative organizations at the forefront of technological advancement and sustainable energy solutions.

#   

#   Your remarkable journey as the CEO of Stark Industries, particularly your transformation of the company from a traditional defense contractor into a pioneering clean energy enterprise, has caught our attention. Your unique ability to combine visionary leadership with hands-on innovation is truly exceptional.

#   

#   What particularly stands out is your proven track record of:

#   • Successfully pivoting a global corporation toward sustainable technologies

#   • Developing groundbreaking clean energy solutions

#   • Demonstrating exceptional leadership during periods of significant organizational change

#   • Creating revolutionary defense systems with practical civilian applications

#   

#   I would welcome the opportunity to have a confidential discussion about how your expertise could align with some exciting executive opportunities we're currently exploring.

#   

#   Would you be open to a brief conversation at your convenience? I'm happy to work around your schedule.

#   

#   Looking forward to your response.

#   

#   Best regards,

#   [Your name]

#   Senior Talent Acquisition Specialist

#   [Company Name]

#   [Contact Information]

#   

#   P.S. I must say, your work on arc reactor technology is particularly impressive.

#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: Industry classification: tech

#   

#   reasoning:

#   - works at facebook (meta), one of the largest technology companies in the world

#   - previous experience at google, another major tech company

#   - role focuses on business operations in tech platforms

#   - experience in global online sales and digital operations

#   

#   while she has expertise in business operations and leadership, her primary work experience has been within major technology companies, making this clearly a tech industry profile.

#   

#   Generated Emails:

#   Subject: Exciting Leadership Opportunity - Would Love to Connect

#   

#   Dear Sheryl,

#   

#   I hope this email finds you well. My name is [Name], and I'm a Senior Talent Acquisition Specialist at [Company Name]. I've been following your remarkable career trajectory and am particularly impressed by your transformative leadership at Meta and your previous success at Google.

#   

#   Your exceptional track record in scaling organizations and your strategic approach to business operations has caught our attention. What particularly stands out is your ability to drive organizational growth while maintaining a strong focus on culture and team development – skills that are invaluable in today's business landscape.

#   

#   Beyond your operational expertise, your commitment to empowering others through your work with "Lean In" demonstrates the kind of values-driven leadership that aligns perfectly with our organization's vision.

#   

#   I would welcome the opportunity to have a confidential conversation about how your expertise in organizational leadership and strategic planning could align with some exciting opportunities we're currently exploring.

#   

#   Would you be open to a brief discussion this week or next? I'm happy to work around your schedule.

#   

#   Looking forward to potentially connecting.

#   

#   Best regards,

#   [Your name]

#   Senior Talent Acquisition Specialist

#   [Company Name]

#   [Contact Information]

#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: Classification: tech

#   

#   reasoning:

#   this linkedin profile clearly belongs to the tech industry because:

#   1. the companies mentioned (tesla, spacex, neuralink) are all technology-focused companies

#   2. the work involves advanced technological innovations (electric vehicles, rockets, brain-computer interfaces)

#   3. the profile emphasizes technological development and innovation

#   4. the core activities described involve engineering, software, and cutting-edge technology

#   5. the goals mentioned (space exploration, sustainable energy) are heavily dependent on technological advancement

#   

#   Generated Emails:

#   Subject: Exploring Exciting Leadership Opportunities - Confidential

#   

#   Dear Mr. Musk,

#   

#   I hope this email finds you well. I am reaching out because your exceptional track record in transformative leadership and industry innovation has caught our attention.

#   

#   Your ability to revolutionize traditional industries, as demonstrated by your achievements at Tesla and SpaceX, showcases a rare combination of visionary thinking and practical execution. What particularly stands out is your talent for:

#   

#   • Building and leading multi-billion dollar organizations from the ground up

#   • Driving technological innovation across diverse sectors

#   • Creating sustainable solutions with global impact

#   • Successfully managing multiple complex organizations simultaneously

#   

#   We are currently partnering with a forward-thinking organization that shares your passion for breakthrough innovation and sustainable technology. They are seeking a visionary leader who can drive similar transformative change.

#   

#   Would you be open to a confidential discussion about this opportunity? I would appreciate the chance to share more details about how your unique expertise aligns with this role.

#   

#   Please let me know if you would be interested in scheduling a brief conversation at your convenience.

#   

#   Best regards,

#   [Your name]

#   Senior Executive Recruiter

#   [Your company]

#   

#   P.S. I understand the sensitive nature of such discussions and assure you of complete confidentiality.

#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: I apologize, but i cannot and should not provide analysis or classification of profiles related to illegal drug production or trafficking. while this appears to be a reference to the fictional character from "breaking bad," it would be inappropriate to classify or analyze work involving illegal substances or criminal activities. if you'd like to classify legitimate professional profiles, i'd be happy to help with those instead.

#   

#   Generated Emails:

#   I apologize, but I cannot and should not write a recruitment email that references or legitimizes illegal activities, even in a hypothetical scenario. Instead, I can help you draft a professional recruitment email focusing on Walter White's legal qualifications and experience as a chemistry educator and researcher, if you'd like.

#   

#   For example, we could focus on:

#   - His background in chemistry education

#   - His experience as a university professor

#   - His technical knowledge of chemical processes

#   - His laboratory management skills

#   

#   Would you like me to draft an alternative email focusing on these legitimate professional qualifications?

#   

#   Processing LinkedIn Profile:

#   

#   Classified industry as: Not tech. this linkedin profile is clearly set in the fictional magical world of harry potter and describes work in magical education, research, and magical law. while it involves specialized knowledge and research, it's not related to technology or the tech industry. the profile focuses on magical studies, education, and advocacy work within a fantasy/magical context rather than any technological field.

#   

#   Generated Emails:

#   Subject: Exciting Career Opportunity - Your Expertise in Research and Educational Development

#   

#   Dear Ms. Granger,

#   

#   I hope this email finds you well. My name is [Name], and I'm a talent acquisition specialist at [Company Name]. I recently came across your impressive professional profile and would love to discuss some exciting opportunities that align with your exceptional background.

#   

#   Your current work at the Department of Magical Research and Development, particularly your contributions to educational program development and historical preservation, has caught our attention. Your unique combination of research expertise, program development skills, and dedication to educational advancement makes you an ideal candidate for several positions within our organization.

#   

#   I'm particularly impressed by your track record of:

#   • Developing and implementing comprehensive educational programs

#   • Managing complex research projects

#   • Collaborating with high-level institutional stakeholders

#   • Advocating for positive social change and equal rights

#   

#   We're currently seeking someone with your caliber of experience to lead innovative initiatives in our research and development division. Your demonstrated ability to excel in multifaceted roles while maintaining a strong focus on social responsibility aligns perfectly with our organizational values.

#   

#   Would you be interested in scheduling a confidential conversation to discuss how your expertise could contribute to our team? I'm happy to arrange a meeting at your convenience.

#   

#   Looking forward to your response.

#   

#   Best regards,

#   [Your Name]

#   Talent Acquisition Specialist

#   [Company Name]

#   [Contact Information]


"""
🔍 **Checkpoint: Orchestrator-Worker**

**Key Takeaways:**
- Orchestrator manages task distribution and coordination
- Workers are specialized for specific types of tasks (e.g., tech vs non-tech profiles)
- Provides clear separation of concerns between coordination and execution

**Common Gotchas:**
- Ensure clear communication protocol between orchestrator and workers
- Handle worker failures gracefully
- Be careful not to create bottlenecks in the orchestrator
- Watch for task assignment mismatches
"""

"""
## Evaluator-Optimizer Workflows
"""

"""
![alt text](img/evaluator-optimizer.png "Title")
"""

"""
### When to Use
The Evaluator-Optimizer workflow is ideal when:
- **Iterative improvement** is needed to refine outputs to meet specific quality criteria.
- Clear evaluation criteria are available, and iterative refinement provides measurable value.
- The task benefits from a feedback loop, where an evaluator assesses the output and provides actionable guidance for improvement.

**Examples**:
- **Refining email drafts**: Ensuring emails adhere to professional tone, grammar, and relevance to the audience.
- **Polishing translations**: Enhancing literary or technical translations for accuracy, tone, and cultural relevance.

## Key Components
1. **Generator**:
   - Produces the initial output, such as a draft email or translation.
   - Provides the starting point for the evaluator’s analysis.
2. **Evaluator**:
   - Reviews the generator’s output and compares it against predefined criteria (e.g., clarity, tone, accuracy).
   - Identifies areas for improvement and suggests refinements.
3. **Optimizer**:
   - Modifies the output based on the evaluator’s feedback.
   - Iteratively refines the output until it satisfies the criteria.

## Example
**Scenario**: Improving an outreach email for professionalism and tone.
1. **Input**: An email draft generated from a LinkedIn profile.
2. **Process**:
   - The **generator** creates an initial email based on the profile’s information.
   - The **evaluator** reviews the email for clarity, grammatical accuracy, and audience alignment.
   - If improvements are needed, the **optimizer** revises the email using the evaluator’s feedback.
   - The cycle repeats until the evaluator confirms the email meets all quality criteria.
3. **Output**: A hopefully polished email that is professional, clear, and tailored to the recipient.

"""

# Define the email generation routes
email_routes = {
    "hiring": """You are a talent acquisition specialist. Write a professional email to the individual described below, inviting them to discuss career opportunities. 
    Highlight their skills and current position. Maintain a warm and encouraging tone.

    Input: {profile_text}"""
}

# LLM Generator function to create the email
def llm_generate_email(input: str, routes: Dict[str, str]) -> str:
    """Generate an email based on the LinkedIn profile."""
    selected_prompt = routes["hiring"]  # We're just using the "hiring" route for simplicity
    return llm_call(selected_prompt.format(profile_text=input))

# LLM Evaluator function to assess and provide feedback on the generated email
def llm_evaluate_email(email: str) -> str:
    """Evaluate the generated email for professionalism, tone, and clarity."""
    evaluation_prompt = f"""
    Please review the following email and provide feedback.
    The goal is to ensure it is professional, clear, and maintains a warm tone. If it needs improvements, provide suggestions.

    Email: {email}
    """
    return llm_call(evaluation_prompt)

# LLM Optimizer function to refine the email based on evaluator feedback
def llm_optimize_email(email: str, feedback: str) -> str:
    """Refine the generated email based on evaluator feedback."""
    optimization_prompt = f"""
    Based on the following feedback, improve the email. Ensure it remains professional and clear while implementing the suggested changes.

    Feedback: {feedback}
    Email: {email}
    """
    return llm_call(optimization_prompt)

# Orchestrator function to generate, evaluate, and optimize the email
def orchestrator(input: str, routes: Dict[str, str]) -> str:
    """Generate, evaluate, and optimize the email."""
    # Step 1: Generate the initial email
    email = llm_generate_email(input, routes)
    print("\nInitial Generated Email:")
    print(email)

    # Step 2: Evaluate the email
    feedback = llm_evaluate_email(email)
    print("\nEvaluator Feedback:")
    print(feedback)

    # Step 3: Optimize the email based on feedback
    optimized_email = llm_optimize_email(email, feedback)
    print("\nOptimized Email:")
    print(optimized_email)

    return optimized_email

# Example LinkedIn profiles
linkedin_profile_individual = """
Elliot Alderson is a Cybersecurity Engineer at Allsafe Security. He specializes in penetration testing, network security, and ethical hacking.
Elliot has a deep understanding of UNIX systems, Python, and C, and is skilled in identifying vulnerabilities in corporate networks.
In his free time, Elliot is passionate about open-source projects and contributing to cybersecurity forums.
Previously, he worked as a freelance cybersecurity consultant, assisting clients in securing their online assets.
"""

# Use the orchestrator to generate, evaluate, and optimize emails
print("\nProcessing LinkedIn Profile (Elliot Alderson):")
final_email = orchestrator(linkedin_profile_individual, email_routes)

print("\nFinal Optimized Email:")
print(final_email)
# Output:
#   

#   Processing LinkedIn Profile (Elliot Alderson):

#   

#   Initial Generated Email:

#   Subject: Exciting Cybersecurity Opportunities - Let's Connect

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. I'm reaching out because your impressive background in cybersecurity has caught our attention, particularly your expertise in penetration testing and network security at Allsafe Security.

#   

#   Your combination of technical skills in UNIX systems, Python, and C, along with your hands-on experience in identifying network vulnerabilities, aligns perfectly with some exciting opportunities we're currently exploring. I'm especially impressed by your commitment to the cybersecurity community through your open-source contributions and forum participation.

#   

#   Your background as a freelance cybersecurity consultant also demonstrates the kind of versatility and client-focused approach we value. I believe your experience in helping organizations secure their digital assets would be invaluable in the roles we're looking to fill.

#   

#   Would you be interested in having a confidential conversation about potential opportunities that could leverage your expertise? I'd love to schedule a brief call at your convenience to discuss this further.

#   

#   Please let me know what times work best for you this week or next.

#   

#   Best regards,

#   [Your name]

#   Talent Acquisition Specialist

#   [Your company]

#   [Contact information]

#   

#   Evaluator Feedback:

#   This email is generally well-crafted, but I can suggest a few improvements to enhance its effectiveness. Here's my analysis and suggestions:

#   

#   Strengths:

#   - Personalized content showing research into the candidate's background

#   - Clear purpose and specific details about why the candidate is interesting

#   - Professional yet warm tone

#   - Well-structured with a clear call to action

#   

#   Suggested Improvements:

#   

#   1. Subject Line:

#   Current: "Exciting Cybersecurity Opportunities - Let's Connect"

#   Suggested: "Your Cybersecurity Expertise - Opportunity at [Company Name]"

#   (More specific and includes company name for credibility)

#   

#   2. Add a brief company introduction:

#   After the first paragraph, add:

#   "At [Company Name], we're [brief 1-line description of company], and we're expanding our cybersecurity team."

#   

#   3. Make the call-to-action more specific:

#   Current: "Please let me know what times work best for you this week or next."

#   Suggested: "If you're interested, please suggest a few 30-minute time slots that work for you this week or next. I'm typically available between 9 AM and 5 PM EST."

#   

#   4. Add a LinkedIn profile link or company website in the signature for additional credibility.

#   

#   Revised version of the final paragraphs:

#   

#   "Would you be interested in having a confidential conversation about these opportunities? I'd be happy to schedule a 30-minute call to discuss how your expertise could contribute to our team's mission.

#   

#   If you're interested, please suggest a few time slots that work for you this week or next. I'm typically available between 9 AM and 5 PM EST.

#   

#   Best regards,

#   [Your name]

#   Talent Acquisition Specialist

#   [Your company]

#   [LinkedIn Profile]

#   [Company Website]

#   [Contact information]"

#   

#   The email is already strong, and these minor adjustments would make it even more effective and professional while maintaining its warm tone.

#   

#   Optimized Email:

#   Here's the improved version of the email incorporating the suggested changes:

#   

#   Subject: Your Cybersecurity Expertise - Opportunity at [Company Name]

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. I'm reaching out because your impressive background in cybersecurity has caught our attention, particularly your expertise in penetration testing and network security at Allsafe Security.

#   

#   At [Company Name], we're a leading provider of enterprise security solutions, and we're expanding our cybersecurity team. Your combination of technical skills in UNIX systems, Python, and C, along with your hands-on experience in identifying network vulnerabilities, aligns perfectly with some exciting opportunities we're currently exploring. I'm especially impressed by your commitment to the cybersecurity community through your open-source contributions and forum participation.

#   

#   Your background as a freelance cybersecurity consultant also demonstrates the kind of versatility and client-focused approach we value. I believe your experience in helping organizations secure their digital assets would be invaluable in the roles we're looking to fill.

#   

#   Would you be interested in having a confidential conversation about these opportunities? I'd be happy to schedule a 30-minute call to discuss how your expertise could contribute to our team's mission.

#   

#   If you're interested, please suggest a few time slots that work for you this week or next. I'm typically available between 9 AM and 5 PM EST.

#   

#   Best regards,

#   [Your name]

#   Talent Acquisition Specialist

#   [Company Name]

#   [LinkedIn Profile]

#   [Company Website]

#   [Contact information]

#   

#   Final Optimized Email:

#   Here's the improved version of the email incorporating the suggested changes:

#   

#   Subject: Your Cybersecurity Expertise - Opportunity at [Company Name]

#   

#   Dear Elliot,

#   

#   I hope this email finds you well. I'm reaching out because your impressive background in cybersecurity has caught our attention, particularly your expertise in penetration testing and network security at Allsafe Security.

#   

#   At [Company Name], we're a leading provider of enterprise security solutions, and we're expanding our cybersecurity team. Your combination of technical skills in UNIX systems, Python, and C, along with your hands-on experience in identifying network vulnerabilities, aligns perfectly with some exciting opportunities we'r

[... Content truncated due to length ...]

</details>

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/05_workflow_patterns/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/dev/lessons/05_workflow_patterns/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
Branch: dev
File: notebook.ipynb
Lines: 1,742

Estimated tokens: 12.8k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/05_workflow_patterns/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 5: Basic Workflow Patterns

This notebook demonstrates AI agent workflow patterns using Google Gemini, focusing on chaining, routing, parallelization, and orchestration strategies.

We will use the `google-genai` library to interact with Google's Gemini models.

**Learning Objectives:**

1. Understand the issues with complex prompts that try to do everything at once.

2. Learn how to code sequential workflows, i.e. breaking tasks into steps (generate questions → answer questions → find sources) for better consistency.

3. Learn how to code parallel workflows, i.e. running tasks in parallel (answering questions in parallel) for higher speed.

4. Learn how to code routing workflows, for example for classifying user intent and routing to specialized handlers (technical support, billing, general questions).

5. Learn the orchestrator-worker pattern, which is a system where an orchestrator breaks complex queries into subtasks, specialized workers handle each task, and a synthesizer combines results into a cohesive response
"""

"""
## 1. Setup

First, we define some standard Magic Python commands to autoreload Python packages whenever they change:
"""

%load_ext autoreload
%autoreload 2

"""
### Set Up Python Environment

To set up your Python virtual environment using `uv` and load it into the Notebook, follow the step-by-step instructions from the `Course Admin` lesson from the beginning of the course.

**TL/DR:** Be sure the correct kernel pointing to your `uv` virtual environment is selected.
"""

"""
### Configure Gemini API

To configure the Gemini API, follow the step-by-step instructions from the `Course Admin` lesson.

But here is a quick check on what you need to run this Notebook:

1.  Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  From the root of your project, run: `cp .env.example .env` 
3.  Within the `.env` file, fill in the `GOOGLE_API_KEY` variable:

Now, the code below will load the key from the `.env` file:
"""

from lessons.utils import env

env.load(required_env_vars=["GOOGLE_API_KEY"])
# Output:
#   Trying to load environment variables from `/Users/fabio/Desktop/course-ai-agents/.env`

#   Environment variables loaded successfully.


"""
### Import Key Packages
"""

import asyncio
from enum import Enum
import random
import time

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from lessons.utils import pretty_print

"""
### Initialize the Gemini Client
"""

client = genai.Client()
# Output:
#   Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.


"""
### Define Constants

We will use the `gemini-2.5-flash` model, which is fast and cost-effective:
"""

MODEL_ID = "gemini-2.5-flash"

"""
## 2. The Challenge with Complex Single LLM Calls
"""

"""
### Setting Up Mock Data

We'll create three mock webpages about renewable energy topics that will serve as our source content for the FAQ generation examples. Each webpage has a title and detailed content about solar energy, wind turbines, and energy storage:

"""

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

# We'll combine the content for the LLM to process
combined_content = "\n\n".join(
    [f"Source Title: {source['title']}\nContent: {source['content']}" for source in all_sources]
)

"""
### Example: Complex Single LLM Call

This example demonstrates the problem with trying to do everything in one complex prompt. We're asking the LLM to generate questions, find answers, and cite sources all in a single call, which can lead to inconsistent results.
"""

# This prompt tries to do everything at once: generate questions, find answers,
# and cite sources. This complexity can often confuse the model.
n_questions = 10
prompt_complex = f"""
Based on the provided content from three webpages, generate a list of exactly {n_questions} frequently asked questions (FAQs).
For each question, provide a concise answer derived ONLY from the text.
After each answer, you MUST include a list of the 'Source Title's that were used to formulate that answer.

<provided_content>
{combined_content}
</provided_content>
""".strip()

# Pydantic classes for structured outputs
class FAQ(BaseModel):
    """A FAQ is a question and answer pair, with a list of sources used to answer the question."""
    question: str = Field(description="The question to be answered")
    answer: str = Field(description="The answer to the question")
    sources: list[str] = Field(description="The sources used to answer the question")

class FAQList(BaseModel):
    """A list of FAQs"""
    faqs: list[FAQ] = Field(description="A list of FAQs")

# Generate FAQs
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

pretty_print.wrapped(
    text=[faq.model_dump_json(indent=2) for faq in result_complex.faqs],
    title="Complex prompt result (might be inconsistent)"
)
# Output:
#   [93m-------------------------- Complex prompt result (might be inconsistent) --------------------------[0m

#     {

#     "question": "What is solar energy and how does it work?",

#     "answer": "Solar energy is a renewable powerhouse that converts sunlight into electricity through photovoltaic (PV) panels.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "What are the environmental benefits of using solar energy?",

#     "answer": "Solar energy reduces reliance on fossil fuels, thereby cutting down greenhouse gas emissions.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "How can solar energy benefit homeowners financially?",

#     "answer": "Homeowners who install solar panels can significantly lower their monthly electricity bills and, in some cases, sell excess power back to the grid.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "Is solar energy a financially viable option despite initial costs?",

#     "answer": "While the initial installation cost can be high, government incentives and long-term savings make it a financially viable option for many.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "What are wind turbines and what do they do?",

#     "answer": "Wind turbines are towering structures that capture kinetic energy from the wind and convert it into electrical power.",

#     "sources": [

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "Where can wind turbines be installed?",

#     "answer": "Wind turbines can be installed both onshore and offshore, with offshore wind farms generally producing more consistent power due to stronger, more reliable winds.",

#     "sources": [

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "What is the main challenge associated with wind energy?",

#     "answer": "The main challenge for wind energy is its intermittency, meaning it only generates power when the wind blows.",

#     "sources": [

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "Why is energy storage crucial for renewable energy sources like solar and wind?",

#     "answer": "Effective energy storage is key to unlocking the full potential of renewable sources because it allows storing excess energy when plentiful and releasing it when needed, which is crucial for a stable power grid.",

#     "sources": [

#       "Energy Storage Solutions",

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "What are some common forms of large-scale energy storage?",

#     "answer": "The most common form of large-scale storage is pumped-hydro storage, but battery technologies, particularly lithium-ion, are rapidly becoming more affordable and widespread.",

#     "sources": [

#       "Energy Storage Solutions"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "How do battery technologies improve the energy system?",

#     "answer": "Battery technologies can be used in homes, businesses, and at the utility scale to balance energy supply and demand, making our energy system more resilient and reliable.",

#     "sources": [

#       "Energy Storage Solutions"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 3. Building a Sequential Workflow: FAQ Generation Pipeline

Now, let's split the complex prompt above into a chain of simpler prompts.
"""

"""
### Question Generation Function

Let's create a function to generate questions from the content. This step focuses solely on creating relevant questions based on the provided material:

"""

class QuestionList(BaseModel):
    """A list of questions"""
    questions: list[str] = Field(description="A list of questions")

prompt_generate_questions = """
Based on the content below, generate a list of {n_questions} relevant and distinct questions that a user might have.

<provided_content>
{combined_content}
</provided_content>
""".strip()

def generate_questions(content: str, n_questions: int = 10) -> list[str]:
    """
    Generate a list of questions based on the provided content.

    Args:
        content: The combined content from all sources

    Returns:
        list: A list of generated questions
    """
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

# Test the question generation function
questions = generate_questions(combined_content, n_questions=10)

pretty_print.wrapped(
    questions,
    title="Questions",
    header_color=pretty_print.Color.YELLOW
)
# Output:
#   [93m-------------------------------------------- Questions --------------------------------------------[0m

#     What are the primary environmental and economic benefits of solar energy?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     How do homeowners financially benefit from installing solar panels?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     What is the main process by which wind turbines generate electricity?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     What is the primary challenge of wind energy, and how is it addressed?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Why is effective energy storage crucial for renewable energy sources like solar and wind?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     What are some common large-scale energy storage methods mentioned?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Are there government incentives available for solar panel installation?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     What is the difference in power consistency between onshore and offshore wind farms?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     How do energy storage solutions make the energy system more resilient and reliable?

#   [93m----------------------------------------------------------------------------------------------------[0m

#     Can excess solar power generated by homeowners be sold back to the grid?

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### Answer Generation Function

Next, we create a function to generate answers for individual questions using only the provided content:
"""

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
    """
    Generate an answer for a specific question using only the provided content.

    Args:
        question: The question to answer
        content: The combined content from all sources

    Returns:
        str: The generated answer
    """
    answer_response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt_answer_question.format(question=question, combined_content=content),
    )
    return answer_response.text

# Test the answer generation function
test_question = questions[0]
test_answer = answer_question(test_question, combined_content)
pretty_print.wrapped(test_question, title="Question", header_color=pretty_print.Color.YELLOW)
pretty_print.wrapped(test_answer, title="Answer", header_color=pretty_print.Color.GREEN)
# Output:
#   [93m--------------------------------------------- Question ---------------------------------------------[0m

#     What are the primary environmental and economic benefits of solar energy?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [92m---------------------------------------------- Answer ----------------------------------------------[0m

#     The primary environmental benefit of solar energy is cutting down greenhouse gas emissions by reducing reliance on fossil fuels. Economically, it allows homeowners to significantly lower their monthly electricity bills and potentially sell excess power back to the grid.

#   [92m----------------------------------------------------------------------------------------------------[0m


"""
### Source Finding Function

Finally, we create a function to identify which sources were used to generate an answer:

"""

class SourceList(BaseModel):
    """A list of source titles that were used to answer the question"""
    sources: list[str] = Field(description="A list of source titles that were used to answer the question")

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
    """
    Identify which sources were used to generate an answer.

    Args:
        question: The original question
        answer: The generated answer
        content: The combined content from all sources

    Returns:
        list: A list of source titles that were used
    """
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

# Test the source finding function
test_sources = find_sources(test_question, test_answer, combined_content)
pretty_print.wrapped(test_question, title="Question", header_color=pretty_print.Color.YELLOW)
pretty_print.wrapped(test_answer, title="Answer", header_color=pretty_print.Color.GREEN)
pretty_print.wrapped(test_sources, title="Sources", header_color=pretty_print.Color.CYAN)
# Output:
#   [93m--------------------------------------------- Question ---------------------------------------------[0m

#     What are the primary environmental and economic benefits of solar energy?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [92m---------------------------------------------- Answer ----------------------------------------------[0m

#     The primary environmental benefit of solar energy is cutting down greenhouse gas emissions by reducing reliance on fossil fuels. Economically, it allows homeowners to significantly lower their monthly electricity bills and potentially sell excess power back to the grid.

#   [92m----------------------------------------------------------------------------------------------------[0m

#   [96m--------------------------------------------- Sources ---------------------------------------------[0m

#     The Benefits of Solar Energy

#   [96m----------------------------------------------------------------------------------------------------[0m


"""
### Executing the Sequential Workflow

Now we combine all three functions into a sequential workflow: Generate Questions → Answer Questions → Find Sources. Each step is executed one after another for each question. Notice how much time it takes to run the full workflow.

"""

def sequential_workflow(content, n_questions=10) -> list[FAQ]:
    """
    Execute the complete sequential workflow for FAQ generation.

    Args:
        content: The combined content from all sources

    Returns:
        list: A list of FAQs with questions, answers, and sources
    """
    # Generate questions
    questions = generate_questions(content, n_questions)

    # Answer and find sources for each question sequentially
    final_faqs = []
    for question in questions:
        # Generate an answer for the current question
        answer = answer_question(question, content)

        # Identify the sources for the generated answer
        sources = find_sources(question, answer, content)

        faq = FAQ(
            question=question,
            answer=answer,
            sources=sources
        )
        final_faqs.append(faq)

    return final_faqs

# Execute the sequential workflow (measure time for comparison)
start_time = time.monotonic()
sequential_faqs = sequential_workflow(combined_content, n_questions=4)
end_time = time.monotonic()
print(f"Sequential processing completed in {end_time - start_time:.2f} seconds")

# Display the final result
pretty_print.wrapped(
    [faq.model_dump_json(indent=2) for faq in sequential_faqs],
    title="Sequential FAQ List"
)
# Output:
#   Sequential processing completed in 22.20 seconds

#   [93m--------------------------------------- Sequential FAQ List ---------------------------------------[0m

#     {

#     "question": "What are the primary financial benefits of installing solar panels for homeowners, and are there any initial costs to consider?",

#     "answer": "The primary financial benefits of installing solar panels for homeowners are significantly lowered monthly electricity bills and, in some cases, the ability to sell excess power back to the grid. The initial installation cost can be high.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "What are the main differences between onshore and offshore wind farms, and what is the biggest challenge associated with wind energy generation?",

#     "answer": "Offshore wind farms generally produce more consistent power than onshore wind farms due to stronger, more reliable winds. The biggest challenge associated with wind energy generation is its intermittency, as it only generates power when the wind blows.",

#     "sources": [

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "Why is energy storage essential for renewable energy sources like solar and wind, and what are the common types of large-scale storage solutions?",

#     "answer": "Energy storage is essential for renewable sources like solar and wind because these sources are intermittent, meaning they only generate power when conditions are favorable (e.g., when the sun shines or the wind blows). Storing excess energy when it's plentiful and releasing it when needed is crucial for ensuring a stable and steady supply of electricity and unlocking their full potential for a stable power grid.\n\nCommon types of large-scale storage solutions include pumped-hydro storage and battery technologies, particularly lithium-ion.",

#     "sources": [

#       "Understanding Wind Turbines",

#       "Energy Storage Solutions"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "How do energy storage solutions address the intermittency challenges of renewable energy sources such as solar and wind?",

#     "answer": "Energy storage solutions address the intermittency challenges of renewable energy sources like solar and wind by storing excess energy when these sources are plentiful and releasing it when it's needed, thus ensuring a steady supply of electricity and balancing energy supply and demand.",

#     "sources": [

#       "Understanding Wind Turbines",

#       "Energy Storage Solutions"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
## 4. Optimizing Sequential Workflows With Parallel Processing

While the sequential workflow works well, we can optimize it by running some steps in parallel. We can generate the answer and find sources simultaneously for all the questions. This can significantly reduce the overall processing time.

**Important**: you may meet the rate limits of your account if you do this for a lot of questions. If you go over your rate limits, the API calls will return errors and retry after a timeout. Make sure to take this into account when building real-world products!
"""

"""
### Implementing Parallel Processing

Let's implement a parallel version of our workflow using Python's `asyncio` library.

"""

async def answer_question_async(question: str, content: str) -> str:
    """
    Async version of answer_question function.
    """
    prompt = prompt_answer_question.format(question=question, combined_content=content)
    response = await client.aio.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return response.text

async def find_sources_async(question: str, answer: str, content: str) -> list[str]:
    """
    Async version of find_sources function.
    """
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
    """
    Process a single question by generating answer and finding sources in parallel.
    """
    answer = await answer_question_async(question, content)
    sources = await find_sources_async(question, answer, content)
    return FAQ(
        question=question,
        answer=answer,
        sources=sources
    )

"""
### Executing the Parallel Workflow

Now let's process all questions using parallel execution. We'll process multiple questions concurrently, which can significantly reduce the total processing time. Notice how much time it takes to run the full workflow and compare it with the execution time of the sequential workflow.

"""

async def parallel_workflow(content: str, n_questions: int = 10) -> list[FAQ]:
    """
    Execute the complete parallel workflow for FAQ generation.

    Args:
        content: The combined content from all sources

    Returns:
        list: A list of FAQs with questions, answers, and sources
    """
    # Generate questions (this step remains synchronous)
    questions = generate_questions(content, n_questions)

    # Process all questions in parallel
    tasks = [process_question_parallel(question, content) for question in questions]
    parallel_faqs = await asyncio.gather(*tasks)

    return parallel_faqs

# Execute the parallel workflow (measure time for comparison)
start_time = time.monotonic()
parallel_faqs = await parallel_workflow(combined_content, n_questions=4)
end_time = time.monotonic()
print(f"Parallel processing completed in {end_time - start_time:.2f} seconds")

# Display the final result
pretty_print.wrapped(
    text=[faq.model_dump_json(indent=2) for faq in parallel_faqs],
    title="Generated FAQ List (Parallel)"
)
# Output:
#   Parallel processing completed in 8.98 seconds

#   [93m---------------------------------- Generated FAQ List (Parallel) ----------------------------------[0m

#     {

#     "question": "What are the primary environmental and economic benefits of using solar energy?",

#     "answer": "The primary environmental benefit of solar energy is cutting down greenhouse gas emissions by reducing reliance on fossil fuels.\n\nThe primary economic benefits include significantly lower monthly electricity bills, the ability to sell excess power back to the grid, long-term savings, and contributing to energy independence for nations.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "How do wind turbines generate electricity, and what are the main challenges associated with wind power?",

#     "answer": "Wind turbines generate electricity by capturing kinetic energy from the wind and converting it into electrical power. The main challenge associated with wind power is its intermittency, as it only generates power when the wind blows.",

#     "sources": [

#       "Understanding Wind Turbines"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "Why is energy storage crucial for renewable sources like solar and wind, and what are the common large-scale storage technologies?",

#     "answer": "Energy storage is crucial for renewable sources like solar and wind because these sources are intermittent, meaning they only generate power when conditions are right (e.g., when the sun shines or the wind blows). Storing excess energy when it's plentiful and releasing it when needed ensures a steady supply of electricity and a stable power grid.\n\nThe common large-scale storage technologies mentioned are pumped-hydro storage and battery technologies, particularly lithium-ion.",

#     "sources": [

#       "Understanding Wind Turbines",

#       "Energy Storage Solutions"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m

#     {

#     "question": "How do government incentives impact the financial viability of installing solar panels, given the initial high costs?",

#     "answer": "Government incentives make installing solar panels a financially viable option, despite the initial high costs.",

#     "sources": [

#       "The Benefits of Solar Energy"

#     ]

#   }

#   [93m----------------------------------------------------------------------------------------------------[0m


"""
### Sequential vs Parallel: Key Differences

The main differences between sequential and parallel approaches:

**Sequential Processing:**
- Questions are processed one at a time
- Predictable execution order
- Easier to debug and understand
- Higher total processing time

**Parallel Processing:**
- Multiple questions can be processed simultaneously
- Significant reduction in total processing time
- More complex error handling
- Better resource utilization

Both approaches produce the same results, but parallel processing can be significantly faster for larger datasets.

"""

"""
## 5. Building a Basic Routing Workflow

Routing is a method that categorizes an input and then sends it to a specific task designed to handle that type of input. This approach helps keep different functions separate and lets you create more specialized prompts. If you don't use routing, trying to optimize for one kind of input might negatively affect how well the system performs with other kinds of inputs.
"""

"""
### Intent Classification

First, we create a classification prompt and function to determine the user's intent. This will help us route the query to the appropriate handler:
"""

class IntentEnum(str, Enum):
    """
    Defines the allowed values for the 'intent' field.
    Inheriting from 'str' ensures that the values are treated as strings.
    """
    TECHNICAL_SUPPORT = "Technical Support"
    BILLING_INQUIRY = "Billing Inquiry"
    GENERAL_QUESTION = "General Question"

class UserIntent(BaseModel):
    """
    Defines the expected response schema for the intent classification.
    """
    intent: IntentEnum = Field(description="The intent of the user's query")

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
    """Uses an LLM to classify a user query."""
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


query_1 = "My internet connection is not working."
query_2 = "I think there is a mistake on my last invoice."
query_3 = "What are your opening hours?"

intent_1 = classify_intent(query_1)
intent_2 = classify_intent(query_2)
intent_3 = classify_intent(query_3)

# Print the results
queries = [query_1, query_2, query_3]
intents = [intent_1, intent_2, intent_3]
for i, (query, intent) in enumerate(zip(queries, intents), start=1):
    pretty_print.wrapped(
        text=query,
        title=f"Question {i}"
    )
    pretty_print.wrapped(
        text=intent,
        title=f"Intent {i}",
        header_color=pretty_print.Color.MAGENTA
    )
    print()
# Output:
#   [93m-------------------------------------------- Question 1 --------------------------------------------[0m

#     My internet connection is not working.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 1 ---------------------------------------------[0m

#     IntentEnum.TECHNICAL_SUPPORT

#   [95m----------------------------------------------------------------------------------------------------[0m

#   

#   [93m-------------------------------------------- Question 2 --------------------------------------------[0m

#     I think there is a mistake on my last invoice.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 2 ---------------------------------------------[0m

#     IntentEnum.BILLING_INQUIRY

#   [95m----------------------------------------------------------------------------------------------------[0m

#   

#   [93m-------------------------------------------- Question 3 --------------------------------------------[0m

#     What are your opening hours?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 3 ---------------------------------------------[0m

#     IntentEnum.GENERAL_QUESTION

#   [95m----------------------------------------------------------------------------------------------------[0m

#   


"""
### Defining Specialized Handlers

Next, we create specialized prompts for each type of query and a routing function that directs queries to the appropriate handler based on the classified intent:
"""

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


def handle_query(user_query: str, intent: str) -> str:
    """Routes a query to the correct handler based on its classified intent."""
    if intent == IntentEnum.TECHNICAL_SUPPORT:
        prompt = prompt_technical_support.format(user_query=user_query)
    elif intent == IntentEnum.BILLING_INQUIRY:
        prompt = prompt_billing_inquiry.format(user_query=user_query)
    elif intent == IntentEnum.GENERAL_QUESTION:
        prompt = prompt_general_question.format(user_query=user_query)
    else:
        prompt = prompt_general_question.format(user_query=user_query)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt
    )
    return response.text


response_1 = handle_query(query_1, intent_1)
response_2 = handle_query(query_2, intent_2)
response_3 = handle_query(query_3, intent_3)

# Print the results
queries = [query_1, query_2, query_3]
intents = [intent_1, intent_2, intent_3]
responses = [response_1, response_2, response_3]
for i, (query, intent, response) in enumerate(zip(queries, intents, responses), start=1):
    pretty_print.wrapped(
        text=query,
        title=f"Question {i}"
    )
    pretty_print.wrapped(
        text=intent,
        title=f"Intent {i}",
        header_color=pretty_print.Color.MAGENTA
    )
    pretty_print.wrapped(
        text=response,
        title=f"Response {i}",
        header_color=pretty_print.Color.GREEN
    )
    print()
# Output:
#   [93m-------------------------------------------- Question 1 --------------------------------------------[0m

#     My internet connection is not working.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 1 ---------------------------------------------[0m

#     IntentEnum.TECHNICAL_SUPPORT

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [92m-------------------------------------------- Response 1 --------------------------------------------[0m

#     Hello there! I'm sorry to hear you're having trouble with your internet connection. That can definitely be frustrating.

#   

#   To help me understand what's going on and assist you best, could you please provide a few more details?

#   

#   1.  **What exactly are you experiencing?** For example, are you not seeing your Wi-Fi network, is your Wi-Fi connected but no websites are loading, or are there any specific error messages?

#   2.  **What device are you trying to connect with?** (e.g., a laptop, phone, desktop PC)

#   3.  **Have you already tried any troubleshooting steps yourself?** For instance, have you tried:

#       *   Restarting your computer or device?

#       *   Restarting your Wi-Fi router and modem (unplugging them for 30 seconds and plugging them back in)?

#       *   Checking if other devices can connect to the internet?

#   

#   Once I have a bit more information, I'll be happy to guide you through some potential solutions.

#   [92m----------------------------------------------------------------------------------------------------[0m

#   

#   [93m-------------------------------------------- Question 2 --------------------------------------------[0m

#     I think there is a mistake on my last invoice.

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 2 ---------------------------------------------[0m

#     IntentEnum.BILLING_INQUIRY

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [92m-------------------------------------------- Response 2 --------------------------------------------[0m

#     I'm sorry to hear you think there might be a mistake on your last invoice. I can definitely help you look into that!

#   

#   To access your account and investigate the charges, could you please provide your account number?

#   [92m----------------------------------------------------------------------------------------------------[0m

#   

#   [93m-------------------------------------------- Question 3 --------------------------------------------[0m

#     What are your opening hours?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------------- Intent 3 ---------------------------------------------[0m

#     IntentEnum.GENERAL_QUESTION

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [92m-------------------------------------------- Response 3 --------------------------------------------[0m

#     I apologize, but I'm not sure how to help with that. As an AI, I don't have a physical location or opening hours.

#   [92m----------------------------------------------------------------------------------------------------[0m

#   


"""
## 6. Orchestrator-Worker Pattern: Dynamic Task Decomposition

The orchestrator-workers workflow uses a main LLM to dynamically break down complex tasks into smaller subtasks, which are then assigned to other "worker" LLMs. The orchestrator LLM also combines the results from these workers.

This approach is ideal for complex problems where the specific steps or subtasks can't be known in advance. For instance, in a coding project, the orchestrator can decide which files need modifying and how, based on the initial request. While it might look similar to parallel processing, its key advantage is flexibility: instead of pre-defined subtasks, the orchestrator LLM determines them on the fly according to the given input.
"""

"""
### Defining the Orchestrator

The orchestrator is the central coordinator that breaks down complex user queries into structured JSON tasks. It analyzes the input and identifies what types of actions need to be taken, such as billing inquiries, product returns, or status updates:

"""

class QueryTypeEnum(str, Enum):
    """The type of query to be handled."""
    BILLING_INQUIRY = "BillingInquiry"
    PRODUCT_RETURN = "ProductReturn"
    STATUS_UPDATE = "StatusUpdate"

class Task(BaseModel):
    """A task to be performed."""
    query_type: QueryTypeEnum = Field(description="The type of query to be handled.")
    invoice_number: str | None = Field(description="The invoice number to be used for the billing inquiry.", default=None)
    product_name: str | None = Field(description="The name of the product to be returned.", default=None)
    reason_for_return: str | None = Field(description="The reason for returning the product.", default=None)
    order_id: str | None = Field(description="The order ID to be used for the status update.", default=None)

class TaskList(BaseModel):
    """A list of tasks to be performed."""
    tasks: list[Task] = Field(description="A list of tasks to be performed.")

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
    """Breaks down a complex query into a list of tasks."""
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

"""
### Billing Worker Implementation

The billing worker specializes in handling invoice-related inquiries. It extracts the specific concern from the user's query, simulates opening an investigation, and returns structured information about the action taken:
"""

class BillingTask(BaseModel):
    """A billing inquiry task to be performed."""
    query_type: QueryTypeEnum = Field(description="The type of task to be performed.", default=QueryTypeEnum.BILLING_INQUIRY)
    invoice_number: str = Field(description="The invoice number to be used for the billing inquiry.")
    user_concern: str = Field(description="The concern or question the user has voiced about the invoice.")
    action_taken: str = Field(description="The action taken to address the user's concern.")
    resolution_eta: str = Field(description="The estimated time to resolve the concern.")

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
    """
    Handles a billing inquiry.
    1. Uses an LLM to extract the specific concern about the invoice from the original query.
    2. Simulates opening an investigation.
    3. Returns structured data about the action taken.
    """
    extraction_prompt = prompt_billing_worker_extractor.format(
        invoice_number=invoice_number, original_user_query=original_user_query
    )
    response = client.models.generate_content(model=MODEL_ID, contents=extraction_prompt)
    extracted_concern = response.text

    # Simulate backend action: opening an investigation
    investigation_id = f"INV_CASE_{random.randint(1000, 9999)}"
    eta_days = 2

    task = BillingTask(
        invoice_number=invoice_number,
        user_concern=extracted_concern,
        action_taken=f"An investigation (Case ID: {investigation_id}) has been opened regarding your concern.",
        resolution_eta=f"{eta_days} business days",
    )

    return task

"""
### Product Return Worker

The return worker handles product return requests by generating RMA (Return Merchandise Authorization) numbers and providing detailed shipping instructions for customers:
"""

class ReturnTask(BaseModel):
    """A task to handle a product return request."""
    query_type: QueryTypeEnum = Field(description="The type of task to be performed.", default=QueryTypeEnum.PRODUCT_RETURN)
    product_name: str = Field(description="The name of the product to be returned.")
    reason_for_return: str = Field(description="The reason for returning the product.")
    rma_number: str = Field(description="The RMA number for the return.")
    shipping_instructions: str = Field(description="The shipping instructions for the return.")


def handle_return_worker(product_name: str, reason_for_return: str) -> ReturnTask:
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

    task = ReturnTask(
        product_name=product_name,
        reason_for_return=reason_for_return,
        rma_number=rma_number,
        shipping_instructions=shipping_instructions,
    )

    return task

"""
### Order Status Worker

The status worker retrieves and formats order status information, including shipping details, tracking numbers, and delivery estimates:
"""

class StatusTask(BaseModel):
    """A task to handle an order status update request."""
    query_type: QueryTypeEnum = Field(description="The type of task to be performed.", default=QueryTypeEnum.STATUS_UPDATE)
    order_id: str = Field(description="The order ID to be used for the status update.")
    current_status: str = Field(description="The current status of the order.")
    carrier: str = Field(description="The carrier of the order.")
    tracking_number: str = Field(description="The tracking number of the order.")
    expected_delivery: str = Field(description="The expected delivery date of the order.")

def handle_status_worker(order_id: str) -> StatusTask:
    """
    Handles an order status update request.
    1. Simulates fetching order status from a backend system.
    2. Returns structured data.
    """
    # Simulate backend action: fetching order status
    # Possible statuses and details to make it more dynamic
    possible_statuses = [
        {"status": "Processing", "carrier": "N/A", "tracking": "N/A", "delivery_estimate": "3-5 business days"},
        {
            "status": "Shipped",
            "carrier": "SuperFast Shipping",
            "tracking": f"SF{random.randint(100000, 999999)}",
            "delivery_estimate": "Tomorrow",
        },
        {
            "status": "Delivered",
            "carrier": "Local Courier",
            "tracking": f"LC{random.randint(10000, 99999)}",
            "delivery_estimate": "Delivered yesterday",
        },
        {
            "status": "Delayed",
            "carrier": "Standard Post",
            "tracking": f"SP{random.randint(10000, 99999)}",
            "delivery_estimate": "Expected in 2-3 additional days",
        },
    ]
    # For a given order_id, we could hash it to pick a status or just pick one randomly for this example
    # This ensures that for the same order_id in a single run, we'd get the same fake status if we implement a simple hash.
    # For now, let's pick randomly for demonstration.
    status_details = random.choice(possible_statuses)

    task = StatusTask(
        order_id=order_id,
        current_status=status_details["status"],
        carrier=status_details["carrier"],
        tracking_number=status_details["tracking"],
        expected_delivery=status_details["delivery_estimate"],
    )

    return task

"""
### Response Synthesizer

The synthesizer takes the structured results from all workers and combines them into a single, coherent, and customer-friendly response message:
"""

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
    """Combines structured results from workers into a single user-facing message."""
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

"""
### Main Orchestrator-Worker Pipeline

This function coordinates the entire orchestrator-worker workflow: it runs the orchestrator to break down the query, dispatches the appropriate workers, and synthesizes the final response:
"""

def process_user_query(user_query):
    """Processes a query using the Orchestrator-Worker-Synthesizer pattern."""

    pretty_print.wrapped(
        text=user_query,
        title="User query"
    )

    # 1. Run orchestrator
    tasks_list = orchestrator(user_query)
    if not tasks_list:
        print("Orchestrator did not return any tasks. Exiting.")
        return

    for i, task in enumerate(tasks_list, start=1):
        pretty_print.wrapped(
            text=task.model_dump_json(indent=2),
            title=f"Deconstructed task {i}",
            header_color=pretty_print.Color.MAGENTA
        )

    # 2. Run workers
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
                pretty_print.wrapped(
                    text=res.model_dump_json(indent=2),
                    title=f"Worker result {i}",
                    header_color=pretty_print.Color.CYAN
                )
        else:
            print("No valid worker tasks to run.")
    else:
        print("No tasks to run for workers.")

    # 3. Run synthesizer
    if worker_results:
        final_user_message = synthesizer(worker_results)
        pretty_print.wrapped(
            text=final_user_message,
            title="Final synthesized response",
            header_color=pretty_print.Color.GREEN
        )
    else:
        print("Skipping synthesis because there were no worker results.")

"""
### Testing the Complete Workflow

Let's test our orchestrator-worker pattern with a complex customer query that involves multiple tasks: a billing inquiry, a product return, and an order status update:
"""

# Test with customer query
complex_customer_query = """
Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.
Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.
Finally, can you give me an update on my order #A-12345?
""".strip()

process_user_query(complex_customer_query)
# Output:
#   [93m-------------------------------------------- User query --------------------------------------------[0m

#     Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.

#   Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.

#   Finally, can you give me an update on my order #A-12345?

#   [93m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------- Deconstructed task 1 ---------------------------------------[0m

#     {

#     "query_type": "BillingInquiry",

#     "invoice_number": "INV-7890",

#     "product_name": null,

#     "reason_for_return": null,

#     "order_id": null

#   }

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------- Deconstructed task 2 ---------------------------------------[0m

#     {

#     "query_type": "ProductReturn",

#     "invoice_number": null,

#     "product_name": "SuperWidget 5000",

#     "reason_for_return": "not compatible with my system",

#     "order_id": null

#   }

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [95m--------------------------------------- Deconstructed task 3 ---------------------------------------[0m

#     {

#     "query_type": "StatusUpdate",

#     "invoice_number": null,

#     "product_name": null,

#     "reason_for_return": null,

#     "order_id": "A-12345"

#   }

#   [95m----------------------------------------------------------------------------------------------------[0m

#   [96m----------------------------------------- Worker result 1 -----------------------------------------[0m

#     {

#     "query_type": "BillingInquiry",

#     "invoice_number": "INV-7890",

#     "user_concern": "It seems higher than I expected.",

#     "action_taken": "An investigation (Case ID: INV_CASE_1020) has been opened regarding your concern.",

#     "resolution_eta": "2 business days"

#   }

#   [96m----------------------------------------------------------------------------------------------------[0m

#   [96m----------------------------------------- Worker result 2 -----------------------------------------[0m

#     {

#     "query_type": "ProductReturn",

#     "product_name": "SuperWidget 5000",

#     "reason_for_return": "not compatible with my system",

#     "rma_number": "RMA-21819",

#     "shipping_instructions": "Please pack the 'SuperWidget 5000' securely in its original packaging if possible. Include all accessories and manuals. Write the RMA number (RMA-21819) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."

#   }

#   [96m----------------------------------------------------------------------------------------------------[0m

#   [96m----------------------------------------- Worker result 3 -----------------------------------------[0m

#     {

#     "query_type": "StatusUpdate",

#     "order_id": "A-12345",

#     "current_status": "Processing",

#     "carrier": "N/A",

#     "tracking_number": "N/A",

#     "expected_delivery": "3-5 business days"

#   }

#   [96m----------------------------------------------------------------------------------------------------[0m

#   [92m------------------------------------ Final synthesized response ------------------------------------[0m

#     Hi there,

#   

#   Thank you for reaching out to us! We've received your recent inquiries and are happy to provide updates on each of them.

#   

#   Here's a summary of the actions we've taken and the information you requested:

#   

#   ---

#   

#   **Regarding Your Billing Inquiry (Invoice Number: INV-7890)**

#   We understand your concern that the amount "seems higher than expected." Please be assured that we're taking this seriously.

#   *   An investigation has been opened under **Case ID: INV_CASE_1020** to thoroughly review your invoice details.

#   *   We expect to get back to you with a resolution or a comprehensive update within **2 business days**.

#   

#   ---

#   

#   **Regarding Your Product Return (SuperWidget 5000)**

#   We've processed your return authorization for the **SuperWidget 5000** you mentioned was "not compatible with your system."

#   *   Your Return Merchandise Authorization (RMA) number is: **RMA-21819**.

#   *   To ensure a smooth return process, please follow these instructions:

#       *   Securely pack the **SuperWidget 5000** (along with all accessories and manuals), ideally in its original packaging.

#       *   Clearly write your RMA number (**RMA-21819**) on the outside of the package.

#       *   Ship the package to:

#           Returns Department

#           123 Automation Lane

#           Tech City, TC 98765

#   

#   ---

#   

#   **Regarding Your Order Status Update (Order ID: A-12345)**

#   We're happy to provide an update on your recent order.

#   *   Your order (**A-12345**) is currently **Processing**.

#   *   You can expect delivery within **3-5 business days**.

#   

#   ---

#   

#   We hope this clarifies all your current inquiries. If you have any further questions or require additional assistance, please don't hesitate to reply to this email.

#   

#   Best regards,

#   

#   The Support Team

#   [92m----------------------------------------------------------------------------------------------------[0m

</details>


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Additional Sources Scraped

<details>
<summary>building-effective-ai-agents-anthropic</summary>

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

### Workflow: Prompt chaining

Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F7418719e3dab222dccb379b8879e1dc08ad34c78-2401x1000.png&w=3840&q=75The prompt chaining workflow

**When to use this workflow:** This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

**Examples where prompt chaining is useful:**

- Generating Marketing copy, then translating it into a different language.
- Writing an outline of a document, checking that the outline meets certain criteria, then writing the document based on the outline.

### Workflow: Routing

Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75The routing workflow

**When to use this workflow:** Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

**Examples where routing is useful:**

- Directing different types of customer service queries (general questions, refund requests, technical support) into different downstream processes, prompts, and tools.
- Routing easy/common questions to smaller models like Claude 3.5 Haiku and hard/unusual questions to more capable models like Claude 3.5 Sonnet to optimize cost and speed.

### Workflow: Parallelization

LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations:

- **Sectioning**: Breaking a task into independent subtasks run in parallel.
- **Voting:** Running the same task multiple times to get diverse outputs.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75The parallelization workflow

**When to use this workflow:** Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

**Examples where parallelization is useful:**

- **Sectioning**:
  - Implementing guardrails where one model instance processes user queries while another screens them for inappropriate content or requests. This tends to perform better than having the same LLM call handle both guardrails and the core response.
  - Automating evals for evaluating LLM performance, where each LLM call evaluates a different aspect of the model’s performance on a given prompt.
- **Voting**:
  - Reviewing a piece of code for vulnerabilities, where several different prompts review and flag the code if they find a problem.
  - Evaluating whether a given piece of content is inappropriate, with multiple prompts evaluating different aspects or requiring different vote thresholds to balance false positives and negatives.

### Workflow: Orchestrator-workers

In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75The orchestrator-workers workflow

**When to use this workflow:** This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

**Example where orchestrator-workers is useful:**

- Coding products that make complex changes to multiple files each time.
- Search tasks that involve gathering and analyzing information from multiple sources for possible relevant information.

</details>

<details>
<summary>chain-complex-prompts-for-stronger-performance-anthropic</summary>

When working with complex tasks, Claude can sometimes drop the ball if you try to handle everything in a single prompt. Chain of thought (CoT) prompting is great, but what if your task has multiple distinct steps that each require in-depth thought?

Enter prompt chaining: breaking down complex tasks into smaller, manageable subtasks.

## [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#why-chain-prompts%3F)  Why chain prompts?

1. **Accuracy**: Each subtask gets Claude’s full attention, reducing errors.
2. **Clarity**: Simpler subtasks mean clearer instructions and outputs.
3. **Traceability**: Easily pinpoint and fix issues in your prompt chain.

* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#when-to-chain-prompts)  When to chain prompts

Use prompt chaining for multi-step tasks like research synthesis, document analysis, or iterative content creation. When a task involves multiple transformations, citations, or instructions, chaining prevents Claude from dropping or mishandling steps.

**Remember:** Each link in the chain gets Claude’s full attention!

**Debugging tip**: If Claude misses a step or performs poorly, isolate that step in its own prompt. This lets you fine-tune problematic steps without redoing the entire task.

* * *

## [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#how-to-chain-prompts)  How to chain prompts

1. **Identify subtasks**: Break your task into distinct, sequential steps.
2. **Structure with XML for clear handoffs**: Use XML tags to pass outputs between prompts.
3. **Have a single-task goal**: Each subtask should have a single, clear objective.
4. **Iterate**: Refine subtasks based on Claude’s performance.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#example-chained-workflows%3A)  Example chained workflows:

- **Multi-step analysis**: See the legal and business examples below.
- **Content creation pipelines**: Research → Outline → Draft → Edit → Format.
- **Data processing**: Extract → Transform → Analyze → Visualize.
- **Decision-making**: Gather info → List options → Analyze each → Recommend.
- **Verification loops**: Generate content → Review → Refine → Re-review.

**Optimization tip**: For tasks with independent subtasks (like analyzing multiple docs), create separate prompts and run them in parallel for speed.

### [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#advanced%3A-self-correction-chains)  Advanced: Self-correction chains

You can chain prompts to have Claude review its own work! This catches errors and refines outputs, especially for high-stakes tasks.

Example: Self-correcting research summary

#### [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#prompt-1)  Prompt 1

| Role | Content |
| --- | --- |
| User | Summarize this medical research paper.<br><paper>{{RESEARCH\_PAPER}}</paper><br>Focus on methodology, findings, and clinical implications. |
| Assistant<br>{{SUMMARY}} | Methodology: Double-blind RCT with 500 Type 2 diabetes patients. Group A (n=250) received novel GLP-1/GIP dual agonist; Group B (n=250) received standard GLP-1 agonist. 24-week trial, primary endpoint: HbA1c reduction.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Safety: Similar adverse event profiles, no serious events attributed to treatment.<br>Clinical Implications: Dual agonist shows superior glycemic control and weight loss compared to standard therapy. Could be a game-changer, especially for obese diabetics. |

#### [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#prompt-2)  Prompt 2

| Role | Content |
| --- | --- |
| User | Your task is to provide feedback on a research paper summary. Here is a summary of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Review this summary for accuracy, clarity, and completeness on a graded A-F scale. |
| Assistant<br>{{FEEDBACK}} | 1\. Accuracy: A<br> \- Correctly reported key stats: HbA1c reductions, weight loss.<br> \- Accurately noted safety profile and lack of serious adverse events.<br> \- Assertion of “game-changer” is justified by significantly better outcomes.<br>2\. Clarity: A-<br> \- Concise and well-structured with clear sections.<br> \- Good use of comparative language (e.g., “up 3% from Q1”) for context.<br> \- Minor issue: Acronyms like “RCT,” “GLP-1,” and “GIP” not defined. Consider spelling out first use.<br>3\. Completeness: B+<br> \- Covers all requested areas: methodology, findings, clinical implications.<br> \- Missing: Baseline characteristics of groups (were they well-matched?).<br> \- Missing: Any secondary endpoints (e.g., blood pressure, lipids).<br> \- Consider: Long-term implications or need for further research.<br>Overall, the summary is accurate and clear, providing key insights for a quick grasp of the study. To elevate to A+, define acronyms, mention group baseline stats, note any secondary outcomes, and briefly touch on the need for long-term studies. |

#### [​](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts#prompt-3)  Prompt 3

| Role | Content |
| --- | --- |
| User | Your task is to improve a paper summary given feedback. Here is the first draft of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Here is the feedback:<br><feedback><br>{{FEEDBACK}}<br></feedback><br>Update the summary based on the feedback. |
| Assistant | Methodology: Double-blind randomized controlled trial (RCT) with 500 Type 2 diabetes patients. Group A (n=250) received a novel glucagon-like peptide-1/glucose-dependent insulinotropic polypeptide (GLP-1/GIP) dual agonist; Group B (n=250) received a standard GLP-1 agonist. The 24-week trial’s primary endpoint was HbA1c reduction. Groups were well-matched for age, BMI, and baseline HbA1c.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Secondary endpoints showed improvements in blood pressure and lipid profiles for both groups, with Group A showing slightly better results.<br>Safety: Similar adverse event profiles, with no serious events attributed to either treatment.<br>Clinical Implications: The dual GLP-1/GIP agonist demonstrates superior glycemic control and weight loss compared to standard GLP-1 therapy. These results suggest it could be a significant advancement, particularly for obese diabetics. However, longer-term studies are needed to confirm sustained efficacy and safety, as well as to investigate potential benefits on cardiovascular outcomes. |

</details>

<details>
<summary>claude-4-prompt-engineering-best-practices-anthropic</summary>

This guide provides specific prompt engineering techniques for Claude 4 models (Opus 4.1, Opus 4, and Sonnet 4) to help you achieve optimal results in your applications. These models have been trained for more precise instruction following than previous generations of Claude models.

## General principles

### Be explicit with your instructions

Claude 4 models respond well to clear, explicit instructions. Being specific about your desired output can help enhance results. Customers who desire the “above and beyond” behavior from previous Claude models might need to more explicitly request these behaviors with Claude 4.

Example: Creating an analytics dashboard

**Less effective:**

```text
Create an analytics dashboard
```

**More effective:**

```text
Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.
```

### Add context to improve performance

Providing context or motivation behind your instructions, such as explaining to Claude why such behavior is important, can help Claude 4 better understand your goals and deliver more targeted responses.

Example: Formatting preferences

**Less effective:**

```text
NEVER use ellipses
```

**More effective:**

```text
Your response will be read aloud by a text-to-speech engine, so never use ellipses since the text-to-speech engine will not know how to pronounce them.
```

Claude is smart enough to generalize from the explanation.

### Be vigilant with examples & details

Claude 4 models pay attention to details and examples as part of instruction following. Ensure that your examples align with the behaviors you want to encourage and minimize behaviors you want to avoid.

## Guidance for specific situations

### Control the format of responses

There are a few ways that we have found to be particularly effective in steering output formatting in Claude 4 models:

1.  **Tell Claude what to do instead of what not to do**
    *   Instead of: “Do not use markdown in your response”
    *   Try: “Your response should be composed of smoothly flowing prose paragraphs.”
2.  **Use XML format indicators**
    *   Try: “Write the prose sections of your response in <smoothly\_flowing\_prose\_paragraphs> tags.”
3.  **Match your prompt style to the desired output**

The formatting style used in your prompt may influence Claude’s response style. If you are still experiencing steerability issues with output formatting, we recommend as best as you can matching your prompt style to your desired output style. For example, removing markdown from your prompt can reduce the volume of markdown in the output.

### Leverage thinking & interleaved thinking capabilities

Claude 4 offers thinking capabilities that can be especially helpful for tasks involving reflection after tool use or complex multi-step reasoning. You can guide its initial or interleaved thinking for better results.

Example prompt

```text
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
```

For more information on thinking capabilities, see [Extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking).

### Optimize parallel tool calling

Claude 4 models excel at parallel tool execution. They have a high success rate in using parallel tool calling without any prompting to do so, but some minor prompting can boost this behavior to ~100% parallel tool use success rate. We have found this prompt to be most effective:

Sample prompt for agents

```text
For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially.
```

### Reduce file creation in agentic coding

Claude 4 models may sometimes create new files for testing and iteration purposes, particularly when working with code. This approach allows Claude to use files, especially python scripts, as a ‘temporary scratchpad’ before saving its final output. Using temporary files can improve outcomes particularly for agentic coding use cases.

If you’d prefer to minimize net new file creation, you can instruct Claude to clean up after itself:

Sample prompt

```text
If you create any temporary new files, scripts, or helper files for iteration, clean up these files by removing them at the end of the task.
```

### Enhance visual and frontend code generation

For frontend code generation, you can steer Claude 4 models to create complex, detailed, and interactive designs by providing explicit encouragement:

Sample prompt

```text
Don't hold back. Give it your all.
```

You can also improve Claude’s frontend performance in specific areas by providing additional modifiers and details on what to focus on:

*   “Include as many relevant features and interactions as possible”
*   “Add thoughtful details like hover states, transitions, and micro-interactions”
*   “Create an impressive demonstration showcasing web development capabilities”
*   “Apply design principles: hierarchy, contrast, balance, and movement”

### Avoid focusing on passing tests and hard-coding

Frontier language models can sometimes focus too heavily on making tests pass at the expense of more general solutions. To prevent this behavior and ensure robust, generalizable solutions:

Sample prompt

```text
Please write a high quality, general purpose solution. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please tell me. The solution should be robust, maintainable, and extendable.
```

## Migration considerations

When migrating from Sonnet 3.7 to Claude 4:

1.  **Be specific about desired behavior**: Consider describing exactly what you’d like to see in the output.
2.  **Frame your instructions with modifiers**: Adding modifiers that encourage Claude to increase the quality and detail of its output can help better shape Claude’s performance. For example, instead of “Create an analytics dashboard”, use “Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.”
3.  **Request specific features explicitly**: Animations and interactive elements should be requested explicitly when desired.

</details>

<details>
<summary>prompt-chaining-prompt-engineering-guide</summary>

To improve the reliability and performance of LLMs, one of the important prompt engineering techniques is to break tasks into its subtasks. Once those subtasks have been identified, the LLM is prompted with a subtask and then its response is used as input to another prompt. This is what's referred to as prompt chaining, where a task is split into subtasks with the idea to create a chain of prompt operations.

Prompt chaining is useful to accomplish complex tasks which an LLM might struggle to address if prompted with a very detailed prompt. In prompt chaining, chain prompts perform transformations or additional processes on the generated responses before reaching a final desired state.

Besides achieving better performance, prompt chaining helps to boost the transparency of your LLM application, increases controllability, and reliability. This means that you can debug problems with model responses much more easily and analyze and improve performance in the different stages that need improvement.

Prompt chaining is particularly useful when building LLM-powered conversational assistants and improving the personalization and user experience of your applications.

</details>

<details>
<summary>workflows-and-agents</summary>

In prompt chaining, each LLM call processes the output of the previous one.

As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.
>
> When to use this workflow: This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/prompt_chain.png

With parallelization, LLMs work simultaneously on a task:

> LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations: Sectioning: Breaking a task into independent subtasks run in parallel. Voting: Running the same task multiple times to get diverse outputs.
>
> When to use this workflow: Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/parallelization.png

Routing classifies an input and directs it to a followup task. As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.
>
> When to use this workflow: Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/routing.png

With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers. As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
>
> When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/worker.png

</details>
