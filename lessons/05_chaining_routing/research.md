# Research

## Research Results

<details>
<summary>What empirical studies document the “lost in the middle” effect in long-context LLMs, and what design recommendations do they provide?</summary>

### Source [2]: https://arxiv.org/pdf/2403.04797

Query: What empirical studies document the “lost in the middle” effect in long-context LLMs, and what design recommendations do they provide?

Answer: This paper provides a quantitative analysis of the “lost-in-the-middle” effect in LLMs by using controlled experiments on ranking and key-value (KV) retrieval tasks. The authors systematically **move the position of important information** from the beginning to the end of long contexts and **measure model accuracy** at each position. They introduce a metric called **Gap accuracy**, which quantifies the difference in model accuracy depending on the position of critical information.

The results demonstrate that **LLMs consistently show lower accuracy** when important information is placed in the middle of the context. The paper proposes a solution called **Multi-scale Positional Encoding (Ms-PoE)**, which aims to enhance the model’s ability to utilize information from all positions, including the middle. Experiments show that Ms-PoE **reduces the gap in accuracy by roughly 2–5%** and generally improves performance even when information is located at the beginning or end. This provides a **design recommendation**: incorporating advanced positional encoding strategies, such as Ms-PoE, can **alleviate the lost-in-the-middle effect** in LLMs.

-----

-----

-----

### Source [3]: https://proceedings.neurips.cc/paper_files/paper/2024/file/6ffdbbe354893979367f93e2121e37dd-Paper-Conference.pdf

Query: What empirical studies document the “lost in the middle” effect in long-context LLMs, and what design recommendations do they provide?

Answer: This source explores the **bias of LLMs to focus on the beginning and local context** within input sequences, often neglecting crucial information in the middle—a phenomenon termed the “lost-in-the-middle” challenge. The authors identify **positional index rescaling and a head-wise position-awareness property** as contributing factors. 

They introduce **Multi-scale Positional Encoding (Ms-PoE)** as a **plug-and-play strategy** to mitigate this effect. Ms-PoE **improves model performance in extracting information from the middle of long contexts** without needing to retrain the model. The effectiveness of this technique is validated through extensive experiments on benchmarks such as **Zero-SCROLLS, multi-document QA, and key-value retrieval tasks**. The findings suggest that **adopting improved positional encoding schemes** is a practical recommendation for LLM designers aiming to reduce middle-position information loss.

-----

-----

-----

### Source [4]: https://cs.stanford.edu/~nfliu/papers/lost-in-the-middle.arxiv2023.pdf

Query: What empirical studies document the “lost in the middle” effect in long-context LLMs, and what design recommendations do they provide?

Answer: The **original “Lost in the Middle” study** by Stanford researchers provides foundational empirical evidence for the phenomenon in LLMs. Through tasks such as **multi-document question answering and key-value retrieval**, the study demonstrates that **performance degrades significantly when the relevant information is located in the middle of long input contexts**. This **degradation occurs even in models explicitly trained for long contexts**.

The analysis shows that **models perform best when information is at the beginning or end** of the context, highlighting a **robust positional bias**. The paper introduces **new evaluation protocols** to systematically probe this issue and stresses that **current LLMs do not uniformly utilize their full context window**. The study’s primary design recommendation is to **develop and adopt improved evaluation methods** that can reveal these biases, and to **pursue architectural or training modifications** to address the uneven context utilization.

-----

-----

</details>

<details>
<summary>Which benchmarking articles or papers compare single-prompt approaches with multi-step prompt chaining in terms of accuracy, cost, and debuggability?</summary>

### Source [9]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts

Query: Which benchmarking articles or papers compare single-prompt approaches with multi-step prompt chaining in terms of accuracy, cost, and debuggability?

Answer: This official documentation recommends **prompt chaining** for multi-step tasks such as research synthesis, document analysis, or iterative content creation.

- **How-to**: Tasks should be broken into distinct, sequential steps. Outputs are passed between prompts (e.g., via XML tags for structured handoffs), supporting transparency and debuggability.
- **Performance**: Chaining prompts for complex tasks is recommended to achieve **stronger performance** (improved accuracy and output quality), as each subtask is explicitly defined and managed.
- **Debuggability**: The clear structure of chained prompts makes it easier to identify and fix issues at each step, rather than searching for errors within a single, complex prompt.
- **Cost**: While not explicitly detailed, the documentation suggests that chaining can optimize resource use by focusing compute on discrete, manageable steps.

No direct benchmarking statistics are provided, but the documentation’s guidance is based on observed best practices from real-world usage.

-----

-----

</details>

<details>
<summary>Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?</summary>

### Source [18]: https://huggingface.co/blog/dcarpintero/design-patterns-for-building-ai-systems

Query: Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?

Answer: This Hugging Face blog post introduces the **Orchestrator-Workers-Workflow** pattern for building AI systems, where a central LLM (the Orchestrator) dynamically **decomposes tasks** and delegates them to specialized worker agents. The Orchestrator is responsible for intelligent coordination: it analyzes incoming requests in real-time, determines necessary subtasks based on task complexity and resource availability, and selects appropriate workers for each subtask. Each worker is given specific objectives and tool guidance, and, after execution, their outputs are collected and synthesized by the Orchestrator into a unified result.

The workflow is particularly suited to scenarios where subtask boundaries are *not pre-defined* but must be dynamically determined, such as:
- Business intelligence (gathering and synthesizing data from multiple sources)
- Software development (identifying files to modify, coordinating code generation, testing, and documentation)

The post discusses the challenges of orchestration, including resource management, control loops, error handling, and avoiding single points of failure. It also highlights complexity management (requiring mechanisms like memory compression and observability) and evaluation challenges (such as multi-turn conversations and the need for specialized evaluation). The architecture and considerations described provide an engineering blueprint for planner–executor systems with LLM-based task decomposition[1].

-----

-----

-----

### Source [19]: https://arxiv.org/html/2412.00239v1

Query: Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?

Answer: This research paper discusses **Task Decomposition** patterns, including the orchestrator–worker (planner–executor) approach. It describes a structure where a component (either AI or non-AI) serves as the **orchestrator**, dividing the main task into sub-tasks and distributing them to one or more foundation models (FMs), which act as **workers**. The orchestrator can itself be an FM (AI agent) or traditional software, and it is responsible for creating sub-task inputs and managing the overall workflow.

The paper notes that this pattern is suitable for tasks requiring **complex structured outputs**, such as code generation and essay writing. The orchestrator–worker setup is contrasted with "single-shot" approaches, where the model attempts the entire task at once. The orchestrator–worker pattern is particularly beneficial for decomposing large, complex tasks into manageable segments, improving correctness and efficiency. The paper also addresses the impact on data labeling, evaluation, and deployment, emphasizing that model deployment becomes more complex when multiple FMs are involved. The orchestrator–worker (planner–executor) pattern is explicitly linked to AI Agents frameworks, where the orchestrator agent manages sub-task assignment and worker agents solve the subtasks[2].

-----

-----

-----

### Source [20]: https://arxiv.org/html/2410.22457v1

Query: Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?

Answer: This paper presents an **advanced agentic framework** for dynamic task decomposition and execution using LLMs. The architecture features an **Orchestrator**, which generates a task graph for multi-hop user queries. A component called the **Delegator** receives this task graph and assigns tasks to the most suitable agents or tools. The Delegator manages both intra-task and inter-task communication using memory buffers, ensuring that each subtask has the necessary context and data from its dependencies.

Agents in this framework are dynamic and typically powered by LLMs, executing specific tasks such as generating Python functions on-the-fly. Tools, in contrast, are static components (pre-defined Python functions) managed by a tool manager, which performs semantic filtering to present only the most relevant tools to the LLM based on the current task. The framework demonstrates how dynamic decomposition, assignment, and result consolidation are orchestrated to handle complex, multi-step workflows. This design closely matches the planner–executor (orchestrator–worker) paradigm, providing mechanisms for dynamic planning and specialized subtask execution[3].

-----

-----

-----

### Source [21]: https://dev.to/jamesli/agent-task-orchestration-system-from-design-to-production-1kof

Query: Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?

Answer: This engineering blog post outlines the development of an **Agent Task Orchestration System** for dynamic task decomposition. It describes a scenario where a user requests an agent to perform complex, multi-step tasks such as compiling a market research report, which requires decomposition into subtasks (e.g., data collection, competitor analysis, summarization).

The system is built around a **central agent** (the orchestrator) that analyzes incoming user requests, breaks them into discrete subtasks, and assigns these to **worker agents** specializing in each area. The orchestrator coordinates execution, monitors progress, and aggregates results into a final deliverable. The blog details considerations for productionization, including error handling, parallel execution, resource management, and observability. The outlined architecture exemplifies the planner–executor (orchestrator–worker) pattern applied in real-world software systems, demonstrating its advantages in flexibility and scalability for complex workflows[4].

-----

-----

-----

### Source [22]: https://spring.io/blog/2025/01/21/spring-ai-agentic-patterns

Query: Which research papers or engineering blog posts outline the orchestrator-worker (planner–executor) pattern for dynamic task decomposition with LLMs?

Answer: The Spring AI blog post introduces **agentic design patterns** for building AI systems, emphasizing the use of a **central LLM** to orchestrate complex tasks. The post demonstrates how the **orchestrator–worker (planner–executor) pattern** enables a central LLM to analyze a user request, decompose it into manageable subtasks, and delegate those to specialized agents or tools for execution.

The orchestrator is responsible for ensuring that each subtask is properly defined, tracked, and that output is collected and synthesized into a final response. The post discusses the importance of maintaining control and observability, handling errors, and supporting extensibility as new types of subtasks and agents are introduced. The pattern is positioned as essential for building scalable, robust AI-driven applications that require dynamic response to complex, variable user requests[5].

-----

-----

</details>

<details>
<summary>What real-world engineering case studies demonstrate measurable latency or throughput improvements from running multiple LLM calls in parallel (e.g., sectioning or voting) in production workflows?</summary>

### Source [33]: https://arxiv.org/html/2403.03699v1

Query: What real-world engineering case studies demonstrate measurable latency or throughput improvements from running multiple LLM calls in parallel (e.g., sectioning or voting) in production workflows?

Answer: This survey on model parallelism in distributed infrastructure highlights major real-world engineering case studies that demonstrate improvements in latency and throughput through parallel execution of large language model (LLM) workflows. The document details how **Google’s Gopher model** employed advanced parallelism strategies for LLM training and inference. Specifically, the Gopher team used "4-way inter-layer parallelism over four TPU pods," along with intra-layer and data parallelism within a 1024-chip TPU pod, to train a model with up to 280 billion parameters. These forms of parallelism allowed the workload to be distributed and executed simultaneously across multiple hardware units, effectively **reducing the time required for each training or inference cycle**. The use of parallelism was essential to handling the model's immense size and enabled **measurable improvements in throughput and latency**, making it feasible to deploy such large models in production environments. The paper lists Gopher as a case where **production-scale LLMs benefited directly from parallel execution strategies**, but does not go into detail on sectioning or voting techniques specifically.

-----

-----

</details>

<details>
<summary>Which technical articles or white papers outline debugging and reliability challenges—such as error propagation or context loss—when decomposing tasks into chained LLM calls, and how do they recommend mitigating these issues?</summary>

### Source [41]: https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517582

Query: Which technical articles or white papers outline debugging and reliability challenges—such as error propagation or context loss—when decomposing tasks into chained LLM calls, and how do they recommend mitigating these issues?

Answer: This paper presents the concept of **LLM Chaining**, where tasks are decomposed into smaller subtasks executed by a sequence of LLM calls, with each output feeding into the next step. The authors emphasize that this approach can improve **explainability and debuggability** by exposing intermediate results and allowing user interaction at each stage. 

Key points regarding debugging and reliability challenges:
- **Error Propagation:** The paper notes that chaining exposes intermediate steps, making it easier to identify where errors are introduced. By breaking down tasks, users can pinpoint which sub-component produces faulty output.
- **Context Loss:** By keeping each sub-task "well-scoped," the risk of losing context between steps is mitigated. The interactive interface lets users review and edit intermediate outputs, thus preserving or restoring context if needed.
- **Mitigation Recommendations:** The authors recommend:
  - Designing **primitive LLM operations** that are narrowly focused, limiting potential for compounding errors.
  - Providing an **interactive interface** for users to inspect, edit, and rewire steps, giving granular control over the chain.
  - Allowing **step-by-step execution** to observe the behavior of each module and intervene early if problems are detected.
- **Empirical Results:** The study reports that this chaining approach increases **system transparency, controllability, and task outcomes**, supporting more reliable and debuggable AI-driven applications without requiring changes to the underlying LLM.

Overall, this work directly addresses the debugging and reliability challenges of chaining LLM calls and presents concrete system design principles to mitigate these issues.

-----

-----

-----

### Source [43]: https://openreview.net/pdf?id=KuPixIqPiq

Query: Which technical articles or white papers outline debugging and reliability challenges—such as error propagation or context loss—when decomposing tasks into chained LLM calls, and how do they recommend mitigating these issues?

Answer: This paper introduces **SELF-DEBUGGING**, a technique where LLMs are taught to debug their own outputs, especially for code generation tasks. The discussion is highly relevant to error propagation and reliability in chained LLM scenarios:

- **Error Propagation:** The paper demonstrates that enabling LLMs to review and explain their previous outputs (akin to "rubber duck debugging") allows identification and correction of errors that may otherwise propagate through a chain of tasks.
- **Context Loss:** By having the LLM explain its reasoning and output, context is maintained and explicitly surfaced, reducing the likelihood of loss across steps.
- **Mitigation Recommendations:**
  - **Self-generated feedback:** LLMs can be instructed to critique and revise their outputs, catching errors before passing results to subsequent steps.
  - **Unit test feedback:** Where available, using automatic tests to validate outputs ensures that only correct results are propagated.
  - **Few-shot prompting:** Teaching LLMs to self-debug via example prompts can be applied even if the model is not specifically fine-tuned for debugging.

The authors report **state-of-the-art performance** using these self-debugging strategies, suggesting that embedding systematic review and feedback within LLM chains is an effective approach to mitigating reliability issues.

-----

-----

-----

### Source [44]: https://hadess.io/web-llm-attacks/

Query: Which technical articles or white papers outline debugging and reliability challenges—such as error propagation or context loss—when decomposing tasks into chained LLM calls, and how do they recommend mitigating these issues?

Answer: This source addresses **security and reliability challenges** in chaining LLM API calls, particularly focusing on **prompt injection and error propagation risks**:

- **Chained Prompt Injection:** When multiple LLM calls are chained, malicious input at any stage can propagate or escalate errors through the workflow, potentially leading to exploitation or system compromise.
- **Mitigation Recommendations:**
  - **Input Validation:** Rigorously validate and sanitize all prompts before they are processed or passed along a chain.
  - **Access Control:** Restrict who can modify or add prompts to the system, limiting the surface for accidental or malicious error injection.
  - **Anomaly Detection:** Monitor for unusual behaviors during chained calls to detect errors or attacks early.
  - **Model Verification:** Thoroughly test and validate the model’s behavior, especially for chained scenarios, to ensure it handles unexpected or malformed input gracefully.

These suggestions are designed to address both **intentional attacks** and **inadvertent error propagation**, directly supporting more robust and reliable LLM chaining.

-----

</details>

<details>
<summary>Which empirical studies or benchmark reports quantify latency and throughput gains when running LLM tasks in parallel (e.g., sectioning or voting) compared with strictly sequential execution?</summary>

### Source [50]: https://pssg.cs.umd.edu/research/llms-for-hpc/

Query: Which empirical studies or benchmark reports quantify latency and throughput gains when running LLM tasks in parallel (e.g., sectioning or voting) compared with strictly sequential execution?

Answer: The ParEval benchmark, introduced by the Parallel Software and Systems Group at the University of Maryland, provides empirical evaluation of LLMs' capabilities in **parallel code generation** for high-performance computing (HPC) tasks. ParEval consists of 420 code generation problems spanning seven execution models and twelve computational problem types. The benchmark includes novel metrics specifically designed to assess **runtime performance and scaling behavior** of LLM-generated code.

Key findings relevant to your question include:

- **ParEval directly compares serial and parallel code generation abilities of LLMs** by measuring pass@1 scores (the likelihood that a model generates correct code on the first attempt) for both types of tasks.
- The results show that **all tested LLMs perform worse at generating parallel code than serial code**, indicating current limitations in their parallelization capabilities.
- There is a performance gap between commercial and open-source LLMs, with commercial models generating parallel code more effectively.
- The ParEval benchmark and its leaderboard provide up-to-date, quantitative data on model performance, including metrics that can be used to infer **latency and throughput differences** when executing parallel tasks versus sequential tasks.

While the benchmark is focused on code generation and not directly on LLM inference strategies like parallel voting or sectioning at the task level, it is one of the few empirical sources quantifying LLM performance on *parallel versus serial* execution in a concrete, reproducible setting.

Details on metrics, results, and analysis can be found in their HPDC '24 paper "Can Large Language Models Write Parallel Code?" and on the ParEval leaderboard.

-----

-----

-----

### Source [51]: https://www.databasemart.com/blog/vllm-gpu-benchmark-v100-3

Query: Which empirical studies or benchmark reports quantify latency and throughput gains when running LLM tasks in parallel (e.g., sectioning or voting) compared with strictly sequential execution?

Answer: This benchmark report evaluates **vLLM inference performance** under different parallelization strategies using three NVIDIA V100 GPUs. It assesses models under 50 and 100 concurrent requests, focusing on both **latency** and **throughput**.

Key quantified findings:

- **Tensor-parallelism (TP)** outperforms pipeline-parallelism (PP) in terms of throughput and latency reduction. TP splits model computation horizontally across GPUs, reducing individual computation time but requiring high memory bandwidth.
- **Pipeline-parallelism (PP)** enables inference on larger models by splitting layers across GPUs, but introduces inter-GPU communication overhead that can increase latency compared to TP.
- The report provides specific benchmark metrics: number of successful requests, total benchmark duration (seconds), total input tokens, total generated tokens, and latency measurements under different parallelization configurations.
- **Best practice recommendations** include balancing concurrency and latency, choosing even tensor parallel sizes, and selecting appropriate numerical precision (e.g., float16) to maximize throughput and minimize latency.
- Quantitative results show that **parallel execution** (with multiple GPUs and concurrency settings) significantly boosts throughput and lowers latency compared to sequential (single-GPU, non-parallel) execution.

This report provides detailed operational and empirical data on how **parallel execution strategies** impact LLM inference performance, with direct implications for real-world deployment scenarios where latency and throughput are critical.

-----

-----

-----

### Source [52]: https://arxiv.org/html/2411.00136v1

Query: Which empirical studies or benchmark reports quantify latency and throughput gains when running LLM tasks in parallel (e.g., sectioning or voting) compared with strictly sequential execution?

Answer: The paper "Inference Benchmarking of Large Language Models on AI Hardware" introduces **LLM-Inference-Bench**, a benchmarking suite for evaluating LLM inference performance across hardware and software stacks. The study analyzes how model architecture, batch size, and parallelization impact **latency and throughput**.

Relevant findings:

- **Throughput and latency are highly sensitive to parallelization settings.** Smaller models (e.g., 7B parameters) typically yield higher throughput and lower latency, but the impact of parallel execution becomes more pronounced at scale.
- The benchmarking suite measures **first-token latency** and **input-to-token latency (ITL)** across different models and hardware acceleration frameworks, including those that support advanced parallelization (TensorRT-LLM, vLLM) and those that do not (Deepspeed-MII, llama.cpp).
- Results show that **platforms with advanced parallelization optimizations** (e.g., vLLM, TensorRT-LLM) deliver significantly higher throughput and lower latency compared to less optimized frameworks, especially under high concurrency (i.e., when running many inference tasks in parallel).
- The empirical data is visualized in Figures 21 and 22, demonstrating that parallelization reduces the time to first token and increases the rate of token generation compared to sequential execution, particularly at larger batch sizes and model scales.

Although the paper does not specifically benchmark sectioning or voting strategies, it provides comprehensive **quantitative evidence** on the gains from parallelized LLM inference compared to strictly sequential approaches.

-----

-----

</details>

<details>
<summary>What peer-reviewed papers or technical articles compare single “all-in-one” prompts to multi-step prompt chaining in terms of accuracy, token costs, and debuggability across diverse tasks?</summary>

### Source [57]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts

Query: What peer-reviewed papers or technical articles compare single “all-in-one” prompts to multi-step prompt chaining in terms of accuracy, token costs, and debuggability across diverse tasks?

Answer: This documentation from Anthropic discusses using **prompt chaining** for complex, multi-step tasks such as research synthesis or document analysis. While the full content is not shown, the summary indicates that prompt chaining is recommended for tasks requiring multiple stages, as it allows for stronger performance and more manageable outputs compared to single, all-in-one prompts.

The documentation implies that chaining enables:
- **Improved accuracy** by focusing the model on one subtask at a time.
- **Better token management** as each prompt is tailored to its specific function.
- **Easier debugging**, since each step can be independently evaluated and adjusted.

No direct peer-reviewed quantitative comparison is provided, but the guidance is aligned with best practices in prompt engineering for accuracy, efficiency, and maintainability.

-----

-----

</details>

<details>
<summary>Where can I find authoritative tutorials or Google-authored guides that show how to configure a Google Colab environment for Gemini, including setting the GOOGLE_API_KEY via `from google.colab import userdata`, installing `google-genai`, and securing credentials?</summary>

### Source [62]: https://cloud.google.com/gemini/docs/colab/set-up-gemini

Query: Where can I find authoritative tutorials or Google-authored guides that show how to configure a Google Colab environment for Gemini, including setting the GOOGLE_API_KEY via `from google.colab import userdata`, installing `google-genai`, and securing credentials?

Answer: This official Google Cloud guide focuses on **enabling and configuring Gemini for Google Colab Enterprise** within a Google Cloud project. Key steps include:

- **Activate Gemini API:** An administrator must enable the Gemini for Google Cloud API in the Google Cloud Console. This is done by selecting the project, navigating to the Gemini for Google Cloud page, and clicking "Enable". The status will update to "Enabled" once complete.
- **Assign IAM Roles:** The administrator must grant the "Gemini for Google Cloud User" role to users who require access. This is managed via the IAM & Admin page in the console. By editing a principal's roles and adding "Gemini for Google Cloud User," users are authorized to utilize Gemini features in Colab Enterprise.
- **Scope:** The documentation emphasizes these are administrative steps for enabling Gemini access at the project level. It does not provide direct code examples for user-side API key management or SDK installation, but it is authoritative for the initial setup and permissioning workflow.

-----

-----

-----

### Source [63]: https://ai.google.dev/gemini-api/docs/quickstart

Query: Where can I find authoritative tutorials or Google-authored guides that show how to configure a Google Colab environment for Gemini, including setting the GOOGLE_API_KEY via `from google.colab import userdata`, installing `google-genai`, and securing credentials?

Answer: This is the **official Gemini API quickstart guide from Google**, which details how to configure a development environment—including Google Colab—for Gemini API access:

- **API Key Requirement:** Obtain a Gemini API key from Google AI Studio.
- **Install the Google GenAI SDK** via pip in Colab:
  ```python
  pip install -q -U google-genai
  ```
- **API Key Configuration in Apps Script:** For Apps Script, the GEMINI_API_KEY is set in project settings under Script Properties, but for Python/Colab, users must set the key in their environment or directly in code.
- **No direct Colab code example** is given, but the Python SDK installation and key requirements are clearly stated.

Relevant to securing credentials and setting environment variables, the documentation suggests keeping the API key private and not exposing it in shared code. This guide is authoritative for installing the SDK and understanding the key requirements for Gemini in Python environments such as Colab.

-----

-----

</details>


## Sources Scraped From Research Results

<details>
<summary>Abstract</summary>

# Abstract

This paper aims to overcome the “lost-in-themiddle” challenge of large language models (LLMs). While recent advancements have successfully enabled LLMs to perform stable language modeling with up to 4 million tokens, the persistent difficulty faced by most LLMs in identifying relevant information situated in the middle of the context has not been adequately tackled. To address this problem, this paper introduces Multiscale Positional Encoding (Ms-PoE) which is a simple yet effective plug-and-play approach to enhance the capacity of LLMs to handle the relevant information located in the middle of the context, without fine-tuning or introducing any additional overhead. Ms-PoE leverages the position indice rescaling to relieve the long-term decay effect introduced by RoPE, while meticulously assigning distinct scaling ratios to different attention heads to preserve essential knowledge learned during the pre-training step, forming a multi-scale context fusion from short to long distance. Extensive experiments with a wide range of LLMs demonstrate the efficacy of our approach. Notably, MsPoE achieves an average accuracy gain of up to 3.8 on the Zero-SCROLLS benchmark over the original LLMs. Code are available at https: //github.com/VITA-Group/Ms-PoE

# 1\. Introduction

Effective long-sequence reasoning in large language models (LLMs) is crucial for a wide range of applications (R´e et al., 2022; Li et al., 2023), from understanding extensive texts (Tay et al., 2020; Kry´sci´nski et al., 2021) and managing day-long conversations (Zhang et al., 2021; Zhong et al., 2022) to code generation (Du et al., 2023; Zheng et al., 2023) and science discoveries (Varadi et al., 2022; Song et al., 2023b). Recent system support advancements (Dao, 2023; Jacobs et al., 2023) have enabled training transformers for any $L$ sequence length even with $O ( L ^ { 2 } )$ computational complexity. This is exemplified by models such as MPT (Team, 2023) and Mistral (Jiang et al., 2024) pre-trained with sequence lengths $1 6 \\mathrm { k }$ and $3 2 \\mathrm { k }$ respectively.

https://arxiv.org/pdf/images/3e382f197f650c5895428032ddf5ea6ba80603dceb89c1785e18973d5bdad82e.jpg

Figure 1. The x-axis illustrates the placement of essential information within the prompt, ranging from start to end. The green bar serves as a standard baseline, illustrating the “lost-in-the-middle” phenomenon. We introduce our method, Multi-scale Position Encoding (Ms-PoE), which requires neither additional fine-tuning nor increased memory usage. Instead, it involves a simple remapping of the position embedding depicted in Figure 2, which enables the important information in the middle to be detected effectively (brown bars). For more details, see Section 4.2 and Figure 5.

Nevertheless, emerging research reveals the constrained efficacy of LLMs in managing tasks requiring long contextual understanding. Particularly, Liu et al. (2023) demonstrated a substantial degradation in LLMs’ performance when crucial information is positioned amidst a lengthy context, a phenomenon they refer to as “lost-in-the-middle”. One explanation is about the use of rotary positional embedding (RoPE) (Su et al., 2024), a prevalent positional encoding technique used in open-source LLMs. As a relative position embedding, RoPE incorporates a long-term decay property, predisposing the model to prioritize current/nearby tokens while paying less attention to further ones. Xiao et al. (2023) identified a surprising trend attributed to the Softmax operation where attention scores are disproportionately allocated into initial tokens, irrespective of their relevance to the language modeling task. Despite the presence of considerable redundancy in long-context inputs (Zhang et al., 2023), crucial information may be located across different positions. The inclination of LLMs to overlook the middle section presents a challenge for their applications, particularly in the context of long-context reasoning. Several approaches successfully extend pre-trained LLMs with context up to extreme token length, either through sparse selection of crucial tokens during generation (Xiao et al., 2023; Zhang et al., 2023; Han et al., 2023) or by modifying positional encoding (Chen et al., 2023c; Jin et al., 2024). Nevertheless, these approaches primarily aim to extend the context length of LLMs and, consequently, fall short in addressing the “lost-in-the-middle” problem when applied out-of-the-box.

Efforts have been made to enhance LLMs’ capacity to capture vital information located within the middle of the context. These include extra memory bank (Wang et al., 2023), reordering the input context based on relevance (Peysakhovich & Lerer, 2023; Chen et al., 2023d), enhancing the information searching and reflection ability via attention strengthening tasks (Junqing et al., 2023; Xu et al., 2023), splitting the input into short segments and applying short-text models (Ivgi et al., 2023). For example, Peysakhovich & Lerer (2023) empirically discovered that LLMs tend to emphasize more on the current window while still paying more attention to the relevant text than distracting content. They subsequently introduced “attention sorting” where the main idea is iteratively sorting documents based on their attention scores, such that critical information will likely be placed at the end, to fit the position-biased nature of RoPE. Chen et al. (2023d) conducted parallel runs of LLMs with different RoPE angles, thereby mitigating the risk of overlooking crucial information through a weighted sum of the outputs. These approaches usually require additional memory or multiple inference runs, which can be expensive for LLMs.

In this paper, we aim to address the “lost-in-the-middle” problem by reintroducing the concept of multi-scale features from computer vision into the context of Transformer-based LLMs. Multi-scale features, well-established in Inceptionstyle models (Szegedy et al., 2015; 2016; Guo et al., 2022), utilize parallel employment of kernels with different sizes to fuse multi-scale information, spanning short to long distances. Introducing multi-scale operations into LLMs intuitively can help compensate for crucial information located in the middle, which might be easily overlooked by full attention operation. Unlike modifying the attention module to form multi-scale attention, we choose to re-scale the indices of positional encoding. This decision is grounded not only in its effectiveness in easily adjusting the scale of the context window by simply changing the position indices (Chen et al., 2023c) but also in the potential of down-scaling the position indices to relieve the long-term decay property introduced by RoPE. However, this approach was initially introduced to extend context windows, and its performance regarding the “lost-in-the-middle” problem remains uncertain for several reasons: (i) Indice re-scaling forces position embeddings of original context window to reside in a narrower region, leading to performance degradation in the original context window as shown in Chen et al. (2023c). (ii) Uniformly applying the same scaling ratio throughout the entire model might be sub-optimal to preserve essential knowledge learned during pre-training; (ii) Fine-tuning is necessary for the original approach, albeit minimal. The impact without fine-tuning remains unknown.

To this end, we systematically visit the position indices scaling regarding the “lost-in-the-middle” problem and counterintuitively discover that it is possible to slightly mitigate the “lost-in-the-middle” issue if we carefully choose the scaling ratio to be around 1.5-2. Additionally, we observe that different attention heads exhibit varying sensitivity to the position shift of the relevant document. Some attention heads are “position-aware”, consistently capturing relevant information even with position shifts, while others may occasionally capture position changes, and some heads are completely insensitive to position changes. This highlights the need to treat attention heads separately when re-scaling position indices.

Contribution. Inspired by the above observations, we introduce Multi-scale Positional Encoding (Ms-PoE), a simple yet effective plug-and-play approach that can enhance the long-context reasoning capability of pre-trained LLMs without requiring fine-tuning or introducing any additional overhead. Ms-PoE meticulously assigns distinct scaling ratios to different attention heads, with the scaling factor monotonically increasing from “position-aware” heads to “position-unaware” heads. This enables us to improve longcontext ability by re-scaling position indices to shorter values while preserving essential knowledge acquired during the pre-training phase. The efficacy of Ms-PoE is substantiated through extensive experiments. By simply re-scaling the indices of positional encoding, Ms-PoE consistently enhances the performance of various LLMs including Llama2 (Touvron et al., 2023), StableBeluga (Mahan et al.) and Vicuna (Chiang et al., 2023) on the ZeroSCROLLS benchmark (Shaham et al., 2023), achieving a notable average accuracy gain of up to 3.8.

# 2\. Background and Related Works

In this section, we provide a concise overview of the background knowledge and recent literature about the generative inference process of Large Language Models (LLMs), their abilities for long-context reasoning, and details of positional encoding.

# 2.1. Generative Inference of LLMs

The generative inference process in LLMs can be categorized into two distinct phases: $\\textcircled{1}$ Prefilling Stage: In this initial phase, LLMs receive an input sequence containing detailed instructions that define a specific generation goal. Throughout this stage, intermediate Key and Value embeddings are generated at each layer and stored in memory, commonly referred to as the KV cache. $\\textcircled{2}$ Decoding Stage: This phase involves retrieving embeddings from the KV cache to generate new tokens. The decoding process is inherently iterative, where each newly generated token serves as input for the subsequent token generation. In real-world LLM deployment, the cumulative length of input sequences and the subsequently generated text can reach several thousand or even millions of tokens, presenting significant challenges for the LLMs’ long-context reasoning capability.

# 2.2. Long Context Reasoning

Two challenges lie ahead for LLMs in handling long-context reasoning tasks. One is to extend the context window to process sentences that exceed the pre-trained window length. Another is the “lost-in-the-window” issue where LLMs will likely overlook the information located in the middle of the sentences.

The reason for the former challenge is that open-source LLMs are usually pre-trained with fixed sequence lengths, such as 4096 for Llama-2 (Touvron et al., 2023). When the sequence length surpasses the predefined context length used in pre-training, LLMs often suffer from performance collapses and thus generate incoherent or fragmented text. Recent efforts to address this issue can be broadly categorized into two streams. Recently, several works have been proposed to address this issue, which can be broadly categorized into two streams. The first one explores from the expansion of positional encoding, with notable contributions including PI (Chen et al., 2023c), CLEX (Chen et al., 2023a), YaRN (Peng et al., 2023), Self-Extend (Jin et al., 2024). On the other hand, some works modify the attention mechanism, such as StreamingLLM (Xiao et al., 2023), LM-Inifinite (Han et al., 2023), $\\mathrm { H \_ { 2 } O }$ (Zhang et al., 2023), TOVA (Oren et al., 2024), Zebra (Song et al., 2023a), and Activation Beacon (Zhang et al., 2024). These approaches have successfully expanded the contextual window with minimal or no additional training overhead.

Despite the extended context window, LLMs still face a significant challenge in long-context inference due to the uneven utilization of lengthy inputs. Liu et al. (2023) conducted a pivotal investigation, revealing that LLMs tend to overlook the middle portion of the input. This bias compromises the practical application of LLMs, as critical information may be located in the middle part of the input, leading to unreliable outputs. To tackle this issue, Peysakhovich & Lerer (2023) introduced ‘attention sorting’ to reorder inputs, placing critical information at the end. However, this method’s reliance on potentially biased attention scores to identify crucial content may compromise its reliability, and the prerequisite knowledge of document count in inputs may affect its effectiveness. Chen et al. (2023d) utilize Attention Buckets, an ensemble approach that combines multiple forward processes with positional modifications. However, this technique necessitates a considerably higher computational cost. Other general approaches for enhancing long-context reasoning include prompt compression (Jiang et al., 2023b), retrieval augmentation (Xu et al., 2023), and inference refinement by constructing memory trees (Chen et al., 2023b) while these approaches typically necessitate extra LLMs’ assistance or bring extra computational cost.

# 2.3. Positional Encoding

For effective processing of long contexts, LLMs necessitate the explicit encoding of positional information. Common techniques include absolute positional embedding and relative positional encoding. Absolute positional embedding integrates word embeddings with an additional positional vector based on the token’s absolute position, which can be either fixed (Vaswani et al., 2017) or learnable (Devlin et al., 2018; Lan et al., 2019; Clark et al., 2020; Radford et al., 2019; 2018). In contrast, relative positional encoding, increasingly popular in contemporary LLMs, encodes the relative distances between tokens instead of their absolute positions. Notable among these are Rotary Position Embedding (RoPE) (Su et al., 2024) that widely implemented in models like Llama (Touvron et al., 2023), Falcon (Penedo et al., 2023), Mistral (Jiang et al., 2023a), and ALiBi (Press et al., 2021), which used in MPT (Team, 2023).

RoPE. The primary goal of RoPE (Su et al., 2024) is to encode positional information such that the inner product of the query and key embeddings inherently contains the relative position information, that is:

$$
f ( \\mathbf { q } \_ { m } , m ) ^ { T } f ( \\mathbf { k } \_ { n } , n ) = g ( \\mathbf { q } \_ { m } , \\mathbf { k } \_ { n } , m - n )
$$

Here, $f$ is the positional encoding function applied to the query and key embeddings at positions $m$ and $n$ , respectively. To satisfy this condition, the function $f$ is defined as a vector-valued complex function, as follows:

$$
\\begin{array} { l } { f ( \\mathbf { x } , m ) = \\mathbf { x } e ^ { i m \\theta } } \ { \ = \[ ( x \_ { 1 } + i x \_ { 2 } ) e ^ { i m \\theta \_ { 1 } } , ( x \_ { 3 } + i x \_ { 4 } ) e ^ { i m \\theta \_ { 2 } } , } \ { \ \\dots , ( x \_ { l - 1 } + i x \_ { l } ) e ^ { i m \\theta \_ { l / 2 } } \] ^ { T } } \\end{array}
$$

In this equation, $l$ represents the dimension of the embeddings, $\\theta \_ { k } = 1 0 0 0 0 ^ { - 2 k / l }$ , and $i$ is the imaginary unit. For calculating the attention score, RoPE considers the real part of the product, specifically $\\mathrm { R e } ( f ( \\mathbf { q } \_ { m } , m ) ^ { T } f ( \\mathbf { k } \_ { n } , n ) )$ .

https://arxiv.org/pdf/images/f763f0332a95eb8cabbf06bac4fc405bfa8a14f17342fa5bc50ab19cb9134b5c.jpg

Figure 2. Illustration of our Multi-scale Positional Encoding (MsPoE) framework. The top figure demonstrates the implementation of Ms-PoE with various scaling ratios in different attention heads, marked with different colors. The bottom figure shows the position details of each head, in which the first matrix $\\begin{array} { r } { r \_ { i } = 1 \\dot { } } \\end{array}$ ) represents the original RoPE.

This approach allows RoPE to effectively integrate relative positional information into the attention mechanism of transformer models.

# 3\. Methodology

In this section, we present the details of our Multi-Scale Positional Encoding (Ms-PoE) approach. Section 3.1 demonstrates that the context utilization of LLMs can be directly enhanced by re-scaling the positional information without incurring extra training costs. Then, Section 3.2 analyzes the properties of various attention heads in LLMs. Section 3.3 outlines the detailed pipeline of Ms-PoE.

# 3.1. Positional Re-scaling Improves Context Utilization

Current LLMs tend to neglect information located in the middle of the context, despite its potential relevance. This “lost in the middle” phenomenon likely arises from two contributing factors: (i) Casual Attention, where preceding tokens undergo a higher number of attention processes, leading LLMs to disproportionately favor initial tokens. This phenomenon has been demonstrated in recent research which highlights the pivotal role of the initial tokens in model generation (Han et al., 2023; Xiao et al., 2023), with these starting tokens consistently accumulating higher attention scores (Zhang et al., 2023). (ii) The utilization of RoPE (Su et al., 2024) introduces a long-term decay effect, diminishing the attention score of distantly positioned yet semantically meaningful tokens. The combination of these factors contributes to LLMs neglecting the context in the middle part. To tackle this issue and improve the context utilization of LLMs, a seemingly unreasonable yet remarkably effective strategy is to down-scale positional information (Song et al., 2023a). Formally, RoPE encodes the position as $f ( \\mathbf { x } , m ) = \\mathbf { x } e ^ { i m \\theta }$ . By substituting the position $m$ with $\\textstyle { \\frac { m } { r } }$ , we can force the long-distance tokens to reside in the shorted range, which can potentially alleviate the long-term decay effects by a factor of $r$ . In the following sections, we conduct experiments to evaluate how LLMs’ context utilization responds to varying re-scaling ratios $r$ .

Details. Experiments are conducted using Llama-2- 7B-Chat (Touvron et al., 2023) and Vicuna-7B (Chiang et al., 2023) on the Multi-Document Question Answering (MDQA) task (Liu et al., 2023). Each question includes ten documents, with only one relevant to the question. By varying the position of the relevant document, we can evaluate LLMs’ context utilization properties. For each position of the key document, we calculate the accuracy over 500 samples. And results show in Figure 3 include both the Average accuracy over the 10 documents as well as Gap accuracy, i.e., the difference between the best and worst accuracy when varying the positions of the relevant document.

https://arxiv.org/pdf/images/d0ede36fa7204cff7bbc96721db46e45ef7823e54ceae5573c410e2cbee3e0a3.jpg

Figure 3. Results of the relationship between positional re-scaling and context utilization. The upper curve illustrates the average accuracy when placing the key document in various positions. The bottom curve indicates the gap between the best and worst accuracy.

Results. Figure 3 demonstrates that the gap accuracy can be alleviated via appropriate positional re-scaling. Particularly, we see that the Gap between the best and the worst accuracy is greatly reduced when increasing the re-scaling ratio. An enhanced average accuracy can be observed with a scaling ratio equals near 1.5. Additionally, changing the scaling ratio also affects the favored zone of LLMs. With a small scaling ratio (e.g., 0.5), LLMs tend to focus more on the most recent part of the context, while with a large ratio (e.g., 2.5), LLMs favour the beginning part.

Improving context reasoning via positional re-scaling. Building upon this, we introduce a plug-and-play treatment for RoPE by re-scaling the position of each token. This approach seamlessly enhances the context utilization of LLMs without requiring additional training or inference overhead. However, there is a trade-off in terms of LLMs favoring certain context regions. For instance, when $r = 0 . 5$ , LLMs achieve peak accuracy when the relevant document is located at the end of the input, while at the beginning for $r = 1 . 5$ . It remains challenging to decide which rescaling ratio to use, given that we lack prior knowledge of the location of relevant information in real-world applications. Moreover, as the re-scaling ratio increases, LLMs may face the positional out-of-distribution (O.O.D) issue (Jin et al., 2024; Chen et al., 2023c), where many position values do not directly exist during pretraining (e.g., using $0 . 1 , 0 . 2 , . . . , 0 . 9$ for position when LLMs only recognize $1 , 2 , . . . , 9$ during pretraining), potentially reducing their average reasoning ability. To tackle these challenges, we investigate the head-wise properties of LLMs and propose a multi-scale positional encoding approach.

# 3.2. Position-Aware Head-Wise Re-scaling Ratio

Inspired by recent works that leverage attention patterns to identify most crucial tokens and optimize inference efficiency (Oren et al., 2024; Zhang et al., 2023; Ge et al., 2023), we carry out a preliminary study to investigate the interaction between attention patterns and token positions.

Details. We visualize the attention patterns of the most recent query with results collected from Vicuna-7B on the MDQA task, following (Oren et al., 2024). In the same input sample, we manually switch the position of the relevant document from the beginning to the end and illustrate the attention scores across different positions.

Observation. We observe the presence of “position-aware” attention heads capable of capturing relevant information even when its position is shifted. As an example, we select the eighth attention head in the fifteenth layer, depicted in the bottom of Figure 4, while consistent observations can be drawn across different layers and input samples. Firstly, most attention scores are near zero and can be ignored, consistent with other studies highlighting high sparsity in attention blocks (Zhang et al., 2023; Likhosherstov et al., 2021; Edelman et al., 2022). For the remaining positions, these “position-aware” attention heads can capture important information across positions, with attention patterns shifting as the position of relevant tokens changes. However, for other attention heads (upper subfigure in Figure 4), they fail to capture relevant tokens and only attend to the beginning and end words, contributing to the “lost-in-the-middle” issue.

Based on this observation, we devise a position-aware strategy to adaptively determine the re-scaling ratio via the inherent properties of different attention heads. For the “positionaware” attention heads, we assign a re-scaling ratio close to one to avoid changing their functionality significantly, as altering them too much could degrade performance due to the positional O.O.D issue. On the other heads, we condense their position indices to a higher degree, providing more opportunity to alleviate the persistent bias toward the beginning and recent tokens. To identify the properties of $n \_ { h }$ attention heads, we introduce a Position-Awareness Score $S \_ { P } \\in \\mathbf { R } ^ { n \_ { h } }$ formulated as:

$$
S \_ { P } = { \\frac { 1 } { l } } \\sum \_ { i = 1 } ^ { l } ( A \_ { i } \\geq \\alpha { \\frac { 1 } { l } } \\sum \_ { i = 1 } ^ { l } A \_ { i } )
$$

In Equation 1, $A$ represents the attention score vector of the most recent query, and $\\alpha$ is a hyperparameter determining the threshold of effective attention scores. In all experiments, we default to using $\\alpha = 3$ , and the corresponding important tokens are highlighted in Figure 4, which are shown in red. In the spirit of numerous studies that investigate the outlier properties in LLMs (Xiao et al., 2023; Lin et al., 2023; Yin et al., 2023), we utilize $ { \\boldsymbol { S } } \_ { P }$ to evaluate the ratio of effective attention tokens, where a larger $ { \\boldsymbol { S } } \_ { P }$ value implies better positional awareness.

# 3.3. Inference with Multi-Scale Positional Encoding

# Algorithm 1 LLM Inference with Ms-PoE

Require: A pre-trained LLM $f ( \\theta )$ with positional encoding RoPE, input content $X$ , generation length $l$ , number of attention heads $n \_ { h }$ , number of layers $n \_ { l }$ , re-scaling ratios $\\mathbf { r } = \[ r \_ { 1 } , r \_ { 2 } , \\ldots , r \_ { n \_ { h } } \]$ , calculated in Equation 2.

# Ensure: Generated text $T$ .

1: Set Prefilling $\ c =$ True;

2: while $i < l$ do

3: if Prefilling then

4: for $\\mathrm { ~ k ~ }$ in 1, 2, ..., $n \_ { l }$ do

5: Get the last query $Q \_ { l }$ and all key $K$ in layer $\\mathrm { k \\Omega }$ ;

6: $A = \\mathrm { S o f t m a x } ( Q \_ { l , \* } ( K \_ { \* } ) ^ { T } )$ ;

7: Calculate Position-Awareness Score $S \_ { \\mathbf { P } } ( A )$ for

each head, with Equation 1;

8: # Initial re-scaling ratio in each layer

9: $\\mathbf { r } \_ { k } = \\mathbf { r } \[ S \_ { \\mathbf { P } }$ .argsort(reverse=True)\];

10: Replace RoPE in layer $\\mathrm { ~ k ~ }$ with Ms-PoE $\\left( \\mathbf { r } \_ { k } \\right)$ ;

11: end for

12: Prefilling $\ c =$ False;

13: end if

14: Generate $T$ with implemented Ms-PoE;

15: end while

The pipeline for utilizing Multi-Scale Positional Encoding (Ms-PoE) in LLM inference is outlined in Algorithm 1. Given a pre-trained LLM, we initially replace the original rotary positional encoding with Ms-PoE. As illustrated in Figure 2, Ms-PoE condenses the positional indices of RoPE and employs different re-scaling ratios for each attention head. The re-scaling ratios are assigned during the prefilling stage, where we first calculate the distribution of attention scores for the most recent query and obtain the corresponding position-awareness score for each attention head. Larger re-scaling ratios are subsequently allocated to attention heads exhibiting smaller position-awareness scores. And the set of re-scaling ratios $\\mathbf { r }$ defaults to a linear range from 1.2 to 1.8. For example, the $i$ th sorted-head would be using re-scaling ratio

https://arxiv.org/pdf/images/829fbf88a9c59f4b502d901e3b9793733b66595c44d7e8b86b78ee1c6b896787.jpg

Figure 4. Visualization of attention pattern of the most recent query within two different attention heads. Top: Results of the 12th attention head in the 15th layer. Bottom: Results of the 8th attention head in the 15th layer. The most recent query remains unchanged while varying the position of the crucial document. More examples are reported in Figure 6 in the appendix.

$$
r \_ { i } = 1 . 2 + ( i - 1 ) ( 1 . 8 - 1 . 2 ) / ( n \_ { h } - 1 )
$$

Once the re-scaling ratios are assigned, they remain fixed in the subsequent decoding stage.

# 4\. Experiments

The goal of this section is to demonstrate Ms-PoE, a plugand-play positional encoding capable of enhancing the context utilization of LLMs, and consequently improving the quality of generation across diverse models and downstream reasoning tasks. Our main results can be summarized below.

In Section 4.1, we demonstrate that Ms-PoE consistently enhances reasoning over long contexts for a range of tasks in the ZeroSCROLLS benchmarks (Shaham et al., 2023), all without the need for additional training. Additionally, Ms-PoE exhibits superior performance when compared to other methods in the field, including PI (Chen et al., 2023c) and Self-Extend (Jin et al., 2024).1 Detailed results of these comparisons are shown in Tables 1 and 2.

In section 4.2, we highlight that Ms-PoE improves the context utilization and achieves consistent improvement when varying the position of critical information within the input context, as shown in Figure 1 & 5.

In Section 4.3, we conduct multiple ablation studies to assess the effectiveness of Ms-PoE under different scaling ratios and selection strategies. Results are reported in Table 3 & 4.

# 4.1. Enhanced Generation Quality

We empirically validate the ability of Ms-PoE to enhance long-context reasoning with a noteworthy improvement up to 13.4 without additional training overhead. Notably, our approach surpasses other competitive baselines, demonstrating improvements from 2.64 to 43.72.

Experimental Setup. In our experiments, we select seven representative LLMs, including Llama-2-chat-7B and 13B (Touvron et al., 2023), StableBeluga-7B and 13B (Mahan et al.), and Vicuna-7B (Chiang et al., 2023), along with its longer-context version (Vicuna-7B-16K). To comprehensively evaluate the long-context reasoning abilities of LLMs, we choose seven tasks from ZeroSCROLLS (Shaham et al., 2023), spanning all four task categories: $\\textcircled{1}$ Document Summarization (Government and SummScreenFD), $\\textcircled{2}$ Query-Based Summarization (QMSum and SQuALITY), $\\textcircled{3}$ Question Answering (Qasper and NarrativeQA), and $\\textcircled{4}$ Information Aggregation (BookSumSort). Furthermore, we compare Ms-PoE with other competitive methods on additional generation tasks, including Multi-document Question Answering (MDQA) and Key-Value Retrieval (Liu et al., 2023).

Main Results. Table 1 summarizes the main results, yielding several key observations: (i) By simply substituting the original positional encoding module with our Ms-PoE, the performance of LLMs consistently improves across all tasks without additional training, resulting in an average performance enhancement ranging from 0.6 to 3.8; (ii) These improvements hold consistently across different model sizes of 7 billion and 13 billion parameters; (iii) The efficacy extends to LLMs with varying sequence lengths, such as Vicuna-7B and its extended version, Vicuna-7B-16K, both showing improvements from 3.3 to 3.8.

Table 1. Comparsion results on ZeroSCROLLS (Shaham et al., 2023) benchmarks. The evaluation metrics for various tasks are tailored as follows: GovReport, SummScreenFD, QMSum, and SQuALITY utilize the geometric mean of Rouge-1/2/L scores. Qasper and NarrativeQA are assessed through the F1 score, while BookSumSort employs the concordance index.

|     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Models | Methods | GovReport | SummScreenFD | QMSum | SQuALITY | Qasper | NarrativeQA | BookSumSort | Average |
| Llama-2-7B-Chat Llama-2-7B-Chat | Baseline Ours | 16.8 17.7 (+0.9) | 14.1 14.2 (+0.1) | 15.2 15.8 (+0.6) | 19.5 19.9 (+0.4) | 21.9 25.1 (+3.2) | 14.4 17.7 (+3.3) | 3.1 5.8 (+2.7) | 15.0 16.6 (+1.6) |
| Llama-2-13B-Chat Llama-2-13B-Chat | Baseline Ours | 15.4 16.5 (+1.1) | 12.3 13.1 (+0.8) | 15.1 15.5 (+0.4) | 18.9 19.2 (+0.3) | 19.0 20.8 (+1.8) | 15.0 17.0 (+2.0) | 5.7 5.9 (+0.2) | 14.5 15.4 (+0.9) |
| StableBeluga-7B StableBeluga-7B | Baseline Ours | 14.9 16.6 (+1.7) | 13.8 14.2 (+0.4) | 14.7 15.2 (+0.5) | 17.9 18.7 (+0.8) | 28.1 36.9 (+8.8) | 16.8 18.0 (+1.2) | 9.2 14.2 (+5.0) | 16.5 19.1 (+2.6) |
| StableBeluga-13B StableBeluga-13B | Baseline Ours | 5.7 7.4 (+1.7) | 7.1 7.4 (+0.3) | 12.9 12.8 (-0.1) | 13.3 13.2 (-0.1) | 19.2 20.8 (+1.6) | 13.4 13.4 (+0) | 4.8 5.6 (+0.8) | 10.9 11.5 (+0.6) |
| Vicuna-7B Vicuna-7B | Baseline Ours | 16.2 20.2 (+4.0) | 13.7 14.5 (+1.8) | 15.1 15.4 (+0.3) | 18.9 19.8 (+0.9) | 24.3 34.7 (+13.4) | 13.7 16.2 (+2.5) | 3.3 10.5 (+7.2) | 15.0 18.8 (+3.8) |
| Vicuna-7B-16K Vicuna-7B-16K | Baseline Ours | 20.2 21.4 (+1.2) | 13.9 14.3 (+0.4) | 16.2 16.2 (+0) | 20.1 20.2 (+0.1) | 32.3 37.8 (+5.5) | 18.8 21.0 (+2.2) | 29.9 43.3 (+13.4) | 21.6 24.9 (+3.3) |

Table 2. Comparsion results with other competitive methods on MDQA and Key-Value Retrival. Results are reported in accuracy.

|     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Models | Methods | MDQA |
| 1 | 3 | 5 | 7 | 10 | Average |
| Vicuna-7B | Baseline PI | 64.0 | 61.0 | 57.4 | 58.4 | 64.8 | 61.12 |
|  | 65.2 | 62.4 | 60.0 | 60.4 | 64.0 | 62.40 |
| Self-Extend Ms-PoE | 64.7 65.6 | 63.7 64.2 | 61.4 63.0 | 59.8 65.2 | 62.0 67.2 | 62.32 65.04 |
| Models | Methods |  |  | Key-ValueRetrival |  |  |  |
|  |  | 1 | 15 | 30 | 40 | 50 | Average |
| Baseline | 92.0 | 25.8 | 8.0 | 25.4 | 30.0 | 36.24 |
| PI | 96.4 | 76.4 | 61.4 | 64.6 | 57.8 | 67.60 |
| Vicuna-7B | Self-Extend | 88.6 | 63.8 | 76.2 | 59.4 | 42.0 | 66.00 |
| Ms-PoE | 97.0 | 83.4 | 75.0 | 86.6 | 57.8 | 79.96 |

Outperform other competitive methods. We conduct a thorough comparison between Ms-PoE and other competitive methods, including Positional Interpolation (PI) (Chen et al., 2023c) and Self-Extend (Jin et al., 2024), both of which modify position indices without utilizing head-wise properties. For PI, we employ the scaling ratio as the average value of our method while for Self-Extend, we set the group size as 2 with the local window size as 1024. The results presented in Table 2 consistently showcase the superiority of our approach over other baselines, demonstrating improvements of up to 3.92 and 43.72 for MDQA and Key-Value Retrival, respectively. Such improvements might come from two primary factors. Firstly, the incorporation of head-wise properties offers a more adaptive strategy for positional modification. Secondly, our approach enhances the general context utilization ability. Notably, our approach demonstrates superiority even when the core document or key is positioned at the end of the input, surpassing other baselines with improvements ranging from 2.4 to 27.8. This performance surpasses the recent work (Peysakhovich & Lerer, 2023), which addresses the “lost-in-the-middle” effect by reordering key documents and placing them at the end of the input. When the identified core document is already located at the recent area, such method can not gain further improvements, while our approach offers a fine-grained strategy to improve context utilization.

https://arxiv.org/pdf/images/3e360ff5f93ad85442d4c74cab1081c8941b172f75f88fcfff5b1dffacab1211.jpg

Figure 5. Comparison results for the multi-document question answering (MDQA) and key-value retrieval (KV retrieval) tasks. Each subfigure depicts the comparison when varying the position of critical information from the beginning to the end. For Vicuna7B, please refer to Figure 1.

# 4.2. Superior Context Utilization

We assess the context utlization ability of our approaches on two tasks, including multi-document question answering (MDQA) and key-value retrieval (KV retrieval) tasks from (Liu et al., 2023). Such tasks provide a good input structure and offers the flexibility to switch the position of crusial information, thus evaluate the context utilization ability of LLMs.

Experimental Setup. In the MDQA task, each input sample comprises ten documents and one question, with only one document being relevant to the question. For the KV retrieval tasks, there are 50 key-value pairs with one question querying the value of the chosen key. In both tasks, we systematically switch the important document or key-value pair from the beginning to the end and report the accuracy of the generated context. All results are averaged across 500 samples. The Gap accuracy metric is employed to assess the context utilization ability of LLMs, defined as the gap between the best and worst accuracy when varying the position of important information.

Main Results. As depicted in Figure 5 and 1, Ms-PoE demonstrates consistent improvement across different models, tasks and critical positions. Even when the important information exists in the sweet region (beginning and end) of the input, Ms-PoE achieves significant performance improvements ranging from $3 %$ to $6 %$ , highlighting its efficacy in enhancing generation quality. Moreover, the “lost-in-themiddle” issue is notably alleviated, with Ms-PoE quantitatively reducing the gap accuracy by approximately $2 %$ to $4 %$ , showcasing improved context utilization.

# 4.3. Ablation Study and More Investigation

This section conducts a further evaluation of the effectiveness of Ms-PoE by addressing the following questions: $Q l$ : How does the effectiveness of Ms-PoE relate to the head-wise selection strategy of the scaling ratio? $Q 2$ : How does the model perform with different scaling ratios?

A1: Positional awareness metrics achieve superior performance compared to other strategies. For a set of scaling ratios $\\mathbf { r } \\in R ^ { n \_ { h } }$ , where $n \_ { h }$ is the number of attention heads, and using scaling ratios linearly ranging from 1.2 to 1.8, we evaluate various strategies for assigning these ratios to different attention heads. These strategies include: $\\textcircled{1}$ Random, which randomly assigns the scaling ratios to each head within each layer; $\\textcircled{2}$ Sequential, performing the assignment based on the original head order; $\\textcircled{3}$ Entropy, where we follow metrics measuring the sparsity level of attention scores (Tian et al., 2023). Larger entropy implies less sparse attention scores, indicating the model attends to more tokens rather than just the beginning and end words, so we assign a scaling ratio near to 1, and vice versa for larger ratios. Results in Table 3 demonstrate that the proposed position-awareness effectively captures the head-wise properties of LLMs, enhancing performance when critical information is located at various positions—beginning, middle, or end. This leads to an average accuracy gain of 3.2 (65.3 v.s. 62.1).

A2: Ablation study of the scaling ratios. We first examined the effect of uniform scaling ratios across all heads on model performance. Our findings, outlined in Table 4, indicate that adjusting the scaling ratio between 0.5 and 2.5 can significantly enhance generative performance and mitigate the ”lost-in-the-middle” effect by $1 . 0 %$ ( $6 3 . 1 %$ v.s. $6 2 . 1 %$ ), particularly with a ratio of 1.5. Further testing with an average ratio of 1.5 across all heads revealed that an optimal range exists between 1.2 and 1.8, leading to an additional $2 . 2 %$ ( $6 5 . 3 %$ v.s. $6 3 . 1 %$ ) accuracy improvement with our approach, Ms-PoE. Based on these results, we established these ratios as our experimental standard.

Table 3. Ablation results of different ordering metrics. Experiments are conducted on Multi-Documents Question Answering task with the Vicuna-7B model.

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| Methods | Begin | Middle | End | Average |
| Baseline | 64.0 | 57.4 | 64.8 | 62.1 |
| Random | 64.5 | 55.0 | 65.5 | 61.7 |
| Sequential | 60.5 | 54.5 | 58.5 | 57.8 |
| Entropy | 63.5 | 59.5 | 64.0 | 62.3 |
| Position-Awareness | 65.6 | 63.0 | 67.2 | 65.3 |

Table 4. Ablation results of the condensing ratios. Experiments are conducted on Multi-Documents Question Answering task with the Vicuna-7B model.

|     |     |     |
| --- | --- | --- |
| Scaling Ratio | Begin Middle End | Average |
| 1 | 64.0 57.4 64.8 | 62.1 |
| 0.5 1.5 | 56.0 51.0 65.2 60.0 | 68.0 58.3 64.0 63.1 |
| 2 2.5 | 61.5 59.0 59.5 57.5 | 62.5 61.0 57.0 58.0 |
| 0.8→2.2 1→2 | 53.5 59.5 | 67.5 60.2 63.0 |
| 1.2 → 1.8 | 61.0 57.0 65.6 63.0 | 60.3 67.2 65.3 |
| 1.4 → 1.6 | 65.5 59.0 | 63.0 62.5 |

# 5\. Conclusion

In this paper, we present a plug-and-play strategy designed to address the “lost-in-the-middle” challenge observed in LLMs. This challenge stems from the persistent bias exhibited by LLMs towards the beginning and local content within the input, leading to the neglect of crucial information in the middle. Our investigation reveals the effects of position indice rescaling and the head-wise positionawareness property, leading to the introduction of Multiscale Positional Encoding (Ms-PoE). This approach enhances the capability of LLMs to effectively capture information in the middle of the context without the need for additional fine-tuning. Comprehensive experiments conducted on Zero-SCROLLS benchmarks, multi-document question-answering tasks, and key-value retrieval tasks confirm the effectiveness of Ms-PoE.

# 6\. Impact Statement

The introduction of Multi-scale Positional Encoding (MsPoE) offers a simple yet effective approach to tackling the challenge of processing long contexts in large language models (LLMs). This enhancement significantly improves LLMs’ capabilities to understand and reason over extensive textual contexts. Ms-PoE can potentially impact numerous fields where processing long-context data is crucial, such as in analyzing vast amounts of case law or detailed patient histories. However, as we advance the boundaries of Machine Learning and Artificial Intelligence, making them increasingly powerful and applicable, it comes with potential challenges and problems that must be carefully addressed. For instance, there is a risk of exacerbating existing biases within AI, as processing longer contexts could amplify the biases present in the training data. Additionally, the misuse of predictive insights derived from LLMs presents a significant worry. To mitigate these concerns, we advocate for ongoing ethical evaluations and the development of guidelines to ensure that the applications of LLM advancements contribute positively to society.

# 7\. Acknowledgements

We thank Dr. Yuandong Tian for interesting discussions on this work.

# References

Chen, G., Li, X., Meng, Z., Liang, S., and Bing, L. Clex: Continuous length extrapolation for large language models. arXiv preprint arXiv:2310.16450, 2023a.

Chen, H., Pasunuru, R., Weston, J., and Celikyilmaz, A. Walking down the memory maze: Beyond context limit through interactive reading. arXiv preprint arXiv:2310.05029, 2023b.

Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023c.

Chen, Y., Lv, A., Lin, T.-E., Chen, C., Wu, Y., Huang, F., Li, Y., and Yan, R. Fortify the shortest stave in attention: Enhancing context awareness of large language models for effective tool use. arXiv preprint arXiv:2312.04455, 2023d.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with $9 0 % ^ { \* }$ chatgpt quality,

March 2023. URL [https://lmsys.org/blog/](https://lmsys.org/blog/) 2023-03-30-vicuna/.

Clark, K., Luong, M.-T., Le, Q. V., and Manning, C. D. Electra: Pre-training text encoders as discriminators rather than generators. arXiv preprint arXiv:2003.10555, 2020.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Du, X., Liu, M., Wang, K., Wang, H., Liu, J., Chen, Y., Feng, J., Sha, C., Peng, X., and Lou, Y. Classeval: A manually-crafted benchmark for evaluating llms on classlevel code generation. arXiv preprint arXiv:2308.01861, 2023.

Edelman, B. L., Goel, S., Kakade, S., and Zhang, C. Inductive biases and variable creation in self-attention mechanisms. In International Conference on Machine Learning, pp. 5793–5831. PMLR, 2022.

Ge, S., Zhang, Y., Liu, L., Zhang, M., Han, J., and Gao, J. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.

Guo, M.-H., Lu, C.-Z., Hou, Q., Liu, Z., Cheng, M.-M., and Hu, S.-M. Segnext: Rethinking convolutional attention design for semantic segmentation. Advances in Neural Information Processing Systems, 35:1140–1156, 2022.

Han, C., Wang, Q., Xiong, W., Chen, Y., Ji, H., and Wang, S. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv preprint arXiv:2308.16137, 2023.

Ivgi, M., Shaham, U., and Berant, J. Efficient long-text understanding with short-text models. Transactions of the Association for Computational Linguistics, 11:284– 299, 2023.

Jacobs, S. A., Tanaka, M., Zhang, C., Zhang, M., Song, S. L., Rajbhandari, S., and He, Y. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. arXiv preprint arXiv:2309.14509, 2023.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. d. l., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023a.

Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Jiang, H., Wu, Q., Luo, X., Li, D., Lin, C.-Y., Yang, Y., and Qiu, L. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. arXiv preprint arXiv:2310.06839, 2023b.

Jin, H., Han, X., Yang, J., Jiang, Z., Liu, Z., Chang, C.- Y., Chen, H., and Hu, X. Llm maybe longlm: Selfextend llm context window without tuning. arXiv preprint arXiv:2401.01325, 2024.

Junqing, H., Kunhao, P., Xiaoqun, D., Zhuoyang, S., Yibo, L., Yuxin, L., Hao, W., Qianguo, S., Songxin, Z., Zejian, X., et al. Never lost in the middle: Improving large language models via attention strengthening question answering. arXiv preprint arXiv:2311.09198, 2023.

Kry´sci´nski, W., Rajani, N., Agarwal, D., Xiong, C., and Radev, D. Booksum: A collection of datasets for long-form narrative summarization. arXiv preprint arXiv:2105.08209, 2021.

Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., and Soricut, R. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942, 2019.

Li, J., Wang, M., Zheng, Z., and Zhang, M. Loogle: Can long-context language models understand long contexts? arXiv preprint arXiv:2311.04939, 2023.

Likhosherstov, V., Choromanski, K., and Weller, A. On the expressive power of self-attention matrices. arXiv preprint arXiv:2106.03764, 2021.

Lin, J., Tang, J., Tang, H., Yang, S., Dang, X., and Han, S. Awq: Activation-aware weight quantization for llm compression and acceleration. arXiv preprint arXiv:2306.00978, 2023.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023.

Mahan, D., Carlow, R., Castricato, L., Cooper, N., and Laforte, C. Stable beluga models. URL \[ [https://huggingface.co/stabilityai/](https://huggingface.co/stabilityai/) StableBeluga2\]( [https://huggingface](https://huggingface/). co/stabilityai/StableBeluga2).

Oren, M., Hassid, M., Adi, Y., and Schwartz, R. Transformers are multi-state rnns. arXiv preprint arXiv:2401.06104, 2024.

Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., and Launay, J. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023. URL [https://arxiv.org/abs/2306.01116](https://arxiv.org/abs/2306.01116).

Peng, B., Quesnelle, J., Fan, H., and Shippole, E. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

Peysakhovich, A. and Lerer, A. Attention sorting combats recency bias in long context language models. arXiv preprint arXiv:2310.01427, 2023.

Press, O., Smith, N. A., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

Radford, A., Narasimhan, K., Salimans, T., Sutskever, I., et al. Improving language understanding by generative pre-training. 2018.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Re´, C., Dao, T., Fu, D., and Goel, K. Can longer sequences help take the next leap in ai?, June 2022. URL https: //hazyresearch.stanford.edu/blog/ 2022-06-09-longer-sequences-next-leap-ai. Accessed: 2024-01-29.

Shaham, U., Ivgi, M., Efrat, A., Berant, J., and Levy, O. Zeroscrolls: A zero-shot benchmark for long text understanding. arXiv preprint arXiv:2305.14196, 2023.

Song, K., Wang, X., Cho, S., Pan, X., and Yu, D. Zebra: Extending context window with layerwise grouped local-global attention. arXiv preprint arXiv:2312.08618, 2023a.

Song, S. L., Kruft, B., Zhang, M., Li, C., Chen, S., Zhang, C., Tanaka, M., Wu, X., Rasley, J., Awan, A. A., et al. Deepspeed4science initiative: Enabling large-scale scientific discovery through sophisticated ai system technologies. arXiv preprint arXiv:2310.04610, 2023b.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., and Rabinovich, A. Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.

Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2818–2826, 2016.

Tay, Y., Dehghani, M., Abnar, S., Shen, Y., Bahri, D., Pham, P., Rao, J., Yang, L., Ruder, S., and Metzler, D. Long range arena: A benchmark for efficient transformers. In International Conference on Learning Representations, 2020.

Team, M. N. Introducing mpt-7b: A new standard for open-source, commercially usable llms, 2023. URL www. mosaicml.com/blog/mpt-7b. Accessed: 2023- 05-05.

Tian, Y., Wang, Y., Zhang, Z., Chen, B., and Du, S. Joma: Demystifying multilayer transformers via joint dynamics of mlp and attention. arXiv preprint arXiv:2310.00535, 2023.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. Llama 2: Open foundation and finetuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Varadi, M., Anyango, S., Deshpande, M., Nair, S., Natassia, C., Yordanova, G., Yuan, D., Stroe, O., Wood, G., Laydon, A., et al. Alphafold protein structure database: massively expanding the structural coverage of protein-sequence space with high-accuracy models. Nucleic acids research, 50(D1):D439–D444, 2022.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Wang, W., Dong, L., Cheng, H., Liu, X., Yan, X., Gao, J., and Wei, F. Augmenting language models with long-term memory. arXiv preprint arXiv:2306.07174, 2023.

Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

Xu, P., Ping, W., Wu, X., McAfee, L., Zhu, C., Liu, Z., Subramanian, S., Bakhturina, E., Shoeybi, M., and Catanzaro, B. Retrieval meets long context large language models. arXiv preprint arXiv:2310.03025, 2023.

Yin, L., Wu, Y., Zhang, Z., Hsieh, C.-Y., Wang, Y., Jia, Y., Pechenizkiy, M., Liang, Y., Wang, Z., and Liu, S. Outlier weighed layerwise sparsity (owl): A missing secret sauce for pruning llms to high sparsity, 2023.

Zhang, P., Liu, Z., Xiao, S., Shao, N., Ye, Q., and Dou, Z. Soaring from 4k to 400k: Extending llm’s context with activation beacon. arXiv preprint arXiv:2401.03462, 2024.

Zhang, Y., Ni, A., Mao, Z., Wu, C. H., Zhu, C., Deb, B., Awadallah, A. H., Radev, D., and Zhang, R. Summˆ n: A multi-stage summarization framework for long input dialogues and documents. arXiv preprint arXiv:2110.10150, 2021.

Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., R´e, C., Barrett, C., et al. H 2 o: Heavy-hitter oracle for efficient generative inference of large language models. arXiv preprint arXiv:2306.14048, 2023.

Zheng, Q., Xia, X., Zou, X., Dong, Y., Wang, S., Xue, Y., Wang, Z., Shen, L., Wang, A., Li, Y., Su, T., Yang, Z., and Tang, J. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. In KDD, 2023.

Zhong, M., Liu, Y., Xu, Y., Zhu, C., and Zeng, M. Dialoglm: Pre-trained model for long dialogue understanding and summarization. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 11765–11773, 2022.

# A. More Experiment Results A.1. Position-Aware Attention Heads

https://arxiv.org/pdf/images/2ece15d495708afee3f896193528181258b9f461af34bcc24b4f55ba0a8b6615.jpg

Figure 6. Visualization of ”position-aware” attention heads. Each row contains the attention pattern for the same heads when varying the key documents within the inputs.

Figure 6 illustrates the attention patterns of ”position-aware” heads. Each row represents the attention pattern of the same head. As the key document is positioned from the beginning to the end, the attention peak gradually shifts, indicating robust positional awareness. It’s important to note that we randomly selected 9 attention heads with these ”position-aware” properties, and these results were validated with different input samples and layers.

</details>

<details>
<summary>Nelson F. Liu1∗ Kevin Lin2 John Hewitt1 Ashwin Paranjape3 Michele Bevilacqua3 Fabio Petroni3 Percy Liang1</summary>

Nelson F. Liu1∗ Kevin Lin2 John Hewitt1 Ashwin Paranjape3 Michele Bevilacqua3 Fabio Petroni3 Percy Liang1

1Stanford University 2University of California, Berkeley 3Samaya AI [nfliu@cs.stanford.edu](mailto:nfliu@cs.stanford.edu)

# Abstract

While recent language models have the ability to take long contexts as input, relatively little is known about how well they use longer context. We analyze the performance of language models on two tasks that require identifying relevant information in their input contexts: multi-document question answering and key-value retrieval. We find that performance can degrade significantly when changing the position of relevant information, indicating that current language models do not robustly make use of information in long input contexts. In particular, we observe that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts, even for explicitly long-context models. Our analysis provides a better understanding of how language models use their input context and provides new evaluation protocols for future long-context language models.

https://cs.stanford.edu/~nfliu/papers/images/128ee6e1d16bfbb2c506be5ddc55195b06528584947c6b81bad032ed8288fbbe.jpg

Figure 1: Changing the location of relevant information (in this case, the position of the passage that answers an input question) within the language model’s input context results in a U-shaped performance curve—models are better at using relevant information that occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performance degrades significantly when models must access and use information located in the middle of its input context.

# 1 Introduction

Language models have become an important and flexible building block in a variety of user-facing language technologies, including conversational interfaces, search and summarization, and collaborative writing (Shuster et al., 2022; Thoppilan et al., 2022; Lee et al., 2022, inter alia). These models perform downstream tasks primarily via prompting: all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion. These input contexts can contain thousands of tokens, especially when language models are used to process long documents (e.g., legal or scientific documents, conversation histories, etc.) or when language models are augmented with external information (e.g., relevant documents from a search engine, database query results, etc; Petroni et al., 2020; Ram et al., 2023; Shi et al., 2023; Mallen et al., 2023; Schick et al., 2023, inter alia).

Handling these use-cases requires language models to successfully operate over long sequences. Existing language models are generally implemented with Transformers (Vaswani et al., 2017), which require memory and compute that increases quadratically in sequence length. As a result, Transformer language models were often trained with relatively small context windows (between 512- 2048 tokens). Recent improvements in hardware (e.g., faster GPUs with more memory) and algorithms (Dai et al., 2019; Dao et al., 2022; Poli et al.,

2023; Rubin and Berant, 2023, inter alia) have resulted in language models with larger context windows (e.g., 4096, 32K, and even 100K tokens), but it remains unclear how these extended-context language models make use of their input contexts when performing downstream tasks.

We empirically investigate this question via controlled experiments with a variety of state-ofthe-art open (MPT-30B-Instruct, LongChat-13B (16K)) and closed (OpenAI’s GPT-3.5-Turbo and Anthropic’s Claude-1.3) language models in settings that require accessing and using information within an input context. In particular, our experiments make controlled changes to the input context size and the position of the relevant information within the input context and study their effects on language model performance. If language models can robustly use information within long input contexts, then their performance should be minimally affected by the position of the relevant information in the input context.

We first experiment with multi-document question answering, which requires models to reason over provided documents to find relevant information and use it to answer a given question; this task mimics the retrieval-augmented generation setup underlying many commercial generative search and question answering applications (e.g., Bing Chat). In this setting, we control (i) the input context length by changing the number of documents in the input context (akin to retrieving more or less documents in retrieval-augmented generation), and (ii) control the position of the relevant information within the input context by changing the order of the documents to place the relevant document at the beginning, middle or end of the context.

We find that changing the position of relevant information in the input context can substantially affect model performance, indicating that current language models do not robustly access and use information in long input contexts. Furthermore, we observe a distinctive U-shaped performance curve (Figure 1); language model performance is highest when relevant information occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performance significantly degrades when models must access and use information in the middle of their input context (§2.3). For example, when relevant information is placed in the middle of its input context, GPT-3.5-Turbo’s performance on the multidocument question task is lower than its performance when predicting without any documents (i.e., the closed-book setting; $56 . 1 %$ ). Furthermore, we find that models often have identical performance to their extended-context counterparts, indicating that extended-context models are not necessarily better at using their input context (§2.3).

Given that language models struggle to retrieve and use relevant information in the multi-document question answering task, to what extent can language models even retrieve from their input contexts? We study this question with a synthetic keyvalue retrieval task, which is designed to be a minimal testbed for the basic ability to retrieve matching tokens from the input context. In this task, models are given a collection of JSON-formatted key-value pairs and must return the value associated with a specific key. Similar to the multi-document QA task, the key-value retrieval task admits controlled changes to the input context length (adding more key-value pairs) and the position of relevant information. Although some models perform the synthetic key-value retrieval task perfectly, other models struggle to simply retrieve matching tokens that occur in the middle of their input context and continue to exhibit a U-shaped performance curve.

To better understand why language models struggle to robustly access and use information in their input contexts, we study the role of model architecture (decoder-only vs. encoder-decoder), queryaware contextualization, and instruction fine-tuning (§4). We find that:

• Encoder-decoder models are relatively robust to changes in the position of relevant information within their input context, but only when evaluated on sequences within its trainingtime sequence length. When evaluated on sequences longer than those seen during training, we observe a U-shaped performance curve (§4.1). • Query-aware contextualization (placing the query before and after the documents or keyvalue pairs) enables near-perfect performance on the synthetic key-value task, but minimally changes trends in multi-document QA (§4.2). • Even base language models (i.e., without instruction fine-tuning) show a U-shaped performance curve as we vary the position of relevant information in the input context.

Our results indicate that prompting language models with longer input contexts is a trade-off— providing the language model with more information may help it perform the downstream task, but it also increases the amount of content that the model must reason over, potentially decreasing accuracy. To better understand this trade-off in practice, we perform a case study with retriever-reader models on open-domain question answering (§5). In contrast to our controlled multi-document QA task, where the context always contains exactly one document that answers the question, none or many of the top $k$ documents may contain the answer in the open-domain QA setting. When retrieving from Wikipedia to answer queries from NaturalQuestions-Open, we find that model performance saturates long before retriever recall saturates, indicating that current models fail to effectively use additional retrieved documents—using 50 documents instead of 20 retrieved documents only marginally improves performance $( \\sim 1 . 5 %$ for GPT-3.5-Turbo and ${ \\sim } 1 %$ for claude-1.3).

Our analysis provides a better understanding of how language models use their input context and introduces new evaluation protocols for future longcontext models; to claim that a language model can robustly use information within long input contexts, it is necessary to show that its performance is minimally affected by the position of the relevant information in the input context (e.g., minimal difference in best- and worst-case performance). To facilitate further work on understanding and improving how language models use their input context, we release our code and evaluation data.1

# 2 Multi-Document Question Answering

Our goal is to better understand how language models use their input context. To this end, we analyze model performance on multi-document question answering, which requires models to find relevant information within an input context and use it to answer the question. In particular, we make controlled changes to the length of the input context and the position of the relevant information and measure changes in task performance.

# 2.1 Experimental Setup

In the multi-document question answering task, the model inputs are (i) a question to answer and (ii) $k$ documents (e.g., passages from Wikipedia), where exactly one of the documents contains the answer to the question and $k - 1$ “distractor” documents do not. This task requires the model to access the document that contains the answer within its input context and use it to answer the question. Figure 2 presents an example.

We instantiate this task with data from NaturalQuestions-Open (Lee et al., 2019; Kwiatkowski et al., 2019), which contains historical queries issued to the Google search engine, coupled with human-annotated answers extracted from Wikipedia. In particular, we take the 2655 queries where the annotated long answer is a paragraph (as opposed to a list or a table). We use passages (chunks of at most 100 tokens) from Wikipedia as documents within our input contexts. For each of the queries, we need a document that contains the answer and $k - 1$ distractor documents that do not contain the answer. To obtain a document that answers the question, we use the Wikipedia paragraph that contains the answer from the NaturalQuestions annotations.

To collect $k - 1$ distractor documents that do not contain the answer, we use a retrieval system (Contriever, fine-tuned on MS-MARCO; Izacard et al., 2021) to retrieve the $k - 1$ Wikipedia chunks that are most relevant to the query and do not contain any of the NaturalQuestions-annotated answers.2,3 In the input context, the distractor documents are presented in order of decreasing relevance.4

To modulate the position of relevant information within the input context, we adjust the order of the documents to change the position of the document that contains the answer (Figure 3). To modulate the input context length in this task, we increase or decrease the number of retrieved documents that do not contain the answer (Figure 4).

Following Kandpal et al. (2022) and Mallen et al. (2023), we use accuracy as our primary evaluation metric, judging whether any of the correct answers (as taken from the NaturalQuestions annotations) appear in the predicted output.

https://cs.stanford.edu/~nfliu/papers/images/a58360090e62d9372e6a4bbed1575886810f110c8a76a3ff8d326cc51d0440dc.jpg

Figure 2: Example of the multi-document question answering task, with an input context and the desired model answer. The document containing the answer is bolded within the input context here for clarity.

https://cs.stanford.edu/~nfliu/papers/images/dbeb34613fa4843ed7bd72e1065e183ba58891fa7de1e6b8f68cc56e4865e4ad.jpg

Figure 3: Modulating the position of relevant information within the input context for the multi-document question answering example presented in Figure 2. Reordering the documents in the input context does not affect the desired output.

Our experimental setup is similar to the needlein-a-haystack experiments of Ivgi et al. (2023), who compare question answering performance when the relevant paragraph is placed (i) at the beginning of the input or (ii) a random position within the input. They find that encoder-decoder models have significantly higher performance when relevant information is placed at the start of the input context. In contrast, we study finer-grained changes in the position of relevant information.

# 2.2 Models

We analyze several state-of-the-art open and closed language models. We use greedy decoding when generating outputs and leave exploration of other decoding methods to future work. We use a standard set of prompts for each model (Figure 2).

https://cs.stanford.edu/~nfliu/papers/images/dc7ef792a7153b170e7c41581ffcc2c8919a5dfcd89d7da80c74d1f325fae0d6.jpg

Figure 4: Modulating the input context length of the multi-document question answering example presented in Figure 2. Adding documents that do not contain the answer increases the length of the input context, but does not affect the desired output.

Open models. We experiment with MPT-30BInstruct, which has a maximum context length of 8192 tokens. The model was initially pre-trained on 1 trillion tokens using 2048-token sequences, followed by an additional sequence length adaptation pre-training phase on 50 billion tokens using 8192-token sequences. MPT-30B-Instruct uses ALiBi (Press et al., 2022) to represent positional information. We also evaluate LongChat-13B (16K) (Li et al., 2023), which extends the LLaMA-13B (Touvron et al., 2023a) context window from 2048 to 16384 tokens by using condensed rotary positional embeddings before fine-tuning with 16384-token sequences.

Closed models. We use the OpenAI API to experiment with GPT-3.5-Turbo and GPT-3.5-Turbo (16K).5 GPT-3.5-Turbo has a maximum context length of 4K tokens, and GPT-3.5-Turbo (16K) is a version with an extended maximum context length of 16K tokens. We evaluate Claude-1.3 and Claude1.3 (100K) with the Anthropic API; Claude-1.3 has a maximum context length of 8K tokens, and Claude-1.3 (100K) has an extended context length of 100K tokens. 6

https://cs.stanford.edu/~nfliu/papers/images/6e12214d976fc59db00a018a0fef0d7965fe6c17643c2a9dd939ff984f551f8e.jpg

Figure 5: The effect of changing the position of relevant information (document containing the answer) on multidocument question answering performance. Lower positions are closer to the start of the input context. Performance is highest when relevant information occurs at the very start or end of the context, and rapidly degrades when models must reason over information in the middle of their input context.

# 2.3 Results and Discussion

We experiment with input contexts containing 10, 20, and 30 total documents. Figure 5 presents multidocument question answering performance when varying the position of relevant information within the input context. To contextualize model performance, we also evaluate on the closed-book and oracle settings (Table 1). In the closed-book setting, models are not given any documents in their input context, and must rely on their parametric memory to generate the correct answer. On the other hand, in the oracle setting, language models are given the single document that contains the answer and must use it to answer the question.

Model performance is highest when relevant information occurs at the beginning or end of its input context. As illustrated in Figure 5, changing the position of relevant information in the input context leads to substantial decreases in model performance. In particular, we see a distinctive Ushaped performance curve—models are often much better at using relevant information that occurs at the very beginning (primacy bias) and very end of contexts (recency bias), and suffer degraded performance when forced to use information within the middle of its input context. For example, GPT-3.5- Turbo’s multi-document QA performance can drop by more than $20 %$ —in the worst case, performance in 20- and 30-document settings is lower than performance without any input documents (i.e., closedbook performance; $56 . 1 %$ . These results indicate that current models cannot effectively reason over their entire context window when prompted for downstream tasks.

Table 1: Closed-book and oracle accuracy of language models on the multi-document question answering task.

|     |     |     |
| --- | --- | --- |
| Model | Closed-Book | Oracle |
| LongChat-13B (16K) | 35.0% | 83.4% |
| MPT-30B-Instruct | 31.5% | 81.9% |
| GPT-3.5-Turbo | 56.1% | 88.3% |
| GPT-3.5-Turbo (16K) | 56.0% | 88.6% |
| Claude-1.3 | 48.3% | 76.1% |
| Claude-1.3 (100K) | 48.2% | 76.4% |

Extended-context models are not necessarily better at using input context. When the input context fits in the context window of both a model and its extended-context counterpart, we see that performance between them is nearly identical. For example, the 10- and 20-document settings both fit in the context window of GPT-3.5-Turbo and GPT-3.5-Turbo (16K), and we observe that their performance as a function of position of relative information is nearly superimposed (solid purple and dashed brown series in Figure 5). These results indicate that extended-context models are not necessarily better than their non-extended counterparts at using their input context.

https://cs.stanford.edu/~nfliu/papers/images/7c7bd468614d0dea7de58d3a53496f8dc2fb3544f7e1e30abcef8dca3f4cbcbf.jpg

Figure 6: Example of the key-value retrieval task, with an input context and the desired model output. Given a key, the goal is to return the associated value. All keys and values are 128-bit UUIDs. The relevant key-value pair for answering the query is bolded here within the input context for clarity.

# 3 How Well Can Language Models Retrieve From Input Contexts?

Given that language models struggle to retrieve and use information from the middle of their input contexts in the multi-document question answering task, to what extent can they simply retrieve from input contexts? We study this question with a synthetic key-value retrieval task, which is designed to provide a minimal testbed for the basic ability to retrieve matching tokens from an input context.

# 3.1 Experimental Setup

In our synthetic key-value retrieval task, the inputs are (i) a string-serialized JSON object with $k$ keyvalue pairs, where each of the keys and values are unique, randomly-generated UUIDs and (ii) a key within the aforementioned JSON object. The goal is to return the value associated with the specified key. Thus, each JSON object contains one relevant key-value pair (where the value is to be returned), and $k - 1$ irrelevant “distractor” key-value pairs. Figure 6 provides an example input context and its corresponding desired output. We again measure accuracy by evaluating whether the correct value appears in the predicted output.

Our synthetic key-value retrieval task shares similar goals with the Little Retrieval Test of Papailiopoulos et al. (2023) and the fine-grained line retrieval task of Li et al. (2023), but we explicitly seek to distill and simplify the task by removing as much natural language semantics as possible (using random UUIDs instead), since language features may present potential confounders. For example, Transformer language models may have varying sensitivity to different linguistic features in their input (O’Connor and Andreas, 2021).

To modulate the position of relevant information within the input context, we change the position of the key to retrieve within the serialized JSON object. To modulate the input context length, we change the number of input JSON key-value pairs $k$ by adding or removing random keys, changing the number of distractor key-value pairs.

# 3.2 Results and Discussion

We experiment with input contexts containing 75, 140, and 300 key-value pairs (500 examples each). We use the same set of models as the multidocument question answering experiments, see $\ S 2 . 2$ for more details.

Figure 7 presents key-value retrieval performance. Claude-1.3 and Claude-1.3 (100K) do nearly perfectly on all evaluated input context lengths, but other models struggle, especially when contexts have 140 or 300 key-value pairs— although the synthetic key-value retrieval task only requires identifying exact match within the input context, not all models achieve high performance.

Similar to our multi-document QA results, GPT3.5-Turbo, GPT-3.5-Turbo (16K), and MPT-30BInstruct have the lowest performance when they must access key-value pairs in the middle of their input context. LongChat-13B (16K) exhibits a different trend in the 140 key-value setting; we qualitatively observe that when relevant information is placed at the start of the input context, LongChat13B (16K) tends to generate code to retrieve the key, rather than outputting the value directly.

https://cs.stanford.edu/~nfliu/papers/images/68b8f153bb570120002b8d0b36943a11d07d8a182c9f17d48816451303ee60b7.jpg

Figure 7: The effect of changing the input context length and the position of relevant information on key-value retrieval performance. Lower positions are closer to the start of the input context. Although some models show perfect accuracy on this synthetic task (e.g., Claude-1.3 and Claude-1.3 (100K)), we see again that performance is often highest when relevant information is occurs at the very start or end of the context, and rapidly degrades when models must retrieve from the middle of the input context.

# 4 Why Are Language Models Not Robust to Changes in the Position of Relevant Information?

Our multi-document question answering and keyvalue retrieval results show that language models struggle to robustly access and use information in long input contexts, since performance degrades significantly when changing the position of relevant information. To better understand why, we perform some preliminary investigations into the role of model architecture (decoder-only vs. encoderdecoder), query-aware contextualization, and instruction fine-tuning.

# 4.1 Effect of Model Architecture

The open models we evaluated are all decoder-only models—at each timestep, they may only attend to prior tokens. To better understand the potential effects of model architecture on how language model use context, we compare decoder-only and encoder-decoder language models.

We experiment with Flan-T5-XXL (Raffel et al., 2020; Chung et al., 2022) and Flan-UL2 (Tay et al., 2023). Flan-T5-XXL is trained with a sequences of 512 tokens (encoder and decoder). Flan-UL2 is initially trained with sequences of 512 tokens (encoder and decoder), but is then pre-trained for an extra 100K steps with 1024 tokens (encoder and decoder) before instruction fine-tuning on sequences with 2048 tokens in the encoder and 512 tokens in the decoder. However, since these models use relative positional embeddings, they can (in principle) extrapolate beyond these maximum context lengths; Shaham et al. (2023) find that both models can perform well with sequences of up to 8K tokens.

Figure 8 compares the performance of decoderonly and encoder-decoder models. When Flan-UL2 is evaluated on sequences within its 2048-token training-time context window (Figure 8; left subplot), its performance is relatively robust to changes in the position of relevant information within the input context ( $1 . 9 %$ absolute difference between best- and worst-case performance). When evaluated on settings with sequences longer than 2048 tokens (Figure 8; center and right), Flan-UL2 performance begins to degrade when relevant information is placed in the middle. Flan-T5-XXL shows a similar trend, where longer input contexts result in a greater performance degradation when placing relevant information in the middle of the input context. We hypothesize that encoder-decoder models may make better use of their context windows because their bidirectional encoder allows processing each document in the context of future documents, potentially improving relative importance estimation between documents.

# 4.2 Effect of Query-Aware Contextualization

Our multi-document QA and key-value retrieval experiments place the query (i.e., question to answer or key to retrieve) after the data to process (i.e., the documents or the key-value pairs). As a result, decoder-only models cannot attend to query tokens when contextualizing documents or keyvalue pairs, since the query only appears at the end of the prompt and decoder-only models can only attend to prior tokens at each timestep. In contrast, encoder-decoder models (which seem more robust to changes in the position of relevant information; $\ S 4 . 1 \ r ,$ ) use a bidirectional encoder to contextualize input contexts—can we use this observation to improve decoder-only models by placing the query before and after the data, enabling query-aware contextualization of documents (or key-value pairs)?

https://cs.stanford.edu/~nfliu/papers/images/b60d50233e8740aa703a3e40ee35b0ff8093e7ba901ae0fc9c2ea80ca69b2ca4.jpg

Figure 8: When encoder-decoder models (Flan-UL2 and Flan-T5-XXL) evaluated on sequences that are shorter than their encoder’s training-time maximum sequence length (2048 and 512 tokens, respectively), they are relatively robust to changes in the position of relevant information within their input context (left subplot). In contrast, when these models are evaluated on sequences longer than those seen during training (center and right subplots), we observe a U-shaped performance curve—performance is higher when relevant information occurs at the beginning or end of the input context, as opposed to the middle of the input context.

https://cs.stanford.edu/~nfliu/papers/images/441bd021b73f591549ca7216dbb46743392ecad270c8e10aac2de41db4e747f8.jpg

Figure 9: Query-aware contextualization (placing the query before and after the documents) does not substantially improve robustness of language models to changing the position of relevant information in multidocument QA; performance slightly increases when relevant information occurs at the very beginning, but otherwise slightly decreases.

We find that query-aware contextualization dramatically improves performance on the key-value retrieval task—all models achieve near-perfect performance on the 75, 140, and 300 key-value pair settings. For example, GPT-3.5-Turbo (16K) with query-aware contextualization achieves perfect performance when evaluated with 300 key-value pairs.

In contrast, without query-aware contextualization, the worst-case performance is $4 5 . 6 %$ (Figure 7). Despite the significant impact on keyvalue retrieval performance, query-aware contextualization minimally affects performance trends in the multi-document question answering task (Figure 9); it slightly improves performance when the relevant information is located at the very beginning of the input context, but slightly decreases performance in other settings.

# 4.3 Effect of Instruction Fine-Tuning

The models we evaluated are all instruction finetuned—after their initial pre-training, they undergo supervised fine-tuning on a dataset of instructions and responses. The task specification and/or instruction is commonly placed at the beginning of the input context in supervised instruction finetuning data, which might lead instruction finetuned language models to place more weight on the start of the input context. To better understand the potential effects of instruction fine-tuning on how language models use long input contexts, we compare the multi-document question answering performance of MPT-30B-Instruct against its base model (i.e., before instruction fine-tuning) MPT30B. We use the same experimental setup as $\ S 2$ .

Figure 10 compares the multi-document QA performance of MPT-30B and MPT-30B-Instruct as a function of the position of the relevant information in the input context. Surprisingly, we see that both MPT-30B and MPT-30B-Instruct exhibit a U-shaped performance curve, where performance is highest when relevant information occurs at the very beginning or very end of the context. Although the absolute performance of MPT-30BInstruct is uniformly higher than that of MPT-30B, their overall performance trends are similar. We also observe that instruction fine-tuning slightly reduces the worst-case performance disparity from nearly $10 %$ between the base model best- and worst-case performance to around $4 %$ .

https://cs.stanford.edu/~nfliu/papers/images/f60dd1f1ed4ba004904ff48f57f175fb1637a68e0ca14c56811576ac0bd0036f.jpg

20 Total Retrieved Documents (~4K tokens)

Figure 10: Multi-document QA performance of MPT30B-Instruct compared against its base model (i.e., before instruction fine-tuning) MPT-30B. Both models have a U-shaped performance curve, where performance is much higher when relevant information occurs at the start or end of the input context, indicating that the instruction fine-tuning process itself is not necessarily responsible for these performance trends.

These observations complement prior work, which found that non-instruction fine-tuned language models are biased towards recent tokens (i.e., the end of the input context; Khandelwal et al., 2018; Press et al., 2021). This recency bias has been observed in past work when evaluating models on next-word prediction of contiguous text, a setting where language models minimally benefit from long-range information (Sun et al., 2021). In contrast, our results show that language models are capable of using longer-range information (i.e., the beginning of the input context) when prompted with instruction-formatted data. We hypothesize that non-instruction fine-tuned language models learn to use these long contexts from similarlyformatted data that may occur in Internet text seen during pre-training, e.g., StackOverflow questions

and answers.

To better understand the effect of additional finetuning and model scale, we also experimented with Llama-2 models of varying sizes (7B, 13B, and 70B) with and without additional supervised fine-tuning and reinforcement learning from human feedback (Appendix E). We find that the Ushaped performance curve only appears in sufficiently large language models (with or without additional fine-tuning)—the 7B Llama-2 models are solely recency biased, while the 13B and 70B models exhibit a U-shaped performance curve. In addition, we see that the Llama-2 supervised fine-tuning and reinforcement learning from human feedback procedure slightly mitigates the positional bias in smaller models (13B, akin to trends shown when comparing MPT-30B and MPT-30B-Instruct), but minimally affects trends on larger models (70B).

# 5 Is More Context Is Always Better? A Case Study With Open-Domain QA

Our results indicate that prompting language models with longer input contexts is a trade-off— providing the language model with more information may help it perform the downstream task, but it also increases the amount of content that the model must reason over, potentially decreasing accuracy. Even if a language model can take in 16K tokens, is it actually beneficial to provide 16K tokens of context? The answer to this question is ultimately downstream task-specific since it depends on the marginal value of the added context and the model’s ability to effectively use long input contexts, but we perform a case study with opendomain question answering on NaturalQuestionsOpen to better understand this trade-off in existing language models.

We use language models in a standard retrieverreader setup. A retrieval system (Contriever, finetuned on MS-MARCO) takes an input query from NaturalQuestions-Open and returns the $k$ documents from Wikipedia with the highest relevance score. To condition language models on these retrieved documents, we simply include them in the prompt. We evaluate retriever recall and reader accuracy (whether any of the annotated answers appear in the predicted output) as a function of the number of retrieved documents $k$ . We use a subset of NaturalQuestions-Open where the long answer is a paragraph (as opposed to a table or a list).

Figure 11 presents retriever recall and opendomain QA results. We see that reader model performance saturates long before retriever performance saturates, indicating that readers are not effectively using the extra context. Using more than 20 retrieved documents only marginally improves reader performance $( \\sim 1 . 5 %$ for GPT-3.5- Turbo and ${ \\sim } 1 %$ for Claude-1.3), while significantly increasing the input context length (and thus latency and cost). These results, coupled with the observation that models are often better at retrieving and using information at the start or end of the input contexts, suggest that effective reranking of retrieved documents (pushing relevant information closer to the start of the input context) or ranked list truncation (retrieving fewer documents when appropriate; Arampatzis et al., 2009) may be promising directions for improving how languagemodel-based readers use retrieved context.

https://cs.stanford.edu/~nfliu/papers/images/20d781dfb6992d83e5fb98aa58f72cc137909caba31bddb68ee7d19ede5176cb.jpg

Figure 11: Retriever recall and model performance as a function of the number of retrieved documents. Model performance saturates long before retriever recall, indicating that the models have difficulty making use of the extra retrieved documents.

# 6 Related Work

# 6.1 Long-Context Language Models

There is much prior work in designing performant language models with cheaper scaling than Transformers in the context length. Many lines of work pursue Transformer variants with attention modifications like recurrence (Dai et al., 2019), factorizing attention into computationally less intensive approximations (Beltagy et al., 2020; Zaheer et al., 2020), or low-rank approximations (Wang et al., 2020; Peng et al., 2021). Dao et al. (2022) instead provide a faster exact attention by a carefullycrafted IO-aware CUDA kernel. Separately, there are attempts to do away with attention entirely to remove quadratic sequence length complexity, often through convolution and/or linear RNNs, e.g., in RWKV (Peng, 2023), S4 (Gu et al., 2022), or Hyena (Poli et al., 2023). Many prior efforts evaluate perplexity on a diverse web corpus as a proxy for the ability to process long contexts; this work shows that precise knowledge access on long contexts may be an added challenge.

# 6.2 How Do Language Models Use Context?

The pioneering work of Khandelwal et al. (2018) showed that small LSTM language models make increasingly coarse use of longer-term context; Sankar et al. (2019) found similar results in dialogue models. In a similar vein, Daniluk et al. (2017) find that attentive LSTM language models tend to mainly use recent history. Petroni et al. (2020) were among the first to demonstrate the potential of combining context from an information retrieval system with a pretrained language models for unsupervised question answering. O’Connor and Andreas (2021) found that many information-destroying operations had marginal effects on Transformer LMs’ predictions. Krishna et al. (2022) found that long-context neural generation in modestly-sized Transformer language models degenerates because models fail to properly condition on long context. Finally, studying long-context models, Sun et al. (2021) found that longer contexts improves prediction of only a few tokens, an empirical finding consistent with the theory of Sharan et al. (2018), who showed that sequence distributions with bounded mutual information necessarily lead to marginal average prediction benefits from increasingly long context. Qin et al. (2023) analyze how efficient Transformers perform on a variety of long-context downstream NLP tasks, finding that long-context transformers are recency-biased and do not effectively use longrange context.

# 6.3 The Serial-Position Effect

The U-shaped curve we observe in this work has a connection in psychology known as the serialposition effect (Ebbinghaus, 1913; Murdock Jr, 1962), that states that in free-association recall of elements from a list, humans tend to best remember the first and last elements of the list. The serial-position effect plays a role in understanding how humans develop short- and long-term memory. Observing a serial-position-like effect in language models is perhaps surprising, since the selfattention mechanisms underlying Transformer language models is technically equally capable of retrieving any token from their contexts.

# 7 Conclusion

We empirically study how language models use long input contexts via a series of controlled experiments. We show that language model performance degrades significantly when changing the position of relevant information, indicating that models struggle to robustly access and use information in long input contexts. In particular, performance is often lowest when models must use information in the middle of long input contexts. We conduct a preliminary investigation of the role of (i) model architecture, (ii) query-aware contextualization, and (iii) instruction fine-tuning to better understand how they affect how language models use context. Finally, we conclude with a practical case study of open-domain question answering, finding that the performance of language model readers saturates far before retriever recall. Our results and analysis provide a better understanding of how language models use their input context and provides new evaluation protocols for future long-context models.

</details>

<details>
<summary>⚠️ Error scraping https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517582: Failed to parse Firecrawl error response as JSON. Status code: 502</summary>

⚠️ Error scraping https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517582: Failed to parse Firecrawl error response as JSON. Status code: 502

</details>

<details>
<summary>Set up Gemini in Colab Enterprise for a project</summary>

# Set up Gemini in Colab Enterprise for a project

Before you can use [Gemini in Colab Enterprise](https://cloud.google.com/gemini/docs/colab/overview),
your team needs to
[activate Gemini in Colab Enterprise in a Google Cloud project](https://cloud.google.com/gemini/docs/colab/set-up-gemini#activate-gemini),
as described in this document. An administrator typically performs this step.

## Activate Gemini in Colab Enterprise in a Google Cloud project

To activate Gemini in Colab Enterprise in a Google Cloud project, an
administrator needs to enable the Gemini for Google Cloud API and grant the
Gemini for Google Cloud User role to users.

1. To enable the Gemini for Google Cloud API, in the Google Cloud console,
go to the **Gemini for Google Cloud** page.

[Go to Gemini for Google Cloud](https://console.cloud.google.com/marketplace/product/google/cloudaicompanion.googleapis.com)

2. In the project selector, select a project.

3. Click **Enable**.

The page updates and shows a status of **Enabled**.
Gemini is now available in the selected
Google Cloud project to all users who have the required
IAM roles.

4. To grant the IAM roles that are required to use
Gemini, go to the **IAM & Admin** page.

[Go to IAM & Admin](https://console.cloud.google.com/projectselector/iam-admin/iam?supportedpurview=)

5. In the **Principal** column, find a
[principal](https://cloud.google.com/iam/docs/overview#concepts_related_identity) for which you want
to give access to Gemini in Colab Enterprise, and then click
edit **Edit principal** in that row.

6. In the **Edit access** pane, click
add **Add another role**.

7. In **Select a role**, select **Gemini for Google Cloud User**.

8. Click **Save**.


All of the users who have been granted these roles can access Gemini for Google Cloud
features in the Google Cloud console within the specified project. For more
information, see
[Use Gemini for AI assistance and development](https://cloud.google.com/gemini/docs/overview).

## Advanced setup tasks

Instead of using the Google Cloud console or the gcloud CLI to
grant predefined IAM roles, you can do any of the following:

- Use [IAM REST APIs](https://cloud.google.com/iam/docs/reference/rest)
or [IAM client libraries](https://cloud.google.com/iam/docs/reference/libraries)
to grant roles.

If you use these interfaces, use the fully qualified role name,
`roles/cloudaicompanion.user`.

For more information about granting roles, see
[Manage access to projects, folders, and organizations](https://cloud.google.com/iam/docs/granting-changing-revoking-access).

- Create and grant custom roles.

Any [custom roles](https://cloud.google.com/iam/docs/creating-custom-roles) that you create need the
following permissions to access Gemini:

  - `cloudaicompanion.instances.generateCode`
  - `cloudaicompanion.instances.completeCode`
  - `cloudaicompanion.entitlements.get`

## Billing

Gemini in Colab Enterprise is currently available to try at no additional cost. See
[Colab Enterprise pricing](https://cloud.google.com/colab/pricing).

## What's next

- Learn how to
[write code with Gemini assistance](https://cloud.google.com/colab/docs/use-code-completion).
- Learn more about the
[types of generative AI assistance available in Gemini for Google Cloud](https://cloud.google.com/gemini/docs/overview).
- Learn
[how Gemini for Google Cloud uses your data](https://cloud.google.com/gemini/docs/discover/data-governance).
- Learn
[how to access and manage Gemini admin controls](https://cloud.google.com/gemini/docs/admin).

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

#   Classified industry as: I apologize, but i cannot and should not provide anal

[... Content truncated due to length ...]

</details>

<details>
<summary>Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/03_chaining_routing/notebook.ipynb</summary>

# Repository analysis for https://github.com/towardsai/course-ai-agents/blob/main/lessons/03_chaining_routing/notebook.ipynb

## Summary
Repository: towardsai/course-ai-agents
File: notebook.ipynb
Lines: 1,446

Estimated tokens: 10.2k

## File tree
```Directory structure:
└── notebook.ipynb

```

## Extracted content
================================================
FILE: lessons/03_chaining_routing/notebook.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Lesson 3: Basic Workflow Ingredients: Chaining, Routing, Parallelization, and Orchestration

This Jupyter notebook demonstrates AI agent workflow patterns using Google Gemini, focusing on chaining, routing, and parallelization strategies.

The sections are:

1. Single LLM Call Problems - Demonstrates issues with complex prompts that try to do everything at once.

2. Sequential Workflows - Breaking tasks into steps (generate questions → answer questions → find sources) for better consistency.

3. Parallel Workflows - Running tasks in parallel (answering questions in parallel) for higher speed.

4. Routing Workflows - Classifying user intent and routing to specialized handlers (technical support, billing, general questions).

5. Orchestrator-Worker Pattern - A system where an orchestrator breaks complex queries into subtasks, specialized workers handle each task, and a synthesizer combines results into a cohesive response

The examples include FAQ generation from multiple sources and a customer service system handling billing inquiries, product returns, and order status updates simultaneously.
"""

"""
## Setup: Installing Required Dependencies

First, we need to install the Google Generative AI library to interact with Gemini models:
"""

%pip install -q google-genai

"""
## Configuring the Gemini Client

Next, we import the necessary libraries and configure the Gemini client. The API key is retrieved from Google Colab's userdata.

"""

import os
import json
import asyncio
import random
from pydantic import BaseModel, RootModel, Field
from typing import List, Optional
import time
from enum import Enum
from google import genai


# Initialize the Gemini client
# The client uses the GOOGLE_API_KEY from the environment
try:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
except ImportError:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found. Set it in your environment or Colab userdata.")

# Create Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)
print("Gemini client initialized successfully.")
# Output:
#   Gemini client initialized successfully.


"""
# The problem with a single, large LLM call
"""

"""
## Setting Up Mock Data

We'll create three mock webpages about renewable energy topics that will serve as our source content for the FAQ generation examples. Each webpage has a title and detailed content about solar energy, wind turbines, and energy storage:

"""

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
## Example: Complex Single LLM Call

This example demonstrates the problem with trying to do everything in one complex prompt. We're asking the LLM to generate questions, find answers, and cite sources all in a single call, which can lead to inconsistent results:

"""

# This prompt tries to do everything at once: generate questions, find answers,
# and cite sources. This complexity can often confuse the model.
n_questions = 10
prompt_complex = f"""
Based on the provided content from three webpages, generate a list of exactly {n_questions} frequently asked questions (FAQs).
For each question, provide a concise answer derived ONLY from the text.
After each answer, you MUST include a list of the 'Source Title's that were used to formulate that answer.

Your final output should be a JSON array where each object has three keys: "question", "answer", and "sources" (which is an array of strings).

<provided_content>
{combined_content}
</provided_content>
""".strip()

# Pydantic classes for structured outputs
class FAQ(BaseModel):
    question: str
    answer: str
    sources: List[str]

class FAQList(BaseModel):
    faqs: List[FAQ]

# Generate FAQs
response_complex = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_complex,
    config={
        "response_mime_type": "application/json",
        "response_schema": FAQList
    },
)
result_complex = response_complex.parsed

print("Complex prompt result (might be inconsistent):")
print(result_complex.model_dump_json(indent=2))
# Output:
#   Complex prompt result (might be inconsistent):

#   {

#     "faqs": [

#       {

#         "question": "How does solar energy work?",

#         "answer": "Solar energy converts sunlight into electricity through photovoltaic (PV) panels.",

#         "sources": [

#           "The Benefits of Solar Energy"

#         ]

#       },

#       {

#         "question": "What are the environmental advantages of using solar energy?",

#         "answer": "Solar energy reduces reliance on fossil fuels and cuts down greenhouse gas emissions.",

#         "sources": [

#           "The Benefits of Solar Energy"

#         ]

#       },

#       {

#         "question": "What economic benefits can homeowners gain from installing solar panels?",

#         "answer": "Homeowners can significantly lower their monthly electricity bills and potentially sell excess power back to the grid.",

#         "sources": [

#           "The Benefits of Solar Energy"

#         ]

#       },

#       {

#         "question": "What is the primary function of wind turbines?",

#         "answer": "Wind turbines capture kinetic energy from the wind and convert it into electrical power.",

#         "sources": [

#           "Understanding Wind Turbines"

#         ]

#       },

#       {

#         "question": "What is the main challenge associated with wind energy?",

#         "answer": "The main challenge for wind energy is its intermittency, as it only generates power when the wind blows.",

#         "sources": [

#           "Understanding Wind Turbines"

#         ]

#       },

#       {

#         "question": "Where can wind turbines be installed?",

#         "answer": "Wind turbines can be installed both onshore and offshore.",

#         "sources": [

#           "Understanding Wind Turbines"

#         ]

#       },

#       {

#         "question": "Why is energy storage essential for renewable sources like solar and wind?",

#         "answer": "Energy storage is crucial for a stable power grid because these renewable sources are intermittent, requiring storage of excess energy and release when needed.",

#         "sources": [

#           "Energy Storage Solutions"

#         ]

#       },

#       {

#         "question": "What are some common large-scale energy storage technologies?",

#         "answer": "Common forms of large-scale storage include pumped-hydro storage and battery technologies, especially lithium-ion.",

#         "sources": [

#           "Energy Storage Solutions"

#         ]

#       },

#       {

#         "question": "How do offshore wind farms differ in power production from onshore ones?",

#         "answer": "Offshore wind farms generally produce more consistent power due to stronger, more reliable winds compared to onshore installations.",

#         "sources": [

#           "Understanding Wind Turbines"

#         ]

#       },

#       {

#         "question": "What is the role of energy storage solutions in the overall energy system?",

#         "answer": "Energy storage solutions balance energy supply and demand, making the energy system more resilient and reliable.",

#         "sources": [

#           "Energy Storage Solutions"

#         ]

#       }

#     ]

#   }


"""
# Building a sequential workflow

Now, let's split the complex prompt above into a chain of simpler prompts.
"""

"""
## Question Generation Function

Let's create a function to generate questions from the content. This step focuses solely on creating relevant questions based on the provided material:

"""

class QuestionList(BaseModel):
    questions: List[str]

prompt_generate_questions = """
Based on the content below, generate a list of {n_questions} relevant and distinct questions that a user might have.
Return these questions as a JSON array of strings.

<provided_content>
{combined_content}
</provided_content>
""".strip()

def generate_questions(content, n_questions=10):
    """
    Generate a list of questions based on the provided content.

    Args:
        content: The combined content from all sources

    Returns:
        list: A list of generated questions
    """
    response_questions = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_generate_questions.format(n_questions=n_questions, combined_content=content),
        config={
            "response_mime_type": "application/json",
            "response_schema": QuestionList
        }
    )

    return response_questions.parsed.questions

# Test the question generation function
questions = generate_questions(combined_content, n_questions=10)
print(f"Successfully generated {len(questions)} questions.")
print(f"\nFirst few questions:")
for i, q in enumerate(questions[:3]):
    print(f"{i+1}. {q}")
# Output:
#   Successfully generated 10 questions.

#   

#   First few questions:

#   1. What are the environmental and economic advantages of solar energy?

#   2. How do photovoltaic (PV) panels generate electricity from sunlight?

#   3. What are the financial considerations for homeowners interested in solar panels, including initial costs and potential savings?


"""
## Answer Generation Function

Next, we create a function to generate answers for individual questions using only the provided content:
"""

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

def answer_question(question, content):
    """
    Generate an answer for a specific question using only the provided content.

    Args:
        question: The question to answer
        content: The combined content from all sources

    Returns:
        str: The generated answer
    """
    answer_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_answer_question.format(question=question, combined_content=content),
    )
    return answer_response.text

# Test the answer generation function
test_question = questions[0]
test_answer = answer_question(test_question, combined_content)
print(f"Question: {test_question}")
print(f"Answer: {test_answer}")
# Output:
#   Question: What are the environmental and economic advantages of solar energy?

#   Answer: Environmental advantages of solar energy include reduced reliance on fossil fuels and a decrease in greenhouse gas emissions. Economic advantages include significantly lower monthly electricity bills for homeowners, the ability to sell excess power back to the grid, long-term savings, and contributing to energy independence for nations.


"""
## Source Finding Function

Finally, we create a function to identify which sources were used to generate an answer:

"""

class SourceList(BaseModel):
    sources: List[str]

prompt_find_sources = """
You will be given a question and an answer that was generated from a set of documents.
Your task is to identify which of the original documents were used to create the answer.
Return a JSON object with a single key "sources" which is a list of the titles of the relevant documents.

Question: "{question}"
Answer: "{answer}"

<provided_content>
{combined_content}
</provided_content>
""".strip()

def find_sources(question, answer, content):
    """
    Identify which sources were used to generate an answer.

    Args:
        question: The original question
        answer: The generated answer
        content: The combined content from all sources

    Returns:
        list: A list of source titles that were used
    """
    sources_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_find_sources.format(question=question, answer=answer, combined_content=content),
        config={
            "response_mime_type": "application/json",
            "response_schema": SourceList
        }
    )
    return sources_response.parsed.sources

# Test the source finding function
test_sources = find_sources(test_question, test_answer, combined_content)
print(f"Question: {test_question}")
print(f"Answer: {test_answer}")
print(f"Sources: {test_sources}")
# Output:
#   Question: What are the environmental and economic advantages of solar energy?

#   Answer: Environmental advantages of solar energy include reduced reliance on fossil fuels and a decrease in greenhouse gas emissions. Economic advantages include significantly lower monthly electricity bills for homeowners, the ability to sell excess power back to the grid, long-term savings, and contributing to energy independence for nations.

#   Sources: ['The Benefits of Solar Energy']


"""
## Executing the Sequential Workflow

Now we combine all three functions into a sequential workflow: Generate Questions → Answer Questions → Find Sources. Each step is executed one after another for each question. Notice how much time it takes to run the full workflow.

"""

def sequential_workflow(content, n_questions=10):
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
        print(f"  - Processing: '{question[:50]}...'")

        # Generate an answer for the current question
        answer = answer_question(question, content)

        # Identify the sources for the generated answer
        sources = find_sources(question, answer, content)

        final_faqs.append({"question": question, "answer": answer, "sources": sources})

    return final_faqs

# Execute the sequential workflow (measure time for comparison)
start_time = time.time()
sequential_faqs = sequential_workflow(combined_content, n_questions=4)
end_time = time.time()
print(f"\nSequential processing completed in {end_time - start_time:.2f} seconds")

# Display the final result
print("\nGenerated FAQ List (Sequential):")
print(json.dumps(sequential_faqs, indent=2))
# Output:
#     - Processing: 'What are the primary economic and environmental be...'

#     - Processing: 'What are the main challenges associated with wind ...'

#     - Processing: 'Why is effective energy storage crucial for integr...'

#     - Processing: 'What are the different types of large-scale energy...'

#   

#   Sequential processing completed in 20.49 seconds

#   

#   Generated FAQ List (Sequential):

#   [

#     {

#       "question": "What are the primary economic and environmental benefits of adopting solar energy?",

#       "answer": "The primary economic benefits of adopting solar energy include significantly lower monthly electricity bills for homeowners, the potential to sell excess power back to the grid, and contributing to energy independence for nations. The primary environmental benefit is the reduction of greenhouse gas emissions by decreasing reliance on fossil fuels.",

#       "sources": [

#         "The Benefits of Solar Energy"

#       ]

#     },

#     {

#       "question": "What are the main challenges associated with wind power, and how can these be addressed?",

#       "answer": "The main challenge associated with wind power is its intermittency, meaning it only generates power when the wind blows. This can be addressed through the use of energy storage solutions, such as large-scale batteries, pumped-hydro storage, and lithium-ion batteries, to ensure a steady supply of electricity.",

#       "sources": [

#         "Understanding Wind Turbines",

#         "Energy Storage Solutions"

#       ]

#     },

#     {

#       "question": "Why is effective energy storage crucial for integrating renewable sources like solar and wind into the power grid?",

#       "answer": "Effective energy storage is crucial because renewable sources like solar and wind are intermittent, only generating power when conditions allow. Storage allows for storing excess energy when plentiful and releasing it when needed to ensure a steady supply of electricity and a stable, resilient, and reliable power grid.",

#       "sources": [

#         "Understanding Wind Turbines",

#         "Energy Storage Solutions"

#       ]

#     },

#     {

#       "question": "What are the different types of large-scale energy storage solutions commonly used today?",

#       "answer": "The different types of large-scale energy storage solutions commonly used today are pumped-hydro storage and battery technologies, particularly lithium-ion.",

#       "sources": [

#         "Energy Storage Solutions"

#       ]

#     }

#   ]


"""
# Parallelization: Improving Efficiency

While the sequential workflow works well, we can optimize it by running some steps in parallel. We can generate the answer and find sources simultaneously for all the questions. This can significantly reduce the overall processing time.

**Important**: you may meet the rate limits of your account if you do this for a lot of questions. If you go over your rate limits, the API calls will return errors and retry after a timeout. Make sure to take this into account when building real-world products!

## Implementing Parallel Processing

Let's implement a parallel version of our workflow using Python's `asyncio` library.

"""

async def answer_question_async(question, content):
    """
    Async version of answer_question function.
    """
    prompt = prompt_answer_question.format(question=question, combined_content=content)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

async def find_sources_async(question, answer, content):
    """
    Async version of find_sources function.
    """
    prompt = prompt_find_sources.format(question=question, answer=answer, combined_content=content)
    response = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": SourceList
        }
    )
    return response.parsed.sources

async def process_question_parallel(question, content):
    """
    Process a single question by generating answer and finding sources in parallel.
    """
    answer = await answer_question_async(question, content)
    sources = await find_sources_async(question, answer, content)
    return {"question": question, "answer": answer, "sources": sources}

"""
## Executing the Parallel Workflow

Now let's process all questions using parallel execution. We'll process multiple questions concurrently, which can significantly reduce the total processing time. Notice how much time it takes to run the full workflow and compare it with the execution time of the sequential workflow.

"""

async def parallel_workflow(content, n_questions=10):
    """
    Execute the complete parallel workflow for FAQ generation.

    Args:
        content: The combined content from all sources

    Returns:
        list: A list of FAQs with questions, answers, and sources
    """
    # Generate questions (this step remains sequential)
    questions = generate_questions(content, n_questions)

    # Process all questions in parallel
    tasks = [process_question_parallel(question, content) for question in questions]

    # Execute all tasks concurrently
    parallel_faqs = await asyncio.gather(*tasks)

    return parallel_faqs

# Execute the parallel workflow (measure time for comparison)
start_time = time.time()
parallel_faqs = await parallel_workflow(combined_content, n_questions=4)
end_time = time.time()
print(f"\nParallel processing completed in {end_time - start_time:.2f} seconds")

# Display the final result
print("\nGenerated FAQ List (Parallel):")
print(json.dumps(parallel_faqs, indent=2))
# Output:
#   

#   Parallel processing completed in 9.15 seconds

#   

#   Generated FAQ List (Parallel):

#   [

#     {

#       "question": "What are the primary environmental and economic benefits of utilizing solar energy?",

#       "answer": "The primary environmental benefit of utilizing solar energy is cutting down greenhouse gas emissions by reducing reliance on fossil fuels. The economic benefits include significantly lowering monthly electricity bills, the potential to sell excess power back to the grid, and contributing to national energy independence.",

#       "sources": [

#         "The Benefits of Solar Energy"

#       ]

#     },

#     {

#       "question": "What is the main challenge associated with wind energy, and how is this issue typically addressed?",

#       "answer": "The main challenge for wind energy is its intermittency, meaning it only generates power when the wind blows. This issue is typically addressed through the use of energy storage solutions, such as large-scale batteries.",

#       "sources": [

#         "Understanding Wind Turbines",

#         "Energy Storage Solutions"

#       ]

#     },

#     {

#       "question": "Why is effective energy storage crucial for integrating intermittent renewable sources like solar and wind into the power grid?",

#       "answer": "Effective energy storage is crucial because intermittent renewable sources like solar and wind only generate power when conditions are favorable, so storing excess energy when it's plentiful and releasing it when needed ensures a steady supply of electricity and a stable power grid by balancing supply and demand.",

#       "sources": [

#         "Understanding Wind Turbines",

#         "Energy Storage Solutions"

#       ]

#     },

#     {

#       "question": "What are the most common and emerging technologies used for large-scale energy storage in conjunction with renewable energy?",

#       "answer": "The most common form of large-scale energy storage is pumped-hydro storage. Battery technologies, particularly lithium-ion, are rapidly becoming more affordable and widespread.",

#       "sources": [

#         "Energy Storage Solutions"

#       ]

#     }

#   ]


"""
## Sequential vs Parallel: Key Differences

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
# Building a routing workflow

Routing is a method that categorizes an input and then sends it to a specific task designed to handle that type of input. This approach helps keep different functions separate and lets you create more specialized prompts. If you don't use routing, trying to optimize for one kind of input might negatively affect how well the system performs with other kinds of inputs.
"""

"""
## Intent Classification

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
    intent: IntentEnum

prompt_classification = """
Classify the user's query into one of the following categories:
{categories}

Return only the category name and nothing else.

User Query: "{user_query}"
""".strip()


def classify_intent(user_query):
    """Uses an LLM to classify a user query."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_classification.format(
            user_query=user_query,
            categories=[intent.value for intent in IntentEnum]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": UserIntent
        }
    )
    return response.parsed.intent


query_1 = "My internet connection is not working."
query_2 = "I think there is a mistake on my last invoice."
query_3 = "What are your opening hours?"

intent_1 = classify_intent(query_1)
print(f"Query: {query_1}\nIntent: {intent_1}\n")
intent_2 = classify_intent(query_2)
print(f"Query: {query_2}\nIntent: {intent_2}\n")
intent_3 = classify_intent(query_3)
print(f"Query: {query_3}\nIntent: {intent_3}\n")
# Output:
#   Query: My internet connection is not working.

#   Intent: IntentEnum.TECHNICAL_SUPPORT

#   

#   Query: I think there is a mistake on my last invoice.

#   Intent: IntentEnum.BILLING_INQUIRY

#   

#   Query: What are your opening hours?

#   Intent: IntentEnum.GENERAL_QUESTION

#   


"""
## Defining Specialized Handlers

Next, we create specialized prompts for each type of query and a routing function that directs queries to the appropriate handler based on the classified intent:
"""

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
    if intent == IntentEnum.TECHNICAL_SUPPORT:
        prompt = prompt_technical_support.format(user_query=user_query)
    elif intent == IntentEnum.BILLING_INQUIRY:
        prompt = prompt_billing_inquiry.format(user_query=user_query)
    elif intent == IntentEnum.GENERAL_QUESTION:
        prompt = prompt_general_question.format(user_query=user_query)
    else:
        prompt = prompt_general_question.format(user_query=user_query)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


response_1 = handle_query(query_1, intent_1)
print(f"Query: {query_1}\nIntent: {intent_1}\nResponse: {response_1}\n")

response_2 = handle_query(query_2, intent_2)
print(f"Query: {query_2}\nIntent: {intent_2}\nResponse: {response_2}\n")

response_3 = handle_query(query_3, intent_3)
print(f"Query: {query_3}\nIntent: {intent_3}\nResponse: {response_3}\n")
# Output:
#   Query: My internet connection is not working.

#   Intent: IntentEnum.TECHNICAL_SUPPORT

#   Response: I'm sorry to hear your internet connection isn't working! That's definitely frustrating.

#   

#   To help me understand what might be going on and avoid suggesting steps you've already tried, could you tell me a bit more about it?

#   

#   For example:

#   *   **What troubleshooting steps have you already attempted** (like restarting your modem/router, checking cables, etc.)?

#   *   Are you seeing any specific error messages, or are there any unusual lights on your modem or router?

#   *   Is it affecting all your devices (computer, phone, tablet), or just one?

#   

#   Query: I think there is a mistake on my last invoice.

#   Intent: IntentEnum.BILLING_INQUIRY

#   Response: "I understand you're concerned about a mistake on your last invoice. I'd be happy to look into that for you. To access your billing details, could you please provide your account number?"

#   

#   Query: What are your opening hours?

#   Intent: IntentEnum.GENERAL_QUESTION

#   Response: I'm sorry, I'm not sure how to help with that. Could you please rephrase your question?

#   


"""
# Orchestrator-worker pattern

The orchestrator-workers workflow uses a main LLM to dynamically break down complex tasks into smaller subtasks, which are then assigned to other "worker" LLMs. The orchestrator LLM also combines the results from these workers.

This approach is ideal for complex problems where the specific steps or subtasks can't be known in advance. For instance, in a coding project, the orchestrator can decide which files need modifying and how, based on the initial request. While it might look similar to parallel processing, its key advantage is flexibility: instead of pre-defined subtasks, the orchestrator LLM determines them on the fly according to the given input.
"""

"""
## Defining the Orchestrator

The orchestrator is the central coordinator that breaks down complex user queries into structured JSON tasks. It analyzes the input and identifies what types of actions need to be taken, such as billing inquiries, product returns, or status updates:

"""

# Orchestrator
class QueryTypeEnum(str, Enum):
    BILLING_INQUIRY = "BillingInquiry"
    PRODUCT_RETURN = "ProductReturn"
    STATUS_UPDATE = "StatusUpdate"

class Task(BaseModel):
    query_type: str
    invoice_number: Optional[str] = None
    product_name: Optional[str] = None
    reason_for_return: Optional[str] = None
    order_id: Optional[str] = None

class TaskList(BaseModel):
    tasks: List[Task]

prompt_orchestrator = f"""
You are a master orchestrator. Your job is to break down a complex user query into a JSON array of objects.
Each object represents one sub-task and must have a "query_type" and relevant parameters.

The possible "query_type" values are:
1. "{QueryTypeEnum.BILLING_INQUIRY.value}": Requires "invoice_number".
2. "{QueryTypeEnum.PRODUCT_RETURN.value}": Requires "product_name" and "reason_for_return".
3. "{QueryTypeEnum.STATUS_UPDATE.value}": Requires "order_id".

User Query:
---
{{query}}
---
""".strip()


def orchestrator(query):
    """Breaks down a complex query into a list of tasks."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_orchestrator.format(query=query),
        config={
            "response_mime_type": "application/json",
            "response_schema": TaskList
        }
    )
    return response.parsed.tasks

"""
## Billing Worker Implementation

The billing worker specializes in handling invoice-related inquiries. It extracts the specific concern from the user's query, simulates opening an investigation, and returns structured information about the action taken:
"""

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


def handle_billing_worker(invoice_number, original_user_query):
    """
    Handles a billing inquiry.
    1. Uses an LLM to extract the specific concern about the invoice from the original query.
    2. Simulates opening an investigation.
    3. Returns structured data about the action taken.
    """
    extraction_prompt = prompt_billing_worker_extractor.format(
        invoice_number=invoice_number, original_user_query=original_user_query
    )
    response = client.models.generate_content(model="gemini-2.5-flash", contents=extraction_prompt)
    extracted_concern = response.text

    # Simulate backend action: opening an investigation
    print(f"  [Billing Worker] Action: Investigating invoice {invoice_number} for concern: '{extracted_concern}'")
    investigation_id = f"INV_CASE_{random.randint(1000, 9999)}"
    eta_days = 2

    return {
        "task": "Billing Inquiry",
        "invoice_number": invoice_number,
        "user_concern": extracted_concern,
        "action_taken": f"An investigation (Case ID: {investigation_id}) has been opened regarding your concern.",
        "resolution_eta": f"{eta_days} business days",
    }

"""
## Product Return Worker

The return worker handles product return requests by generating RMA (Return Merchandise Authorization) numbers and providing detailed shipping instructions for customers:
"""

# Worker for Product Return
def handle_return_worker(product_name, reason_for_return):
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
        "shipping_instructions": shipping_instructions,
    }

"""
## Order Status Worker

The status worker retrieves and formats order status information, including shipping details, tracking numbers, and delivery estimates:
"""

# Worker for Status Update
def handle_status_worker(order_id):
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
    print(f"  [Status Worker] Action: Fetched status for order {order_id}: {status_details['status']}")

    return {
        "task": "Status Update",
        "order_id": order_id,
        "current_status": status_details["status"],
        "carrier": status_details["carrier"],
        "tracking_number": status_details["tracking"],
        "expected_delivery": status_details["delivery_estimate"],
    }

"""
## Response Synthesizer

The synthesizer takes the structured results from all workers and combines them into a single, coherent, and customer-friendly response message:
"""

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


def synthesizer(results):
    """Combines structured results from workers into a single user-facing message."""
    bullet_points = []
    for res in results:
        point = f"Regarding your {res['task']}:\n"
        if res["task"] == "Billing Inquiry":
            point += f"  - Invoice Number: {res['invoice_number']}\n"
            point += f'  - Your Stated Concern: "{res["user_concern"]}"\n'
            point += f"  - Our Action: {res['action_taken']}\n"
            point += f"  - Expected Resolution: We will get back to you within {res['resolution_eta']}."
        elif res["task"] == "ProductReturn":
            point += f"  - Product: {res['product_name']}\n"
            point += f'  - Reason for Return: "{res["reason_for_return"]}"\n'
            point += f"  - Return Authorization (RMA): {res['rma_number']}\n"
            point += f"  - Instructions: {res['shipping_instructions']}"
        elif res["task"] == "Status Update":
            point += f"  - Order ID: {res['order_id']}\n"
            point += f"  - Current Status: {res['current_status']}\n"
            if res["carrier"] != "N/A":
                point += f"  - Carrier: {res['carrier']}\n"
            if res["tracking_number"] != "N/A":
                point += f"  - Tracking Number: {res['tracking_number']}\n"
            point += f"  - Delivery Estimate: {res['expected_delivery']}"
        bullet_points.append(point)

    formatted_results = "\n\n".join(bullet_points)
    prompt = prompt_synthesizer.format(formatted_results=formatted_results)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

"""
## Main Orchestrator-Worker Pipeline

This function coordinates the entire orchestrator-worker workflow: it runs the orchestrator to break down the query, dispatches the appropriate workers, and synthesizes the final response:
"""

def process_user_query(user_query):
    """Processes a query using the Orchestrator-Worker-Synthesizer pattern."""

    print(f"User query:\n---\n{user_query}\n---")

    # 1. Run orchestrator
    tasks_list = orchestrator(user_query)
    if not tasks_list:
        print("\nOrchestrator did not return any tasks. Exiting.")
        return

    print("\nDeconstructed tasks from Orchestrator:")
    for task in tasks_list:
        print(task.model_dump_json(indent=2))

    # 2. Run workers
    worker_results = []
    if tasks_list:
        print(f"\nDispatching {len(tasks_list)} workers...")
        for task in tasks_list:
            if task.query_type == QueryTypeEnum.BILLING_INQUIRY:
                worker_results.append(handle_billing_worker(task.invoice_number, user_query))
            elif task.query_type == QueryTypeEnum.PRODUCT_RETURN:
                # Ensure reason_for_return is present, provide a default if not (though orchestrator should capture it)
                worker_results.append(handle_return_worker(task.product_name, task.reason_for_return))
            elif task.query_type == QueryTypeEnum.STATUS_UPDATE:
                worker_results.append(handle_status_worker(task.order_id))
            else:
                print(f"Warning: Unknown query_type '{task.query_type}' found in orchestrator tasks.")

        if worker_results:
            print(f"Ran {len(worker_results)} workers sequentially.")
            print("\nWorkers finished their jobs. Results:")
            for i, res in enumerate(worker_results):
                print(f"--- Worker Result {i + 1} ---")
                print(json.dumps(res, indent=2))
                print("----------------------")
        else:
            print("\nNo valid worker tasks to run.")
    else:
        print("\nNo tasks to run for workers.")

    # 3. Run synthesizer
    if worker_results:
        print("\nSynthesizing final response...")
        final_user_message = synthesizer(worker_results)
        print("\n--- Final Synthesized Response ---")
        print(final_user_message)
        print("---------------------------------")
    else:
        print("\nSkipping synthesis because there were no worker results.")

"""
## Testing the Complete Workflow

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
#   User query:

#   ---

#   Hi, I'm writing to you because I have a question about invoice #INV-7890. It seems higher than I expected.

#   Also, I would like to return the 'SuperWidget 5000' I bought because it's not compatible with my system.

#   Finally, can you give me an update on my order #A-12345?

#   ---

#   

#   Deconstructed tasks from Orchestrator:

#   {

#     "query_type": "BillingInquiry",

#     "invoice_number": "INV-7890",

#     "product_name": null,

#     "reason_for_return": null,

#     "order_id": null

#   }

#   {

#     "query_type": "ProductReturn",

#     "invoice_number": null,

#     "product_name": "SuperWidget 5000",

#     "reason_for_return": "not compatible with my system",

#     "order_id": null

#   }

#   {

#     "query_type": "StatusUpdate",

#     "invoice_number": null,

#     "product_name": null,

#     "reason_for_return": null,

#     "order_id": "A-12345"

#   }

#   

#   Dispatching 3 workers...

#     [Billing Worker] Action: Investigating invoice INV-7890 for concern: 'It seems higher than I expected.'

#     [Return Worker] Action: Generated RMA RMA-91987 for SuperWidget 5000 (Reason: not compatible with my system)

#     [Status Worker] Action: Fetched status for order A-12345: Processing

#   Ran 3 workers sequentially.

#   

#   Workers finished their jobs. Results:

#   --- Worker Result 1 ---

#   {

#     "task": "Billing Inquiry",

#     "invoice_number": "INV-7890",

#     "user_concern": "It seems higher than I expected.",

#     "action_taken": "An investigation (Case ID: INV_CASE_2400) has been opened regarding your concern.",

#     "resolution_eta": "2 business days"

#   }

#   ----------------------

#   --- Worker Result 2 ---

#   {

#     "task": "Product Return",

#     "product_name": "SuperWidget 5000",

#     "reason_for_return": "not compatible with my system",

#     "rma_number": "RMA-91987",

#     "shipping_instructions": "Please pack the 'SuperWidget 5000' securely in its original packaging if possible. Include all accessories and manuals. Write the RMA number (RMA-91987) clearly on the outside of the package. Ship to: Returns Department, 123 Automation Lane, Tech City, TC 98765."

#   }

#   ----------------------

#   --- Worker Result 3 ---

#   {

#     "task": "Status Update",

#     "order_id": "A-12345",

#     "current_status": "Processing",

#     "carrier": "N/A",

#     "tracking_number": "N/A",

#     "expected_delivery": "3-5 business days"

#   }

#   ----------------------

#   

#   Synthesizing final response...

#   

#   --- Final Synthesized Response ---

#   Hi there,

#   

#   Thank you for reaching out to us. We're happy to provide you with an update on your recent inquiries.

#   

#   **Regarding your Billing Inquiry (Invoice INV-7890):**

#   We understand your concern that your invoice seems higher than expected. We've initiated a thorough investigation into this for you (Case ID: INV_CASE_2400). Our team is actively looking into the details, and we anticipate getting back to you with a resolution within **2 business days**.

#   

#   **Regarding your Order Status Update (Order ID: A-12345):**

#   We're pleased to confirm that your order is currently **processing**. You can expect delivery within **3-5 business days**.

#   

#   We hope this update is helpful. If you have any further questions in the meantime, please don't hesitate to reply to this email.

#   

#   Best regards,

#   

#   The [Your Company/Support Team Name] Team

#   ---------------------------------

</details>


## YouTube Video Transcripts

_No YouTube video transcripts found._


## Local Files

_No local files found._


## Additional Sources Scraped

<details>
<summary>building-effective-ai-agents-anthropic</summary>

# Building effective agents

Published Dec 19, 2024

We've worked with dozens of teams building LLM agents across industries. Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks.

Over the past year, we've worked with dozens of teams building large language model (LLM) agents across industries. Consistently, the most successful implementations weren't using complex frameworks or specialized libraries. Instead, they were building with simple, composable patterns.

In this post, we share what we’ve learned from working with our customers and building agents ourselves, and give practical advice for developers on building effective agents.

## What are agents?

"Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as **agentic systems**, but draw an important architectural distinction between **workflows** and **agents**:

- **Workflows** are systems where LLMs and tools are orchestrated through predefined code paths.
- **Agents**, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Below, we will explore both types of agentic systems in detail. In Appendix 1 (“Agents in Practice”), we describe two domains where customers have found particular value in using these kinds of systems.

## When (and when not) to use agents

When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

## When and how to use frameworks

There are many frameworks that make agentic systems easier to implement, including:

- [LangGraph](https://langchain-ai.github.io/langgraph/) from LangChain;
- Amazon Bedrock's [AI Agent framework](https://aws.amazon.com/bedrock/agents/);
- [Rivet](https://rivet.ironcladapp.com/), a drag and drop GUI LLM workflow builder; and
- [Vellum](https://www.vellum.ai/), another GUI tool for building and testing complex workflows.

These frameworks make it easy to get started by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together. However, they often create extra layers of abstraction that can obscure the underlying prompts ​​and responses, making them harder to debug. They can also make it tempting to add complexity when a simpler setup would suffice.

We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code. Incorrect assumptions about what's under the hood are a common source of customer error.

See our [cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/patterns/agents) for some sample implementations.

## Building blocks, workflows, and agents

In this section, we’ll explore the common patterns for agentic systems we’ve seen in production. We'll start with our foundational building block—the augmented LLM—and progressively increase complexity, from simple compositional workflows to autonomous agents.

### Building block: The augmented LLM

The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory. Our current models can actively use these capabilities—generating their own search queries, selecting appropriate tools, and determining what information to retain.

We recommend focusing on two key aspects of the implementation: tailoring these capabilities to your specific use case and ensuring they provide an easy, well-documented interface for your LLM. While there are many ways to implement these augmentations, one approach is through our recently released [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol), which allows developers to integrate with a growing ecosystem of third-party tools with a simple [client implementation](https://modelcontextprotocol.io/tutorials/building-a-client#building-mcp-clients).

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

Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully. We expand on best practices for tool development in Appendix 2 ("Prompt Engineering your Tools").

**When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

The autonomous nature of agents means higher costs, and the potential for compounding errors. We recommend extensive testing in sandboxed environments, along with the appropriate guardrails.

**Examples where agents are useful:**

The following examples are from our own implementations:

- A coding Agent to resolve [SWE-bench tasks](https://www.anthropic.com/research/swe-bench-sonnet), which involve edits to many files based on a task description;
- Our [“computer use” reference implementation](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo), where Claude uses a computer to accomplish tasks.

## Combining and customizing these patterns

These building blocks aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases. The key to success, as with any LLM features, is measuring performance and iterating on implementations. To repeat: you should consider adding complexity _only_ when it demonstrably improves outcomes.

## Summary

Success in the LLM space isn't about building the most sophisticated system. It's about building the _right_ system for your needs. Start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when simpler solutions fall short.

When implementing agents, we try to follow three core principles:

1. Maintain **simplicity** in your agent's design.
2. Prioritize **transparency** by explicitly showing the agent’s planning steps.
3. Carefully craft your agent-computer interface (ACI) through thorough tool **documentation and testing**.

Frameworks can help you get started quickly, but don't hesitate to reduce abstraction layers and build with basic components as you move to production. By following these principles, you can create agents that are not only powerful but also reliable, maintainable, and trusted by their users.

### Acknowledgements

Written by Erik Schluntz and Barry Zhang. This work draws upon our experiences building agents at Anthropic and the valuable insights shared by our customers, for which we're deeply grateful.

## Appendix 1: Agents in practice

Our work with customers has revealed two particularly promising applications for AI agents that demonstrate the practical value of the patterns discussed above. Both applications illustrate how agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight.

### A. Customer support

Customer support combines familiar chatbot interfaces with enhanced capabilities through tool integration. This is a natural fit for more open-ended agents because:

- Support interactions naturally follow a conversation flow while requiring access to external information and actions;
- Tools can be integrated to pull customer data, order history, and knowledge base articles;
- Actions such as issuing refunds or updating tickets can be handled programmatically; and
- Success can be clearly measured through user-defined resolutions.

Several companies have demonstrated the viability of this approach through usage-based pricing models that charge only for successful resolutions, showing confidence in their agents' effectiveness.

### B. Coding agents

The software development space has shown remarkable potential for LLM features, with capabilities evolving from code completion to autonomous problem-solving. Agents are particularly effective because:

- Code solutions are verifiable through automated tests;
- Agents can iterate on solutions using test results as feedback;
- The problem space is well-defined and structured; and
- Output quality can be measured objectively.

In our own implementation, agents can now solve real GitHub issues in the [SWE-bench Verified](https://www.anthropic.com/research/swe-bench-sonnet) benchmark based on the pull request description alone. However, whereas automated testing helps verify functionality, human review remains crucial for ensuring solutions align with broader system requirements.

## Appendix 2: Prompt engineering your tools

No matter which agentic system you're building, tools will likely be an important part of your agent. [Tools](https://www.anthropic.com/news/tool-use-ga) enable Claude to interact with external services and APIs by specifying their exact structure and definition in our API. When Claude responds, it will include a [tool use block](https://docs.anthropic.com/en/docs/build-with-claude/tool-use#example-api-response-with-a-tool-use-content-block) in the API response if it plans to invoke a tool. Tool definitions and specifications should be given just as much prompt engineering attention as your overall prompts. In this brief appendix, we describe how to prompt engineer your tools.

There are often several ways to specify the same action. For instance, you can specify a file edit by writing a diff, or by rewriting the entire file. For structured output, you can return code inside markdown or inside JSON. In software engineering, differences like these are cosmetic and can be converted losslessly from one to the other. However, some formats are much more difficult for an LLM to write than others. Writing a diff requires knowing how many lines are changing in the chunk header before the new code is written. Writing code inside JSON (compared to markdown) requires extra escaping of newlines and quotes.

Our suggestions for deciding on tool formats are the following:

- Give the model enough tokens to "think" before it writes itself into a corner.
- Keep the format close to what the model has seen naturally occurring in text on the internet.
- Make sure there's no formatting "overhead" such as having to keep an accurate count of thousands of lines of code, or string-escaping any code it writes.

One rule of thumb is to think about how much effort goes into human-computer interfaces (HCI), and plan to invest just as much effort in creating good _agent_-computer interfaces (ACI). Here are some thoughts on how to do so:

- Put yourself in the model's shoes. Is it obvious how to use this tool, based on the description and parameters, or would you need to think carefully about it? If so, then it’s probably also true for the model. A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools.
- How can you change parameter names or descriptions to make things more obvious? Think of this as writing a great docstring for a junior developer on your team. This is especially important when using many similar tools.
- Test how the model uses your tools: Run many example inputs in our [workbench](https://console.anthropic.com/workbench) to see what mistakes the model makes, and iterate.
- [Poka-yoke](https://en.wikipedia.org/wiki/Poka-yoke) your tools. Change the arguments so that it is harder to make mistakes.

While building our agent for [SWE-bench](https://www.anthropic.com/research/swe-bench-sonnet), we actually spent more time optimizing our tools than the overall prompt. For example, we found that the model would make mistakes with tools using relative filepaths after the agent had moved out of the root directory. To fix this, we changed the tool to always require absolute filepaths—and we found that the model used this method flawlessly.

</details>

<details>
<summary>chain-complex-prompts-for-stronger-performance-anthropic</summary>

While these tips apply broadly to all Claude models, you can find prompting tips specific to extended thinking models [here](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips).

When working with complex tasks, Claude can sometimes drop the ball if you try to handle everything in a single prompt. Chain of thought (CoT) prompting is great, but what if your task has multiple distinct steps that each require in-depth thought?

Enter prompt chaining: breaking down complex tasks into smaller, manageable subtasks.

##  Why chain prompts?

1. **Accuracy**: Each subtask gets Claude’s full attention, reducing errors.
2. **Clarity**: Simpler subtasks mean clearer instructions and outputs.
3. **Traceability**: Easily pinpoint and fix issues in your prompt chain.

---

##  When to chain prompts

Use prompt chaining for multi-step tasks like research synthesis, document analysis, or iterative content creation. When a task involves multiple transformations, citations, or instructions, chaining prevents Claude from dropping or mishandling steps.

**Remember:** Each link in the chain gets Claude’s full attention!

**Debugging tip**: If Claude misses a step or performs poorly, isolate that step in its own prompt. This lets you fine-tune problematic steps without redoing the entire task.

---

##  How to chain prompts

1. **Identify subtasks**: Break your task into distinct, sequential steps.
2. **Structure with XML for clear handoffs**: Use XML tags to pass outputs between prompts.
3. **Have a single-task goal**: Each subtask should have a single, clear objective.
4. **Iterate**: Refine subtasks based on Claude’s performance.

###  Example chained workflows:

- **Multi-step analysis**: See the legal and business examples below.
- **Content creation pipelines**: Research → Outline → Draft → Edit → Format.
- **Data processing**: Extract → Transform → Analyze → Visualize.
- **Decision-making**: Gather info → List options → Analyze each → Recommend.
- **Verification loops**: Generate content → Review → Refine → Re-review.

**Optimization tip**: For tasks with independent subtasks (like analyzing multiple docs), create separate prompts and run them in parallel for speed.

###  Advanced: Self-correction chains

You can chain prompts to have Claude review its own work! This catches errors and refines outputs, especially for high-stakes tasks.

Example: Self-correcting research summary

####  Prompt 1

| Role | Content |
| --- | --- |
| User | Summarize this medical research paper.<br><paper>{{RESEARCH\_PAPER}}</paper><br>Focus on methodology, findings, and clinical implications. |
| Assistant<br>{{SUMMARY}} | Methodology: Double-blind RCT with 500 Type 2 diabetes patients. Group A (n=250) received novel GLP-1/GIP dual agonist; Group B (n=250) received standard GLP-1 agonist. 24-week trial, primary endpoint: HbA1c reduction.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Safety: Similar adverse event profiles, no serious events attributed to treatment.<br>Clinical Implications: Dual agonist shows superior glycemic control and weight loss compared to standard therapy. Could be a game-changer, especially for obese diabetics. |

####  Prompt 2

| Role | Content |
| --- | --- |
| User | Your task is to provide feedback on a research paper summary. Here is a summary of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Review this summary for accuracy, clarity, and completeness on a graded A-F scale. |
| Assistant<br>{{FEEDBACK}} | 1\. Accuracy: A<br> \- Correctly reported key stats: HbA1c reductions, weight loss.<br> \- Accurately noted safety profile and lack of serious adverse events.<br> \- Assertion of “game-changer” is justified by significantly better outcomes.<br>2\. Clarity: A-<br> \- Concise and well-structured with clear sections.<br> \- Good use of comparative language (e.g., “up 3% from Q1”) for context.<br> \- Minor issue: Acronyms like “RCT,” “GLP-1,” and “GIP” not defined. Consider spelling out first use.<br>3\. Completeness: B+<br> \- Covers all requested areas: methodology, findings, clinical implications.<br> \- Missing: Baseline characteristics of groups (were they well-matched?).<br> \- Missing: Any secondary endpoints (e.g., blood pressure, lipids).<br> \- Consider: Long-term implications or need for further research.<br>Overall, the summary is accurate and clear, providing key insights for a quick grasp of the study. To elevate to A+, define acronyms, mention group baseline stats, note any secondary outcomes, and briefly touch on the need for long-term studies. |

####  Prompt 3

| Role | Content |
| --- | --- |
| User | Your task is to improve a paper summary given feedback. Here is the first draft of a medical research paper:<br><summary><br>{{SUMMARY}}<br></summary><br>Here is the research paper:<br><paper><br>{{RESEARCH\_PAPER}}<br></paper><br>Here is the feedback:<br><feedback><br>{{FEEDBACK}}<br></feedback><br>Update the summary based on the feedback. |
| Assistant | Methodology: Double-blind randomized controlled trial (RCT) with 500 Type 2 diabetes patients. Group A (n=250) received a novel glucagon-like peptide-1/glucose-dependent insulinotropic polypeptide (GLP-1/GIP) dual agonist; Group B (n=250) received a standard GLP-1 agonist. The 24-week trial’s primary endpoint was HbA1c reduction. Groups were well-matched for age, BMI, and baseline HbA1c.<br>Findings:<br>\- Group A: 1.8% mean HbA1c reduction<br>\- Group B: 1.1% mean HbA1c reduction<br>\- 68% of Group A vs. 52% of Group B achieved HbA1c <7%<br>\- Group A: 5.2kg mean weight loss<br>\- Group B: 3.1kg mean weight loss<br>Secondary endpoints showed improvements in blood pressure and lipid profiles for both groups, with Group A showing slightly better results.<br>Safety: Similar adverse event profiles, with no serious events attributed to either treatment.<br>Clinical Implications: The dual GLP-1/GIP agonist demonstrates superior glycemic control and weight loss compared to standard GLP-1 therapy. These results suggest it could be a significant advancement, particularly for obese diabetics. However, longer-term studies are needed to confirm sustained efficacy and safety, as well as to investigate potential benefits on cardiovascular outcomes. |

---

##  Examples

Example: Analyzing a legal contract (without chaining)

Note that Claude misses the instruction to provide proposed changes in its email draft.

| Role | Content |
| --- | --- |
| User | You’re our Chief Legal Officer. Review this SaaS contract for risks, focusing on data privacy, SLAs, and liability caps.<br><contract><br>{{CONTRACT}}<br></contract><br>Then draft an email to the vendor with your concerns and proposed changes. |
| Assistant | After reviewing the SaaS contract, I’ve identified several concerns:<br>1\. Data Privacy (Section 7):<br> \- Issue: Vendor has broad rights to use client data for “service improvement.”<br> \- Risk: Could lead to unauthorized data sharing or monetization.<br>2\. SLAs (Section 5):<br> \- Issue: 99.5% uptime guarantee is below industry standard.<br> \- Risk: Excessive downtime could impact our operations.<br>3\. Liability (Section 9):<br> \- Issue: Cap set at 3 months of fees.<br> \- Risk: Inadequate for potential data breaches or service failures.<br>Draft email:<br>Dear {{VENDOR}},<br>After reviewing the SaaS agreement, we have concerns:<br>1\. Data Privacy: We require that our data only be used for direct service provision, not broad “service improvement.” Please amend Section 7.<br>2\. SLAs: 99.5% uptime is insufficient. We need 99.99% with clearly defined penalties. Please revise Section 5.<br>3\. Liability: The current cap of 3 months’ fees is inadequate. We propose a cap of 12 months’ fees, with no cap for data breaches or willful misconduct.<br>We value our potential partnership and believe these changes will set us up for success. Please send a revised agreement.<br>Best regards,<br>{{CLO\_NAME}} |

Example: Analyzing a legal contract (with chaining)

####  Prompt 1

| Role | Content |
| --- | --- |
| User | You’re our Chief Legal Officer. Review this SaaS contract for risks, focusing on data privacy, SLAs, and liability caps.<br><contract><br>{{CONTRACT}}<br></contract><br>Output your findings in <risks> tags. |
| Assistant<br>{{CONCERNS}} | 1\. Data Privacy (Section 7):<br> \- Issue: Vendor has broad rights to use client data for “service improvement.”<br> \- Risk: Could lead to unauthorized data sharing or monetization of client data.<br> \- Impact: Potential breach of client trust, regulatory fines (GDPR, CCPA).<br>2\. SLAs (Section 5):<br> \- Issue: 99.5% uptime guarantee is below industry standard (typically 99.99%).<br> \- Risk: Up to 43.8 hours of downtime per year vs. 52.6 minutes at 99.99%.<br> \- Impact: Operational disruptions, revenue loss, client dissatisfaction.<br>3\. Liability (Section 9):<br> \- Issue: Liability cap set at 3 months of fees.<br> \- Risk: Inadequate for potential data breaches, IP theft, or prolonged service failures.<br> \- Impact: Insufficient compensation for damages that could far exceed the cap, like class-action lawsuits or regulatory fines. |

####  Prompt 2

| Role | Content |
| --- | --- |
| User | Draft an email to a SaaS product vendor outlining the following concerns and proposing changes. Here are the concerns:<br><concerns><br>{{CONCERNS}}<br></concerns> |
| Assistant<br>{{EMAIL}} | Dear {{VENDOR}},<br>After a thorough review of the proposed SaaS agreement, our legal team has identified several high-priority concerns that must be addressed before we can proceed:<br>1\. Data Privacy (Section 7):<br> \- Current: Broad rights to use client data for “service improvement.”<br> \- Concern: Potential unauthorized data sharing or monetization.<br> \- Proposed Change: Limit data use to direct service provision. Any use for improvement, analytics, or marketing must be opt-in, revocable, and anonymized.<br>2\. SLAs (Section 5):<br> \- Current: 99.5% uptime guarantee.<br> \- Concern: Up to 43.8 hours of downtime per year, risking operational disruptions.<br> \- Proposed Change: 99.99% uptime with tiered service credits. E.g., 10% credit for <99.99%, 25% for <99.9%, 50% for <99.5%.<br>3\. Liability (Section 9):<br> \- Current: Cap at 3 months of fees.<br> \- Concern: Grossly inadequate for data breaches, IP theft, or prolonged outages.<br> \- Proposed Change: 12 months’ fees for general liability. Uncapped for data breaches, willful misconduct, IP infringement.<br>These changes are non-negotiable for us. They align with industry standards and our commitment to data security, operational excellence, and fiduciary responsibility. We value the potential of our partnership, and addressing these will set a strong foundation.<br>Please have your legal team review and incorporate these changes. We’re happy to discuss further to ensure mutual understanding and agreement.<br>Best regards,<br>{{CLO\_NAME}} |

####  Prompt 3

| Role | Content |
| --- | --- |
| User | Your task is to review an email and provide feedback. Here is the email:<br><email><br>{{EMAIL}}<br></email><br>Give feedback on tone, clarity, and professionalism. |
| Assistant | 1\. Tone:<br> \- Professional and assertive without being confrontational.<br> \- Use of “non-negotiable” is strong but justified given the criticality of the issues.<br> \- Collaborative language (“we value,” “happy to discuss”) balances the firmness, maintaining a constructive tone.<br>2\. Clarity:<br> \- Excellent structuring: Current state, concern, and proposed change for each issue.<br> \- Technical terms (e.g., “EBITDA,” “tiered service credits”) are appropriate for a B2B SaaS context.<br> \- Action items are clear: vendor’s legal team to review and incorporate changes.<br>3\. Professionalism:<br> \- Focuses on mutual benefit and industry standards, not just our demands.<br> \- Mentions broader responsibilities (data security, operational excellence, fiduciary duty), elevating the discussion.<br> \- Offers further discussion, showing commitment to understanding and resolving issues.<br>Overall, the email strikes the right balance of assertiveness and collaboration. It clearly articulates our concerns, proposes specific solutions, and maintains a professional, partnership-oriented tone. No revisions needed; proceed with sending. |

Example: Multitenancy strategy review

####  Prompt 1

| Role | Content |
| --- | --- |
| User | As a senior solutions architect, review and analyze this multitenancy strategy for our new enterprise SaaS product.<br><strategy><br>{{STRATEGY}}<br></strategy><br>Focus on scalability, security, and cost-effectiveness. |
| Assistant<br>{{ANALYSIS}} | 1\. Scalability:<br> \+ Database sharding by tenant reduces contention, supports high growth.<br> \+ Microservices allow independent scaling of components.<br> \- Shared app servers could bottleneck under extreme load.<br> \- No mention of auto-scaling or load balancing for app tier.<br>2\. Security:<br> \+ Tenant data isolation via sharding mitigates data leaks.<br> \+ JWT with short expiry and refresh tokens is industry best practice.<br> \- No mention of encryption (at rest, in transit) or key management.<br> \- IAM strategy lacks detail on roles, permissions, or RBAC.<br>3\. Cost-effectiveness:<br> \+ Shared app servers reduce idle resource costs.<br> \+ Serverless functions for infrequent tasks optimize costs.<br> \- Per-tenant DB could increase costs at scale vs. optimized multi-tenant DB.<br> \- No discussion of reserved instances or spot instances for cost saving. |

####  Prompt 2

| Role | Content |
| --- | --- |
| User | Draft a strategy review document for engineering leadership based on this analysis of a multitenancy strategy.<br><strategy><br>{{STRATEGY}}<br></strategy><br><analysis><br>{{ANALYSIS}}<br></analysis><br>Include an executive summary, detailed analysis, and recommendations. |
| Assistant<br>{{STRATEGY\_DOC}} | \# Multitenancy Strategy Review: SaaS Enterprise Product<br>\## Executive Summary<br>The proposed strategy demonstrates a solid foundation in scalability and security principles but has gaps in implementation details. It excels in data isolation and component-level scaling but lacks robust app-tier scaling and advanced security measures. Cost optimizations are present but may falter at high scale. With targeted enhancements, this strategy can effectively support our enterprise SaaS ambitions.<br>\## Detailed Analysis<br>\### 1\. Scalability: B+<br>Strengths:<br>\- Database sharding by tenant is excellent. It minimizes query latency, supports high data growth, and facilitates seamless per-tenant scaling.<br>\- Microservices architecture allows scaling individual components based on demand, crucial for handling diverse enterprise workloads.<br>Weaknesses:<br>\- Shared application servers risk becoming a bottleneck. As tenant count and traffic grow, these could degrade performance across all tenants.<br>\- Absence of auto-scaling and load balancing for the app tier is a critical oversight. These are essential for handling traffic spikes without over-provisioning.<br>\### 2\. Security: B-<br>Strengths:<br>\- Sharded databases provide strong tenant data isolation, mitigating risks of data leaks or unauthorized access between tenants.<br>\- JWT with short expiry and refresh tokens align with OWASP best practices, reducing token vulnerability windows.<br>Weaknesses:<br>\- No mention of data encryption at rest or in transit (TLS 1.3). This is non-negotiable for enterprise data.<br>\- Key management strategy is missing. How are encryption keys, JWT secrets, and DB credentials managed and rotated?<br>\- IAM lacks depth. We need RBAC with well-defined roles, attribute-based access control (ABAC) for fine-grained permissions, and integration with enterprise identity providers (SAML, OIDC).<br>\### 3\. Cost-effectiveness: B<br>Strengths:<br>\- Shared app servers reduce idle capacity costs, good for steady-state traffic.<br>\- Serverless for infrequent tasks (e.g., reporting) is smart; pay only when these resources are used.<br>Weaknesses:<br>\- Per-tenant databases may not be cost-effective at scale compared to a well-designed multi-tenant database with robust data partitioning.<br>\- No mention of cost-saving measures like reserved instances for predictable workloads or spot instances for fault-tolerant tasks.<br>\- Lack of app-tier auto-scaling could lead to over-provisioning, increasing costs.<br>\## Recommendations<br>1\. **App Tier Resilience**: Implement auto-scaling with Kubernetes or cloud-native services (EKS, GKE). Deploy across availability zones with a CDN for static assets.<br>2\. **Security Hardening**:<br> a. Encrypt data at rest (AES-256) and in transit (TLS 1.3).<br> b. Use a managed KMS (AWS KMS, GCP KMS) for key management.<br> c. Enhance IAM with RBAC, ABAC, and SAML/OIDC for enterprise SSO.<br>3\. **Cost Optimization**:<br> a. Evaluate columnar multi-tenant DB (e.g., Snowflake) vs. sharding at scale.<br> b. Use reserved instances for app servers, spot instances for non-critical workloads.<br> c. Implement proactive monitoring and right-sizing to avoid overprovisioning.<br>4\. **Proof of Concept (PoC)**: Before full implementation, conduct a PoC with a high-scale simulation:<br> a. Test sharded DB vs. optimized multi-tenant DB for cost and performance.<br> b. Stress-test app tier with simulated enterprise-level traffic spikes.<br> c. Conduct third-party security audit, focusing on data isolation and access controls.<br>With these enhancements, our multi-tenancy strategy will be enterprise-grade: scalable, secure, and cost-effective. The PoC will validate our design choices and instill confidence in our enterprise clients. |

####  Prompt 3

| Role | Content |
| --- | --- |
| User | Grade this strategy review document for clarity, actionability, and alignment with enterprise priorities. <br><priorities><br>{{PRIORITIES}}<br></priorities><br><strategy\_doc><br>{{STRATEGY\_DOC}}<br></strategy\_doc> |
| Assistant | 1\. Clarity: A<br> \+ Clear structure with Executive Summary, Detailed Analysis, and Recommendations.<br> \+ Analysis uses a grading system (B+, B-) for quick grasp of strengths/weaknesses.<br> \+ Technical terms (RBAC, ABAC, TLS) are explained or contextualized, aiding non-technical leadership.<br> \+ Visual aids (bullet points, indentation) enhance readability.<br>2\. Actionability: A-<br> \+ Recommendations are specific and technical (e.g., “Implement auto-scaling with Kubernetes”).<br> \+ PoC plan with clear objectives provides a roadmap.<br> \+ Missing: Timelines, resource requirements, or team assignments for each recommendation.<br> \+ Consider: Add a “Next Steps” section with immediate actions, roles, and a high-level timeline.<br>3\. Enterprise Alignment: A+<br> \+ Directly addresses enterprise priorities:<br> \- Scalability: Discusses handling enterprise-level traffic spikes.<br> \- Security: Focuses on data isolation, encryption, and enterprise IAM.<br> \- Cost: Strategies align with enterprise FinOps (reserved instances, right-sizing).<br> \+ PoC with “high-scale simulation” and security audit shows commitment to enterprise-grade quality.<br> \+ Language resonates with enterprise concerns: “unacceptable risk,” “enterprise-grade,” “instill confidence.”<br>Overall, the document excels in clarity and enterprise alignment. The actionability is strong but could be elevated to A+ with a “Next Steps” section. The technical depth, coupled with clear explanations, makes this highly effective for both technical and non-technical leadership. Great work! |

</details>

<details>
<summary>claude-4-prompt-engineering-best-practices-anthropic</summary>

This guide provides specific prompt engineering techniques for Claude 4 models (Opus 4 and Sonnet 4) to help you achieve optimal results in your applications. These models have been trained for more precise instruction following than previous generations of Claude models.

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

1. **Tell Claude what to do instead of what not to do**
   - Instead of: “Do not use markdown in your response”
   - Try: “Your response should be composed of smoothly flowing prose paragraphs.”
2. **Use XML format indicators**
   - Try: “Write the prose sections of your response in <smoothly\_flowing\_prose\_paragraphs> tags.”
3. **Match your prompt style to the desired output**

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

- “Include as many relevant features and interactions as possible”
- “Add thoughtful details like hover states, transitions, and micro-interactions”
- “Create an impressive demonstration showcasing web development capabilities”
- “Apply design principles: hierarchy, contrast, balance, and movement”

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

1. **Be specific about desired behavior**: Consider describing exactly what you’d like to see in the output.

2. **Frame your instructions with modifiers**: Adding modifiers that encourage Claude to increase the quality and detail of its output can help better shape Claude’s performance. For example, instead of “Create an analytics dashboard”, use “Create an analytics dashboard. Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation.”

3. **Request specific features explicitly**: Animations and interactive elements should be requested explicitly when desired.

</details>

<details>
<summary>prompt-chaining-prompt-engineering-guide</summary>

# Prompt Chaining

## Introduction to Prompt Chaining

To improve the reliability and performance of LLMs, one of the important prompt engineering techniques is to break tasks into its subtasks. Once those subtasks have been identified, the LLM is prompted with a subtask and then its response is used as input to another prompt. This is what's referred to as prompt chaining, where a task is split into subtasks with the idea to create a chain of prompt operations.

Prompt chaining is useful to accomplish complex tasks which an LLM might struggle to address if prompted with a very detailed prompt. In prompt chaining, chain prompts perform transformations or additional processes on the generated responses before reaching a final desired state.

Besides achieving better performance, prompt chaining helps to boost the transparency of your LLM application, increases controllability, and reliability. This means that you can debug problems with model responses much more easily and analyze and improve performance in the different stages that need improvement.

Prompt chaining is particularly useful when building LLM-powered conversational assistants and improving the personalization and user experience of your applications.

## Use Cases for Prompt Chaining

### Prompt Chaining for Document QA

Prompt chaining can be used in different scenarios that could involve several operations or transformations. For instance, one common use case of LLMs involves answering questions about a large text document. It helps if you design two different prompts where the first prompt is responsible for extracting relevant quotes to answer a question and a second prompt takes as input the quotes and original document to answer a given question. In other words, you will be creating two different prompts to perform the task of answering a question given in a document.

The first prompt below extracts the relevant quotes from the document given the question. Note that for simplicity, we have added a placeholder for the document `{{document}}`. To test the prompt you can copy and paste an article from Wikipedia such as this page for [prompt engineering (opens in a new tab)](https://en.wikipedia.org/wiki/Prompt_engineering). Due to larger context used for this task, we are using the `gpt-4-1106-preview` model from OpenAI. You can use the prompt with other long-context LLMs like Claude.

Prompt 1:

```
You are a helpful assistant. Your task is to help answer a question given in a document. The first step is to extract quotes relevant to the question from the document, delimited by ####. Please output the list of quotes using <quotes></quotes>. Respond with "No relevant quotes found!" if no relevant quotes were found.

####
{{document}}
####
```

Here is a screenshot of the entire prompt including the question which is passed using the `user` role.

https://www.promptingguide.ai/_next/image?url=%2F_next%2Fstatic%2Fmedia%2Fprompt-chaining-1.4bdd376c.png&w=1920&q=75

Output of Prompt 1:

```
<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>
```

The quotes that were returned in the first prompt can now be used as input to the second prompt below. Note that you can clean up the quotes a bit more, i.e., remove the citations. Those citations could be removed or utilized as part of another prompt in the chain but you can ignore this for now. The second prompt then takes the relevant quotes extracted by prompt 1 and prepares a helpful response to the question given in the document and those extracted quotes. The second prompt can be the following:

Prompt 2:

```
Given a set of relevant quotes (delimited by <quotes></quotes>) extracted from a document and the original document (delimited by ####), please compose an answer to the question. Ensure that the answer is accurate, has a friendly tone, and sounds helpful.

####
{{document}}
####

<quotes>
- Chain-of-thought (CoT) prompting[27]
- Generated knowledge prompting[37]
- Least-to-most prompting[38]
- Self-consistency decoding[39]
- Complexity-based prompting[41]
- Self-refine[42]
- Tree-of-thought prompting[43]
- Maieutic prompting[45]
- Directional-stimulus prompting[46]
- Textual inversion and embeddings[59]
- Using gradient descent to search for prompts[61][62][63][64]
- Prompt injection[65][66][67]
</quotes>
```

Output of Prompt 2:

```
The prompting techniques mentioned in the document include:

1. Chain-of-thought (CoT) prompting[27]
2. Generated knowledge prompting[37]
3. Least-to-most prompting[38]
4. Self-consistency decoding[39]
5. Complexity-based prompting[41]
6. Self-refine[42]
7. Tree-of-thought prompting[43]
8. Maieutic prompting[45]
9. Directional-stimulus prompting[46]
10. Textual inversion and embeddings[59]
11. Using gradient descent to search for prompts[61][62][63][64]
12. Prompt injection[65][66][67]

Each of these techniques employs unique strategies to enhance or specify the interactions with large language models to produce the desired outcomes.
```

As you can see, simplifying and creating prompt chains is a useful prompting approach where the responses need to undergo several operations or transformations. As an exercise, feel free to design a prompt that removes the citations (e.g., \[27\]) from the response before sending this as a final response to the user of your application.

You can also find more examples of prompt chaining in this [documentation (opens in a new tab)](https://docs.anthropic.com/claude/docs/prompt-chaining) that leverages the Claude LLM. Our example is inspired and adapted from their examples.

</details>

<details>
<summary>workflows-and-agents</summary>

# Workflows and Agents

This guide reviews common patterns for agentic systems. In describing these systems, it can be useful to make a distinction between "workflows" and "agents". One way to think about this difference is nicely explained [here](https://www.anthropic.com/research/building-effective-agents) by Anthropic:

> Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
> Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Here is a simple way to visualize these differences:

https://langchain-ai.github.io/langgraphjs/concepts/img/agent_workflow.png

When building agents and workflows, LangGraph [offers a number of benefits](https://langchain-ai.github.io/langgraphjs/concepts/high_level/) including persistence, streaming, and support for debugging as well as deployment.

## Set up

Note

The Functional API requires `@langchain/langgraph>=0.2.24`.

You can use [any chat model](https://js.langchain.com/docs/integrations/chat/) that supports structured outputs and tool calling. Below, we show the process of installing the packages, setting API keys, and testing structured outputs / tool calling for Anthropic.

Install dependencies

```md-code__content
yarn add @langchain/langgraph @langchain/anthropic @langchain/core

```

Initialize an LLM

```md-code__content
import { ChatAnthropic } from "@langchain/anthropic";

process.env.ANTHROPIC_API_KEY = "<your_anthropic_key>";

const llm = new ChatAnthropic({
  model: "claude-3-5-sonnet-latest",
});

```

## Building Blocks: The Augmented LLM

LLM have [augmentations](https://www.anthropic.com/research/building-effective-agents) that support building workflows and agents. These include [structured outputs](https://js.langchain.com/docs/concepts/structured_outputs/) and [tool calling](https://js.langchain.com/docs/concepts/tool_calling/), as shown in this image from the Anthropic [blog](https://www.anthropic.com/research/building-effective-agents):

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/augmented_llm.png

```md-code__content
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const searchQuerySchema = z.object({
  searchQuery: z.string().describe("Query that is optimized web search."),
  justification: z.string("Why this query is relevant to the user's request."),
});

// Augment the LLM with schema for structured output
const structuredLlm = llm.withStructuredOutput(searchQuerySchema, {
  name: "searchQuery",
});

// Invoke the augmented LLM
const output = await structuredLlm.invoke(
  "How does Calcium CT score relate to high cholesterol?"
);

const multiply = tool(
  async ({ a, b }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "multiplies two numbers together",
    schema: z.object({
      a: z.number("the first number"),
      b: z.number("the second number"),
    }),
  }
);

// Augment the LLM with tools
const llmWithTools = llm.bindTools([multiply]);

// Invoke the LLM with input that triggers the tool call
const message = await llmWithTools.invoke("What is 2 times 3?");

console.log(message.tool_calls);

```

## Prompt chaining

In prompt chaining, each LLM call processes the output of the previous one.

As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one. You can add programmatic checks (see "gate” in the diagram below) on any intermediate steps to ensure that the process is still on track.
>
> When to use this workflow: This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/prompt_chain.png

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
  improvedJoke: Annotation<string>,
  finalJoke: Annotation<string>,
});

// Define node functions

// First LLM call to generate initial joke
async function generateJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a short joke about ${state.topic}`);
  return { joke: msg.content };
}

// Gate function to check if the joke has a punchline
function checkPunchline(state: typeof StateAnnotation.State) {
  // Simple check - does the joke contain "?" or "!"
  if (state.joke?.includes("?") || state.joke?.includes("!")) {
    return "Pass";
  }
  return "Fail";
}

  // Second LLM call to improve the joke
async function improveJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${state.joke}`
  );
  return { improvedJoke: msg.content };
}

// Third LLM call for final polish
async function polishJoke(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${state.improvedJoke}`
  );
  return { finalJoke: msg.content };
}

// Build workflow
const chain = new StateGraph(StateAnnotation)
  .addNode("generateJoke", generateJoke)
  .addNode("improveJoke", improveJoke)
  .addNode("polishJoke", polishJoke)
  .addEdge("__start__", "generateJoke")
  .addConditionalEdges("generateJoke", checkPunchline, {
    Pass: "improveJoke",
    Fail: "__end__"
  })
  .addEdge("improveJoke", "polishJoke")
  .addEdge("polishJoke", "__end__")
  .compile();

// Invoke
const state = await chain.invoke({ topic: "cats" });
console.log("Initial joke:");
console.log(state.joke);
console.log("\n--- --- ---\n");
if (state.improvedJoke !== undefined) {
  console.log("Improved joke:");
  console.log(state.improvedJoke);
  console.log("\n--- --- ---\n");

  console.log("Final joke:");
  console.log(state.finalJoke);
} else {
  console.log("Joke failed quality gate - no punchline detected!");
}

```

```md-code__content
import { task, entrypoint } from "@langchain/langgraph";

// Tasks

// First LLM call to generate initial joke
const generateJoke = task("generateJoke", async (topic: string) => {
  const msg = await llm.invoke(`Write a short joke about ${topic}`);
  return msg.content;
});

// Gate function to check if the joke has a punchline
function checkPunchline(joke: string) {
  // Simple check - does the joke contain "?" or "!"
  if (joke.includes("?") || joke.includes("!")) {
    return "Pass";
  }
  return "Fail";
}

  // Second LLM call to improve the joke
const improveJoke = task("improveJoke", async (joke: string) => {
  const msg = await llm.invoke(
    `Make this joke funnier by adding wordplay: ${joke}`
  );
  return msg.content;
});

// Third LLM call for final polish
const polishJoke = task("polishJoke", async (joke: string) => {
  const msg = await llm.invoke(
    `Add a surprising twist to this joke: ${joke}`
  );
  return msg.content;
});

const workflow = entrypoint(
  "jokeMaker",
  async (topic: string) => {
    const originalJoke = await generateJoke(topic);
    if (checkPunchline(originalJoke) === "Pass") {
      return originalJoke;
    }
    const improvedJoke = await improveJoke(originalJoke);
    const polishedJoke = await polishJoke(improvedJoke);
    return polishedJoke;
  }
);

const stream = await workflow.stream("cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Parallelization

With parallelization, LLMs work simultaneously on a task:

> LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically. This workflow, parallelization, manifests in two key variations: Sectioning: Breaking a task into independent subtasks run in parallel. Voting: Running the same task multiple times to get diverse outputs.
>
> When to use this workflow: Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results. For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/parallelization.png

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  joke: Annotation<string>,
  story: Annotation<string>,
  poem: Annotation<string>,
  combinedOutput: Annotation<string>,
});

// Nodes
// First LLM call to generate initial joke
async function callLlm1(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a joke about ${state.topic}`);
  return { joke: msg.content };
}

// Second LLM call to generate story
async function callLlm2(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a story about ${state.topic}`);
  return { story: msg.content };
}

// Third LLM call to generate poem
async function callLlm3(state: typeof StateAnnotation.State) {
  const msg = await llm.invoke(`Write a poem about ${state.topic}`);
  return { poem: msg.content };
}

// Combine the joke, story and poem into a single output
async function aggregator(state: typeof StateAnnotation.State) {
  const combined = `Here's a story, joke, and poem about ${state.topic}!\n\n` +
    `STORY:\n${state.story}\n\n` +
    `JOKE:\n${state.joke}\n\n` +
    `POEM:\n${state.poem}`;
  return { combinedOutput: combined };
}

// Build workflow
const parallelWorkflow = new StateGraph(StateAnnotation)
  .addNode("callLlm1", callLlm1)
  .addNode("callLlm2", callLlm2)
  .addNode("callLlm3", callLlm3)
  .addNode("aggregator", aggregator)
  .addEdge("__start__", "callLlm1")
  .addEdge("__start__", "callLlm2")
  .addEdge("__start__", "callLlm3")
  .addEdge("callLlm1", "aggregator")
  .addEdge("callLlm2", "aggregator")
  .addEdge("callLlm3", "aggregator")
  .addEdge("aggregator", "__end__")
  .compile();

// Invoke
const result = await parallelWorkflow.invoke({ topic: "cats" });
console.log(result.combinedOutput);

```

```md-code__content
import { task, entrypoint } from "@langchain/langgraph";

// Tasks

// First LLM call to generate initial joke
const callLlm1 = task("generateJoke", async (topic: string) => {
  const msg = await llm.invoke(`Write a joke about ${topic}`);
  return msg.content;
});

// Second LLM call to generate story
const callLlm2 = task("generateStory", async (topic: string) => {
  const msg = await llm.invoke(`Write a story about ${topic}`);
  return msg.content;
});

// Third LLM call to generate poem
const callLlm3 = task("generatePoem", async (topic: string) => {
  const msg = await llm.invoke(`Write a poem about ${topic}`);
  return msg.content;
});

// Combine outputs
const aggregator = task("aggregator", async (params: {
  topic: string;
  joke: string;
  story: string;
  poem: string;
}) => {
  const { topic, joke, story, poem } = params;
  return `Here's a story, joke, and poem about ${topic}!\n\n` +
    `STORY:\n${story}\n\n` +
    `JOKE:\n${joke}\n\n` +
    `POEM:\n${poem}`;
});

// Build workflow
const workflow = entrypoint(
  "parallelWorkflow",
  async (topic: string) => {
    const [joke, story, poem] = await Promise.all([\
      callLlm1(topic),\
      callLlm2(topic),\
      callLlm3(topic),\
    ]);

    return aggregator({ topic, joke, story, poem });
  }
);

// Invoke
const stream = await workflow.stream("cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Routing

Routing classifies an input and directs it to a followup task. As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> Routing classifies an input and directs it to a specialized followup task. This workflow allows for separation of concerns, and building more specialized prompts. Without this workflow, optimizing for one kind of input can hurt performance on other inputs.
>
> When to use this workflow: Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/routing.png

```md-code__content
import { StateGraph, Annotation } from "@langchain/langgraph";
import { z } from "zod";

// Schema for structured output to use as routing logic
const routeSchema = z.object({
  step: z.enum(["poem", "story", "joke"]).describe(
    "The next step in the routing process"
  ),
});

// Augment the LLM with schema for structured output
const router = llm.withStructuredOutput(routeSchema);

// Graph state
const StateAnnotation = Annotation.Root({
  input: Annotation<string>,
  decision: Annotation<string>,
  output: Annotation<string>,
});

// Nodes
// Write a story
async function llmCall1(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert storyteller.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

// Write a joke
async function llmCall2(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert comedian.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

// Write a poem
async function llmCall3(state: typeof StateAnnotation.State) {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert poet.",\
  }, {\
    role: "user",\
    content: state.input\
  }]);
  return { output: result.content };
}

async function llmCallRouter(state: typeof StateAnnotation.State) {
  // Route the input to the appropriate node
  const decision = await router.invoke([\
    {\
      role: "system",\
      content: "Route the input to story, joke, or poem based on the user's request."\
    },\
    {\
      role: "user",\
      content: state.input\
    },\
  ]);

  return { decision: decision.step };
}

// Conditional edge function to route to the appropriate node
function routeDecision(state: typeof StateAnnotation.State) {
  // Return the node name you want to visit next
  if (state.decision === "story") {
    return "llmCall1";
  } else if (state.decision === "joke") {
    return "llmCall2";
  } else if (state.decision === "poem") {
    return "llmCall3";
  }
}

// Build workflow
const routerWorkflow = new StateGraph(StateAnnotation)
  .addNode("llmCall1", llmCall1)
  .addNode("llmCall2", llmCall2)
  .addNode("llmCall3", llmCall3)
  .addNode("llmCallRouter", llmCallRouter)
  .addEdge("__start__", "llmCallRouter")
  .addConditionalEdges(
    "llmCallRouter",
    routeDecision,
    ["llmCall1", "llmCall2", "llmCall3"],
  )
  .addEdge("llmCall1", "__end__")
  .addEdge("llmCall2", "__end__")
  .addEdge("llmCall3", "__end__")
  .compile();

// Invoke
const state = await routerWorkflow.invoke({
  input: "Write me a joke about cats"
});
console.log(state.output);

```

```md-code__content
import { z } from "zod";
import { task, entrypoint } from "@langchain/langgraph";

// Schema for structured output to use as routing logic
const routeSchema = z.object({
  step: z.enum(["poem", "story", "joke"]).describe(
    "The next step in the routing process"
  ),
});

// Augment the LLM with schema for structured output
const router = llm.withStructuredOutput(routeSchema);

// Tasks
// Write a story
const llmCall1 = task("generateStory", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert storyteller.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Write a joke
const llmCall2 = task("generateJoke", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert comedian.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Write a poem
const llmCall3 = task("generatePoem", async (input: string) => {
  const result = await llm.invoke([{\
    role: "system",\
    content: "You are an expert poet.",\
  }, {\
    role: "user",\
    content: input\
  }]);
  return result.content;
});

// Route the input to the appropriate node
const llmCallRouter = task("router", async (input: string) => {
  const decision = await router.invoke([\
    {\
      role: "system",\
      content: "Route the input to story, joke, or poem based on the user's request."\
    },\
    {\
      role: "user",\
      content: input\
    },\
  ]);
  return decision.step;
});

// Build workflow
const workflow = entrypoint(
  "routerWorkflow",
  async (input: string) => {
    const nextStep = await llmCallRouter(input);

    let llmCall;
    if (nextStep === "story") {
      llmCall = llmCall1;
    } else if (nextStep === "joke") {
      llmCall = llmCall2;
    } else if (nextStep === "poem") {
      llmCall = llmCall3;
    }

    const finalResult = await llmCall(input);
    return finalResult;
  }
);

// Invoke
const stream = await workflow.stream("Write me a joke about cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Orchestrator-Worker

With orchestrator-worker, an orchestrator breaks down a task and delegates each sub-task to workers. As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.
>
> When to use this workflow: This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task). Whereas it’s topographically similar, the key difference from parallelization is its flexibility—subtasks aren't pre-defined, but determined by the orchestrator based on the specific input.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/worker.png

```md-code__content
import { z } from "zod";

// Schema for structured output to use in planning
const sectionSchema = z.object({
  name: z.string().describe("Name for this section of the report."),
  description: z.string().describe(
    "Brief overview of the main topics and concepts to be covered in this section."
  ),
});

const sectionsSchema = z.object({
  sections: z.array(sectionSchema).describe("Sections of the report."),
});

// Augment the LLM with schema for structured output
const planner = llm.withStructuredOutput(sectionsSchema);

```

```md-code__content
import { Annotation, StateGraph, Send } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  topic: Annotation<string>,
  sections: Annotation<Array<z.infer<typeof sectionSchema>>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b),
  }),
  finalReport: Annotation<string>,
});

// Worker state
const WorkerStateAnnotation = Annotation.Root({
  section: Annotation<z.infer<typeof sectionSchema>>,
  completedSections: Annotation<string[]>({
    default: () => [],
    reducer: (a, b) => a.concat(b),
  }),
});

// Nodes
async function orchestrator(state: typeof StateAnnotation.State) {
  // Generate queries
  const reportSections = await planner.invoke([\
    { role: "system", content: "Generate a plan for the report." },\
    { role: "user", content: `Here is the report topic: ${state.topic}` },\
  ]);

  return { sections: reportSections.sections };
}

async function llmCall(state: typeof WorkerStateAnnotation.State) {
  // Generate section
  const section = await llm.invoke([\
    {\
      role: "system",\
      content: "Write a report section following the provided name and description. Include no preamble for each section. Use markdown formatting.",\
    },\
    {\
      role: "user",\
      content: `Here is the section name: ${state.section.name} and description: ${state.section.description}`,\
    },\
  ]);

  // Write the updated section to completed sections
  return { completedSections: [section.content] };
}

async function synthesizer(state: typeof StateAnnotation.State) {
  // List of completed sections
  const completedSections = state.completedSections;

  // Format completed section to str to use as context for final sections
  const completedReportSections = completedSections.join("\n\n---\n\n");

  return { finalReport: completedReportSections };
}

// Conditional edge function to create llm_call workers that each write a section of the report
function assignWorkers(state: typeof StateAnnotation.State) {
  // Kick off section writing in parallel via Send() API
  return state.sections.map((section) =>
    new Send("llmCall", { section })
  );
}

// Build workflow
const orchestratorWorker = new StateGraph(StateAnnotation)
  .addNode("orchestrator", orchestrator)
  .addNode("llmCall", llmCall)
  .addNode("synthesizer", synthesizer)
  .addEdge("__start__", "orchestrator")
  .addConditionalEdges(
    "orchestrator",
    assignWorkers,
    ["llmCall"]
  )
  .addEdge("llmCall", "synthesizer")
  .addEdge("synthesizer", "__end__")
  .compile();

// Invoke
const state = await orchestratorWorker.invoke({
  topic: "Create a report on LLM scaling laws"
});
console.log(state.finalReport);

```

```md-code__content
import { z } from "zod";
import { task, entrypoint } from "@langchain/langgraph";

// Schema for structured output to use in planning
const sectionSchema = z.object({
  name: z.string().describe("Name for this section of the report."),
  description: z.string().describe(
    "Brief overview of the main topics and concepts to be covered in this section."
  ),
});

const sectionsSchema = z.object({
  sections: z.array(sectionSchema).describe("Sections of the report."),
});

// Augment the LLM with schema for structured output
const planner = llm.withStructuredOutput(sectionsSchema);

// Tasks
const orchestrator = task("orchestrator", async (topic: string) => {
  // Generate queries
  const reportSections = await planner.invoke([\
    { role: "system", content: "Generate a plan for the report." },\
    { role: "user", content: `Here is the report topic: ${topic}` },\
  ]);

  return reportSections.sections;
});

const llmCall = task("sectionWriter", async (section: z.infer<typeof sectionSchema>) => {
  // Generate section
  const result = await llm.invoke([\
    {\
      role: "system",\
      content: "Write a report section.",\
    },\
    {\
      role: "user",\
      content: `Here is the section name: ${section.name} and description: ${section.description}`,\
    },\
  ]);

  return result.content;
});

const synthesizer = task("synthesizer", async (completedSections: string[]) => {
  // Synthesize full report from sections
  return completedSections.join("\n\n---\n\n");
});

// Build workflow
const workflow = entrypoint(
  "orchestratorWorker",
  async (topic: string) => {
    const sections = await orchestrator(topic);
    const completedSections = await Promise.all(
      sections.map((section) => llmCall(section))
    );
    return synthesizer(completedSections);
  }
);

// Invoke
const stream = await workflow.stream("Create a report on LLM scaling laws", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

## Evaluator-optimizer

In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop:

> In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop.
>
> When to use this workflow: This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value. The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback. This is analogous to the iterative writing process a human writer might go through when producing a polished document.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/evaluator_optimizer.png

```md-code__content
import { z } from "zod";
import { Annotation, StateGraph } from "@langchain/langgraph";

// Graph state
const StateAnnotation = Annotation.Root({
  joke: Annotation<string>,
  topic: Annotation<string>,
  feedback: Annotation<string>,
  funnyOrNot: Annotation<string>,
});

// Schema for structured output to use in evaluation
const feedbackSchema = z.object({
  grade: z.enum(["funny", "not funny"]).describe(
    "Decide if the joke is funny or not."
  ),
  feedback: z.string().describe(
    "If the joke is not funny, provide feedback on how to improve it."
  ),
});

// Augment the LLM with schema for structured output
const evaluator = llm.withStructuredOutput(feedbackSchema);

// Nodes
async function llmCallGenerator(state: typeof StateAnnotation.State) {
  // LLM generates a joke
  let msg;
  if (state.feedback) {
    msg = await llm.invoke(
      `Write a joke about ${state.topic} but take into account the feedback: ${state.feedback}`
    );
  } else {
    msg = await llm.invoke(`Write a joke about ${state.topic}`);
  }
  return { joke: msg.content };
}

async function llmCallEvaluator(state: typeof StateAnnotation.State) {
  // LLM evaluates the joke
  const grade = await evaluator.invoke(`Grade the joke ${state.joke}`);
  return { funnyOrNot: grade.grade, feedback: grade.feedback };
}

// Conditional edge function to route back to joke generator or end based upon feedback from the evaluator
function routeJoke(state: typeof StateAnnotation.State) {
  // Route back to joke generator or end based upon feedback from the evaluator
  if (state.funnyOrNot === "funny") {
    return "Accepted";
  } else if (state.funnyOrNot === "not funny") {
    return "Rejected + Feedback";
  }
}

// Build workflow
const optimizerWorkflow = new StateGraph(StateAnnotation)
  .addNode("llmCallGenerator", llmCallGenerator)
  .addNode("llmCallEvaluator", llmCallEvaluator)
  .addEdge("__start__", "llmCallGenerator")
  .addEdge("llmCallGenerator", "llmCallEvaluator")
  .addConditionalEdges(
    "llmCallEvaluator",
    routeJoke,
    {
      // Name returned by routeJoke : Name of next node to visit
      "Accepted": "__end__",
      "Rejected + Feedback": "llmCallGenerator",
    }
  )
  .compile();

// Invoke
const state = await optimizerWorkflow.invoke({ topic: "Cats" });
console.log(state.joke);

```

```md-code__content
import { z } from "zod";
import { task, entrypoint } from "@langchain/langgraph";

// Schema for structured output to use in evaluation
const feedbackSchema = z.object({
  grade: z.enum(["funny", "not funny"]).describe(
    "Decide if the joke is funny or not."
  ),
  feedback: z.string().describe(
    "If the joke is not funny, provide feedback on how to improve it."
  ),
});

// Augment the LLM with schema for structured output
const evaluator = llm.withStructuredOutput(feedbackSchema);

// Tasks
const llmCallGenerator = task("jokeGenerator", async (params: {
  topic: string;
  feedback?: z.infer<typeof feedbackSchema>;
}) => {
  // LLM generates a joke
  const msg = params.feedback
    ? await llm.invoke(
        `Write a joke about ${params.topic} but take into account the feedback: ${params.feedback.feedback}`
      )
    : await llm.invoke(`Write a joke about ${params.topic}`);
  return msg.content;
});

const llmCallEvaluator = task("jokeEvaluator", async (joke: string) => {
  // LLM evaluates the joke
  return evaluator.invoke(`Grade the joke ${joke}`);
});

// Build workflow
const workflow = entrypoint(
  "optimizerWorkflow",
  async (topic: string) => {
    let feedback: z.infer<typeof feedbackSchema> | undefined;
    let joke: string;

    while (true) {
      joke = await llmCallGenerator({ topic, feedback });
      feedback = await llmCallEvaluator(joke);

      if (feedback.grade === "funny") {
        break;
      }
    }

    return joke;
  }
);

// Invoke
const stream = await workflow.stream("Cats", {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
  console.log("\n");
}

```

## Agent

Agents are typically implemented as an LLM performing actions (via tool-calling) based on environmental feedback in a loop. As noted in the [Anthropic blog](https://www.anthropic.com/research/building-effective-agents):

> Agents can handle sophisticated tasks, but their implementation is often straightforward. They are typically just LLMs using tools based on environmental feedback in a loop. It is therefore crucial to design toolsets and their documentation clearly and thoughtfully.
>
> When to use agents: Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path. The LLM will potentially operate for many turns, and you must have some level of trust in its decision-making. Agents' autonomy makes them ideal for scaling tasks in trusted environments.

https://langchain-ai.github.io/langgraphjs/tutorials/workflows/img/agent.png

```md-code__content
import { tool } from "@langchain/core/tools";
import { z } from "zod";

// Define tools
const multiply = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a * b;
  },
  {
    name: "multiply",
    description: "Multiply two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const add = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a + b;
  },
  {
    name: "add",
    description: "Add two numbers together",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

const divide = tool(
  async ({ a, b }: { a: number; b: number }) => {
    return a / b;
  },
  {
    name: "divide",
    description: "Divide two numbers",
    schema: z.object({
      a: z.number().describe("first number"),
      b: z.number().describe("second number"),
    }),
  }
);

// Augment the LLM with tools
const tools = [add, multiply, divide];
const toolsByName = Object.fromEntries(tools.map((tool) => [tool.name, tool]));
const llmWithTools = llm.bindTools(tools);

```

```md-code__content
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  SystemMessage,
  ToolMessage
} from "@langchain/core/messages";

// Nodes
async function llmCall(state: typeof MessagesAnnotation.State) {
  // LLM decides whether to call a tool or not
  const result = await llmWithTools.invoke([\
    {\
      role: "system",\
      content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."\
    },\
    ...state.messages\
  ]);

  return {
    messages: [result]
  };
}

const toolNode = new ToolNode(tools);

// Conditional edge function to route to the tool node or end
function shouldContinue(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages.at(-1);

  // If the LLM makes a tool call, then perform an action
  if (lastMessage?.tool_calls?.length) {
    return "Action";
  }
  // Otherwise, we stop (reply to the user)
  return "__end__";
}

// Build workflow
const agentBuilder = new StateGraph(MessagesAnnotation)
  .addNode("llmCall", llmCall)
  .addNode("tools", toolNode)
  // Add edges to connect nodes
  .addEdge("__start__", "llmCall")
  .addConditionalEdges(
    "llmCall",
    shouldContinue,
    {
      // Name returned by shouldContinue : Name of next node to visit
      "Action": "tools",
      "__end__": "__end__",
    }
  )
  .addEdge("tools", "llmCall")
  .compile();

// Invoke
const messages = [{\
  role: "user",\
  content: "Add 3 and 4."\
}];
const result = await agentBuilder.invoke({ messages });
console.log(result.messages);

```

```md-code__content
import { task, entrypoint, addMessages } from "@langchain/langgraph";
import { BaseMessageLike, ToolCall } from "@langchain/core/messages";

const callLlm = task("llmCall", async (messages: BaseMessageLike[]) => {
  // LLM decides whether to call a tool or not
  return llmWithTools.invoke([\
    {\
      role: "system",\
      content: "You are a helpful assistant tasked with performing arithmetic on a set of inputs."\
    },\
    ...messages\
  ]);
});

const callTool = task("toolCall", async (toolCall: ToolCall) => {
  // Performs the tool call
  const tool = toolsByName[toolCall.name];
  return tool.invoke(toolCall.args);
});

const agent = entrypoint(
  "agent",
  async (messages: BaseMessageLike[]) => {
    let llmResponse = await callLlm(messages);

    while (true) {
      if (!llmResponse.tool_calls?.length) {
        break;
      }

      // Execute tools
      const toolResults = await Promise.all(
        llmResponse.tool_calls.map((toolCall) => callTool(toolCall))
      );

      messages = addMessages(messages, [llmResponse, ...toolResults]);
      llmResponse = await callLlm(messages);
    }

    messages = addMessages(messages, [llmResponse]);
    return messages;
  }
);

// Invoke
const messages = [{\
  role: "user",\
  content: "Add 3 and 4."\
}];

const stream = await agent.stream([messages], {
  streamMode: "updates",
});

for await (const step of stream) {
  console.log(step);
}

```

#### Pre-built

LangGraph also provides a **pre-built method** for creating an agent as defined above (using the [`createReactAgent`](https://langchain-ai.github.io/langgraphjs/reference/functions/langgraph_prebuilt.createReactAgent.html) function):

[https://langchain-ai.github.io/langgraphjs/how-tos/create-react-agent/](https://langchain-ai.github.io/langgraphjs/how-tos/create-react-agent/)

```md-code__content
import { createReactAgent } from "@langchain/langgraph/prebuilt";

// Pass in:
// (1) an LLM instance
// (2) the tools list (which is used to create the tool node)
const prebuiltAgent = createReactAgent({
  llm: llmWithTools,
  tools,
});

// invoke
const result = await prebuiltAgent.invoke({
  messages: [\
    {\
      role: "user",\
      content: "Add 3 and 4.",\
    },\
  ],
});
console.log(result.messages);

```

</details>
