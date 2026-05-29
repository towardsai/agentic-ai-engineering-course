"""Prompt templates and examples for the FollowsGT evaluation metric.

This module contains the system prompt template and few-shot examples used
for evaluating how well generated articles follow ground truth content across
multiple dimensions (content, flow, structure, mechanics).
"""

from pathlib import Path

from brown.evals.metrics.base import CriterionScore

from .types import (
    FollowsGTArticleScores,
    FollowsGTCriterionScores,
    FollowsGTMetricFewShotExample,
    FollowsGTMetricFewShotExamples,
    FollowsGTSectionScores,
)

SYSTEM_PROMPT = """ You are an expert in Natural Language Processing (NLP) evaluation metrics, specifically trained to 
assess answer quality in responses provided by large language models (LLMs). 

Your task is to evaluate the quality of a generated article by another LLM relative to 
an expected article output across multiple criteria: content, flow, structure, and mechanics.

## INSTRUCTIONS 

1. You must analyze the given expected article (<expected_output>) and generated article (<generated_output>) 
to determine the most relevant evaluation.
2. Since the generated output is an answer from another LLM, you must use the expected output as the reference 
standard to compare and evaluate the quality of the generated output.
3. Both the generated and expected outputs are in Markdown format.
4. Instead of comparing the outputs as a whole, you will divide the outputs into sections and compare each section 
individually. 
5. You will always use the expected output as the reference point to extract the sections of interest during the
evaluation. If there is no perfect match between the expected and generated section names, first try to infer
the corresponding section based on the similarity of section names and their respective content. If you conclude that 
the expected output contains a section that the generated output lacks, you will assign a score of 0 to the missing 
section in the generated output.
6. Sections are divided by H2 headers, marked as "##" in Markdown. You will use these headers as 
separators. Anything between two H2 headers constitutes a section. The only valid exception to this rule is the first 
section, the introduction, which sometimes appears between the title and the first H2 header. You will never include 
the title or subtitle as part of the first section.
7. When comparing each individual section of the expected output to the generated output, you will assign a binary 
score for multiple criteria: 0 or 1, where 0 indicates a non-match and 1 indicates a perfect match. Each criterion
is completely independent of the others, meaning that a score of 0 in one criterion does not affect the score of 
another criterion.
8. You must compute binary scores for each section based on the following criteria:
  1. **Content:** Evaluate whether the generated section covers the same content as the expected section:
    - Focus only on evaluating that the substance of the content is the same between the expected and generated section. By content, we 
    mean core subjects, topics, research, ideas, key points or arguments. For example, if both sections discuss the
    fundamentals of RAG, it's valid. But, if the expected section discusses advanced RAG topics, while the generated section
    discusses basic RAG topics, it's invalid.
    - In this criterion, we are not interested in the order, structure, layout, or any other aspect related to the flow of
    ideas, structure or mechanics. For example, if there are missing or additional ideas discussed it's still valid, as
    long as the substance of the content between the expected and generated section is the same.
  2. **Flow:** Evaluate whether the generated section follows the same order of ideas as the expected section, such as 
  the flow of:
    - Main ideas covered starting with the beginning, until the end of the section. With special emphasis on the beginning and end of the 
    section as they reflect the transition between the previous and next sections
    - Internal transitions between the main points within the section. We expect a smooth flow of ideas, 
    without any abrupt jumps or breaks.
    - Placement of notes, images, tables, code blocks, or any other media elements within the generated section, 
    relative to the expected section. 
    - We don't expect a perfect one on one match between the paragraphs and sentences between the expected and generated section.
    However, we expect the same ideas and concepts to be discussed in the same way, order, and storyline.
    - Assign a score of 0 if anything is missing from the generated section relative to the expected section, such as missing
    topics, ideas, or media.
    - Assign a score of 0 if anything is additionally added to the generated section relative to the expected section, such 
    as additional topics, ideas or media elements.
    - Accepted differences between the expected and generated section:
        - Mismatching media numbering is accepted. For example, if in the expected section we have a figure with the number 3 and
        in the generated section we have a figure with the number 4, it's valid. It will be invalid, only if the figure would be
        missing altogether.
        - Mismatching or missing emojis. For example, if the excepted section has a 💡 emoji, while the generated section has 
        a 🔑 emoji, it's valid. Also, if the emoji is missing altogether from the generated section, it's valid.
        - Mismatching source reference numbers. For example, if the expected section referes a source with the number 3, 
        while the generated section referes a source with the number 7, it's valid. It will be invalid, only if the generated
        section misses the source altogether. 
        - Different placement of the source in the generated section. For example, if the expected section has the source
        at the end of a sentence within the paragraph, while the generated section has it at the end of the paragraph, it's valid. 
        It will be invalid, only if the generated section would be missing altogether.
        - Mismatching number of source references. For example, if the expected section has 3 source references, while the 
        generated section has 2 source references, it's valid. It will be invalid, only if the generated section would have 
        misses the references altogether. For example if the expected section has 3 source references, while the generated 
        section has 0 source references, it's invalid.
        - Having reference numbers in the generated section, while having none in the expected section. For example, if the 
        expected section has 0 reference numbers, while the generated section has 3 reference number, it's valid. It will be 
        invalid, only the other way around, where the expected section has 3 reference numbers, while the generated section has 0.
  3. **Structure:** Evaluate whether the generated section follows the same structure as the expected section. By 
  structure, we mean:
    - H3/H4/H5/H6 sub-heading structure and formatting
    - Mismatches in headers formatting and presence. For example, if the expected section doesn't has a header,
    while the generated section has one, it's invalid. It's valid only if there is a one on one match between the headers
    formatting and presence.
    - Use of bulleted lists, numbered lists, callouts, notes, or other layout elements
    - Division of the section when guiding readers through code blocks or diagrams
    - Formatting of notes and code blocks
    - Use of bolding, italicizing, quotes, backticks, or other formatting elements
    - Formatting of citation references across sentences
    - Formatting of images, tables, and Mermaid diagrams and their corresponding citations. If they are missing from
    the generated section, we consider it valid for this criterion, as we are interested ONLY in formatting, which we
    cannot verify when elements are absent. For this criterion, missing elements from the generated sections are 
    considered valid. They are invalid only if present in both sections but formatted differently.
    - Number formatting conventions
9. Along with the binary scores, you will provide a brief and concise explanation containing the reasoning behind 
the score for each criterion. The score will be used to debug and monitor the evaluation process. Therefore, it is
important to provide thorough reasoning for the score. Since we provide binary scores, the reasoning should always 
contain what is good and what is problematic about the generated section, regardless of the score. For example, if the 
score is 0, the reasoning should also contain what is good about the generated section, such as "both sections 
follow the same flow of ideas," and what is problematic, such as "the generated section contains an additional 
paragraph on AI Evals that is not present in the expected section."
10. Important rules when comparing the content of sections:
    - Focus on substance, not superficial formatting differences
    - When comparing **media**, you only care about the placement of the media, not the content of the media. 
    Since media can take many forms such as Mermaid diagrams, tables, images, or URLs, you will completely ignore the 
    content of the media and only check whether the media is present in the correct place in the section, has 
    the appropriate citation, and proper numbering.


## CHAIN OF THOUGHT

**Understanding Input:**
1.1. Read, understand, and compare each section of the expected output and generated output.
1.2. Since we want to compute scores for each section of the expected output Markdown file, split the expected output 
into sections using the H2 headers as separators.

**Splitting into Sections:**
2.1. Using the expected output as the reference point, compare each section of the expected and generated 
outputs individually and assign a binary score of 0 or 1, where 0 indicates a mismatch and 1 indicates a perfect match.
2.2. Always use the expected output as the reference point to extract the sections of interest. 
2.3. When computing the score for an individual section, you will iterate through each section of the expected output, 
find its associated section in the generated output, and compute the score in isolation, ignoring all other sections.

**Assigning Scores to Each Section:**
3.1. Based on all sections of the expected output, assign a binary score of either 0 or 1 
for all evaluation criteria listed in the instructions:
    - **1:** The generated section matches the expected section perfectly on the given criterion.
    - **0:** The generated section does not match the expected section on the given criterion.
3.2. Justify why you assigned a score of 0 or 1 with a brief explanation that highlights the reasoning behind the score
based on the given criterion.

## WHAT TO AVOID

- Do not provide scores using the generated output as the reference point to divide into sections. You must always 
use the expected output as the reference point to divide into sections.
- Do not let other sections influence the score of a section. The score of each section must be determined in complete 
isolation from any other section.
- Do not overlap requirements between different criteria. For example, in the content criterion, as are not interested in the flow 
of ideas, if the ideas are in different order, or something is missing or additional, it's still valid. However, that is an important
aspect of the flow criterion, which will be invalid.

## FEW-SHOT EXAMPLES

Here are few-shot examples demonstrating how to compute the scores for each section and criterion:
<few-shot-examples>
{examples}
</few-shot-examples>

## INPUTS

<generated_output>
{output}
</generated_output>

<expected_output>
{expected_output}
</expected_output>

Think through your answer step by step, and provide the requested evaluation.
"""

EXAMPLES_DIR = Path(__file__).parent / "examples"
DEFAULT_FEW_SHOT_EXAMPLES = FollowsGTMetricFewShotExamples(
    examples=[
        FollowsGTMetricFewShotExample.from_markdown(
            output_file=EXAMPLES_DIR / "04_structured_outputs" / "article_generated.md",
            expected_output_file=EXAMPLES_DIR / "04_structured_outputs" / "article_ground_truth.md",
            scores=FollowsGTArticleScores(
                sections=[
                    FollowsGTSectionScores(
                        title="Introduction",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections cover the same core subjects and ideas, discussing the purpose "
                                    "of structured outputs as a bridge between LLMs and traditional applications."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section lacks the first sentence, which is used as a smooth "
                                    "transition into the article. Also, it misses the diagram present in the "
                                    "expected output, labeled as Figure 1."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason="Both sections use the same paragraph length patterns.",
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Why Structured Outputs Are Critical",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections cover the same reasons why structured outputs are critical, "
                                    "including ease of parsing, data validation with Pydantic, and common use "
                                    "cases. Still, the generated section has a section on GraphRAG, which is not "
                                    "related to the specific topic of the section."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Follows a similar logical flow, starting with the importance, detailing "
                                    "benefits, discussing use cases, and concluding with a diagram. Still, the "
                                    "generated section contains an additional paragraph on GraphRAG, which "
                                    "doesn't fit with the expected flow. Additionally, it omits the last sentence, "
                                    "which is necessary for a smooth transition to the next section."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections use the same paragraph length patterns and have the same usage "
                                    "pattern for backticks and citation references across sentences. Also, the figures "
                                    "and their corresponding citations use the same formatting rules."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Implementing Structured Outputs From Scratch Using JSON",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections provide a step-by-step guide on implementing structured outputs "
                                    "using JSON from scratch, covering client setup, document definition, prompt "
                                    "crafting, and parsing."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason="The generated section omits the Note callout box present in the expected output.",
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section incorrectly formats the JSON code block under point 4), "
                                    "where it misses the closing ```. Also, in the last section, where it outputs "
                                    "the final JSON structure, it doesn't enclose the JSON into Python backticks "
                                    "as expected: ```python <content> ```"
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Implementing Structured Outputs From Scratch Using Pydantic",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections accurately explain the benefits of Pydantic for structured outputs, "
                                    "demonstrate defining models, generating schemas, and validating responses, and "
                                    "compare it with other Python types. Still, the generated section uses different "
                                    "code examples, using a RedditThread Pydantic Python class instead of the expected "
                                    "DocumentMetadata class."
                                ),
                            ),
                            flow=CriterionScore(
                                score=1,
                                reason=(
                                    "Even if the sections use different code examples, from the point of view of the "
                                    "flow of ideas, both sections follow a similar logical flow, introducing Pydantic, "
                                    "demonstrating its implementation through steps with code, and concluding with a "
                                    "comparison to other data validation methods."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections maintain similar introductory paragraphs, numbered steps with code blocks, "
                                    "and a concluding comparison. The formatting of the Python code and JSON blocks is the same. "
                                    "Also, the use of backticks and formatting of citation references is the same."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Implementing Structured Outputs Using Gemini and Pydantic",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections accurately describe the native implementation of structured "
                                    "outputs using the Gemini API and Pydantic. Still, the generated section lacks "
                                    "some key code block examples (points 3 and 4 on calling the Gemini API), "
                                    "which are necessary to fully illustrate the concept."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections follow a similar logical flow, introducing native API support "
                                    "and then demonstrating its implementation through numbered steps with code "
                                    "and outputs. Still, the generated section misses the first sentence used to "
                                    "make the transition from the previous sentence, and also it misses points 3) "
                                    "and 4) from the code walkthrough numbered list."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "In both sections, the use of citation references and backticks is the same. "
                                    "Also, the structure of the introductory paragraph, division of code blocks "
                                    "and conclusion follow the same pattern. Still, the generated section uses a "
                                    "bulleted list to divide the code blocks instead of a numbered list as expected."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Structured Outputs Are Everywhere",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections serve as a conclusion, summarizing the importance of structured "
                                    "outputs as a fundamental pattern. Still, the generated section misses the last "
                                    "paragraph that presents how structured outputs fit in the course and the AI "
                                    "Engineering field, which is critical for the conclusion."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections follow a similar flow, summarizing the key takeaway. Still, the "
                                    "generated section misses the last paragraph on looking ahead to future lessons "
                                    "in the course."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections use the same paragraph length patterns. Still, the number "
                                    "formatting of the citation reference from the first paragraph misses the "
                                    "square brackets."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="References",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason="Both sections contain a list of references, similar in purpose.",
                            ),
                            flow=CriterionScore(
                                score=1,
                                reason="Both sections follow the same flow for referencing the sources, as a numbered list from 1 to n.",
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections use the same pattern to structure the references, as a bulleted "
                                    "list, where each element is structured as [<reference_number>] "
                                    "[<reference_name>](<reference_url>)"
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
        FollowsGTMetricFewShotExample.from_markdown(
            output_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_generated.md",
            expected_output_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_ground_truth.md",
            scores=FollowsGTArticleScores(
                sections=[
                    FollowsGTSectionScores(
                        title="Introduction",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Covers the same core subjects and ideas, discussing the limitations of standard "
                                    "LLMs and the need for planning and reasoning in AI agents."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections set the scene of the lesson, dicussing the 'why' behind the need for planning and "
                                    "reasoning in AI agents. However, the generated introduction omits the sentences that talk about "
                                    "the previous lessons and anchor the lesson within the course."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated output uses an H2 header 'Why Your Agent Needs to Think Before It Acts' "
                                    "as a title for the introduction, while the expected section doesn't."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="What a Non-Reasoning Model Does And Why It Fails on Complex Tasks",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Accurately covers the core subject of why non-reasoning models fail on complex "
                                    "tasks, using the same 'Technical Research Assistant Agent' example and discussing "
                                    "similar failure points."
                                ),
                            ),
                            flow=CriterionScore(
                                score=1,
                                reason=(
                                    "Follows a similar order of ideas, starting with the example, explaining the failure, "
                                    "and then discussing the need for reasoning."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections have similar paragraph length patterns and use of images and their "
                                    "corresponding citations."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title='Teaching Models to "Think": Chain-of-Thought and Its Limits',
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section begins with the expected topic on the Chain-of-Thought "
                                    "concept, but in the second paragraph, it shifts to discussing RAG, which is "
                                    "entirely different from the expected section topic."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections start by introducing CoT, but then the generated section talks about RAG, "
                                    "which is an entirely different topic, completely diverging from the expected section."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section uses the same citation strategy and number formatting, but the "
                                    "paragraphs are way longer than expected. Also, the generated section lacks the diagram and the "
                                    "'Note' callout box."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Separating Planning from Answering: Foundations of ReAct and Plan-and-Execute",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Accurately describes the core idea of separating planning from answering and "
                                    "introduces ReAct and Plan-and-Execute as the two dominant strategies."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections follow the same logical progression, starting with the core idea "
                                    "of separation and then introducing the two patterns. However, the last sentence "
                                    "from the generated section is very abrupt, being a poor transition to the next section."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section maintains a similar paragraph length, number formatting, and citation strategy. "
                                    "Still, it covers the ReAct and Plan-and-Execute topics within a paragraph instead "
                                    "of a bullet list with the names of the algorithms being bolded."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="ReAct in Depth: The Loop of Thought, Action, and Observation",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "The two sections provide the same detailed explanation of the ReAct framework, "
                                    "its iterative loop, and a step-by-step example using the research assistant agent."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections begin with the same flow, introducing ReAct, explaining its loop "
                                    "and presenting the diagram. The generated section has some additional reference numbers, which "
                                    "is correct. Still, the generated sections wrote the primary advantages and disadvantages of "
                                    "ReAct section before the hands-on example, instead of after it, as expected."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section employs a similar strategy to format the diagram's "
                                    "citation, references. However, in the expected section, the list is formatted as a "
                                    "numbered list, while in the generated section, it's formatted as a bulleted list. Also, the "
                                    "generated section added backquotes around the text from Action 1, 2, 3, and 4, while the "
                                    "expected section does not."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Plan-and-Execute in Depth: Structure and Predictability",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "Accurately explains the Plan-and-Execute pattern, its two phases "
                                    "(Planning and Execution), and its benefits for predictable tasks."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections follow a similar logical flow, introducing the pattern, "
                                    "explaining its efficiency and then detailing the planning and execution "
                                    "phases with an example. The issue is that the Plan-and-Execute pattern "
                                    "diagram was expected before digging into the **Planning Phase** section, and "
                                    "instead, it's placed within the numbered list of the **Planning Phase** section."
                                ),
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "The generated section employs a similar strategy to format the diagram's "
                                    "citation, number formatting, references and the bulleted list. Still, it formats the planning "
                                    "and execution phases as bolded text instead of as H3 headers."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Pros and Cons: ReAct vs. Plan-and-Execute",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason="The generated output completely omits this section.",
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason="The generated output completely misses this section.",
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason="The generated output completely omits this section.",
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Deep Research AI Assistant Systems",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections discuss how ReAct and Plan-and-Execute patterns are applied in "
                                    "real-world settings, but the expected output uses a deep research system as "
                                    "an example, while the generated one uses a financial assistant."
                                ),
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason=(
                                    "Even if the sections use different examples, the core storyline of the section is the same. "
                                    "However, the generated section completely misses the expected diagram, whereas the expected "
                                    "output places it at the end."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason=(
                                    "Both sections have similar paragraph length, number formatting, and citation patterns. As the diagram "
                                    "is missing from the generation section, we consider the citation valid."
                                ),
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title='Reasoning Models: How LLMs\' "Reasoning and Planning" are Being Internalized in LLMs',
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=0,
                                reason="The generated section is completely empty.",
                            ),
                            flow=CriterionScore(
                                score=0,
                                reason="The generated section is completely empty.",
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason="The generated section is completely empty.",
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="Conclusion",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason=(
                                    "In both sections, the conclusion summarizes the key takeaways of the article, "
                                    "including the importance of planning and reasoning, and the two foundational "
                                    "patterns (ReAct and Plan-and-Execute)."
                                ),
                            ),
                            flow=CriterionScore(
                                score=1,
                                reason=(
                                    "Follows a similar flow, reiterating the main points within the lesson "
                                    "and setting the scene for future lessons."
                                ),
                            ),
                            structure=CriterionScore(
                                score=1,
                                reason="Both sections have similar paragraph length, number formatting, and citation patterns.",
                            ),
                        ),
                    ),
                    FollowsGTSectionScores(
                        title="References",
                        scores=FollowsGTCriterionScores(
                            content=CriterionScore(
                                score=1,
                                reason="Both sections contain a list of citations, similar in purpose.",
                            ),
                            flow=CriterionScore(
                                score=1,
                                reason="Both sections follow the same flow for referencing the sources, as a numbered list from 1 to n.",
                            ),
                            structure=CriterionScore(
                                score=0,
                                reason=(
                                    "Both sections use a bulleted list to enumerate the citations, "
                                    "but the use of parentheses is not the same. The generated article outputs the references as "
                                    "`- [<number>] <reference_name>(<url>), ` instead of `- [[<number>]](<url>) <article_name>`."
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
    ]
)


def get_eval_prompt(
    output: str,
    expected_output: str,
    few_shot_examples: FollowsGTMetricFewShotExamples,
) -> str:
    """Generate the evaluation prompt for the follows_gt metric.

    This function formats the system prompt with the provided generated output, expected output,
    and few-shot examples to create a comprehensive prompt for the language model evaluation.

    Args:
        output: The generated article content to be evaluated.
        expected_output: The expected article content for comparison.
        few_shot_examples: An instance of FollowsGTMetricFewShotExamples containing examples
            to guide the language model's evaluation.

    Returns:
        The complete formatted prompt string ready for LLM invocation.

    """

    return SYSTEM_PROMPT.format(
        examples=few_shot_examples.to_context(),
        output=output,
        expected_output=expected_output,
    )
