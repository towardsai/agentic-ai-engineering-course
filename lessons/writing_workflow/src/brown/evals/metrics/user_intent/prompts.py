"""Prompt templates and examples for the UserIntent evaluation metric.

This module contains the system prompt template and few-shot examples used
for evaluating how well generated articles follow guidelines and are anchored
in research across two dimensions (guideline adherence, research anchoring).
"""

from pathlib import Path

from brown.evals.metrics.base import CriterionScore, SectionCriteriaScores

from .types import (
    UserIntentArticleScores,
    UserIntentCriteriaScores,
    UserIntentExample,
    UserIntentMetricFewShotExamples,
)

SYSTEM_PROMPT = """You are an expert in Natural Language Processing (NLP) evaluation metrics, specifically trained to 
assess how well generated content follows specific guidelines and incorporates provided research material. 

The AI system you will evaluate takes as input an article guideline (<input>) and research material (<context>), and 
based on these inputs, another large language model (LLM) generates an article (<output>).

Your task is to evaluate whether the generated article (<output>) adheres to the given
user intent structured as an article guideline and is properly anchored in the provided research.

## INSTRUCTIONS 

1. You must analyze the given article guideline (<input>), research material (<context>), and 
generated article (<output>) to determine how well the output follows the input guidelines and utilizes the research.
2. The input, context, and output are in Markdown format.
3. Instead of comparing the outputs as a whole, you will divide the outputs into sections and compare each section 
individually. 
4. Since the article guideline reflects the user expectations and intent, you will always use it as the reference point 
to understand which sections from the generated article should be evaluated. To find the expected sections, look within 
the given article guideline for the keyword "Outline" or actual section titles often marked with H2 headers prefixed
with "Section". You will always use these as the anchor points, the expected sections, when making comparisons between
sections. 
5. When associating a section from the article guideline with one from the generated article, you will first look
for a matching or similar title. If there is no match based solely on the title, you will try to make associations based
on the content of the section. For example, if the expected section has 3 points on how RAG works and
the generated section has 3 points on how RAG works as well, then they are associated even if the titles are different.
If a required section mentioned in the guideline is missing from the generated article, you will assign a score of 
0 to all evaluation criteria.
6. Using the article guideline as an anchor, you will divide the generated article into sections and evaluate each section 
individually against all given criteria.
7. Sections are divided by H2 headers, marked as "##" in Markdown. You will use the headers as separators. 
Anything between two H2 headers constitutes a section. The only valid exception to this rule is the first section, 
the introduction, which sometimes appears between the title and the first H2 header. You will never include the title or 
subtitle as part of the first section.
8. The score can only be an integer value of 0 or 1. For each section, you will assign a binary integer score (0 or 1) based on 
two criteria:
   1. **Guideline Adherence**: For each expected section in the article guideline, you will evaluate whether the generated 
   section follows the specific section requirements outlined in the article guideline:
        - We expect a perfect match between the expected section and the generated section. Intuitively, you can
        think of the section guideline as a sketch, a compressed version of the generated section.
        - Less: If any topic from the expected article guideline is missing from the generated article, you will assign 
        a score of 0.
        - More: If the generated section has any additional topics that are not in the expected article guideline, 
        you will assign a score of 0.
        - Different Order: If the order of ideas from the expected article guideline is not followed in the 
        generated article, you will assign a score of 0.
        - If section constraints are provided, we are looking only for a rough approximation of the length. The exact
        section length criterias are present in the article guideline. Errors of ±100 units are acceptable. Units can
        be words, characters, or reading time. For example, if the expected section length is 100 words and the generated section length 
        is 190 words, you will assign a score of 1. But if the generated section is 230 words, as it exceeds the tolerance range,
        you will assign a score of 0.
   2. **Research Anchoring**: For each expected section in the article guideline, you will evaluate whether the generated 
   section content is based on or derived from the provided research:
        - We expect each section from the generated article to be generated entirely based on the ideas provided
        in the article guideline and research. Thus, you can consider both the context and article guideline as the 
        "research", the single source of truth.
        - If any idea from the generated section is not present in the research, you will assign a score of 0.
        - The generated section does not have to contain all the ideas from the research, just a subset of them.
        - If no research is explicitly referenced through citations, you will manually check if the generated section content
        it's based solely on the research. Missing explicit citations is valid. What it's critical is all the ideas to
        adhere to the research. Thus, if the generated section content is based solely on the research, while missing citations,
        you will assign a score of 1.
9. Along with the binary score, you will provide a brief and concise explanation containing the reasoning behind 
the score for each criterion. The score will be used to debug and monitor the evaluation process. Therefore, it is
important to provide thorough reasoning for the score. Since we provide binary scores, the reasoning should always 
contain what is good and what is problematic about the generated section, regardless of the score. For example, if the score 
is 0, the reasoning should also contain what is good about the generated section, such as "the generated section 
contains all the bulleted points from the expected section guideline," and what is problematic, such as "however, it contains an 
additional section on AI Evals that is not present in the guideline." Also, when generating the reasoning for the
research anchoring criterion, you will always mention if the topic comes from the article guideline, context, or both, while
supporting every single claim with evidence from the research. For example, the generated section is correctly anchored in the research,
while the fundamentals on RAG are based on the context, while the specific details on the RAG architecture are based on the article
guideline.
10. Important rules when evaluating:
   - Focus on substance, not superficial formatting differences
   - When comparing **media**, you only care about the placement and the caption of the media. 
    Since media can take many forms such as Mermaid diagrams, images, or URLs, you will completely ignore the 
    content of the media. Based on the section guideline, you will check whether the media is present in the 
    correct place. Based on the caption of the media, you will check whether it is properly anchored in the research.

## CHAIN OF THOUGHT

**Understanding Input:**
1.1. Read and understand the article guideline (<input>) to identify specific requirements, structure, 
content expectations, constraints, and most importantly the expected sections.
1.2. Read and understand the context (<context>) to identify available information, sources, and key findings.
1.3. Label the article guideline and context as the "research", the single source of truth.
1.4. Read and understand the generated article (<output>) and split it into sections using H2 headers as separators.
1.5. Connect the expected sections from the article guideline to the sections from the generated article.

**Section-by-Section Evaluation:**
2.1. For each section identified in the article guideline, locate its associated section in the generated article, and 
evaluate it against both criteria. If a section is found in the article guideline and is missing in the generated 
article, you will assign a score of 0 to all evaluation criteria.
2.2. Evaluate guideline adherence between each expected section from the article guideline and the associated section 
from the generated article.
2.3. Evaluate research anchoring by first selecting the sections to evaluate from the article guideline and then 
comparing the generated section to the research, found in the context and the associated section guideline.

**Assigning Scores:**
3.1. Based on each section expected from the article guideline, assign a binary score of either 0 or 1 
for all evaluation criteria listed in the instructions:
    - Score 1 if the section clearly follows the requirements detailed in the instructions.
    - Score 0 if it fails to follow the requirements detailed in the instructions.
3.2. Justify why you assigned a score of 0 or 1 with a brief explanation that highlights the reasoning behind the score
based on the given criterion.

## WHAT TO AVOID

- Do not provide scores using the generated output as the reference point to divide into sections. You must always 
use the article guideline as the reference point to divide into sections.
- Do not let other sections influence the score of a section. The score of each section must be determined in complete 
isolation from any other section.

## FEW-SHOT EXAMPLES

Here are few-shot examples demonstrating how to compute the scores for each section and criterion:
<few-shot-examples>
{examples}
</few-shot-examples>

## INPUTS

<input>
{input}
</input>

<context>
{context}
</context>

<output>
{output}
</output>

Think through your answer step by step, and provide the requested evaluation.
"""

EXAMPLES_DIR = Path(__file__).parent / "examples"
DEFAULT_FEW_SHOT_EXAMPLES = UserIntentMetricFewShotExamples(
    examples=[
        UserIntentExample.from_markdown(
            input_file=EXAMPLES_DIR / "04_structured_outputs" / "article_guideline.md",
            context_file=EXAMPLES_DIR / "04_structured_outputs" / "research.md",
            output_file=EXAMPLES_DIR / "04_structured_outputs" / "article_generated.md",
            scores=UserIntentArticleScores(
                sections=[
                    SectionCriteriaScores(
                        title="Introduction",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The introduction successfully follows the guideline requirements by referencing "
                                    "previous lessons ('AI engineering fundamentals including agent landscapes and context engineering') "
                                    "and transitioning to the current lesson topic as specified. It correctly positions "
                                    "structured outputs as bridging 'two worlds' between LLMs and applications, matching "
                                    "the guideline's bridge concept. The section meets the expected 150-word length constraint."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "While the section effectively establishes the conceptual framework for structured outputs, "
                                    "it fails to anchor its claims in the provided research material. The section discusses "
                                    "the bridge concept and benefits but doesn't reference any of the specific research "
                                    "findings about native API features, Pydantic validation, or JSON parsing best practices "
                                    "mentioned in the research sources."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Why Structured Outputs Are Critical",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section excellently covers all three theoretical benefits specified in the guideline: "
                                    "easy parsing and programmatic manipulation ('easy to parse and manipulate programmatically'), "
                                    "data quality checks with Pydantic ('using libraries like Pydantic adds data and type validation'), "
                                    "and reducing fragile parsing ('eliminate the messy task of parsing it with fragile regular "
                                    " expressions'). The section maintains the specified 300-word length and theoretical focus."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates key research findings, particularly about the benefits "
                                    "of structured outputs over manual parsing. While it doesn't cite specific sources, "
                                    "the content aligns well with the research findings about Pydantic's validation capabilities "
                                    "and the formal contract concept between LLMs and applications mentioned in the research sources."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Implementing Structured Outputs From Scratch Using JSON",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "While the section correctly addresses the JSON implementation topic and mentions "
                                    "prompting models to return JSON structures, it significantly lacks the detailed "
                                    "code examples and step-by-step implementation that the guideline specifically requires. "
                                    "The guideline calls for 'code examples to explain how to force the LLM to output data structures "
                                    "in JSON format,' but the generated section only provides high-level descriptions "
                                    "without actual implementation details or code snippets."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "The section appropriately mentions JSON formatting and parsing concepts, which aligns "
                                    "with the general research theme. However, it fails to reference any specific research "
                                    "findings about JSON parsing techniques, validation methods, or best practices detailed "
                                    "in the research sources. The content remains at a conceptual level without grounding "
                                    "in the provided research evidence."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Implementing Structured Outputs Using Pydantic",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section correctly covers Pydantic implementation and explains its validation benefits "
                                    "and fail-fast behavior as required by the guideline. However, it falls significantly short "
                                    "of the guideline's expectations in terms of depth and length. The guideline specifies "
                                    "600 words for this section, making it the most substantial part of the article, but "
                                    "the generated content provides only a brief overview without the detailed implementation "
                                    "examples and comprehensive coverage expected for this critical section."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates research findings about Pydantic's validation "
                                    "capabilities, particularly the 'field and type checking out-of-the-box' and immediate "
                                    "error raising mentioned in Source [2]. The fail-fast behavior description aligns well "
                                    "with the research findings about Pydantic's ability to catch data quality issues early "
                                    "and provide clear validation errors."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Implementing Structured Outputs Using Gemini",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section successfully covers Gemini's native structured outputs functionality "
                                    "as specified in the guideline and presents the key advantages of using native features "
                                    "over custom implementation ('easier, more accurate and cheaper'). It appropriately "
                                    "positions native API usage as the preferred approach and meets the expected 300-word "
                                    "length constraint while maintaining focus on the Gemini-Pydantic integration theme."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates research findings from Source [1] about native "
                                    "API advantages, specifically referencing that native structured outputs are 'easier, "
                                    "more accurate, and cheaper' which directly matches the research findings. The content "
                                    "about eliminating custom parsing logic aligns with the research about reliable data "
                                    "extraction and standardized JSON responses mentioned in the research sources."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Conclusion: Structured Outputs Are Everywhere",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The conclusion perfectly fulfills the guideline requirements by emphasizing "
                                    "the ubiquity of structured outputs ('used everywhere in AI engineering, regardless "
                                    "of what you are building') and providing a clear transition to future lessons "
                                    "('We will leverage structured outputs in almost all future lessons'). The section "
                                    "maintains the appropriate 150-word length and conclusive tone as specified."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "While the conclusion effectively wraps up the lesson's main theme, it remains "
                                    "entirely generic without referencing any specific research findings, best practices, "
                                    "or evidence from the provided research sources. The conclusion does convey the "
                                    "importance of structured outputs but lacks grounding in the research about validation "
                                    "benefits, API advantages, or implementation best practices detailed in the research material."
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
        UserIntentExample.from_markdown(
            input_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_guideline.md",
            context_file=EXAMPLES_DIR / "07_reasoning_planning" / "research.md",
            output_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_generated.md",
            scores=UserIntentArticleScores(
                sections=[
                    SectionCriteriaScores(
                        title="Introduction",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The introduction successfully follows the guideline structure by building on "
                                    "previous lessons ('you learned foundational components for building AI systems') "
                                    "and introducing the core concepts of planning and reasoning as specified. It correctly "
                                    "positions agentic reasoning as the major leap and mentions both ReAct and Plan-and-Execute "
                                    "strategies as required. The introduction appropriately sets up the article's focus "
                                    "without going into implementation details."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "While the introduction effectively establishes the conceptual foundation for agentic "
                                    "reasoning and mentions key strategies, it does not reference any specific research "
                                    "findings from the provided sources. The section discusses planning and reasoning concepts "
                                    "but fails to ground these claims in the research about agent architectures, planning "
                                    "benefits, or the differences between ReAct and Plan-and-Execute detailed in the research material."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="What a Non-Reasoning Model Does And Why It Fails on Complex Tasks",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section excellently follows the guideline by using the required Technical "
                                    "Research Assistant Agent example with a specific task ('comprehensive technical "
                                    "report on edge AI deployment'). It clearly demonstrates non-reasoning model behavior "
                                    "('treats the entire complex request as one big text generation problem') and explains "
                                    "the consequences including superficial outputs and lack of sub-goal breakdown. "
                                    "The section meets the 250-350 word length constraint."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates research findings from Source [5] about "
                                    "non-reasoning model failures, specifically referencing 'lack of mechanism to evaluate "
                                    "information quality' and 'don't break problems down into logical sub-goals,' which "
                                    "directly matches the research findings. The content about treating complex requests "
                                    "as text generation aligns well with the research about models lacking planning capabilities."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title='Teaching Models to "Think": Chain-of-Thought and Its Limits',
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section successfully explains Chain-of-Thought prompting as the solution "
                                    "to non-reasoning model failures and provides a concrete example using the research "
                                    "task as specified in the guideline. It correctly identifies the core limitation "
                                    "that 'the thinking and the answer as one uninterrupted stream' with 'no pause point "
                                    "to run tools or verify claims.' The section maintains the required 250-350 word length."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively draws from research findings in Source [3] about CoT "
                                    "limitations, specifically incorporating that 'the plan and answer appear in the same "
                                    "text stream' and there's 'no pause point to run tools, fetch fresh data, or verify "
                                    "claims,' which directly matches the research findings about CoT's constraints in "
                                    "agent contexts."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Separating Planning from Answering: Foundations of ReAct and Plan-and-Execute",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section perfectly presents the core idea of separating planning from action "
                                    "('explicitly separate the planning process from the action process') as required "
                                    "by the guideline. It correctly positions ReAct and Plan-and-Execute as the two "
                                    "fundamental patterns and provides clear, concise descriptions of their key differences. "
                                    "The section maintains the specified 150-250 word length and serves as an effective "
                                    "bridge between the problem identification and detailed implementation sections."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section accurately incorporates research findings from Source [1] about the "
                                    "fundamental differences between these architectures, specifically that ReAct 'interleaves "
                                    "reasoning and actions in a single loop' while Plan-and-Execute 'separates an LLM-powered "
                                    "planner from the execution runtime.' The descriptions align well with the research "
                                    "about ReAct's reasoning-action loop and Plan-and-Execute's separated phases."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="ReAct in Depth: The Loop of Thought, Action, and Observation",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "While the section correctly covers ReAct's core mechanism and provides a concrete "
                                    "example using the research assistant, it significantly falls short of the guideline's "
                                    "expectations. The guideline specifically requires a 'detailed evolving example,' "
                                    "a 'diagram' showing the loop structure, and 500-600 words, making this the most "
                                    "substantial section. The generated content provides only a brief overview with "
                                    "a simple example, missing the comprehensive depth and visual elements required."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates research findings about ReAct's mechanism, "
                                    "specifically the 'interleaving reasoning and actions with environment feedback' "
                                    "from Source [2], and mentions key advantages like 'high interpretability and natural "
                                    "error recovery' which align with the research about ReAct's benefits. The loop "
                                    "description matches the research findings about ReAct's iterative decision-making process."
                                ),
                            ),
                        ),
                    ),
                    SectionCriteriaScores(
                        title="Plan-and-Execute in Depth: Structure and Predictability",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section successfully covers the core Plan-and-Execute concept with upfront "
                                    "planning followed by systematic execution. It provides a comprehensive structured "
                                    "example with clear numbered steps (1-6) using the same research agent as required. "
                                    "The section includes the required diagram concept ('Planning phase output') and "
                                    "covers both pros ('upfront structure and improved efficiency for well-defined tasks') "
                                    "and cons ('less flexible for exploratory problems'). The length appears appropriate "
                                    "for the 450-550 word target."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section effectively incorporates research findings from Source [1] about "
                                    "Plan-and-Execute advantages, specifically that it 'can be faster because sub-tasks "
                                    "execute without consulting the larger model' and 'can be cheaper by delegating "
                                    "sub-tasks to smaller, domain-specific models.' The structured approach and efficiency "
                                    "benefits mentioned align well with the research about separated planning and execution phases."
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
    input: str,
    context: str,
    output: str,
    few_shot_examples: UserIntentMetricFewShotExamples,
) -> str:
    """Generate the evaluation prompt for the user intent metric.

    This function formats the system prompt with the provided input (article guideline),
    context (research), output (generated article), and few-shot examples to create a
    comprehensive prompt for the language model evaluation.

    Args:
        input: The article guideline content (input).
        context: The research content (context).
        output: The generated article content to be evaluated.
        few_shot_examples: An instance of UserIntentMetricFewShotExamples containing examples
            to guide the language model's evaluation.

    Returns:
        The complete formatted prompt string ready for LLM invocation.

    """
    # Path("user_intent_context.md").write_text(few_shot_examples.to_context())
    return SYSTEM_PROMPT.format(
        examples=few_shot_examples.to_context(),
        input=input,
        context=context,
        output=output,
    )
