"""Prompt templates and examples for the UserIntent evaluation metric.

This module contains the system prompt template and few-shot examples used
for evaluating how well generated articles follow guidelines and are anchored
in research across two dimensions (guideline adherence, research anchoring).
"""

from pathlib import Path

from brown.evals.metrics.base import CriterionScore

from .types import (
    UserIntentArticleScores,
    UserIntentCriteriaScores,
    UserIntentMetricFewShotExample,
    UserIntentMetricFewShotExamples,
    UserIntentSectionScores,
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
        UserIntentMetricFewShotExample.from_markdown(
            input_file=EXAMPLES_DIR / "04_structured_outputs" / "article_guideline.md",
            context_file=EXAMPLES_DIR / "04_structured_outputs" / "research.md",
            output_file=EXAMPLES_DIR / "04_structured_outputs" / "article_generated.md",
            scores=UserIntentArticleScores(
                sections=[
                    UserIntentSectionScores(
                        title="Introduction",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    'The section correctly introduces the "bridge" concept but violates three guidelines: '
                                    'it uses the first-person singular ("I remember...") against the point-of-view rule, '
                                    "adds a new anecdote about sentiment analysis and at ~250 words, it exceeds the "
                                    "150-word length constraint."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    'While the core "bridge" concept between the LLM (software 3.0) and Python '
                                    "(software 1.0) worlds' is anchored in the guideline, the section introduces a "
                                    'significant, un-anchored claim that mastering structured outputs is "the key to '
                                    'unlocking true Artificial General Intelligence (AGI)," which is not supported by '
                                    "the research."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Understanding why structured outputs are critical",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section correctly covers the required benefits but violates the "
                                    '"Different Order" rule by presenting them in a different sequence than the '
                                    'guideline. It also adds an unrequested benefit ("Improved Token Efficiency"), '
                                    'violating the "More" rule.'
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section is well-anchored. The benefits of parsing, Pydantic validation, "
                                    "avoiding fragile parsing and the GraphRAG/knowledge-graph use-case are all "
                                    "supported in the research and the added point on token efficiency are all "
                                    "directly supported by claims in the provided research material"
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Implementing structured outputs from scratch using JSON",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "This section adheres to the guideline. It follows the step list exactly: "
                                    "client/model setup, sample DOCUMENT, XML-tagged prompt, model call, raw output, "
                                    "helper to extract JSON, parsed output, and display. It meets the length and "
                                    "content expectations."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section is correctly anchored. All code examples are taken directly from "
                                    "the required lesson notebook, and the explanation of using XML tags for clarity "
                                    "is supported by prompt engineering best practices found in the research"
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Implementing structured outputs from scratch using Pydantic",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section correctly explains Pydantic's general benefits. However, it "
                                    "critically fails to follow the guideline by omitting the mandatory step of "
                                    "generating the JSON schema from the Pydantic model and injecting it into the "
                                    "prompt. It also adds an unrequested comment on YAML."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "The section introduces a factual contradiction with the provided guideline. "
                                    'It incorrectly states that type hints can be used directly "Starting with '
                                    'Python 10," whereas the guideline explicitly specifies this feature is '
                                    'available from "Python 11," failing the anchoring criterion.'
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Implementing structured outputs using Gemini and Pydantic",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section provides the correct code for using Gemini's native features. "
                                    "However, it fails the guideline by completely omitting the required explanation "
                                    'of the pros of this approach (that it is "easier, more accurate and cheaper").'
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "The section fails to incorporate key research findings. The provided sources "
                                    "clearly state that native API features enhance reliability and reduce "
                                    "development costs (Sources [37], [58]), but none of this required evidence "
                                    "is mentioned."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Conclusion: Structured Outputs Are Everywhere",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section correctly emphasizes the ubiquity of structured outputs and "
                                    "name-checks later topics as the guideline asks for. However, it does not "
                                    "provide a transition to the requested next Lesson 5."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section's claims are general and consistent with the course context. "
                                    "It does not introduce new factual claims that require specific anchoring, "
                                    "and therefore contains no information that contradicts the provided research "
                                    "material."
                                ),
                            ),
                        ),
                    ),
                ]
            ),
        ),
        UserIntentMetricFewShotExample.from_markdown(
            input_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_guideline.md",
            context_file=EXAMPLES_DIR / "07_reasoning_planning" / "research.md",
            output_file=EXAMPLES_DIR / "07_reasoning_planning" / "article_generated.md",
            scores=UserIntentArticleScores(
                sections=[
                    UserIntentSectionScores(
                        title="Introduction",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The introduction aligns with the guideline. It sets the stage by introducing "
                                    "planning and reasoning, explains the problem with standard LLMs, previews ReAct "
                                    "and Plan-and-Execute, and mentions the advanced capabilities, all while staying "
                                    "within the specified word count."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section is fully anchored in the research. All the high-level concepts it "
                                    "introduces—planning, reasoning, ReAct, Plan-and-Execute, and self-correction—are "
                                    "the central themes of the provided research material and are presented accurately."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="What a Non-Reasoning Model Does And Why It Fails on Complex Tasks",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The section meets all requirements from the outline, including the correct agent "
                                    "example, consequences, and transitions. The addition of context on the 'black box "
                                    "problem' is brief and does not detract from or replace any of the mandated points."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "While the core concepts are based on the guideline, the section's added discussion "
                                    "on the 'black box problem' and its relation to 'enterprise adoption' is not "
                                    "supported by any evidence within the provided research material."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title='Teaching Models to "Think": Chain-of-Thought and Its Limits',
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section uses the correct prompt example and most request topics but it fails "
                                    "to address the specific limitation required by the guideline, which is that the "
                                    "'plan and the answer appear in the same text'. Instead, it substitutes this with "
                                    "an unrequested point about computational cost."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section's description of Chain-of-Thought is correctly aligned with the "
                                    "foundational concepts presented in the research materials, such as the canonical "
                                    "paper Sourced."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Separating Planning from Answering: Foundations of ReAct and Plan-and-Execute",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The presents the core idea, details the benefits, correctly positions the two "
                                    "patterns against each other, and provides a clear transition to the next section. "
                                    "It misses an explanation for how it allows different handling for reasoning traces "
                                    "vs. final outputs."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The content is well-anchored in the provided research. The distinction between "
                                    "ReAct's interleaved loop and Plan-and-Execute's separated phases is directly "
                                    "supported by sources"
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="ReAct in Depth: The Loop of Thought and Observation",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section fails on multiple guidelines: it is missing the required Mermaid "
                                    "diagram, and the steps in the evolving example depart from what the guidelines "
                                    "map out and are presented in a logically incorrect order, violating the "
                                    "deliverable requirements."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "Most key claims are sourced but the section introduces a significant claim that "
                                    "ReAct has 'Python implementation difficulty,' which is an assertion not supported "
                                    "by any of the provided research sources."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Plan-and-Execute in Depth: Structuring the Path from Goal to Action",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "Includes mermaid diagram and maps out the plan and execute example as requested. "
                                    "The section fails to adhere to the content guideline for its 'Pros'. It incorrectly "
                                    "emphasizes 'fostering model creativity' instead of the specified advantages of "
                                    "efficiency and reliability for structured tasks."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=(
                                    "A source for inflexibility is found correctly but the primary advantage "
                                    "presented—creativity—is not supported by the research set. The provided sources "
                                    "frame the benefits of this pattern in terms of cost, reliability, and efficiency."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Where This Shows Up in Practice: Deep Research-Style Systems",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "This section, which is explicitly required in the lesson outline, is completely "
                                    "missing from the generated article."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=0,
                                reason=("As the section is not present in the article, it cannot be anchored in research."),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Modern Reasoning Models: Thinking vs. Answer Streams and Interleaved Thinking",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section is incomplete. It introduces the core concepts but omits the mandatory "
                                    "discussion on 'Implications for system design' and fails to provide the required "
                                    "transition to the next section. As a result the section is also well under the "
                                    "word count asked."
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The concepts that are discussed, such as 'thinking' streams and 'interleaved "
                                    "thinking', are correctly anchored in the research, with support from sources."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Advanced Agent Capabilities Enabled by Planning: Goal Decomposition and Self-Correction",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=0,
                                reason=(
                                    "The section includes key discussion on Goal decomposition and self-correction as "
                                    "asked but it violates the guideline by substituting the recurring 'conflicting "
                                    "adoption rates' example with an unrequested flight-booking scenario. It also omits "
                                    "the required discussion on 'Why patterns still matter"
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "Although the specific flight example/analogy was not requested, the key facts and "
                                    "high-level concepts of goal decomposition and self-correction are well-supported "
                                    "by the provided research material and include sources."
                                ),
                            ),
                        ),
                    ),
                    UserIntentSectionScores(
                        title="Conclusion",
                        scores=UserIntentCriteriaScores(
                            guideline_adherence=CriterionScore(
                                score=1,
                                reason=(
                                    "The conclusion adheres to the guideline. It effectively summarizes the lesson's "
                                    "key takeaways, recaps the ReAct and Plan-and-Execute patterns, and provides a "
                                    "clear, accurate preview of the next lessons (8, 9, and 10), all within the "
                                    "specified length"
                                ),
                            ),
                            research_anchoring=CriterionScore(
                                score=1,
                                reason=(
                                    "The section is fully anchored. The summary accurately recaps the core concepts "
                                    "supported by the research, and the preview of future lessons (8, 9, and 10) is "
                                    "taken directly from the article guideline."
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

    return SYSTEM_PROMPT.format(
        examples=few_shot_examples.to_context(),
        input=input,
        context=context,
        output=output,
    )
