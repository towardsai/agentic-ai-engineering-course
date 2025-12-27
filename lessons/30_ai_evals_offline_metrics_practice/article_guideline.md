
## Our Process Of Creating Training Few Shot Examples For the LLM Judges

I prepared the template for labeling 2 articles (Lesson 4 and Lesson 7) on the guideline adherence and research anchoring metrics which don't require a ground truth, but just:
the article guideline
the research file
the generated article
We will label both metrics as the User Intent
Data wise, I prepared everything you need in the GitHub: https://github.com/towardsai/course-agents-writing/tree/main/src/brown/evals/metrics/user_intent/examples
Also, I prepared a template in Google Sheets: https://docs.google.com/spreadsheets/d/1X0jmAvl8BueOwBATVNXzM2HJmFIEIHYYi7-GCYpj_kU/edit?usp=sharing
Under the following sheets:
Lesson 4 - User Intent Metric
Lesson 7 - User Intent Metric
IMPORTANT NOTE: The metrics are already pre-filled by the LLM to help you do this faster. But they are FAR FROM BEING CORRECT. The whole purpose of this labeling is to provide a few-shot example to better guide the LLM judge while doing the evaluation.  The score is binary: 0 or 1. So that's an easy shot. But the most important part of the labeling is the reason section where we always describe what's correct or wrong about the score. Even if we provide a score of 0, because it's binary, some parts of the sections will be correct, which we have to specify, along specifying while we provided a score of 0. Also, it's super important to be detailed here, but not tooo detailed. Like 1-2 sentences per reason should be enough (the current results from the LLM are pretty verbose, so we can shorten them up). These are super important, because these will help us during debugging and monitoring. My plan is to use this LLM judge to automate part of the rewing process. So if we do this right, we automate part of our boring part of the job.
To understand what exactly goes under the article guideline adherenence and research anchoring metrics, read the system prompt. You can find all the instructions there. If you think something is missing, PLEASE write that down as well so I can adapt the system prompt as well: https://github.com/towardsai/course-agents-writing/blob/main/src/brown/evals/metrics/user_intent/prompts.py
In the same Google Sheet, you can find more examples on how I labeled another metric, based on GT, in case you want to see more examples on how to write the Reason. You can find them under:
Lesson 4 - Follows GT Metric
Lesson 7 - Follows GT Metric
Also, what is CRITICAL, is to take the provided generated article from GitHub, and add more noise to it. The goal of these examples is to be as varried as possible. As the scoring is done per section, you can see each section as a sample. Thus, take few samples, and break them, such as:
add random ideas that are not anchored to the research
drop ideas present in the guideline
add more ideas that are not present in the guideline
add new sections
remove a section
You get the gist. The idea is to have as many use cases as possible in these examples.
Thus, take the generated articles that I prepared you in GitHub: https://github.com/towardsai/course-agents-writing/tree/main/src/brown/evals/metrics/user_intent/examples
Copy them somewhere you can later share with me, and tweak them as you see with. Afterward, I will take care of brining all your effort into the code.


 For this task and adding the noise and imperfections to articles - do we end up with just 1 new version of each of these articles that we will keep saved?
And then we update and improve the Binary Scores and Reasons based just on our updated article?
Do you want us to aim for roughly 50-50 0s and 1s scores after we tweak the article?

Yes, exactly. I will store the updated article in GitHub as it will be used directly by the LLM judge when constructing the prompt.
When adding noise to the article, it doesn't need to be perfect like a "production article". That's even better, as we can reflect the imperfections during the labeling.
Thus, adding noise with an LLM gets the job done.
1:44
I would say 70% 0s and 30% 1s as 0s usually contain more information
Also, when labeling something with a 0 is important in the reason to quickly enumerate what is correct (without too many details) and what it did wrong as it got the 0 score. Usually, it's enough to enumerate the first thing it got wrong.