1. The anatomy of a prompt section
2. How the few-shot-examples are actually your "training set"
3. The trade-off between isolating context (v1) + adding everything everything into a one-shot (v2). There has to be balance between "specialized LLM calls" and more generic ones. Experiment. 
    - Inspired from the transition from v1 to v2: As you start having more "specialized LLM calls or agents" you start having more LLM calls + your context get's fragmented. Thus, you have to find the right balance to avoid high latency, costs and overcomplicating the solution
3. Better explain the role of the profiles!!!!
4. Better explain the prompt of the writing agent!!!
5. Better explain our multimodality input: text + images!!!!
6. Code structure: 
    - We talked most about the entity layer
    - Note at the end how we had to glue all our enitity components to actually generate an end-to-end article (here is where the application layer kicks in)
    - How about infra layer related to the loading/rendering MD files?