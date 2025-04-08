question_evaluation_prompt = """
You are tasked with evaluating whether the provided question is correct and appropriate for the given context. Your evaluation should consider the following three parameters:

1. Relevance: How closely the question relates to the information provided in the context.
2. Groundness: Whether the context contains sufficient and clear information to adequately answer the question.
3. Standalone: Whether the question is complete and understandable on its own, without requiring additional context.

For each parameter, provide a rating on a scale from 1 to 5, where 5 indicates the highest level of quality.

Below are some examples (delimited by triple backticks):
#Example 1(Positive): 
## Context: "The Industrial Revolution, which took place between the 18th and 19th centuries, marked a turning point in history. This era saw the emergence of groundbreaking technologies such as the steam engine, mechanized cotton spinning, and the power loom. These innovations not only boosted manufacturing efficiency but also led to rapid urbanization and significant shifts in social structures as populations moved from rural areas to cities in search of work." 
## Question: "How did the technological innovations of the Industrial Revolution influence urbanization and social change?" 
## evaluation: "The question is directly tied to the context, addressing both technological and social impacts. The context provides ample information on manufacturing innovations and their societal implications, and the question is self-contained." 
## relevance_rating: 5 
## groundness_rating: 5 
## standalone_rating: 5

#Example 2(Positive): 
## Context: "In recent years, renewable energy has gained momentum as a viable alternative to fossil fuels. Detailed studies have shown that advancements in solar panels, wind turbines, and energy storage systems are reducing carbon emissions and promoting sustainable energy practices. However, challenges remain, particularly in integrating these technologies into existing power grids and ensuring reliable energy supply during peak demand periods." 
## Question: "What are the main challenges and benefits associated with the adoption of renewable energy technologies?" 
## evaluation: "The question is relevant as it touches on both the benefits and challenges mentioned in the context. Although the context covers several aspects of renewable energy, it could have included more specific examples, which slightly affects the groundness. The question is clear and stands on its own." 
## relevance_rating: 4 
## groundness_rating: 4 
## standalone_rating: 5

#Example 3 (Negative): 
## Context: "The rapid expansion of the internet over the past few decades has transformed communication, commerce, and information sharing. It has enabled global connectivity and revolutionized how people access data and interact in the digital age." 
## Question: "What are the major ingredients in traditional French cuisine?" 
## evaluation: "The question is not relevant to the provided context. The context discusses the expansion of the internet and its impacts, while the question is about traditional French cuisine, which has no connection to the information given. Therefore, the question scores very low on relevance and groundness, and its standalone quality is also poor because it does not make sense without additional context." 
## relevance_rating: 1 
## groundness_rating: 1 
## standalone_rating: 2


Now, here are the inputs:
Context: \"\"\"{context}\"\"\"
Question: \"\"\"{question}\"\"\"

Please return your evaluation in the following JSON format:
{{
  "evaluation": "insert your detailed evaluation here",
  "relevance_rating": "insert rating here (1-5)",
  "groundness_rating": "insert rating here (1-5)",
  "standalone_rating": "insert rating here (1-5)"
}}
Return Output
"""
