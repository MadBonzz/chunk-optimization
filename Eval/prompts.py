evaluation_prompt = """
You are an evaluator determining whether a given chunk of text contains enough information for generating a meaningful context based question for evaluating a RAG system.  
Your evaluation should follow these criteria:  

The evaluation should satisfy the rules below:  
{rules}

Follow these steps to evaluate the context:  
{guidelines}

Here are some examples (delimited by triple backticks):  
{examples}

Now here is the context (delimited by triple quotes):  

The provided chunk is : \"\"\"{context}\"\"\"  
Your answer should include a reasoning and a final score in the range 0 to 1. Do not include any other information.
Your answer should be in the format : 
reasoning: (Yes/No, explain why)  
final-Score: 
"""

rules = """ 
- The context must present a clear subject or main idea.  
- The context must include specific details, facts, or examples.  
- The context must contain claims, arguments, or explanations that could be questioned.  
- The context must have sufficient depth or complexity to allow meaningful questions to be generated."""

guidelines = """ 
1. **Analyze the subject clarity** – Does the context introduce a well-defined topic?  
2. **Check for depth** – Does the context contain enough details, reasoning, or claims?  
3. **Assess questionability** – Can a meaningful question be generated from the context?  
4. **Assign a rating (0-1 scale) and justify it** based on the criteria.  

### Scoring Scale:  
- **0.0 - 0.3**: The context is too vague or generic, making it impossible to form a meaningful question.  
- **0.4 - 0.6**: The context has some useful information but lacks depth or supporting details.  
- **0.7 - 1.0**: The context is rich with details and allows for meaningful question generation."""

examples = """
# Example 1 (Positive, High Score):  
## Context: "The Earth revolves around the Sun in an elliptical orbit, completing one revolution approximately every 365.25 days."  
## Reasoning: The context presents a well-defined topic (Earth's orbit), provides specific details (elliptical path, 365.25 days), and allows for meaningful questions (e.g., "Why is Earth's orbit elliptical?").  
## Evaluation: **0.9**  

# Example 2 (Negative, Low Score):  
## Context: "Apples are a type of fruit."  
## Reasoning: This statement is too general. It lacks details, claims, or explanations that could be questioned.  
## Evaluation: **0.2**  

# Example 3 (Positive, Medium-High Score):  
## Context: "Water boils at 100°C at sea level, but the boiling point decreases at higher altitudes due to lower atmospheric pressure."  
## Reasoning: The statement provides a clear subject (boiling point), specific details (temperature, altitude effects), and a scientific principle that can be questioned.  
## Evaluation: **0.8**  

# Example 4 (Negative, Mid-Low Score):  
## Context: "The Renaissance was a time of cultural rebirth."  
## Reasoning: While the topic is clear (Renaissance), the statement lacks specific details or supporting explanations. It does not provide enough depth to generate a meaningful question beyond general knowledge.  
## Evaluation: **0.4**  
"""

question_prompt = """
You will be provided with a chunk of text. You are tasked with creating context based questions using this chunk. The questions should require deep understanding of the text provided.
You may create one or multiple questions. Make sure all the questions are based on the chunk and the chunk have enough context for this question.
Do not create too many unnecessary questions, generate multiple questions only if they are relevant based on the chunk and can be answered based on it.

Follow these steps:
1. Read the context carefully and extract the key details, facts, and arguments.
2. Formulate questions that are not overly direct but encourages deeper analysis of the context. Avoid questions that result in very short answers.
3. All the questions must be fact-based and grounded solely in the information from the context.

Below are some examples (delimited by triple backticks):
#Example 1
Chunk:
The Black Death, which swept through Europe in the 14th century, dramatically altered the continent’s demographic, economic, and social structures. Estimates suggest that between 30% and 60% of Europe’s population perished as a result of the plague. Labor shortages due to the massive loss of life led to a rise in wages, a shift in the balance of power between peasants and landowners, and the eventual decline of the feudal system. Socially, the trauma of the pandemic fostered changes in religious practices, art, and attitudes toward life and death. Some historians argue that the Black Death accelerated the end of the Middle Ages and paved the way for the Renaissance.
Generated Question:
Question 1 : How did the Black Death act as a catalyst for social and economic transformation in Europe, and what long-term effects did it have on medieval society?

#Example 2
Chunk:
The construction of the Panama Canal in the early 20th century was a monumental engineering achievement that reshaped global trade routes. Prior to its completion, ships traveling between the Atlantic and Pacific Oceans were forced to navigate the lengthy and dangerous journey around the southern tip of South America. The new canal drastically reduced travel time and shipping costs, increasing economic activity and facilitating the rapid movement of goods and military vessels. However, the canal’s construction was also marred by significant political and ethical controversies, including the U.S. role in supporting Panama’s independence from Colombia and the exploitation of laborers from around the world, many of whom faced dangerous working conditions and high mortality rates.
Generated Questions:
Question 1 : How did the completion of the Panama Canal impact international trade and naval strategy in the 20th century?
Question 2 : What political and ethical controversies surrounded the construction of the Panama Canal, and how did they shape its legacy?

#Example 3
Chunk:
In recent decades, the widespread adoption of genetically modified (GM) crops has transformed agricultural practices around the world. Proponents argue that GM crops can increase yields, reduce the need for chemical pesticides, and help farmers adapt to changing environmental conditions. Critics, however, raise concerns about the long-term ecological effects, potential health risks, and the consolidation of seed patents among a few large corporations, which can undermine smallholder farmers’ autonomy. The debate over GM crops involves not only scientific and economic questions, but also ethical and social considerations regarding food security, environmental stewardship, and global equity.
Generated Question:
Question 1 : Discuss the various scientific, economic, and ethical dimensions of the debate over genetically modified crops, as outlined in the text.


Now, here is the provided chunk:

Chunk {context}

Always provide the generated quesiton/questions in the following format even if a single question is generated:
questions : List[questions],
"""

question_eval_prompt = """
You are tasked with evaluating whether the provided question is correct and appropriate for the given context. Your evaluation should consider the following three parameters:

Relevance: How closely the question relates to the information provided in the context.

Groundness: Whether the context contains sufficient and clear information to adequately answer the question.

Standalone: Whether the question is complete and understandable on its own, without requiring additional context.

For each parameter, provide a rating on a scale from 1 to 5, where 5 indicates the highest level of quality.

Below are some examples
#Example 1 (Positive): 
Context: "The Industrial Revolution, which took place between the 18th and 19th centuries, marked a turning point in history. This era saw the emergence of groundbreaking technologies such as the steam engine, mechanized cotton spinning, and the power loom. These innovations not only boosted manufacturing efficiency but also led to rapid urbanization and significant shifts in social structures as populations moved from rural areas to cities in search of work."
Question: "How did the technological innovations of the Industrial Revolution influence urbanization and social change?"
evaluation: "The question is directly tied to the context, addressing both technological and social impacts. The context provides ample information on manufacturing innovations and their societal implications, and the question is self-contained."
relevance_rating: 5
groundness_rating: 5
standalone_rating: 5

#Example 2 (Negative): 
Context: "The rapid expansion of the internet over the past few decades has transformed communication, commerce, and information sharing. It has enabled global connectivity and revolutionized how people access data and interact in the digital age."
Question: "What are the major ingredients in traditional French cuisine?"
evaluation: "The question is not relevant to the provided context. The context discusses the expansion of the internet and its impacts, while the question is about traditional French cuisine, which has no connection to the information given."
relevance_rating: 1
groundness_rating: 1
standalone_rating: 2
Now, here are the inputs:
Context: {context}
Question: {question}

Please return your evaluation in the following format:
evaluation: "insert your detailed evaluation here",
relevance_rating: "insert rating here (1-5)",
groundness_rating: "insert rating here (1-5)",
standalone_rating: "insert rating here (1-5)"
"""

qa_prompt = """
You are tasked with generating a detailed answer for the provided question using only the information given in the context.

Follow these steps:
1. Read the context carefully to understand all relevant details.
2. Read the question and ensure you understand what is being asked.
3. Construct a comprehensive answer that is entirely based on the context. Avoid using external knowledge or assumptions not supported by the context.
4. Ensure that the answer is detailed and fact-based, covering all aspects of the question with sufficient explanation.

Below are some examples (delimited by triple backticks):
# Example 1: 
## Question: "How did the technological advancements during the Industrial Revolution impact urbanization and social structures?"
## Context: "The Industrial Revolution, which took place between the 18th and 19th centuries, brought major technological advancements. Innovations such as the steam engine, mechanized textile production, and improved transportation systems revolutionized manufacturing. These changes led to rapid urbanization, as large numbers of people migrated from rural areas to cities for work, resulting in significant shifts in social structures and labor practices." 
## Generated Answer: "The technological advancements during the Industrial Revolution, including the steam engine and mechanized production, dramatically increased manufacturing efficiency. This efficiency spurred rapid urbanization by drawing large populations from rural areas into cities in search of work. Consequently, traditional social structures were transformed, and new labor practices emerged as a direct result of these urban migrations." 

# Example 2: 
## Question: "What benefits do renewable energy technologies offer, and what challenges must be addressed to ensure their successful implementation?" 
## Context: "Recent developments in renewable energy have underscored the potential of technologies such as solar panels and wind turbines to reduce carbon emissions. Despite these advances, challenges such as efficient energy storage and ensuring a stable power supply during periods of low production continue to hinder widespread adoption. Ongoing research and investment are seen as crucial to overcoming these hurdles and fully realizing the benefits of renewable energy." 
## Generated Answer: "Renewable energy technologies, like solar panels and wind turbines, offer significant benefits including the reduction of carbon emissions and a decreased reliance on fossil fuels. However, for these benefits to be fully realized, challenges such as efficient energy storage and the maintenance of a stable energy supply during low production periods must be addressed. Continued research and investment are key to overcoming these obstacles and ensuring the successful implementation of renewable energy solutions." 

# Example 3: 
## Question: "In what ways has digital technology transformed modern communication?"
## Context: "Digital technology has fundamentally reshaped communication in the modern era. The widespread adoption of the internet and the rise of social media platforms have enabled instant and global exchange of information. These advancements have not only transformed personal communication but also revolutionized business practices, political campaigning, and information dissemination on a global scale." 
## Generated Answer: "Digital technology has transformed modern communication by enabling instantaneous and global information exchange. The adoption of the internet and social media platforms has revolutionized how individuals interact, conduct business, and engage in political processes. These changes have led to more dynamic and interconnected modes of communication, significantly altering traditional communication practices."

Now, similary answer the given question using the context:

Question: \"\"\"{question}\"\"\"
Context: \"\"\"{context}\"\"\"

As the model output, I want a single string containing the answer to the question based on the context provided using the function calling schema tool provided to you.
"answer": "insert the detailed answer here"

Return Output
"""

context_eval = """
"### Instructions\n\n"
"You are a world class expert designed to evaluate the relevance score of a Context chunks in order to answer the Question.\n"
"You will be provided a question, the number of chunks retrieved and the retrieved chunks. You have to provide relevance rating for each chunk."
"Your task is to determine if the Context chunks contain proper information to answer the Question.\n"
"Do not rely on your previous knowledge about the Question.\n"
"Use only what is written in the Context and in the Question.\n"
"Follow the instructions below:\n"
"0. If the context does not contains any relevant information to answer the question, say 0.\n"
"1. If the context partially contains relevant information to answer the question, say 1.\n"
"2. If the context contains any relevant information to answer the question, say 2.\n"
"You must provide the relevance score from 0-2 for each chunk, nothing else.\nDo not explain.\n"
"Provide a rating for each of the context chunk provided. Rating of 1 chunk should not affect the rating of another chunk."

question : {question}
number_of_chunks : {n_chunks}
chunks : {context}

"Do not try to explain.\nProvide the output as a list of ratings.\nDo not give the output as a json or a dict. I want the answer as a python list only.\n"
"Analyzing Context and Question, the Relevance scores are "
ratings : []
"""

answer_eval = """
You are a RAG application response evaluator. Provided a given question, reference contexts and the answer, your task is to provide ratings for 
context precision and faithfulness. The instructions to evaluate for these metrics are as follows : 
Context Precision : 'Given question, answer and context verify if the context was useful in arriving at the given answer. Give rating from 1-10.
Faithfulness : Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement. Provide the rating from 1-10.

The provided question, reference contexts and answer are as follows :
question : {question}
context : {context}
answer : {context}

Finally provide the answer in the following format only : 
{{
    precision : rating,
    faithfulness : rating
}}
"""
